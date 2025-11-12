from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx, re, os, uuid, json

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# --- Local embedding model (CPU) ---
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# Config
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "interworks_docs")
# Keep the env name for transparency; embeddings are local MiniLM by default
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "local-all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GENERATION_MODEL", "llama3.1:8b")

api = FastAPI(title="InterWorks RAG API")
qdrant = QdrantClient(url=QDRANT_URL)

# =========================
# Globals (lazy-loaded)
# =========================
_local_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None

def get_local_embed_model() -> SentenceTransformer:
    global _local_embed_model
    if _local_embed_model is None:
        # Small, fast, reliable, 384-d
        _local_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _local_embed_model

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        # Lightweight cross-encoder for reranking
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# =========================
# Text utils
# =========================
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
WHITESPACE = re.compile(r"\s+")

def normalize(s: str) -> str:
    return WHITESPACE.sub(" ", s).strip()

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = normalize(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        slice_ = text[start:end]
        last_period = slice_.rfind(". ")
        if last_period > int(size * 0.6):
            end = start + last_period + 1
            slice_ = text[start:end]
        chunks.append(slice_)
        start = max(end - overlap, end)
    return [c for c in chunks if c]

# =========================
# Embeddings (LOCAL CPU)
# =========================
async def embed_texts(texts: List[str]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    model = get_local_embed_model()
    vecs = model.encode(texts, normalize_embeddings=True).tolist()
    if not vecs or len(vecs[0]) == 0:
        raise RuntimeError("Local embedding returned empty vectors.")
    return vecs

# =========================
# Generation (Ollama)
# =========================
async def ollama_generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

# =========================
# Qdrant helpers
# =========================
def ensure_collection(collection: str, dim: int):
    """
    Ensure collection exists with the specified vector size.
    Use Qdrant REST (canonical JSON) to avoid client/server schema mismatches.
    """
    try:
        info = qdrant.get_collection(collection_name=collection)
        vectors_cfg = getattr(info.config.params, "vectors", None)
        current_size = None
        if hasattr(vectors_cfg, "size"):
            current_size = int(vectors_cfg.size)
        elif isinstance(vectors_cfg, dict) and vectors_cfg:
            first = next(iter(vectors_cfg.values()))
            current_size = int(first.size)
        if current_size == int(dim):
            return
    except Exception:
        pass

    payload = {"vectors": {"size": int(dim), "distance": "Cosine"}}
    with httpx.Client(timeout=10) as client:
        r = client.put(f"{QDRANT_URL}/collections/{collection}", json=payload)
        if r.status_code >= 300:
            raise RuntimeError(
                f"Qdrant create failed: {r.status_code} {r.text}. "
                f"Payload sent: {json.dumps(payload)}"
            )

# =========================
# Schemas
# =========================
class IngestItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None  # e.g., {"source_id":"...", "title":"..."}

class IngestRequest(BaseModel):
    collection: Optional[str] = Field(default=DEFAULT_COLLECTION)
    items: List[IngestItem]

class QueryRequest(BaseModel):
    collection: Optional[str] = Field(default=DEFAULT_COLLECTION)
    query: str
    top_k: int = 5

# =========================
# Health & Debug
# =========================
@api.get("/health")
async def health():
    out: Dict[str, Any] = {"qdrant": "unknown", "ollama": "unknown"}
    try:
        qdrant.get_collections()
        out["qdrant"] = "ok"
    except Exception as e:
        out["qdrant"] = f"error:{e.__class__.__name__}"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            out["ollama"] = "ok" if r.status_code == 200 else f"bad_status:{r.status_code}"
    except Exception as e:
        out["ollama"] = f"error:{e.__class__.__name__}"
    return out

@api.get("/debug/embeddings")
async def debug_embeddings():
    try:
        model = get_local_embed_model()
        dim = model.get_sentence_embedding_dimension()
        name = getattr(model, "model_name_or_path", "local-embedder")
        return {"model": name, "dim": dim}
    except Exception as e:
        return {"error": e.__class__.__name__, "detail": str(e)}

# =========================
# Rerank helper
# =========================
def rerank(query: str, passages: List[str], top_k: int) -> List[int]:
    if not passages:
        return []
    model = get_reranker()
    pairs = [[query, p] for p in passages]
    scores = model.predict(pairs).tolist()
    idxs = list(range(len(passages)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    return idxs[:top_k]

# =========================
# Routes
# =========================
@api.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        collection = req.collection or DEFAULT_COLLECTION

        # chunk + simple de-dup
        docs, meta = [], []
        seen = set()
        for it in req.items:
            for ch in chunk_text(it.text):
                n = normalize(ch)
                if n in seen:
                    continue
                seen.add(n)
                docs.append(n)
                meta.append(it.metadata or {})

        if not docs:
            return {"status": "no_content", "collection": collection, "chunks": 0}

        # embed
        vectors = await embed_texts(docs)
        dim = len(vectors[0])
        if dim <= 0:
            return {"error": "invalid_embedding_dim", "detail": f"Embedding dim was {dim}."}
        ensure_collection(collection, dim=dim)

        # upsert with UUID and metadata
        points: List[PointStruct] = []
        for v, d, m in zip(vectors, docs, meta):
            payload = {"text": d}
            if m:
                payload.update(m)  # allows source_id/title/etc
            points.append(PointStruct(id=str(uuid.uuid4()), vector=v, payload=payload))

        qdrant.upsert(collection_name=collection, points=points)
        return {"status": "ok", "collection": collection, "chunks": len(points)}
    except httpx.HTTPStatusError as e:
        return {"error": "upstream_http_error", "status": e.response.status_code, "detail": e.response.text}
    except Exception as e:
        return {"error": e.__class__.__name__, "detail": str(e)}

@api.post("/query")
async def query(req: QueryRequest):
    try:
        collection = req.collection or DEFAULT_COLLECTION
        query_text = (req.query or "").strip()
        if not query_text:
            return {"error": "empty_query", "detail": "Query text is empty."}

        # embed query and ensure collection
        qvec = (await embed_texts([query_text]))[0]
        ensure_collection(collection, dim=len(qvec))

        # wider recall, then score floor
        raw_limit = max(req.top_k * 4, 20)
        res = qdrant.search(
            collection_name=collection,
            query_vector=qvec,
            limit=raw_limit,
            with_payload=True,
        )
        res = [h for h in res if (h.score or 0.0) > 0.1]

        # texts for rerank
        raw_texts = [h.payload.get("text", "") for h in res if h.payload]
        order = rerank(query_text, raw_texts, req.top_k)
        res = [res[i] for i in order]

        # build context items with metadata for citations
        items = []
        for h in res:
            p = h.payload or {}
            items.append({
                "text": p.get("text", ""),
                "title": p.get("title", "Untitled"),
                "source_id": p.get("source_id", "unknown"),
                "score": float(h.score) if h.score is not None else None,
            })

        # prompt
        if items:
            ctx_lines = []
            for it in items:
                ctx_lines.append(f"[{it['source_id']}] {it['title']}\n{it['text']}")
            context_block = "\n\n---\n".join(ctx_lines)
            prompt = (
                "Answer using ONLY the context. If something is missing, say what you need.\n\n"
                f"CONTEXT:\n{context_block}\n\n"
                f"QUESTION:\n{query_text}\n\n"
                "Return a concise answer and include bracketed source ids where relevant (e.g., [note-001])."
            )
        else:
            prompt = (
                "Answer the user's question as best you can. "
                "If the information is missing, say what you'd need.\n\n"
                f"QUESTION:\n{query_text}"
            )

        answer = await ollama_generate(prompt)

        return {
            "answer": answer,
            "matches": [
                {
                    "score": it["score"],
                    "source_id": it["source_id"],
                    "title": it["title"],
                    "text": it["text"],
                }
                for it in items
            ],
        }
    except httpx.HTTPStatusError as e:
        return {"error": "upstream_http_error", "status": e.response.status_code, "detail": e.response.text}
    except Exception as e:
        return {"error": e.__class__.__name__, "detail": str(e)}