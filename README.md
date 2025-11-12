# 🧠 Supreme Spork
*A local-first, containerized RAG engine for rapid, offline development.*

---

### 🚀 Overview
Supreme Spork is a **minimal yet powerful Retrieval-Augmented Generation (RAG)** starter template.  
It’s designed to run entirely **on your machine** — fast, private, and portable — so you can build semantic search or contextual LLM pipelines without cloud lock-in.

> ⚙️ Built for developers who value *clarity, control, and composability.*

---

### 🧩 Core Features
- 🧰 **FastAPI** backend with Swagger UI at `/docs`
- 💾 **Qdrant** vector database for fast semantic search
- 🧠 **SentenceTransformers** embeddings (`all-MiniLM-L6-v2` by default)
- 🐳 **Docker Compose** orchestration for instant local spin-up
- 🔁 Pluggable architecture for easy model or backend swaps

---

### 🧱 Project Structure
```
supreme-spork/
├── api/                   # FastAPI application
├── docker-compose.yml     # Local orchestration
├── requirements.txt       # Python dependencies (if included)
└── README.md
```

---

### ⚡ Quickstart

#### 1️⃣ Prerequisites
- Docker Desktop (or Docker Engine + Compose)
- Python 3.11+ (if you prefer running without Docker)

#### 2️⃣ Spin it up locally
```bash
# From the project root
docker compose up --build
```
Then open 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

#### 3️⃣ Or run manually
```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### ⚙️ Environment Variables
Create a `.env` file in the project root:

```
# Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Embeddings
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384

# API
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=info
```

---

### 📡 API Endpoints
| Method | Endpoint | Description |
|:-------|:----------|:-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/embed`  | Generate embeddings for text |
| `POST` | `/ingest` | Upsert document chunks |
| `POST` | `/query`  | Semantic search / RAG retrieval |
| `GET`  | `/docs`   | Swagger UI |

---

### 🧠 Extending the Engine
- Swap embeddings: use any `SentenceTransformers` or local HF model.
- Replace Qdrant with Milvus, Chroma, or pgvector.
- Connect to local LLMs (Ollama, Groq, vLLM, etc.).
- Add observability and analytics for retrieval quality.

---

### 🧪 Development Notes
```bash
# Run tests (if added)
pytest -q

# Format / lint (recommended)
black .
ruff check .
```

---

### 🌍 Roadmap
- [ ] `/chat` endpoint with streaming RAG
- [ ] Hybrid retrieval (BM25 + vector)
- [ ] Document loaders (PDF, HTML, CSV)
- [ ] Evaluation metrics for recall & precision
- [ ] Observability dashboard (Prometheus/Grafana)

---

### 📜 License
MIT.  
Please credit this template if you fork or extend it.

---

**Made with 🧠 + 🧰 + 💡 by Jerry Nguyen**  
> *Local-first AI. Fast. Private. Scalable.*
