# 📚 Uzbek RAG Agent — Hybrid RAG + Tool-calling AI Agent

Production-gilik (production-ready) AI yordamchi **O'zbek talabalari** uchun. Agent
foydalanuvchi yuklagan PDF hujjatlar (Kimyo, Fizika, Biologiya kitoblari) bilan
ishlay oladi, internetdan ma'lumot qidiradi, Notion'ga konspekt saqlaydi va
memory bilan suhbatni davom ettiradi.

> **LLM:** Ollama `llama3.2:3b` (lokal, GPU uchun mos, tool-calling qo'llab-quvvatlaydi).
> OpenAI ixtiyoriy fallback sifatida.

---

## ✨ Features

| # | Imkoniyat | Tavsif |
|---|-----------|--------|
| 1 | **Chat + Memory** | SQLite'da saqlanadigan conversation history |
| 2 | **PDF Upload** | Parse → chunk → embed → vector DB (avtomatik indeks yangilanishi) |
| 3 | **Hybrid RAG** | Dense (FAISS) + Sparse (BM25) → Reciprocal Rank Fusion → Cross-encoder reranker |
| 4 | **Tool Calling** | `web_search` (DuckDuckGo/Tavily), `calculator`, `save_to_notion` |
| 5 | **Agent Reasoning** | LangGraph workflow (classify → retrieve → agent ↔ tools → finalize) |
| 6 | **Memory** | chat history, previous questions/answers (SQLite) |
| 7 | **Configuration** | Barcha kalitlar `.env` orqali |
| 8 | **UI** | Streamlit dark-mode (sidebar + chat + sources + tool logs + chunks + memory) |
| 9 | **API** | FastAPI + Swagger (`/docs`) |
| 10 | **Tests** | pytest unit testlari (calculator, chunker, BM25) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (dark)                      │
│  Sidebar: upload / indexed docs / clear / settings           │
│  Main:    chat / sources / tool logs / chunks / memory       │
└────────────────────────┬────────────────────────────────────┘
                         │  services.chat() / index_pdf()
┌────────────────────────▼────────────────────────────────────┐
│                       FastAPI (app/main.py)                 │
│  /chat  /upload  /documents  /history  /health   (/docs)    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 LangGraph Agent (agent/graph.py)            │
│                                                             │
│   START → classify → [retrieve] → agent ↔ tools → finalize   │
└───────┬───────────────────────────────┬────────────────────┘
        │ RAG pipeline                  │ Tools
┌───────▼───────────────┐   ┌──────────▼──────────────────┐
│  rag/                 │   │  tools/                     │
│  chunker              │   │  calculator (safe AST eval) │
│  embeddings (HF)      │   │  web_search (DDG/Tavily)    │
│  retriever (BM25+FAISS)│  │  notion_tool (Notion API)   │
│  hybrid (RRF fusion)  │   └─────────────────────────────┘
│  reranker (bge-reranker)│
└───────────┬───────────┘
            │
┌───────────▼─────────────────────────────────────────────────┐
│  database/    FAISS index (disk) + SQLite (docs + memory)   │
└─────────────────────────────────────────────────────────────┘
```

### Hybrid RAG pipeline

```
User Question
   │
   ▼
Dense (FAISS) ──┐               ┌── BM25 (sparse)
   │            ▼               ▼
   │     Reciprocal Rank Fusion (weighted)
   │            │
   │            ▼
   │     deduplicate + merge
   │            │
   │            ▼
   │     Cross-encoder reranker (BAAI/bge-reranker)
   │            │
   │            ▼
   └──────► Top-K chunks → LLM → Answer (with sources)
```

---

## 📁 Project Structure

```
.
├── app/
│   ├── main.py            # FastAPI endpoints + Swagger
│   └── services.py        # orchestration: index_pdf / chat / list / clear
├── agent/
│   ├── graph.py            # LangGraph workflow build + run_agent
│   ├── nodes.py            # classify / retrieve / agent / tools / finalize
│   ├── state.py            # AgentState TypedDict
│   └── llm.py              # LLM factory (Ollama / OpenAI)
├── rag/
│   ├── chunker.py          # PyMuPDF parse + overlapping splitter
│   ├── embeddings.py       # HF sentence-transformers (singleton)
│   ├── retriever.py         # BM25 + Dense retrievers + RetrievalResult
│   ├── hybrid.py            # RRF fusion + rerank pipeline
│   └── reranker.py          # bge cross-encoder reranker
├── tools/
│   ├── notion_tool.py       # Notion page creation
│   ├── web_search.py        # DuckDuckGo / Tavily
│   └── calculator.py        # safe AST arithmetic
├── database/
│   ├── vector_store.py      # FAISS persistence + doc registry + BM25 sync
│   └── memory.py            # SQLAlchemy models + ConversationMemory
├── ui/
│   └── streamlit_app.py     # dark-mode UI
├── utils/
│   └── logging.py
├── tests/                   # pytest unit tests
├── config.py                # typed settings (.env-driven)
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Installation

### 1. Talab qilingan dasturlar

- **Python 3.10+** (3.12 tavsiya etiladi)
- **uv** (package manager) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Ollama** — `ollama pull llama3.2:3b`

### 2. Loyihani sozlash

```bash
git clone <repo-url> && cd Agent

# virtual environment
uv venv --python 3.10 .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# dependencies
uv pip install -r requirements.txt

# environment
cp .env.example .env
# .env ichida kerakli kalitlarni to'ldiring
```

### 3. Ollama modelini tayyorlash

```bash
ollama serve &
ollama pull llama3.2:3b
```

---

## 🔑 Environment Variables (`.env`)

| Variable | Default | Tavsif |
|----------|---------|--------|
| `MODEL_PROVIDER` | `ollama` | `ollama` yoki `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MODEL_NAME` | `llama3.2:3b` | Ollama model nomi |
| `OPENAI_API_KEY` | _(bo'sh)_ | OpenAI fallback uchun |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | multilingual embedding |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder reranker |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (kalitsiz) yoki `tavily` |
| `TAVILY_API_KEY` | _(bo'sh)_ | Tavily uchun |
| `NOTION_API_KEY` | _(bo'sh)_ | Notion integratsiya |
| `NOTION_PARENT_PAGE_ID` | _(bo'sh)_ | konspektlar saqlanadigan page |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | chunk parametrlari |
| `RETRIEVAL_TOP_K` | `10` | har retrieever'dan kandidatlar |
| `HYBRID_FINAL_K` | `6` | rerank'dan keyin final chunks |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | `0.5` / `0.5` | RRF vaznlari |
| `SQLITE_DB_PATH` | `./data/agent.db` | conversation + doc registry |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | FAISS index fayli |

---

## ▶️ Run locally

### API (FastAPI + Swagger)

```bash
uvicorn app.main:app --reload
# Swagger UI:  http://localhost:8000/docs
```

### UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
# http://localhost:8501
```

### Docker Compose (Ollama + API + UI)

```bash
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
# Ollama avtomatik ravishda llama3.2:3b'ni yuklab oladi
```

---

## 📸 Screenshots

> _(Loyiha ishga tushirilgandan keyin shu yerga screenshotlar joylanadi.)_

- `assets/sidebar.png` — sidebar (upload, indexed docs, settings)
- `assets/chat.png` — chat + sources + tool logs
- `assets/swagger.png` — FastAPI `/docs`

---

## 🧪 Tests

```bash
pytest -q
```

| Test | Nimani tekshiradi |
|------|-------------------|
| `test_calculator.py` | safe AST evaluator (arithmetic, functions, unsafe input rad etish) |
| `test_chunker.py` | text splitting + overlap + provenance |
| `test_hybrid.py` | BM25 retrieval + scoring + clear |

---

## 📡 API Reference (Swift/Postman uchun)

| Method | Path | Tavsif |
|--------|------|--------|
| `GET`  | `/health` | Liveness probe |
| `POST` | `/chat` | Agent'ga savol berish (`{query, session_id}`) |
| `POST` | `/upload` | PDF yuklash va indekslash |
| `GET`  | `/documents` | Indekslangan hujjatlar ro'yxati |
| `DELETE` | `/documents` | Bazani tozalash |
| `GET`  | `/history/{session_id}` | Suhbat tarixi |
| `DELETE` | `/history/{session_id}` | Suhbat tarixini tozalash |

---

## 🔮 Future Improvements / Bonus

- [ ] Google Calendar tool
- [ ] Gmail tool
- [ ] Multi-agent workflow (researcher + writer + reviewer)
- [ ] Redis cache (LLM javoblari uchun)
- [ ] PostgreSQL + pgvector (production vector DB)
- [ ] Boshqa reranking modellari (bge-reranker-large)
- [ ] Authentication (JWT)
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring (Prometheus / Grafana)
- [ ] Token usage tracking
- [ ] Asinxron FastAPI endpointlar (httpx + async tools)

---

## 📜 Coding Standards

- **Type hints** — har bir funksiyada
- **Docstrings** — har bir modul/klass/funksiyada
- **Logging** — `utils.logging.get_logger`
- **Exception handling** — hech bir node/endpoint exceptionsiz qolmaydi
- **SOLID** — har bir modul bitta mas'uliyat
- **Hardcode yo'q** — barcha tunablelar `config.py` / `.env`

---

## 📄 License

MIT — ta'lim maqsadida erkin ishlatish uchun.
