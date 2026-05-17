<div align="center">

# IsoCortex

**Self-hosted semantic search engine with AI-powered vector embeddings**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com)

[Features](#features) &bull;
[Quick Start](#-quick-start) &bull;
[API Docs](#-api-examples) &bull;
[Architecture](#-architecture) &bull;
[License](#license)

</div>

---

IsoCortex is a high-performance, self-hosted semantic search engine. Index your documents (PDF, DOCX, Markdown, code, and more), then search them by *meaning* — not just keywords. Built with a C++17 HNSW index, Python FastAPI backend, and a modern Next.js dashboard.

**No cloud. No API keys. No data leaving your machine.**

```bash
docker-compose up -d
# API at http://localhost:8900
# Web UI at http://localhost:3000
```

---

## Why IsoCortex?

Traditional search (grep, SQL LIKE, Ctrl+F) only matches exact words. If you search for *"security"* but the document says *"protection"*, it won't find it. IsoCortex uses **vector embeddings** to understand meaning — search *"how does authentication work?"* and find relevant passages even with completely different wording.

This is the core technology behind **RAG (Retrieval-Augmented Generation)** — the pattern that powers modern AI assistants like ChatGPT with your own data.

| Traditional Search | IsoCortex |
|---|---|
| Matches exact words | Matches meaning and intent |
| No understanding of context | Understands semantic similarity |
| Fails with synonyms/paraphrases | Handles synonyms naturally |
| Manual keyword tuning | Works out of the box |

---

## Features

### Core Engine
- **HNSW Index** — C++17 implementation with SIMD acceleration (AVX2, SSE4.1, ARM NEON) for sub-millisecond search
- **Three Distance Metrics** — Cosine similarity, L2 (Euclidean), and Inner Product
- **Sentence-Aware Chunking** — Intelligent document splitting (~120 words, 25-30% overlap, 256 token limit)
- **Incremental Indexing** — Add documents without rebuilding; timestamp + SHA-256 change detection
- **Soft Delete** — Mark vectors as deleted without corrupting the index

### Document Support
- **PDF** (via PyMuPDF), **DOCX** (via python-docx), **Markdown**, **Plain Text**
- **Code files** (.py, .js, .ts, .go, .rs, .java, .cpp, etc.)
- **Data formats** (CSV, JSON, TSV, YAML)
- **OCR** for images (via Tesseract, optional)
- **7 specialized extractors** with automatic format detection

### API & Auth
- **REST API** — Full CRUD for indexes, documents, search, jobs, and admin
- **JWT Authentication** — Access tokens + refresh tokens (SHA-256 hashed at rest)
- **Role-Based Access** — Admin and user roles with per-endpoint authorization
- **Rate Limiting** — Token bucket algorithm with configurable per-minute limits
- **SSE Job Streaming** — Real-time job progress updates via Server-Sent Events
- **Binary Serialization** — Atomic writes for index persistence (vectors.bin + metadata.json)

### Dashboard (Web UI)
- **Modern SPA** — Next.js 16 + TypeScript + Tailwind CSS
- **All-in-one management** — Create/manage indexes, browse documents, run searches
- **Admin panel** — User management, analytics, system stats, rate limit monitoring
- **Responsive design** — Works on desktop, tablet, and mobile
- **Static export** — Serves as plain HTML/CSS/JS — deploy anywhere

### DevOps
- **Docker** — Multi-stage builds for both API and Web UI (`docker-compose up`)
- **Cross-Platform** — Linux, macOS, Windows 10+ (with Docker Desktop)
- **Health Checks** — Built-in `/health` endpoint with component status
- **CLI Tool** — Full command-line interface for scripting and automation
- **SQLite Backend** — Zero external database dependencies

---

## Quick Start

### Option 1: Docker (Recommended)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop) (free)

```bash
# Clone the repository
git clone https://github.com/shaheerdev/isocortex.git
cd isocortex/project

# Start both API server and Web UI
docker-compose up -d

# Wait for services to be healthy (~30 seconds for first start)
docker-compose ps
```

Open your browser:
- **Web UI:** http://localhost:3000
- **API Docs:** http://localhost:8900/docs (Swagger UI)

On first launch, you'll be prompted to create an admin account.

### Option 2: Python (Manual)

**Prerequisites:** Python 3.11+, C++ compiler (GCC/Clang/MSVC)

```bash
# Install the engine
pip install .

# Start the API server
isocortex serve --host 0.0.0.0 --port 8900

# Or run with uvicorn directly
uvicorn isocortex.api:app --host 0.0.0.0 --port 8900
```

### Option 3: CLI

```bash
# Create an index
isocortex index create my-index --dimension 384 --metric cosine

# Scan and index a directory
isocortex index add my-index ./documents/

# Search
isocortex search my-index "how does authentication work?" --top-k 5

# List indexes
isocortex index list
```

---

## API Examples

### Authentication

```bash
# Create admin account (first run only)
curl -X POST http://localhost:8900/api/v1/auth/setup \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secretpassword","email":"admin@example.com"}'

# Login
curl -X POST http://localhost:8900/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secretpassword"}'
```

### Index Management

```bash
TOKEN="your_access_token"

# Create an index
curl -X POST http://localhost:8900/api/v1/indexes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"my-docs","dimension":384,"metric":"cosine","M":16,"ef_construction":128}'

# List indexes
curl http://localhost:8900/api/v1/indexes \
  -H "Authorization: Bearer $TOKEN"

# Get index details
curl http://localhost:8900/api/v1/indexes/my-docs \
  -H "Authorization: Bearer $TOKEN"
```

### Semantic Search

```bash
# Search an index
curl -X POST http://localhost:8900/api/v1/indexes/my-docs/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does the authentication flow work?",
    "top_k": 10,
    "include_metadata": true
  }'
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_1_chunk_3",
      "content": "The authentication flow begins with the client sending credentials...",
      "score": 0.94,
      "metadata": {
        "source": "api docs.md",
        "metric": "cosine",
        "chunk_index": 3
      },
      "chunk_index": 3,
      "source": "api docs.md"
    }
  ],
  "query": "How does the authentication flow work?",
  "total_results": 1,
  "latency_ms": 2.3
}
```

### Document Management

```bash
# List documents in an index (paginated)
curl "http://localhost:8900/api/v1/indexes/my-docs/documents?page=1&page_size=20" \
  -H "Authorization: Bearer $TOKEN"

# Delete a document
curl -X DELETE http://localhost:8900/api/v1/indexes/my-docs/documents/doc_1_chunk_3 \
  -H "Authorization: Bearer $TOKEN"
```

### Admin Operations

```bash
# System statistics
curl http://localhost:8900/api/v1/admin/stats \
  -H "Authorization: Bearer $TOKEN"

# Create a user
curl -X POST http://localhost:8900/api/v1/auth/users/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username":"dev","password":"devpassword","role":"user"}'

# Rate limit status
curl http://localhost:8900/api/v1/admin/rate-limits \
  -H "Authorization: Bearer $TOKEN"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web UI (Next.js)                      │
│              http://localhost:3000                       │
│         Static SPA — TypeScript + Tailwind               │
└─────────────────────┬───────────────────────────────────┘
                      │ REST API + JWT Auth
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  API Server (FastAPI)                    │
│              http://localhost:8900                       │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │  Auth    │ Indexes  │ Search   │  Jobs (SSE)      │  │
│  │  Routes  │  Routes  │  Routes  │  Scheduler       │  │
│  └────┬─────┴────┬─────┴────┬─────┴────────┬─────────┘  │
│       │          │          │               │            │
│  ┌────▼──────────▼──────────▼───────────────▼─────────┐  │
│  │              Middleware Layer                        │  │
│  │  Auth (JWT) │ Rate Limit │ Error Handler │ Req ID   │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                 │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │              Core Engine                             │  │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────────┐   │  │
│  │  │  Extractor  │→│ Chunker  │→│  Embedding     │   │  │
│  │  │  (7 formats)│  │ (~120w)  │  │  (MiniLM-L6)  │   │  │
│  │  └────────────┘  └──────────┘  └───────┬───────┘   │  │
│  │                                       │            │  │
│  │  ┌────────────────────────────────────▼──────────┐  │  │
│  │  │         HNSW Index (C++17 + SIMD)             │  │  │
│  │  │  AVX2 │ SSE4.1 │ NEON │ Scalar fallback      │  │  │
│  │  │  pybind11 Python bindings                     │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                        │                                 │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │         Storage Layer                              │  │
│  │  ~/.isocortex/{name}/                             │  │
│  │  ├── vectors.bin     (binary serialized vectors)    │  │
│  │  ├── metadata.json   (index config + doc metadata)  │  │
│  │  └── models/         (cached embedding model)       │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| Backend API | FastAPI, Uvicorn, Pydantic v2 |
| Search Engine | C++17 HNSW with SIMD (AVX2/SSE4.1/NEON) |
| Embeddings | sentence-transformers, all-MiniLM-L6-v2 (384-dim) |
| Document Parsing | PyMuPDF, python-docx, python-pptx, pytesseract |
| Authentication | JWT (PyJWT), bcrypt, SHA-256 token hashing |
| Database | SQLite (via aiosqlite) for users, tokens, analytics |
| CLI | Click-based with rich formatting |
| Containerization | Docker, docker-compose, multi-stage builds |

---

## Configuration

Configuration is managed via environment variables or the `~/.isocortex/config.json` file.

| Variable | Default | Description |
|---|---|---|
| `ISOCORTEX_HOST` | `0.0.0.0` | API server bind address |
| `ISOCORTEX_PORT` | `8900` | API server port |
| `ISOCORTEX_WORKERS` | `1` | Number of Uvicorn workers |
| `ISOCORTEX_DATA_DIR` | `~/.isocortex` | Data storage directory |
| `ISOCORTEX_LOG_LEVEL` | `INFO` | Logging level |
| `ISOCORTEX_SECRET_KEY` | (auto-generated) | JWT signing key |
| `ISOCORTEX_MAX_MEMORY_MB` | `4096` | Memory limit for index operations |
| `ISOCORTEX_EF_SEARCH` | `40` | Default HNSW search parameter |
| `ISOCORTEX_RATE_LIMIT_RPM` | `60` | Rate limit (requests per minute) |
| `ISOCORTEX_RATE_LIMIT_BURST` | `10` | Rate limit burst size |

---

## Development

### Project Structure

```
isocortex/
├── src/isocortex/           # Python source
│   ├── api/                 # FastAPI routes, middleware, schemas
│   ├── auth/                # JWT auth, user management
│   ├── config/              # Settings, configuration
│   ├── core/                # Core engine
│   │   ├── hnsw/            # C++17 HNSW index + pybind11 bindings
│   │   ├── embedding/       # sentence-transformers wrapper
│   │   ├── extractor/       # Document extractors (PDF, DOCX, etc.)
│   │   └── search/          # Search engine orchestrator
│   ├── engine/              # High-level operations
│   │   ├── indexing/        # Index manager, search utils
│   │   ├── ingestion/       # File scanner, chunker
│   │   ├── jobs/            # Background job scheduler
│   │   └── analytics/       # Usage analytics engine
│   └── storage/             # SQLite database, serializer
├── cli/                     # CLI tool (Click)
├── web/                     # Next.js dashboard SPA
│   ├── app/                 # Pages (dashboard, search, indexes, etc.)
│   ├── components/          # Layout, UI components
│   └── lib/                 # API client, auth, types, utils
├── tests/                   # Python tests (pytest)
├── Dockerfile               # API server Docker image
├── docker-compose.yml       # Full stack orchestration
├── pyproject.toml           # Python project config
└── LICENSE                  # MIT License
```

### Running Tests

```bash
# Python tests
pip install -e ".[dev]"
pytest tests/ -v

# Frontend tests
cd web
npm install
npm test

# C++ tests (requires g++ or clang++)
cd src/isocortex/core/hnsw
g++ -std=c++17 -O2 -o test_hnsw hnsw.cpp && ./test_hnsw
```

### Building Docker Images

```bash
# Build both images
docker-compose build

# Build individually
docker build -t isocortex-api .
docker build -t isocortex-web ./web
```

---

## HNSW Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `M` | 16 | 4–128 | Max connections per node (higher = better recall, more memory) |
| `ef_construction` | 128 | 50–2000 | Build-time search width (higher = better index quality, slower build) |
| `ef_search` | 40 | 10–500 | Query-time search width (higher = better recall, slower search) |
| `dimension` | 384 | 1–4096 | Must match your embedding model output |

**Rule of thumb:** For most use cases with all-MiniLM-L6-v2, the defaults work great. Increase `M` and `ef_construction` only if you need higher recall on very large datasets (1M+ vectors).

---

## Platform Support

| Platform | Status | Notes |
|---|---|---|
| Linux (x86-64) | ✅ Full | Recommended — native AVX2/SSE4.1 |
| Linux (ARM64) | ✅ Full | AWS Graviton, Raspberry Pi 4+ — NEON SIMD |
| macOS (Intel) | ✅ Full | SSE4.1 (AVX2 via Rosetta 2) |
| macOS (Apple Silicon) | ✅ Full | M1/M2/M3 — native NEON SIMD |
| Windows 10+ (x86-64) | ✅ Full | SSE4.1/AVX2, Docker Desktop recommended |

---

## Roadmap

- [x] Core HNSW search engine with SIMD
- [x] Multi-format document ingestion
- [x] REST API with JWT authentication
- [x] Web dashboard (Next.js SPA)
- [x] Docker deployment
- [x] CLI tool
- [x] Background job system with SSE streaming
- [x] Admin panel with analytics
- [x] Rate limiting
- [x] Account lockout
- [ ] **PyPI package** (`pip install isocortex`)
- [ ] **Helm chart** for Kubernetes deployment
- [ ] **Hybrid search** (keyword + semantic)
- [ ] **Multi-modal search** (image + text)
- [ ] **Custom embedding models** (OpenAI, Cohere, local GGUF)
- [ ] **Federated indexes** (search across multiple instances)
- [ ] **Webhook integrations** (Slack, Discord, email)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Free to use, modify, and distribute. No attribution required (but appreciated!).

---

<div align="center">

**Built by [Shaheer Qureshi](https://github.com/shaheerdev)**

If you find IsoCortex useful, consider giving it a ⭐ on GitHub!

</div>
