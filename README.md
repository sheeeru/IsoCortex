<div align="center">

# IsoCortex

**Self-hosted semantic search engine with AI-powered vector embeddings**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/shaheerdev/isocortex/ci.yml?branch=main)](https://github.com/shaheerdev/isocortex/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com)

</div>

---

IsoCortex is a high-performance, self-hosted semantic search engine. Index your documents (PDF, DOCX, Markdown, code, and more), then search them by *meaning* — not just keywords. Built with a C++17 HNSW index, Python FastAPI backend, and a modern Next.js dashboard.

**No cloud. No API keys. No data leaving your machine.**

```bash
git clone https://github.com/shaheerdev/isocortex.git
cd isocortex/project
docker-compose up -d
# Web UI at http://localhost:3000 — API at http://localhost:8900
```

**Desktop App?** IsoCortex also ships as a downloadable desktop application (like Discord/Spotify). Customers install it with one click — no terminal, no Docker, no developer tools needed. See [`project/desktop/README.md`](project/desktop/README.md).

---

## Repository Structure

This repository contains three main components:

```
isocortex/
├── project/          ← Core application (API, engine, dashboard, CLI, Docker, Desktop)
├── website/          ← Marketing landing page (Next.js)
└── docs/             ← Software Requirements Specification (LaTeX)
```

### [`project/`](project/) — Core Application

The main IsoCortex application: C++17 HNSW search engine, FastAPI backend, Next.js dashboard UI, CLI tool, Docker deployment, and test suites.

| Component | Technology |
|---|---|
| Search Engine | C++17 with SIMD (AVX2, SSE4.1, ARM NEON) |
| Backend API | FastAPI + Uvicorn + Pydantic v2 |
| Dashboard UI | Next.js 16 + React 19 + TypeScript + Tailwind |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| Auth | JWT + bcrypt + SHA-256 token hashing |
| Database | SQLite (users, tokens, analytics) |
| CLI | Click-based command-line interface |
| Containerization | Docker + docker-compose |
| Desktop App | Electron + PyInstaller (Windows/Mac/Linux installers) |

**Quick start:** See [`project/README.md`](project/README.md) for full installation, API examples, architecture diagram, configuration, and development guide.

```bash
cd project
docker-compose up -d          # Start API + Dashboard (Docker)
# OR build the desktop app:
bash scripts/build_all.sh     # Build downloadable installer
pytest tests/ -v              # Run tests
npm test --prefix web         # Run frontend tests
```

### [`website/`](website/) — Marketing Landing Page

The public-facing website for IsoCortex — deployed to your domain (e.g., isocortex.dev). Includes hero section, features, architecture overview, interactive demo, pricing tiers, API preview, and call-to-action.

| Section | Description |
|---|---|
| Hero | Tagline, one-liner install command, CTA buttons |
| Features | Grid of core capabilities with icons |
| Architecture | How it works — step-by-step pipeline |
| Demo | Interactive search preview |
| Pricing | Community / Pro / Enterprise tiers |
| API Preview | Live code examples |
| CTA | Download and GitHub links |

```bash
cd website
npm install
npm run dev     # Development server
npm run build   # Static export
```

Deploy to Vercel, Netlify, or any static host.

### [`docs/`](docs/) — SRS (Software Requirements Specification)

The full requirements document for IsoCortex, written in LaTeX. Defines functional requirements, API specifications, non-functional requirements, security constraints, deployment architecture, and monetization strategy.

| Section | Contents |
|---|---|
| FR-API | 23 REST API endpoints with request/response schemas |
| FR-IDX | Index CRUD, configuration, HNSW parameters |
| FR-DOC | Document ingestion, extraction, chunking |
| FR-SRC | Semantic search, batch search, scoring |
| NFR-01 to NFR-18 | Performance, security, scalability, reliability |
| Section 14 | Docker deployment, health checks, CI/CD |
| Section 15 | Pricing tiers, multi-tenancy, billing, compliance |

Compile to PDF:
```bash
cd docs
pdflatex SRS_v5.0.tex
```

---

## One-Command Quick Start

```bash
# 1. Clone
git clone https://github.com/shaheerdev/isocortex.git
cd isocortex/project

# 2. Start (Docker)
docker-compose up -d

# 3. Open
#    Dashboard:  http://localhost:3000
#    API:        http://localhost:8900
#    API Docs:   http://localhost:8900/docs
```

On first launch, create your admin account through the dashboard.

---

## Why IsoCortex?

| Traditional Search | IsoCortex |
|---|---|
| Matches exact words | Matches meaning and intent |
| Fails with synonyms | Handles "security" = "protection" |
| No context understanding | Understands semantic similarity |
| Manual keyword tuning | Works out of the box |

Perfect for **RAG (Retrieval-Augmented Generation)** — the pattern behind modern AI assistants with your own data.

---

## Platform Support

| Platform | Status |
|---|---|
| Linux (x86-64 / ARM64) | ✅ Full |
| macOS (Intel / Apple Silicon) | ✅ Full |
| Windows 10+ | ✅ Full (Docker Desktop) |

---

## License

[MIT License](LICENSE) — Free to use, modify, and distribute.

---

<div align="center">

**Built by [Shaheer Qureshi](https://github.com/shaheerdev)**

If you find IsoCortex useful, consider giving it a ⭐ on GitHub!

</div>
