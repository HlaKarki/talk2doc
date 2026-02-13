# Talk2Doc

> ### **Disclaimer**
> 
> This project is not polished and is currently buggy in multiple areas.
> Expect rough edges, incomplete flows, and behavior changes at this stage.

Tech

- React/TanStack Start frontend
- Python/FastAPI backend
- Multi-agent `/chat` routing (document, knowledge graph, data analysis, general)
- Cloudflare deployment (frontend Worker + backend Cloudflare Containers)

## What Works Today

- Upload PDF documents and ask grounded questions about selected docs.
- Upload CSV/XLS/XLSX datasets and run analysis/modeling prompts through `/chat`.
- Trigger knowledge-graph extraction from `/chat` when a document is selected.
- Show memory, knowledge graph, models, and recent conversations in the frontend.
- Render dataset visualizations returned by backend analysis.

## Repo Layout

```text
talk2doc/
├── backend/                       # FastAPI backend
│   ├── cloudflare/container-worker.js
│   ├── wrangler.containers.jsonc
│   └── CLOUDFLARE_CONTAINERS.md
├── src/                           # Frontend app
├── wrangler.jsonc                 # Frontend Cloudflare Worker config
└── ...
```

## Local Development

### Prerequisites

- Bun (for frontend and deploy scripts)
- Python 3.11+ (project currently uses 3.13 in Docker)
- Docker (required for Cloudflare Containers deploy)

### 1) Install dependencies

```bash
bun install
cd backend
uv sync
cd ..
```

### 2) Backend environment

Create `backend/.env` from `backend/.env.example` and fill real values:

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `r2_account_id`
- `r2_access_key_id`
- `r2_secret_access_key`
- `r2_bucket_name`

### 3) Run backend locally

```bash
cd backend
uv run dev
```

Backend default URL: `http://localhost:8000`

### 4) Run frontend locally

```bash
bun run dev
```

Frontend default URL: `http://localhost:3000`

## API Docs

- Local Swagger UI: `http://localhost:8000/docs`
- Local OpenAPI JSON: `http://localhost:8000/openapi.json`

Deployed backend Swagger UI:

- `https://talk2doc-backend.hla-htuncs.workers.dev/docs`

## Deployment

### Frontend (Cloudflare Worker)

```bash
npm run deploy
```

Current deployed frontend URL:

- `https://tanstack-start-app.hla-htuncs.workers.dev`

### Backend (Cloudflare Containers)

See `backend/CLOUDFLARE_CONTAINERS.md` for full steps.

Minimum flow:

```bash
npx wrangler login
npx wrangler secret put DATABASE_URL -c backend/wrangler.containers.jsonc
npx wrangler secret put OPENAI_API_KEY -c backend/wrangler.containers.jsonc
npx wrangler secret put R2_ACCOUNT_ID -c backend/wrangler.containers.jsonc
npx wrangler secret put R2_ACCESS_KEY_ID -c backend/wrangler.containers.jsonc
npx wrangler secret put R2_SECRET_ACCESS_KEY -c backend/wrangler.containers.jsonc
npm run deploy:backend
```

Current deployed backend URL:

- `https://talk2doc-backend.hla-htuncs.workers.dev`

## Frontend -> Backend URL Resolution

Frontend API base URL resolves in this order:

1. `VITE_BACKEND_URL`
2. `SERVER_URL`
3. Auto-derived `workers.dev` backend URL in production
4. `http://localhost:8000` fallback

Implementation: `src/routes/index.tsx`.

## Known Limitations

- Cloudflare Containers cold starts can intermittently fail the first request right after deploy/idle; retry usually succeeds.
- CORS is currently configured permissively (`CORS_ORIGINS="*"` in backend container vars).
- No user authentication/authorization is implemented yet.

## Testing

Frontend:

```bash
bun run test
```

Backend includes integration-style test scripts under `backend/tests/` that are intended to run against a configured backend environment.

Hey Vishal Bhaiya,

Quick update - I completed all three courses you recommended. To make sure the concepts stuck, I built a project called
Talk2Doc that implements most of what I learned.

What I implemented:

- Multi-agent orchestration with LangGraph - Router agent classifies user intent, routes to specialized agents (document,
  knowledge graph, data scientist), then a synthesizer agent refines the final response
- 4-layer memory system - Short-term (conversation buffer), long-term (user facts/preferences), semantic (vector
  similarity search on past interactions), and graph-based memory (concept relationships)
- Knowledge graph extraction - Using LangChain's LLMGraphTransformer to extract entities/relationships from documents,
  with NetworkX for traversal and pathfinding
- RAG pipeline - Chunking, embeddings, vector search, reranking, and streaming responses
- Data scientist agent - Dataset profiling, NLP analysis (sentiment/keywords with TF-IDF), supervised learning (Random
  Forest, Logistic Regression, SVM), unsupervised learning (K-Means, DBSCAN, Hierarchical clustering), and auto-generated
  visualizations

The stack is FastAPI + React, deployed to Cloudflare (Workers for frontend, Containers for backend).

It's still a buggy MVP - definitely some rough edges - but the core flows work. Here are the links if you want to poke
around:
- Backend API (Swagger): link1
- Live site: link2

Thanks again for the course recommendations - they gave me exactly the foundation I needed to build something like this.

Best,
Kshap
