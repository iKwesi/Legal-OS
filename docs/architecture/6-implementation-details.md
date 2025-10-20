# 6. Implementation Details

## 6.1 Frontend (Next.js + Shadcn/ui)

* **Framework/Setup:** Next.js 14+ (App Router), TypeScript, Tailwind CSS, Shadcn/ui (components installed via CLI).
* **Directory Structure:** Standard Next.js App Router structure with `components/app/` (custom), `components/ui/` (Shadcn), `lib/api.ts` (fetching).
    ```plaintext
    frontend/
    ├── app/                     # Next.js App Router
    │   ├── (main)/              # Main app layout group
    │   │   ├── analysis/[session_id]/ # Main results/chat/checklist hub
    │   │   │   ├── page.tsx       # Default view (e.g., Summary)
    │   │   │   ├── chat/page.tsx  # Chat screen
    │   │   │   └── checklist/page.tsx # Checklist screen
    │   │   └── layout.tsx       # Layout for the main app (post-upload)
    │   ├── page.tsx             # Root page (Document Upload Screen)
    │   └── layout.tsx           # Root layout (includes theme providers)
    ├── components/
    │   ├── app/                 # Custom composite components (e.g., UploadForm, ResultsDisplay, ChatInterface)
    │   └── ui/                  # Shadcn/ui components (e.g., Button, Card, Input)
    ├── lib/
    │   ├── utils.ts             # Tailwind merging, etc.
    │   └── api.ts               # Client-side functions for calling the FastAPI backend
    ├── styles/
    │   └── globals.css          # Tailwind base styles
    └── tailwind.config.ts       # Tailwind configuration
    ```
* **State Management:** Zustand / React Context.
* **Data Fetching:** `fetch`/`axios` client-side, or RSCs.

## 6.2 Backend (FastAPI + LangChain)

* **Framework/Setup:** FastAPI, Python, LangChain/LangGraph, `uv` dependency management.
* **Directory Structure:** Modular structure with `app/api/`, `app/agents/`, `app/core/`, `app/models/`, `app/orchestration/`, `app/rag/`, `app/scripts/`, `tests/`.
    ```plaintext
    backend/
    ├── app/
    │   ├── api/                 # FastAPI routes/endpoints
    │   │   └── v1/
    │   │       ├── endpoints/   # Endpoint files (upload, analyze, chat, etc.)
    │   │       └── router.py    # Main API router
    │   ├── agents/              # Core logic for each of the 6 agents
    │   │   ├── ingestion.py
    │   │   ├── clause_extraction.py
    │   │   ├── risk_scoring.py
    │   │   ├── summary.py
    │   │   ├── provenance.py
    │   │   ├── checklist.py
    │   │   └── tools/           # Specific tools for agents (e.g., vector search)
    │   ├── core/                # Config, logging, shared services
    │   │   └── config.py        # Environment variables
    │   ├── models/              # Pydantic data models (as defined in Sec 4)
    │   ├── orchestration/       # The ReAct/LangGraph orchestrator (supervisor agent) logic
    │   │   └── pipeline.py      # Main graph definition
    │   ├── rag/                 # RAG pipeline components
    │   │   ├── chunking.py      # Chunking strategies (Naive, Semantic)
    │   │   ├── retrievers.py    # Retriever implementations (BM25, Rerank, etc.)
    │   │   └── vector_store.py  # Qdrant client/logic
    │   ├── scripts/             # Utility scripts
    │   │   ├── generate_sgd.py
    │   │   └── evaluate_rag.py
    │   └── main.py              # FastAPI application entry point
    ├── tests/                   # Pytest unit/integration tests
    ├── requirements.txt         # Managed by uv
    └── Dockerfile               # Uses uv to install dependencies
    ```
* **Agent Tools:** Each agent in `app/agents/` will have access to a defined set of tools (e.g., vector search for Clause Agent) needed for its task. The orchestrator (supervisor agent) routes tasks, agents use their tools. Specific tools for each agent will be defined during implementation based on PRD requirements.

## 6.3 RAG Pipeline (Swappable)

* **Location:** `backend/rag/`.
* **Components:** Implementations for Naive/Semantic chunking; Naive, BM25, Multi-Query, Parent-Doc, Rerank (Cohere), Ensemble retrievers.
* **Configuration:** Pipeline selectable via config.
* **Evaluation Plan (Two-Stage):**
    1.  Stage 1 (Story 2.1): Evaluate (Naive/Semantic Chunk) x (Naive/BM25 Retrieve) -> Select **default chunking** & **base retriever**. Justify.
    2.  Stage 2 (Story 2.6): Evaluate all retrievers x both chunking strategies -> Select **final optimal combination** for the app (potentially swapping base). Justify.

## 6.4 Agent Orchestration (ReAct + LangGraph)

* **Pattern:** ReAct orchestration using LangGraph.
* **Location:** `backend/orchestration/pipeline.py`.
* **Structure:** LangGraph graph managing state and routing tasks to specialized agents based on ReAct loop (Reason -> Act -> Observe). Agents use their own tools.
* **Visualization:** Graph structure will be displayable in Jupyter Notebooks using a reusable helper function.
* **Terminology:** The component managing this flow is the **orchestrator (supervisor agent)**.

## 6.5 Notebooks / Evaluation (Jupytext)

* **Location:** `notebooks/`.
* **Format:** Jupytext-compatible `.py` scripts using `# %%` cell markers.
* **Purpose:** Import from `backend/` for testing, evaluation, justification, visualization.
* **Key Scripts:** `E01_Pipeline_Foundation.py`, `E02_Evaluation.py`, `E04_Agent_Collaboration.py`. Evaluation script includes RAGAS metrics, latency/cost, caching, tables, charts.
* **Visualization:** Notebook scripts will contain reusable function(s) to display LangGraph objects interactively.

## 6.6 Containerization (Docker)

* **File:** Root `docker-compose.yml`.
* **Services:** `qdrant`, `backend` (FastAPI, uses `uv`), `frontend` (Next.js).
* **Command:** `docker-compose up` for full `localhost` stack.
* **Makefile:** A root `Makefile` will provide simplified targets (e.g., `make start`, `make evaluate`, `make format`, `make lint`). Creation included in Story 1.1.

---
