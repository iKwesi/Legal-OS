# Legal OS - M&A Diligence Module (AIE8 Challenge) Fullstack Architecture Document

**Document Owner:** Winston, Architect 🏗️
**Status:** Finalized
**Next Step:** Development (following PRD Epics)

## 1. Introduction

### 1.1 Starter Template or Existing Project

* **Decision:** N/A - Greenfield project. We will build the structure from scratch following best practices. (Existing user notebooks may serve as reference if needed later).

### 1.2 Change Log

| Date   | Version | Description                      | Author          |
| :----- | :------ | :------------------------------- | :-------------- |
| (Auto) | 1.0     | Initial Draft                    | Winston (Arch.) |
| (Auto) | 1.1     | Finalized based on user feedback | Winston (Arch.) |

---

## 2. High Level Architecture

### 2.1 Technical Summary

This project implements a full-stack, multi-agent RAG system for M&A contract diligence. The architecture utilizes a **Python/LangChain backend** with six distinct, collaborative agents orchestrated via a **ReAct pattern (using LangGraph)**, retrieving data from **Qdrant**. A separate **Next.js/React frontend application** provides the user interface. The entire system is containerized using **Docker Compose** for consistent `localhost` deployment. The architecture prioritizes modularity (supporting the "Legal OS" vision), testability (via Jupyter notebooks using Jupytext-compatible scripts), and a robust evaluation framework using RAGAS.

### 2.2 Platform and Infrastructure Choice

* **MVP Platform (`localhost`):** Docker Compose.
* **Key Services (Local):** Docker Engine, Docker Compose, Qdrant (container), Backend (container), Frontend (container).
* **Future Vision Platform (Post-MVP):** Cloud-native design (e.g., AWS Lambda/ECS, API Gateway, S3, managed Qdrant). Cloud setup is out of scope for MVP.

### 2.3 Repository Structure

* **Structure:** **Monorepo** (Approved).
* **Monorepo Tool:** Standard **npm/yarn workspaces** (for managing Node.js frontend) combined with Python project structure. A root `Makefile` will orchestrate tasks across packages.
* **Package Organization:** `apps/` (for `backend`, `frontend`), `packages/` (for potential shared code like `types`), root folders for `data/`, `golden_dataset/`, `notebooks/`, `docs/`.

### 2.4 High Level Architecture Diagram

```mermaid
graph TD
    subgraph User Machine (localhost via Docker Compose)
        A[Frontend App (Next.js)] <--> B(Backend API (FastAPI) / Orchestrator (LangGraph));
        B --> C{Agent Logic (LangChain/ReAct)};
        C --> D[Qdrant Vector Store];
        C --> E[Agent Tools];
        F[Jupyter Notebook (.py Script)] --> B;
        G[Source Docs (data/)] --> B;
        H[SGD (golden_dataset/)] --> F;
    end

    User --> A;
    User --> F;

    style D fill:#f9f,stroke:#333,stroke-width:2px;
    style B fill:#ccf,stroke:#333,stroke-width:2px;
    style A fill:#bbf,stroke:#333,stroke-width:2px;
    style F fill:#dfd,stroke:#333,stroke-width:2px;
```

### 2.5 Architectural Patterns

* [cite_start]**Monorepo:** Manages frontend, backend, shared code, and notebooks [cite: 969-970].
    * _Rationale:_ Simplifies setup, ensures type sharing, facilitates full-stack development.
* **Containerization (Docker Compose):** Encapsulates services (backend, frontend, Qdrant) for consistent `localhost` deployment.
    * _Rationale:_ Meets rubric requirement, ensures reproducibility, standardizes environment.
* **Multi-Agent System (ReAct Orchestration):** Backend logic organized into specialized agents coordinated via a ReAct pattern using LangGraph.
    * _Rationale:_ Explicitly required, enables modularity, supports complex reasoning and dynamic tool use.
* **Retrieval-Augmented Generation (RAG):** Core pattern using Qdrant for grounding agent responses.
    * _Rationale:_ Essential for accurate analysis of legal documents.
* **Swappable Components (Retrievers/Chunking):** Designing the RAG pipeline for easy configuration and testing.
    * _Rationale:_ Required for rubric evaluation, facilitates optimization.
* **Backend for Frontend (BFF):** FastAPI backend serves the Next.js frontend.
    * _Rationale:_ Standard pattern for separation of concerns.
* **Notebook-Driven Development/Testing:** Using Jupyter via Jupytext `.py` scripts for testing and evaluation.
    * _Rationale:_ Required by rubric, excellent for AI component iteration.

---

## 3. Tech Stack

| Category               | Technology                | Version      | Purpose                                                     | Rationale                                                                        |
| :--------------------- | :------------------------ | :----------- | :---------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **Frontend Language** | TypeScript                | `~5.x`       | Type safety for UI development                              | Industry standard, improves maintainability, integrates well with React/Next.js |
| **Frontend Framework** | Next.js (React)           | `~14.x`      | Full-stack React framework for UI, routing, build           | Approved: Mature, great DX, SSR/SSG capable, large ecosystem                     |
| **UI Component Lib.** | Shadcn/ui (Radix+Tw)      | `Latest`     | Accessible, customizable components styled with Tailwind    | Approved: Fits "ultra-modern" aesthetic, Radix primitives = A11y, uses Tailwind |
| **State Management** | Zustand / Context API     | `Latest`     | Lightweight global/local state management                 | Simple, less boilerplate than Redux, sufficient for MVP scope                    |
| **Backend Language** | Python                    | `~3.11/3.12` | Core logic, agents, RAG pipeline                          | Approved: Best ecosystem for AI/ML, LangChain, RAGAS                             |
| **Backend Framework** | FastAPI                   | `Latest`     | Web framework for exposing API to frontend                  | Approved: User specified. High performance, async support, great for Python APIs |
| **Agent Framework** | LangChain (inc. LangGraph)| `Latest`     | Building agents, RAG pipeline, ReAct orchestration        | Approved: User preference, powerful, supports ReAct/Graph vis.                   |
| **API Style** | REST                      | `N/A`        | Communication between frontend and backend                  | Simple, well-understood standard, sufficient for MVP needs                     |
| **Database** | **None (MVP)** | `N/A`        | Persistent storage                                          | Approved: User preference to avoid for MVP complexity                            |
| **Vector Store** | Qdrant                    | `Latest`     | Storing and retrieving document embeddings                  | Approved: User specified, performant vector database                             |
| **Reranker** | Cohere Rerank API         | `N/A`        | Improving retrieval relevance (part of advanced retriever) | Approved: State-of-the-art reranking performance                                 |
| **Evaluation** | RAGAS                     | `Latest`     | SGD Generation & RAG pipeline evaluation                  | Required by PRD/Rubric                                                           |
| **Notebook Env.** | Jupyter (via Jupytext)    | `Latest`     | Testing, evaluation, demonstration                        | Approved: Required by PRD, Jupytext for `.py` script compatibility             |
| **Containerization** | Docker & Docker Compose   | `Latest`     | Local deployment, service orchestration                   | Approved: Required by PRD/Rubric                                                 |
| **Dependency Mgmt (Py)**| uv                        | `Latest`     | Installing Python packages                                  | Approved: User specified, faster than pip/conda                                  |
| **CSS Framework** | Tailwind CSS              | `~3.x`       | Utility-first CSS for frontend styling                    | Required by Shadcn/ui, excellent for modern UI design                            |
| **Frontend Testing** | Vitest / React Testing Lib| `Latest`     | Unit/Integration tests for frontend components            | Standard for React/Vite/Next.js ecosystem                                        |
| **Backend Testing** | Pytest                    | `Latest`     | Unit/Integration tests for backend agents/logic           | Standard, powerful testing framework for Python                                  |
| **E2E Testing** | Playwright / Cypress      | `Latest`     | End-to-end testing of full application flow (Post-MVP)    | Out of scope for MVP                                                             |
| **Build Tool (FE)** | Next.js built-in (Webpack/Turbopack) | `N/A` | Bundling, optimizing frontend code                      | Comes integrated with Next.js                                                    |
| **IaC Tool** | N/A (Local Docker only)   | `N/A`        | Infrastructure as Code                                      | Not needed for `localhost` MVP                                                   |
| **CI/CD** | GitHub Actions (Basic)    | `N/A`        | Continuous Integration (linting, basic tests)             | Good practice, easy setup for GitHub repo                                        |
| **Monitoring** | N/A (Local Docker only)   | `N/A`        | Application monitoring                                      | Not needed for `localhost` MVP                                                   |
| **Logging (BE)** | Python `logging` module   | `Built-in`   | Backend application logging                                 | Standard Python library, sufficient for MVP                                      |
| **Logging (FE)** | `console.log` (Dev only)  | `Built-in`   | Frontend logging during development                       | Simple, standard for browser development                                         |

---

## 4. Data Models

### 4.1 Data Models (Conceptual)

1.  **`Document`** (Internal Backend): `document_id: str`, `file_name: str`, `content: str`, `metadata: dict`
2.  **`DocumentChunk`** (Internal): `chunk_id: str`, `document_id: str`, `text: str`, `metadata: dict`
3.  **`ExtractedClause`**: `clause_id: str`, `clause_type: str`, `text: str`, `risk_score: str`, `summary: str`, `provenance: dict`
4.  **`RedFlag`**: `flag_id: str`, `description: str`, `risk_score: str`, `provenance: dict`
5.  **`ChecklistItem`**: `item_id: str`, `text: str`, `related_flag_id: Optional[str]`
6.  **`AnalysisReport`** (API Response): `report_id: str`, `summary_memo: str`, `extracted_clauses: List[ExtractedClause]`, `red_flags: List[RedFlag]`, `checklist: List[ChecklistItem]`
7.  **`ChatMessage`** (API Request/Response): `role: str`, `content: str`, `provenance: Optional[List[dict]]`

### 4.2 API Data Formats (Pydantic / TypeScript)

* API uses **JSON**.
* Backend (FastAPI) uses **Pydantic models** for validation/serialization.
* Frontend (Next.js) uses **TypeScript interfaces** (can be auto-generated from FastAPI spec).

---

## 5. API Endpoints

### 5.1 API Endpoint Table

| Endpoint (URL)             | Method | Request Body (Pydantic Model)     | Response Body (Pydantic Model)                     | Description                                         |
| :------------------------- | :----- | :-------------------------------- | :------------------------------------------------- | :-------------------------------------------------- |
| `/api/v1/upload`           | POST   | `Upload(files: List[UploadFile])` | `UploadResponse(session_id: str, file_names: List[str])` | Uploads docs, starts ingestion, returns session ID. |
| `/api/v1/analyze`          | POST   | `AnalyzeRequest(session_id: str)` | `AnalyzeResponse(status_url: str)`                 | Triggers full agent orchestration.                  |
| `/api/v1/status/{session_id}` | GET    | `None`                            | `StatusResponse(status: str, progress: int)`     | Pollable endpoint for analysis status.              |
| `/api/v1/results/{session_id}`| GET    | `None`                            | `AnalysisReport`                                   | Fetches final analysis results when complete.       |
| `/api/v1/chat/{session_id}`    | POST   | `ChatRequest(message: str)`       | `ChatMessage`                                      | Sends user question to RAG chat pipeline.           |

### 5.2 Authentication and Authorization

* **MVP Scope:** **None.** Open API for `localhost` demo.

### 5.3 Error Handling

* Use standard HTTP status codes (`200`, `201`, `400`, `404`, `500`).
* Return JSON error body: `{"detail": "Error message"}`.

### 5.4 API Documentation

* Auto-generated via FastAPI's built-in OpenAPI (`/docs`) and ReDoc (`/redoc`).

---

## 6. Implementation Details

### 6.1 Frontend (Next.js + Shadcn/ui)

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

### 6.2 Backend (FastAPI + LangChain)

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

### 6.3 RAG Pipeline (Swappable)

* **Location:** `backend/rag/`.
* **Components:** Implementations for Naive/Semantic chunking; Naive, BM25, Multi-Query, Parent-Doc, Rerank (Cohere), Ensemble retrievers.
* **Configuration:** Pipeline selectable via config.
* **Evaluation Plan (Two-Stage):**
    1.  Stage 1 (Story 2.1): Evaluate (Naive/Semantic Chunk) x (Naive/BM25 Retrieve) -> Select **default chunking** & **base retriever**. Justify.
    2.  Stage 2 (Story 2.6): Evaluate all retrievers x both chunking strategies -> Select **final optimal combination** for the app (potentially swapping base). Justify.

### 6.4 Agent Orchestration (ReAct + LangGraph)

* **Pattern:** ReAct orchestration using LangGraph.
* **Location:** `backend/orchestration/pipeline.py`.
* **Structure:** LangGraph graph managing state and routing tasks to specialized agents based on ReAct loop (Reason -> Act -> Observe). Agents use their own tools.
* **Visualization:** Graph structure will be displayable in Jupyter Notebooks using a reusable helper function.
* **Terminology:** The component managing this flow is the **orchestrator (supervisor agent)**.

### 6.5 Notebooks / Evaluation (Jupytext)

* **Location:** `notebooks/`.
* **Format:** Jupytext-compatible `.py` scripts using `# %%` cell markers.
* **Purpose:** Import from `backend/` for testing, evaluation, justification, visualization.
* **Key Scripts:** `E01_Pipeline_Foundation.py`, `E02_Evaluation.py`, `E04_Agent_Collaboration.py`. Evaluation script includes RAGAS metrics, latency/cost, caching, tables, charts.
* **Visualization:** Notebook scripts will contain reusable function(s) to display LangGraph objects interactively.

### 6.6 Containerization (Docker)

* **File:** Root `docker-compose.yml`.
* **Services:** `qdrant`, `backend` (FastAPI, uses `uv`), `frontend` (Next.js).
* **Command:** `docker-compose up` for full `localhost` stack.
* **Makefile:** A root `Makefile` will provide simplified targets (e.g., `make start`, `make evaluate`, `make format`, `make lint`). Creation included in Story 1.1.

---

## 7. Project Plan / Epic Mapping

* **Epic 1: Project Foundation & Core RAG Pipeline:** Setup monorepo, Docker, `uv`, `Makefile`, base RAG, SGD script, Notebook E01.
* **Epic 2: RAG Evaluation & Retriever/Chunking Optimization:** Implement swappable components, RAGAS eval script, instrumentation/caching, all retrievers, Notebook E02 (two-stage eval, charts, justification).
* **Epic 3: Modular Agent Implementation (Backend):** Implement 5 agents (Clause, Risk, Summary, Provenance, Checklist) using LangChain/ReAct, incorporating LangGraph vis.
* **Epic 4: Agent Collaboration & Notebook Validation:** Implement ReAct orchestrator (supervisor agent) using LangGraph, Notebook E04 (tests collaboration, displays graph).
* **Epic 5: Frontend Application & Final Deliverables:** Build Next.js/Shadcn UI (Upload, Results, Checklist, Chat screens), integrate API, finalize Docker, prepare docs/repo/video script.

---