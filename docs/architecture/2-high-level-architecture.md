# 2. High Level Architecture

## 2.1 Technical Summary

This project implements a full-stack, multi-agent RAG system for M&A contract diligence. The architecture utilizes a **Python/LangChain backend** with six distinct, collaborative agents orchestrated via a **ReAct pattern (using LangGraph)**, retrieving data from **Qdrant**. A separate **Next.js/React frontend application** provides the user interface. The entire system is containerized using **Docker Compose** for consistent `localhost` deployment. The architecture prioritizes modularity (supporting the "Legal OS" vision), testability (via Jupyter notebooks using Jupytext-compatible scripts), and a robust evaluation framework using RAGAS.

## 2.2 Platform and Infrastructure Choice

* **MVP Platform (`localhost`):** Docker Compose.
* **Key Services (Local):** Docker Engine, Docker Compose, Qdrant (container), Backend (container), Frontend (container).
* **Future Vision Platform (Post-MVP):** Cloud-native design (e.g., AWS Lambda/ECS, API Gateway, S3, managed Qdrant). Cloud setup is out of scope for MVP.

## 2.3 Repository Structure

* **Structure:** **Monorepo** (Approved).
* **Monorepo Tool:** Standard **npm/yarn workspaces** (for managing Node.js frontend) combined with Python project structure. A root `Makefile` will orchestrate tasks across packages.
* **Package Organization:** `apps/` (for `backend`, `frontend`), `packages/` (for potential shared code like `types`), root folders for `data/`, `golden_dataset/`, `notebooks/`, `docs/`.

## 2.4 High Level Architecture Diagram

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

## 2.5 Architectural Patterns

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
