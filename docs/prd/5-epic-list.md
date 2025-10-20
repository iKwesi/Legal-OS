# 5. Epic List

1.  **Epic 1: Project Foundation & Core RAG Pipeline:**
    * **Goal:** Establish the monorepo structure (`data/`, `golden_dataset/`, `notebooks/`, etc.), core backend services (using Docker, `uv`), basic RAG pipeline (Ingestion -> Naive Retriever -> Generator), Jupyter Notebook environment, and Synthetic Golden Dataset (SGD) generation capability (saving to `golden_dataset/`).
2.  **Epic 2: RAG Evaluation & Retriever/Chunking Optimization:**
    * **Goal:** Implement and evaluate Naive vs. Semantic chunking strategies using RAGAS. Implement comprehensive RAGAS evaluation against the SGD for all specified retriever methods (Naive, BM25, Multi-Query, Parent-Doc, Rerank, Ensemble, Semantic Chunking), including caching, latency/cost metrics, comparative analysis (tables/charts), and selection/documentation of the optimal retriever and default chunking strategy.
3.  **Epic 3: Modular Agent Implementation (Backend):**
    * **Goal:** Develop the remaining five specialized agents (Clause Extraction, Risk Scoring, Summary, Checklist, Provenance) as modular, independently testable components within the backend (using LangChain, ReAct pattern), utilizing the selected optimal retriever. Ensure implementations facilitate LangGraph visualization display.
4.  **Epic 4: Agent Collaboration & Notebook Validation:**
    * **Goal:** Implement the ReAct-based orchestration logic (likely using LangGraph) connecting all six agents. Finalize the Jupyter Notebook environment (via `.py` script) to demonstrate both independent agent testing and the successful execution of the full, collaborative M&A diligence pipeline, including interactive display of the orchestration graph.
5.  **Epic 5: Frontend Application & Final Deliverables:**
    * **Goal:** Build the ultra-modern `localhost` frontend application (including Upload, Results, Checklist, and Chat screens), integrate it with the validated backend API, ensure the full system runs via Docker Compose, and prepare final submission materials (documentation bundle, GitHub repo, video script).

---
