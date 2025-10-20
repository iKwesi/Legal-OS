# 6. User Stories

### Epic 1: Project Foundation & Core RAG Pipeline

* **Story 1.1: Project Setup (Architecture)**
    * **As:** The Architect (Winston),
    * **I want:** To define and initialize the monorepo structure (`data/`, `golden_dataset/`, `backend/`, `frontend/`, `notebooks/`, `docs/`) and add the core project configuration files (Docker, Docker Compose, Python dependencies using `uv`).
    * **So that:** The development team has a clean, standardized environment to begin work, runnable via `docker-compose up`.
    * **Acceptance Criteria:**
        * Monorepo structure is created including `data/` and `golden_dataset/`.
        * `docker-compose.yml` file is created and successfully builds the initial (empty) `backend` and `frontend` services.
        * `backend/` contains an initial `requirements.txt` (or `pyproject.toml`) with core libraries (LangChain, RAGAS, Qdrant, uv).
        * Backend Dockerfile or setup script explicitly uses `uv` to install Python dependencies.
        * `notebooks/` directory is created.
        * Root `Makefile` created with initial targets for `docker-compose up/down/build`.
* **Story 1.2: Ingestion Agent & RAG Pipeline (V1)**
    * **As:** The Developer,
    * **I want:** To implement the **Ingestion Agent** and a basic RAG pipeline.
    * **So that:** Documents can be processed, chunked (using initial default strategy), stored in Qdrant, and a basic retrieval-generation loop is functional.
    * **Acceptance Criteria:**
        * Ingestion Agent can receive a document path (from `data/`).
        * The agent implements an initial **chunking strategy** (e.g., Recursive Character).
        * Chunks are successfully embedded and stored in the **Qdrant** vector store.
        * A basic **RAG chain** (using LangChain and "Naive Retrieval") is created for question answering.
* **Story 1.3: Synthetic Golden Dataset (SGD) Generation**
    * **As:** The Developer,
    * **I want:** To implement a script using **RAGAS** testset generation.
    * **So that:** We can create our SGD benchmark from source documents.
    * **Acceptance Criteria:**
        * A script (e.g., `backend/scripts/generate_sgd.py`) is created.
        * Loads source documents from `data/`.
        * Uses RAGAS `TestsetGenerator`.
        * Saves the generated SGD as a **CSV file** in `golden_dataset/` (e.g., `golden_dataset/sgd_benchmark.csv`).
* **Story 1.4: Notebook Test Script (Epic 1)**
    * **As:** The Developer,
    * **I want:** To create a **Jupytext-compatible Python script (`.py`)**.
    * **So that:** I can validate Story 1.2 (Ingestion/RAG) and 1.3 (SGD Generation) in a notebook.
    * **Acceptance Criteria:**
        * Script `notebooks/E01_Pipeline_Foundation.py` created (Jupytext format).
        * **Cell 1 (Docs):** Explains purpose.
        * **Cell 2 (Ingest):** Runs Ingestion Agent on a sample doc from `data/`.
        * **Cell 3 (RAG Test):** Asks a question to RAG (V1) and prints answer.
        * **Cell 4 (SGD):** Runs SGD Generation script and shows sample output from `golden_dataset/sgd_benchmark.csv`.

### Epic 2: RAG Evaluation & Retriever/Chunking Optimization

* **Story 2.1: Evaluate Chunking Strategies & Select Base Retriever/Chunking**
    * **As:** The Developer,
    * **I want:** To implement and evaluate **Naive (Recursive Character splitting)** versus **Semantic Chunking**, combined with **Naive Retrieval** vs. **BM25 Retrieval**, using **RAGAS** against the SGD.
    * **So that:** We can make data-informed decisions on the initial default chunking strategy and base retriever for the application, and provide a strong justification based on RAGAS metrics.
    * **Acceptance Criteria:**
        * Both **Recursive Character splitting** and **Semantic Chunking** strategies are implemented.
        * Both **Naive Retrieval** and **BM25 Retrieval** are implemented.
        * The evaluation script/notebook cells run the RAG pipeline against the SGD for all 4 combinations.
        * Uses **RAGAS** (`context_precision`, `context_recall`, etc.) to score context effectiveness for each combo.
        * A **default chunking strategy** is chosen based on the RAGAS evaluation results.
        * A **base retriever** (Naive or BM25) is chosen based on the RAGAS evaluation results.
        * The justification for both choices, citing the RAGAS scores, is documented (notebook markdown or `docs/decisions/`).
* **Story 2.2: Implement Swappable Retrievers Architecture**
    * **As:** The Developer,
    * **I want:** To architect the backend for multiple, easily swappable retrieval methods.
    * **So that:** We can evaluate each retriever systematically.
    * **Acceptance Criteria:**
        * Backend refactored for retriever selection via config/params.
        * Base retriever (from 2.1) is integrated into the swappable architecture.
* **Story 2.3: Integrate RAGAS Evaluation Framework**
    * **As:** The Developer,
    * **I want:** To integrate RAGAS into a reusable evaluation script/module.
    * **So that:** We can programmatically run evaluations against the SGD.
    * **Acceptance Criteria:**
        * Evaluation script (e.g., `backend/scripts/evaluate_rag.py`) created.
        * Loads SGD from `golden_dataset/sgd_benchmark.csv`.
        * Takes RAG pipeline config (including retriever & chunking strategy) as input.
        * Runs pipeline against SGD questions.
        * Uses RAGAS for metrics: faithfulness, answer relevancy, context precision, context recall.
        * Returns/stores metrics.
* **Story 2.4: Implement Evaluation Instrumentation & Caching**
    * **As:** The Developer,
    * **I want:** To instrument the pipeline/evaluation script for latency/cost estimation and implement results caching.
    * **So that:** We collect performance data and avoid re-runs.
    * **Acceptance Criteria:**
        * Evaluation script measures/records latency (retrieval, generation).
        * Estimates cost (e.g., token usage).
        * Evaluation results (metrics, latency, cost) are cached based on config.
* **Story 2.5: Implement Advanced Retrievers**
    * **As:** The Developer,
    * **I want:** To implement the specified advanced retriever methods (Multi-Query, Parent-Doc, Rerank [Cohere], Ensemble, plus Semantic Chunking interaction if applicable).
    * **So that:** All required strategies can be evaluated.
    * **Acceptance Criteria:**
        * Each specified advanced retriever implemented and integrated into swappable architecture.
        * Each is runnable within the RAG pipeline.
* **Story 2.6: Notebook Test Script (Epic 2 - Full Evaluation)**
    * **As:** The Developer,
    * **I want:** To create a **Jupytext-compatible Python script (`.py`)** for Epic 2.
    * **So that:** I can run evaluations for all retriever/chunking combos, compare results, document findings, and justify the final selection.
    * **Acceptance Criteria:**
        * Script `notebooks/E02_Evaluation.py` created (Jupytext format).
        * **Cell 1 (Docs):** Explains purpose (comparing chunking/retrievers).
        * **Cell 2 (Chunking Summary - Markdown):** Summarizes Story 2.1 findings and justifies default chunking choice.
        * **Cell 3 (Base Retriever Summary - Markdown):** States the base retriever chosen in Story 2.1.
        * **Cell 4 (Setup):** Loads SGD and evaluation functions.
        * **Subsequent Cells (one per retriever x chunking combo):** Configures pipeline, runs evaluation (using cache), stores results.
        * **Comparison Cell:** Aggregates results into a comparison **table**.
        * **Visualization Cell:** Generates **charts/graphs** comparing metrics.
        * **Decision Cell (Markdown):** States final chosen retriever (and confirms chunking strategy) with **justification** based on data (RAGAS, latency, cost), explicitly comparing advanced vs. base.

### Epic 3: Modular Agent Implementation (Backend)

* **Story 3.1: Clause Extraction Agent**
    * **As:** The Developer,
    * **I want:** To implement the **Clause Extraction Agent** (using LangChain, ReAct, optimal retriever).
    * **So that:** It extracts clauses and detects red flags.
    * **Acceptance Criteria:**
        * Implemented as distinct module (`backend/agents/clause_extraction.py`).
        * Takes processed chunks as input.
        * Uses optimal retriever (from Epic 2 decision).
        * Outputs structured clauses/flags.
        * Testable independently.
        * Includes code to facilitate display of its internal LangGraph visualization.
* **Story 3.2: Risk Scoring Agent**
    * **As:** The Developer,
    * **I want:** To implement the **Risk Scoring Agent** (using LangChain, ReAct).
    * **So that:** It assigns risk scores to clauses.
    * **Acceptance Criteria:**
        * Implemented as distinct module (`backend/agents/risk_scoring.py`).
        * Takes Clause Agent output as input.
        * Applies defined rules/heuristics.
        * Outputs data enriched with risk scores.
        * Testable independently.
        * Includes code to facilitate display of its internal LangGraph visualization.
* **Story 3.3: Summary Agent**
    * **As:** The Developer,
    * **I want:** To implement the **Summary Agent** (using LangChain, ReAct, optimal retriever).
    * **So that:** It generates a concise diligence memo.
    * **Acceptance Criteria:**
        * Implemented as distinct module (`backend/agents/summary.py`).
        * Takes Risk Agent output (+ context) as input.
        * Generates coherent summary (Markdown/JSON).
        * Testable independently.
        * Includes code to facilitate display of its internal LangGraph visualization.
* **Story 3.4: Provenance Agent**
    * **As:** The Developer,
    * **I want:** To implement the **Provenance Agent** functionality.
    * **So that:** Summary statements link back to sources.
    * **Acceptance Criteria:**
        * Agent/logic tracks information sources.
        * Embeds provenance data (doc ID, location) into summary output.
        * Output format supports frontend rendering of links.
        * Testable independently.
        * Includes code to facilitate display of its internal LangGraph visualization (if applicable).
* **Story 3.5: Checklist Agent**
    * **As:** The Developer,
    * **I want:** To implement the **Checklist Agent** (using LangChain, ReAct).
    * **So that:** It generates checklists/follow-up questions.
    * **Acceptance Criteria:**
        * Implemented as distinct module (`backend/agents/checklist.py`).
        * Takes final analysis results as input.
        * Generates structured checklist/questions based on rules.
        * Outputs structured list (JSON/list).
        * Testable independently.
        * Includes code to facilitate display of its internal LangGraph visualization.

### Epic 4: Agent Collaboration & Notebook Validation

* **Story 4.1: Implement ReAct-Based Orchestration (Supervisor Agent)**
    * **As:** The Developer,
    * **I want:** To implement the **orchestrator (supervisor agent)** logic using **LangGraph** managing a **ReAct-style workflow**, routing tasks to specialized agents.
    * **So that:** The system dynamically processes documents end-to-end under the supervisor agent's direction.
    * **Acceptance Criteria:**
        * Orchestrator (supervisor agent) logic (LangGraph graph) managing ReAct loop implemented in `backend/orchestration/`.
        * Maintains state.
        * Routes tasks to correct agents based on state/reasoning.
        * Agents execute using their defined tools.
        * Data/state passed correctly.
        * Handles errors/loops.
        * Facilitates display of the overall orchestration graph.
* **Story 4.2: Notebook Test Script (Epic 4 - Collaboration & Visualization)**
    * **As:** The Developer,
    * **I want:** To create the final **Jupytext-compatible Python script (`.py`)** for demonstrating full collaboration.
    * **So that:** We test the end-to-end workflow, visualize interactions, and confirm requirements.
    * **Acceptance Criteria:**
        * Script `notebooks/E04_Agent_Collaboration.py` created (Jupytext format).
        * **Cell 1 (Docs):** Explains purpose.
        * **Cell 2 (Setup):** Imports modules, loads sample doc from `data/`.
        * **Cell 3 (Orchestration):** Runs full orchestration logic (Story 4.1).
        * **Cell 4 (Results):** Displays final outputs (summary, checklist).
        * **Cell 5 (Graph Visualization):** Includes code to **display the LangGraph visualization** of the overall orchestration.
        * **(Optional) Individual Agent Test Cells:** May include cells testing individual agents.

### Epic 5: Frontend Application & Final Deliverables

* **Story 5.1: Frontend UI Implementation (Core Screens - Upload, Results, Checklist)**
    * **As:** The Developer,
    * **I want:** To implement core frontend screens (Upload, Results, Checklist) based on Architect's framework/library and "ultra-modern" aesthetic.
    * **So that:** Users have a functional interface.
    * **Acceptance Criteria:**
        * Frontend project setup.
        * Upload, Results, Checklist screens implemented.
        * Styling reflects "ultra-modern" theme (Light Mode only).
        * Basic responsiveness.
        * UI follows architecture standards.
* **Story 5.2: Implement Chat UI Screen**
    * **As:** The Developer,
    * **I want:** To implement the **Chat UI screen** with the "ultra-modern" aesthetic.
    * **So that:** Users can interactively query documents.
    * **Acceptance Criteria:**
        * Chat screen/component created.
        * Includes input field, message history.
        * Handles submission, loading, errors.
        * Matches overall theme.
* **Story 5.3: Frontend-Backend API Integration (Including Chat)**
    * **As:** The Developer,
    * **I want:** To integrate the frontend (including Chat) with the backend API.
    * **So that:** Users can use the full workflow (upload, analyze, view, chat).
    * **Acceptance Criteria:**
        * Frontend makes API calls for upload, status, results, chat.
        * Upload connected to Ingestion endpoint.
        * Results/Checklist screens display fetched data.
        * Chat input submits questions to backend chat endpoint.
        * Chat screen displays responses from backend RAG pipeline.
        * Provenance links rendered correctly.
        * Loading states/error handling implemented.
* **Story 5.4: Finalize Docker Compose & Localhost Deployment**
    * **As:** The Developer,
    * **I want:** To finalize `docker-compose.yml` for easy `localhost` execution.
    * **So that:** The application meets the rubric's deployment requirement.
    * **Acceptance Criteria:**
        * `docker-compose up` starts all services (Qdrant, Backend, Frontend).
        * Frontend accessible on `localhost`.
        * Frontend communicates with backend in Docker.
        * Full workflow (Upload -> Analyze -> View -> Chat) functional in Docker.
* **Story 5.5: Prepare Final Submission Materials (Including Video Script)**
    * **As:** The Project Lead (You), assisted by Agents,
    * **I want:** To prepare all final deliverables (docs, code, repo, video script) for the AIE8 Rubric.
    * **So that:** The project can be submitted.
    * **Acceptance Criteria:**
        * **Written Document:** All BMad docs (`project-brief.md`, `prd.md`, `front-end-spec.md`, `architecture.md`) & evaluation notebooks (`.py` scripts) finalized and organized.
        * **Code:** Source code finalized and cleaned up.
        * **GitHub Repo:** Code and docs pushed to public GitHub repo.
        * **Video Script:** Draft script for a <=5 min Loom video demo generated.

---