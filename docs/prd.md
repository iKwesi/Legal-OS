# Legal OS - M&A Diligence Module (AIE8 Challenge) Product Requirements Document (PRD)

**Document Owner:** John, Product Manager 📋
**Status:** Approved
**Next Step:** Handoff to UX Expert (Sally) & Architect (Winston)

## 1. Goals and Background Context

### 1.1 Goals

* **Primary Goal:** Build an end-to-end prototype that satisfies all requirements of the **AIE8 Certification Challenge Rubric**.
* **Rubric-Driven Goals:**
    * Deliver a `localhost` deployable full-stack application.
    * Generate a **Synthetic Golden Dataset (SGD)** using the **RAGAS** framework's testset generation capabilities. This SGD will be the official benchmark for all subsequent RAGAS evaluations of the pipeline.
    * Assess the RAG pipeline using RAGAS for faithfulness, response relevance, context precision, and context recall.
    * Quantify performance improvements from swapping a base retriever with an advanced one, providing results in a table.
    * Deliver a 5-minute Loom demo script and all project code/documentation.
* **Product Goals:**
    * Successfully implement the full "Transactional Cluster" as a modular, six-agent system (Ingestion, Clause Extraction, Risk Scoring, Summary, Checklist, Provenance) using the ReAct pattern where appropriate.
    * Develop a Jupyter Notebook environment (using Jupytext-compatible Python scripts) to independently test each agent and their collaboration.
    * Architect the system for future expansion into the full "Legal OS" (e.g., multi-jurisdictional support).

### 1.2 Background Context

This project represents the foundational "Transactional Cluster" module of a larger "Legal OS" vision. The immediate problem we are solving is the slow, manual, and error-prone nature of M&A due diligence, a critical, non-billable risk for legal teams.

Our solution is a multi-agent RAG application that automates the entire diligence workflow: ingesting documents, extracting clauses, scoring risk, and generating auditable summary reports.

For this AIE8 Certification Challenge, the prototype will be a full-stack `localhost` application built using publicly available U.S. contract datasets (like SEC EDGAR, specific dataset TBD by Architect) as the initial data source. The primary deliverables are not just the working application, but also the verifiable test results from RAGAS, the demonstration of a modular, collaborative multi-agent architecture (using ReAct and LangGraph), and clear documentation addressing the rubric.

### 1.3 Change Log

| Date   | Version | Description                          | Author    |
| :----- | :------ | :----------------------------------- | :-------- |
| (Auto) | 1.0     | Initial PRD draft based on brief | John (PM) |
| (Auto) | 1.1     | Added SGD/RAGAS/Retriever/Notebook details, refined UI goals, added Chat, re-added Video Script | John (PM) |

---

## 2. Requirements

### 2.1 Functional Requirements (FR)

**Document Ingestion & Processing:**
* **FR1:** The system must allow users to upload multiple legal contract documents (e.g., PDF, DOCX) through the frontend application.
* **FR2:** The **Ingestion Agent** must be able to classify uploaded documents by type (e.g., NDA, SPA, Vendor Contract).
* **FR3:** The **Ingestion Agent** must preprocess documents (e.g., cleaning text) and prepare them for the RAG pipeline, assuming input documents are text-readable (OCR is out of scope for MVP).
* **FR4:** The system must implement a configurable chunking strategy for ingested documents, with the default strategy justified by evaluation.

**Clause Extraction & Analysis:**
* **FR5:** The **Clause Extraction Agent** must identify and extract predefined key clauses (e.g., termination, indemnity, non-compete) from ingested documents.
* **FR6:** The **Clause Extraction Agent** must detect potential red flags (e.g., unbounded indemnity, missing clauses) based on predefined rules or patterns.
* **FR7:** The **Risk Scoring Agent** must score the risk level of extracted clauses based on predefined thresholds or heuristics.

**Summarization & Output Generation:**
* **FR8:** The **Summary Agent** must generate a concise diligence memo summarizing the key findings, extracted clauses, and flagged risks.
* **FR9:** The **Provenance Agent** must ensure that all statements in the summary report include clear, verifiable references (e.g., line-level citations) back to the source documents.
* **FR10:** The **Checklist Agent** must generate an automated closing checklist or list of follow-up questions based on the analysis (e.g., missing documents, high-risk clauses needing attention).

**Evaluation & Testing:**
* **FR11:** The system must include functionality to generate a **Synthetic Golden Dataset (SGD)** using the RAGAS framework's testset generation capabilities based on the source documents loaded from the `data/` folder.
* **FR12:** The system must integrate the **RAGAS framework** to evaluate the RAG pipeline's performance against the generated SGD (stored in `golden_dataset/`).
* **FR13:** The RAGAS evaluation must calculate and report key metrics including **faithfulness, response relevance, context precision, and context recall**.
* **FR14:** The system must allow configuring and testing multiple retrieval methods, including **Naive Retrieval, BM25, Multi-Query Retrieval, Parent-Document Retrieval, Contextual Compression (Rerank), Ensemble Retrieval, and Semantic Chunking**. The final implementation will use the retriever demonstrating the best performance based on evaluation results.
* **FR15:** The system must allow running the RAGAS evaluation against the SGD for *each* configured chunking strategy (Naive vs. Semantic) and *each* configured retrieval method to quantify and compare performance changes.
* **FR16:** The backend logic (including all six agents and the RAG pipeline) must be importable as a library into a **Jupyter Notebook** environment.
* **FR17:** The Jupyter Notebook environment must allow for **independent testing** of each of the six agents.
* **FR18:** The Jupyter Notebook environment must allow for testing the **full collaborative workflow** of all six agents using ReAct orchestration.
* **FR19:** The implementation of each agent and the overall orchestration (using LangGraph or similar) must facilitate **displaying visualizations** of their internal graphs within the Jupyter Notebook environment.

**Frontend Application:**
* **FR20:** The system must include a functional **frontend application** accessible via `localhost`, built with a production-ready, ultra-modern aesthetic inspired by Harvey AI.
* **FR21:** The frontend must provide UI elements for uploading documents.
* **FR22:** The frontend must display the generated summary report, including clickable provenance information.
* **FR23:** The frontend must display the generated checklist or follow-up questions.
* **FR24:** The frontend application must include a **chat interface** allowing users to ask questions about the uploaded and processed documents.
* **FR25:** The chat interface must use the selected optimal RAG pipeline (backend) to retrieve relevant context and generate answers based on the processed documents.

**Reporting & Documentation:**
* **FR26:** The RAGAS evaluation results for different chunking strategies and retrieval methods must be stored and presented in a comparative format (including **tables, charts, and graphs** within the Jupyter Notebook), and include metrics for **latency and estimated cost** alongside **RAGAS metrics**.
* **FR27:** The project documentation (specifically within the evaluation Jupyter Notebook or a linked section in the architecture document) must clearly state the **chosen default chunking strategy** and the **chosen retriever** for the final application, along with **justifications** based on the comparative evaluation results.

### 2.2 Non-Functional Requirements (NFR)

* **NFR1 (Deployment):** The entire application (frontend, backend agents, database [if needed], vector store) must be deployable and runnable on a local machine using Docker Compose.
* **NFR2 (Modularity):** The backend must be implemented as a modular system, with each of the six agents developed as distinct, potentially independent components/services suitable for potential future microservice extraction within the "Legal OS" vision.
* **NFR3 (Testability):** The backend architecture must facilitate easy importation and testing within a Jupyter Notebook environment (via Jupytext-compatible `.py` scripts), as specified in FR16-FR19.
* **NFR4 (Performance - Local):** The `localhost` demo should be responsive enough to complete a typical M&A document review cycle (upload -> analyze -> view report -> ask chat question) within a reasonable timeframe for the 5-minute demo video (e.g., under 1-2 minutes per major step).
* **NFR5 (Accuracy - Rubric):** The system's accuracy must be measurable via RAGAS metrics against the SGD, meeting the evaluation criteria specified in the AIE8 Rubric.
* **NFR6 (Provenance):** All generated outputs (summaries, checklists, chat responses where applicable) requiring factual grounding must have clear and verifiable provenance linking back to the source documents.
* **NFR7 (Maintainability):** Code should follow established best practices and conventions defined in the Architecture Document, including use of helper functions to adhere to DRY principles.
* **NFR8 (Caching):** The RAGAS evaluation process should cache results where appropriate (e.g., based on input hash or configuration) to avoid re-running computationally expensive evaluations unnecessarily.
* **NFR9 (Swappability):** The backend architecture must be designed to allow chunking and retrieval components to be easily swapped or configured to facilitate the testing, comparison, ranking, and final selection required by FR4, FR14, FR15, FR26 and FR27.
* **NFR10 (Instrumentation):** The RAG pipeline and evaluation framework must be instrumented to capture **latency** and allow for **cost estimation** during evaluation runs.
* **NFR11 (Dependency Management):** Python dependencies for the backend must be managed using `uv`.

---

## 3. User Interface Design Goals

### 3.1 Overall UX Vision
The user experience should be **polished, intuitive, and professional**, reflecting a production-ready application. The UI must clearly demonstrate the core value proposition: automated diligence report, checklist generation, and document chat with clear provenance, presented in a clean and trustworthy interface suitable for legal professionals.

### 3.2 Key Interaction Paradigms
* **Document Upload:** A seamless drag-and-drop or file selection mechanism with clear feedback on upload progress and success/failure.
* **Process Initiation:** A clear, prominent action to start analysis, with visual indication that processing is underway.
* **Results Display:** A well-structured and easily navigable view for the Summary Report and Checklist. Provenance links must be obvious and functional. Risk levels should be visually distinct.
* **Chat Interaction:** Standard chat interface elements (input box, send button, message history) for querying documents.
* **Status Indication:** Clear, non-intrusive notifications for processing status, completion, and any errors.

### 3.3 Core Screens and Views
Conceptually, the prototype requires the following main views for the `localhost` demo:
1.  **Document Upload Screen:** Interface for selecting and uploading contract files.
2.  **Analysis Results Screen:** Displays the generated Summary Report, including extracted clauses, risk scores, and provenance information.
3.  **Checklist/Follow-up Screen:** Displays the generated checklist or follow-up questions.
4.  **Chat with Documents Screen:** Interface for asking questions about the processed documents.

### 3.4 Accessibility
* **Compliance Target:** WCAG AA (Must adhere to these standards).

### 3.5 Branding
* **Aesthetic Goal:** The UI should have an **ultra-modern, clean, and sophisticated aesthetic**, drawing inspiration from contemporary AI platforms like **Harvey AI**. This implies minimalist design, clear typography, and subtle animations or transitions. Architect may recommend a specific component library to achieve this efficiently.
* **Theme**: **Light Theme only** for this prototype.

### 3.6 Target Device and Platforms
* **Platform:** Web Responsive (Desktop-first focus, but ensuring usability on tablet/mobile is a plus).

---

## 4. Technical Assumptions

### 4.1 Repository Structure
* **Structure:** **Monorepo** (Approved).

### 4.2 Service Architecture
* **Style:** Multi-Agent Backend with a separate Frontend. Modular components/services implementing a ReAct pattern.
* **API:** Backend will expose an API (Style TBD by Architect).

### 4.3 Core Technologies
* **Backend Language:** **Python** (Approved).
* **Agent Framework:** **LangChain** (User preference; Architect to confirm suitability and specific modules, e.g., LangGraph for ReAct orchestration).
* **RAG Implementation:** TBD by Architect, must support multiple swappable retrievers (Naive, BM25, Multi-Query, Parent-Doc, Rerank, Ensemble, Semantic Chunking).
* **Evaluation:** RAGAS (for SGD generation and evaluation).
* **Testing Environment:** Jupyter Notebooks (via Jupytext `.py` scripts).
* **Containerization:** Docker & Docker Compose.
* **Vector Store:** **Qdrant** (User specified).
* **Dependency Management:** `uv` for Python backend.

### 4.4 Key Decisions Required (Architect to Recommend, User Confirm)
* **Frontend Framework:** **Architect to research and recommend** based on the "ultra-modern" UI goal and Harvey AI aesthetic.
* **UI Component Library:** **Architect to research and recommend** based on the chosen framework and Harvey AI aesthetic.
* **Database:** **User preference is NO database for the MVP.** Architect to evaluate if essential and provide final recommendation with justification.

### 4.5 Testing Requirements
* Backend logic testable via Jupyter Notebooks (`.py` scripts).
* RAGAS framework mandatory.
* Standard unit/integration tests.

### 4.6 Additional Technical Assumptions and Requests
* RAGAS evaluation results must be cached.
* Evaluation must include latency and cost estimation.
* Evaluation results (tables, charts) and retriever/chunking choice justification must be documented.

---

## 5. Epic List

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

## 6. User Stories

#### Epic 1: Project Foundation & Core RAG Pipeline

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

#### Epic 2: RAG Evaluation & Retriever/Chunking Optimization

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

#### Epic 3: Modular Agent Implementation (Backend)

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

#### Epic 4: Agent Collaboration & Notebook Validation

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

#### Epic 5: Frontend Application & Final Deliverables

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