# 2. Requirements

## 2.1 Functional Requirements (FR)

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

## 2.2 Non-Functional Requirements (NFR)

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
