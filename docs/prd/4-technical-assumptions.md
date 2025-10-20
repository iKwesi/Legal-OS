# 4. Technical Assumptions

## 4.1 Repository Structure
* **Structure:** **Monorepo** (Approved).

## 4.2 Service Architecture
* **Style:** Multi-Agent Backend with a separate Frontend. Modular components/services implementing a ReAct pattern.
* **API:** Backend will expose an API (Style TBD by Architect).

## 4.3 Core Technologies
* **Backend Language:** **Python** (Approved).
* **Agent Framework:** **LangChain** (User preference; Architect to confirm suitability and specific modules, e.g., LangGraph for ReAct orchestration).
* **RAG Implementation:** TBD by Architect, must support multiple swappable retrievers (Naive, BM25, Multi-Query, Parent-Doc, Rerank, Ensemble, Semantic Chunking).
* **Evaluation:** RAGAS (for SGD generation and evaluation).
* **Testing Environment:** Jupyter Notebooks (via Jupytext `.py` scripts).
* **Containerization:** Docker & Docker Compose.
* **Vector Store:** **Qdrant** (User specified).
* **Dependency Management:** `uv` for Python backend.

## 4.4 Key Decisions Required (Architect to Recommend, User Confirm)
* **Frontend Framework:** **Architect to research and recommend** based on the "ultra-modern" UI goal and Harvey AI aesthetic.
* **UI Component Library:** **Architect to research and recommend** based on the chosen framework and Harvey AI aesthetic.
* **Database:** **User preference is NO database for the MVP.** Architect to evaluate if essential and provide final recommendation with justification.

## 4.5 Testing Requirements
* Backend logic testable via Jupyter Notebooks (`.py` scripts).
* RAGAS framework mandatory.
* Standard unit/integration tests.

## 4.6 Additional Technical Assumptions and Requests
* RAGAS evaluation results must be cached.
* Evaluation must include latency and cost estimation.
* Evaluation results (tables, charts) and retriever/chunking choice justification must be documented.

---
