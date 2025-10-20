# 3. Tech Stack

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
