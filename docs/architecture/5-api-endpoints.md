# 5. API Endpoints

## 5.1 API Endpoint Table

| Endpoint (URL)             | Method | Request Body (Pydantic Model)     | Response Body (Pydantic Model)                     | Description                                         |
| :------------------------- | :----- | :-------------------------------- | :------------------------------------------------- | :-------------------------------------------------- |
| `/api/v1/upload`           | POST   | `Upload(files: List[UploadFile])` | `UploadResponse(session_id: str, file_names: List[str])` | Uploads docs, starts ingestion, returns session ID. |
| `/api/v1/analyze`          | POST   | `AnalyzeRequest(session_id: str)` | `AnalyzeResponse(status_url: str)`                 | Triggers full agent orchestration.                  |
| `/api/v1/status/{session_id}` | GET    | `None`                            | `StatusResponse(status: str, progress: int)`     | Pollable endpoint for analysis status.              |
| `/api/v1/results/{session_id}`| GET    | `None`                            | `AnalysisReport`                                   | Fetches final analysis results when complete.       |
| `/api/v1/chat/{session_id}`    | POST   | `ChatRequest(message: str)`       | `ChatMessage`                                      | Sends user question to RAG chat pipeline.           |

## 5.2 Authentication and Authorization

* **MVP Scope:** **None.** Open API for `localhost` demo.

## 5.3 Error Handling

* Use standard HTTP status codes (`200`, `201`, `400`, `404`, `500`).
* Return JSON error body: `{"detail": "Error message"}`.

## 5.4 API Documentation

* Auto-generated via FastAPI's built-in OpenAPI (`/docs`) and ReDoc (`/redoc`).

---
