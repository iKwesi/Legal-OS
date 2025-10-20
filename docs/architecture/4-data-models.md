# 4. Data Models

## 4.1 Data Models (Conceptual)

1.  **`Document`** (Internal Backend): `document_id: str`, `file_name: str`, `content: str`, `metadata: dict`
2.  **`DocumentChunk`** (Internal): `chunk_id: str`, `document_id: str`, `text: str`, `metadata: dict`
3.  **`ExtractedClause`**: `clause_id: str`, `clause_type: str`, `text: str`, `risk_score: str`, `summary: str`, `provenance: dict`
4.  **`RedFlag`**: `flag_id: str`, `description: str`, `risk_score: str`, `provenance: dict`
5.  **`ChecklistItem`**: `item_id: str`, `text: str`, `related_flag_id: Optional[str]`
6.  **`AnalysisReport`** (API Response): `report_id: str`, `summary_memo: str`, `extracted_clauses: List[ExtractedClause]`, `red_flags: List[RedFlag]`, `checklist: List[ChecklistItem]`
7.  **`ChatMessage`** (API Request/Response): `role: str`, `content: str`, `provenance: Optional[List[dict]]`

## 4.2 API Data Formats (Pydantic / TypeScript)

* API uses **JSON**.
* Backend (FastAPI) uses **Pydantic models** for validation/serialization.
* Frontend (Next.js) uses **TypeScript interfaces** (can be auto-generated from FastAPI spec).

---
