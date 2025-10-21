// API Response Types for Legal-OS Frontend

export interface UploadResponse {
  session_id: string;
  file_names: string[];
  message: string;
}

export interface ExtractedClause {
  clause_type: string;
  text: string;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high';
  page_number?: number;
  confidence?: number;
}

export interface RedFlag {
  title: string;
  description: string;
  risk_score: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  related_clauses?: string[];
  recommendation?: string;
}

export interface ChecklistItem {
  id: string;
  category: string;
  text: string;
  completed: boolean;
  related_red_flags?: string[];
  priority?: 'low' | 'medium' | 'high';
}

export interface AnalysisReport {
  session_id: string;
  document_name: string;
  analysis_date: string;
  summary: string;
  extracted_clauses: ExtractedClause[];
  red_flags: RedFlag[];
  overall_risk_score: number;
  checklist_items: ChecklistItem[];
}

export interface AnalyzeRequest {
  session_id: string;
}

export interface AnalyzeResponse {
  status_url: string;
}

export interface StatusResponse {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message?: string;
}

export interface ApiError {
  detail: string;
  status_code: number;
}
