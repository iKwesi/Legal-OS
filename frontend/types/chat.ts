export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  provenance?: ProvenanceSource[];
}

export interface ProvenanceSource {
  source: string;
  page?: number;
}

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  role: string;
  content: string;
  provenance?: ProvenanceSource[];
}
