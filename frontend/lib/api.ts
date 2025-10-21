// API Client for Legal-OS Frontend

import { UploadResponse, AnalysisReport, ApiError } from '@/types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: 'An unexpected error occurred',
        status_code: response.status,
      }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  async uploadDocument(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<UploadResponse>(response);
  }

  async getAnalysisReport(sessionId: string): Promise<AnalysisReport> {
    const response = await fetch(`${this.baseUrl}/analyze/${sessionId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    return this.handleResponse<AnalysisReport>(response);
  }

  async queryDocument(sessionId: string, query: string): Promise<{ answer: string; sources: string[] }> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ session_id: sessionId, query }),
    });

    return this.handleResponse<{ answer: string; sources: string[] }>(response);
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
