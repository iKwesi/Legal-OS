// API Client for Legal-OS Frontend

import { 
  UploadResponse, 
  AnalysisReport, 
  AnalyzeResponse,
  StatusResponse,
  ApiError 
} from '@/types/api';
import { ChatResponse } from '@/types/chat';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const isDevelopment = process.env.NODE_ENV === 'development';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Log API requests and responses in development mode
   */
  private log(message: string, data?: any) {
    if (isDevelopment) {
      console.log(`[API Client] ${message}`, data || '');
    }
  }

  /**
   * Handle API response with error handling
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: this.getErrorMessage(response.status),
        status_code: response.status,
      }));
      
      this.log(`Error ${response.status}:`, error.detail);
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    this.log('Response:', data);
    return data;
  }

  /**
   * Get user-friendly error message based on status code
   */
  private getErrorMessage(status: number): string {
    switch (status) {
      case 400:
        return 'Invalid request. Please check your input.';
      case 404:
        return 'Session not found. Please upload a document first.';
      case 500:
        return 'Server error. Please try again later.';
      case 503:
        return 'Service unavailable. Please try again later.';
      default:
        return 'An unexpected error occurred.';
    }
  }

  /**
   * Upload a document and get a session ID
   */
  async uploadDocument(file: File): Promise<UploadResponse> {
    this.log('Uploading document:', file.name);
    
    const formData = new FormData();
    // Backend expects 'files' (plural) as the field name for the array
    formData.append('files', file);

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/upload`, {
        method: 'POST',
        body: formData,
      });

      return this.handleResponse<UploadResponse>(response);
    } catch (error) {
      this.log('Upload error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Unable to connect to server. Please check your connection.'
      );
    }
  }

  /**
   * Trigger analysis for a session
   */
  async triggerAnalysis(sessionId: string): Promise<AnalyzeResponse> {
    this.log('Triggering analysis for session:', sessionId);

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      return this.handleResponse<AnalyzeResponse>(response);
    } catch (error) {
      this.log('Analysis trigger error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to start analysis. Please try again.'
      );
    }
  }

  /**
   * Get analysis status for a session
   */
  async getAnalysisStatus(sessionId: string): Promise<StatusResponse> {
    this.log('Checking status for session:', sessionId);

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/status/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      return this.handleResponse<StatusResponse>(response);
    } catch (error) {
      this.log('Status check error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to check analysis status.'
      );
    }
  }

  /**
   * Poll analysis status until completion or failure
   * @param sessionId - Session ID to poll
   * @param onProgress - Callback for progress updates
   * @param pollInterval - Polling interval in milliseconds (default: 2000)
   * @param maxDuration - Maximum polling duration in milliseconds (default: 300000 = 5 minutes)
   */
  async pollAnalysisStatus(
    sessionId: string,
    onProgress?: (status: StatusResponse) => void,
    pollInterval: number = 2000,
    maxDuration: number = 300000
  ): Promise<StatusResponse> {
    this.log('Starting status polling for session:', sessionId);
    
    const startTime = Date.now();
    
    while (true) {
      // Check if max duration exceeded
      if (Date.now() - startTime > maxDuration) {
        throw new Error('Analysis timeout. Please try again.');
      }

      const status = await this.getAnalysisStatus(sessionId);
      
      // Call progress callback if provided
      if (onProgress) {
        onProgress(status);
      }

      // Check if analysis is complete or failed
      if (status.status === 'completed' || status.status === 'failed') {
        this.log('Analysis finished with status:', status.status);
        return status;
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
  }

  /**
   * Get analysis results for a completed session
   */
  async getAnalysisResults(sessionId: string): Promise<AnalysisReport> {
    this.log('Fetching results for session:', sessionId);

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/results/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      return this.handleResponse<AnalysisReport>(response);
    } catch (error) {
      this.log('Results fetch error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to fetch analysis results.'
      );
    }
  }

  /**
   * Send a chat message and get a response
   */
  async sendChatMessage(sessionId: string, message: string): Promise<ChatResponse> {
    this.log('Sending chat message for session:', sessionId);

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/chat/${sessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      return this.handleResponse<ChatResponse>(response);
    } catch (error) {
      this.log('Chat message error:', error);
      throw new Error(
        error instanceof Error 
          ? error.message 
          : 'Failed to send message. Please try again.'
      );
    }
  }

  /**
   * Retry a failed request with exponential backoff
   */
  async retryRequest<T>(
    requestFn: () => Promise<T>,
    maxRetries: number = 3,
    initialDelay: number = 1000
  ): Promise<T> {
    let lastError: Error | null = null;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        this.log(`Retry attempt ${i + 1}/${maxRetries} failed:`, lastError.message);
        
        if (i < maxRetries - 1) {
          const delay = initialDelay * Math.pow(2, i);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError || new Error('Request failed after retries');
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
