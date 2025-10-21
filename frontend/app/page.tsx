'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Header } from '@/components/layout/header';
import { FileDropzone } from '@/components/upload/file-dropzone';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/ui/loading';
import { ErrorMessage } from '@/components/ui/error';
import { apiClient } from '@/lib/api';

export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      const response = await apiClient.uploadDocument(selectedFile);
      
      clearInterval(progressInterval);
      setUploadProgress(100);

      // Navigate to results page after short delay
      setTimeout(() => {
        router.push(`/results/${response.session_id}`);
      }, 500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload document');
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Header />
      
      <main className="flex-grow flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8">
        <div className="w-full max-w-4xl mx-auto flex flex-col items-center text-center">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-gray-900 mb-4">
            Upload Contracts for Analysis
          </h2>
          <p className="max-w-2xl text-gray-600 mb-10">
            Securely upload your legal documents. Our AI will analyze them for key clauses, risks, and opportunities.
          </p>

          <div className="w-full p-4">
            <FileDropzone onFileSelect={handleFileSelect} />
          </div>

          {selectedFile && !isUploading && (
            <div className="w-full max-w-3xl mt-6">
              <div className="flex items-center justify-between p-4 rounded-lg bg-white border border-gray-200 shadow-sm">
                <div className="flex items-center gap-3">
                  <svg className="w-6 h-6 text-primary" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20Z" />
                  </svg>
                  <div className="text-left">
                    <p className="text-sm font-medium text-gray-900">{selectedFile.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {isUploading && (
            <div className="w-full max-w-3xl mt-6">
              <div className="p-6 rounded-lg bg-white border border-gray-200 shadow-sm">
                <div className="flex items-center gap-4 mb-4">
                  <LoadingSpinner size="md" />
                  <div className="flex-1 text-left">
                    <p className="text-sm font-medium text-gray-900">Uploading and analyzing...</p>
                    <p className="text-xs text-gray-500">This may take a few moments</p>
                  </div>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="w-full max-w-3xl mt-6">
              <ErrorMessage message={error} onRetry={() => setError(null)} />
            </div>
          )}

          <div className="w-full mt-10 flex justify-center">
            <Button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              className="min-w-[200px] h-12 px-6 bg-primary text-white text-base font-bold shadow-lg hover:bg-primary/90 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {isUploading ? 'Analyzing...' : 'Start Analysis'}
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
