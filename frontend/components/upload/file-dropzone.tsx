'use client';

import React, { useState, useCallback } from 'react';
import { Card } from '@/components/ui/card';

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number; // in MB
}

export function FileDropzone({ onFileSelect, accept = '.pdf', maxSize = 10 }: FileDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    setError(null);

    // Check file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are allowed');
      return false;
    }

    // Check file size (convert MB to bytes)
    const maxSizeBytes = maxSize * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      setError(`File size must be less than ${maxSize}MB`);
      return false;
    }

    return true;
  };

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        const file = files[0];
        if (validateFile(file)) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect, maxSize]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (validateFile(file)) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect, maxSize]
  );

  return (
    <div className="w-full">
      <Card
        className={`relative flex flex-col items-center justify-center w-full h-64 px-6 py-14 border-2 border-dashed transition-all duration-300 cursor-pointer ${
          isDragging
            ? 'border-primary bg-primary/5'
            : 'border-gray-300 hover:border-primary hover:bg-gray-50'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="flex flex-col items-center gap-4 text-center">
          <svg
            className="w-12 h-12 text-primary"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <div>
            <p className="text-lg font-semibold text-gray-700">Drag & Drop Contract Here</p>
            <p className="text-sm text-gray-500 mt-1">or</p>
          </div>
          <label
            htmlFor="file-input"
            className="cursor-pointer inline-flex items-center justify-center h-10 px-5 rounded-lg bg-primary/10 text-primary text-sm font-semibold hover:bg-primary/20 transition-colors"
          >
            Browse Files
          </label>
          <input
            id="file-input"
            type="file"
            accept={accept}
            onChange={handleFileInput}
            className="sr-only"
          />
          <p className="text-xs text-gray-500 mt-2">PDF files only, max {maxSize}MB</p>
        </div>
      </Card>
      {error && (
        <p className="mt-2 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}
