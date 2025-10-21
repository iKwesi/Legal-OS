'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Header } from '@/components/layout/header';
import { ClauseCard } from '@/components/results/clause-card';
import { RedFlagAlert } from '@/components/results/red-flag-alert';
import { Button } from '@/components/ui/button';
import { Loading } from '@/components/ui/loading';
import { ErrorMessage } from '@/components/ui/error';
import { apiClient } from '@/lib/api';
import { AnalysisReport } from '@/types/api';

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setIsLoading(true);
        const data = await apiClient.getAnalysisResults(sessionId);
        setReport(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load analysis report');
      } finally {
        setIsLoading(false);
      }
    };

    if (sessionId) {
      fetchReport();
    }
  }, [sessionId]);

  if (isLoading) {
    return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        <Header />
        <main className="flex-grow flex items-center justify-center">
          <div className="text-center">
            <Loading />
            <p className="mt-4 text-gray-600">Loading analysis results...</p>
          </div>
        </main>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        <Header />
        <main className="flex-grow flex items-center justify-center p-4">
          <div className="max-w-2xl w-full">
            <ErrorMessage
              message={error || 'Report not found'}
              onRetry={() => window.location.reload()}
            />
            <div className="mt-6 text-center">
              <Button onClick={() => router.push('/')}>
                Back to Upload
              </Button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Header />

      <div className="border-b border-gray-200 bg-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <nav className="flex items-center gap-8 text-sm font-medium">
              <span className="text-primary border-b-2 border-primary pb-3">Summary</span>
              <Link
                href={`/chat/${sessionId}`}
                className="text-gray-600 hover:text-primary transition-colors pb-3"
              >
                Chat
              </Link>
              <Link
                href={`/checklist/${sessionId}`}
                className="text-gray-600 hover:text-primary transition-colors pb-3"
              >
                Checklist
              </Link>
            </nav>
            <Button
              onClick={() => router.push('/')}
              variant="outline"
              className="flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Analysis
            </Button>
          </div>
        </div>
      </div>

      <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar - Key Clauses */}
          <aside className="lg:col-span-1">
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden sticky top-4">
              <div className="p-4 border-b border-gray-200">
                <h3 className="text-lg font-bold text-gray-900">Key Clauses</h3>
              </div>
              <div className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
                {report.extracted_clauses.slice(0, 5).map((clause, idx) => (
                  <div
                    key={idx}
                    className="p-3 hover:bg-gray-50 cursor-pointer transition-colors"
                  >
                    <p className="font-semibold text-sm text-gray-800">{clause.clause_type}</p>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium mt-1 ${
                      clause.risk_level === 'high' ? 'bg-red-100 text-red-800' :
                      clause.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {clause.risk_level.charAt(0).toUpperCase() + clause.risk_level.slice(1)} Risk
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <section className="lg:col-span-3">
            <div className="space-y-8">
              {/* Header */}
              <div>
                <h1 className="text-4xl font-bold text-gray-900 mb-2">
                  AI-Generated Summary Report
                </h1>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span>{report.document_name}</span>
                  <span>•</span>
                  <span>{new Date(report.analysis_date).toLocaleDateString()}</span>
                  <span>•</span>
                  <span>Risk Score: {report.overall_risk_score.toFixed(2)}</span>
                </div>
              </div>

              {/* Summary Section */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4 border-b border-gray-200 pb-2">
                  Executive Summary
                </h2>
                <div className="prose prose-sm max-w-none">
                  <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {report.summary}
                  </p>
                </div>
              </div>

              {/* Red Flags Section */}
              {report.red_flags && report.red_flags.length > 0 && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-4">Red Flags</h2>
                  <div className="space-y-4">
                    {report.red_flags.map((flag, idx) => (
                      <RedFlagAlert key={idx} redFlag={flag} />
                    ))}
                  </div>
                </div>
              )}

              {/* Extracted Clauses Section */}
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Extracted Clauses</h2>
                <div className="space-y-4">
                  {report.extracted_clauses.map((clause, idx) => (
                    <ClauseCard key={idx} clause={clause} />
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
