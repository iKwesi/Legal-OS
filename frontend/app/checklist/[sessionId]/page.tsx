'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Header } from '@/components/layout/header';
import { ChecklistItem } from '@/components/checklist/checklist-item';
import { Button } from '@/components/ui/button';
import { Loading } from '@/components/ui/loading';
import { ErrorMessage } from '@/components/ui/error';
import { apiClient } from '@/lib/api';
import { ChecklistItem as ChecklistItemType } from '@/types/api';

export default function ChecklistPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  const [checklistItems, setChecklistItems] = useState<ChecklistItemType[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChecklist = async () => {
      try {
        setIsLoading(true);
        const report = await apiClient.getAnalysisResults(sessionId);
        setChecklistItems(report.checklist || report.checklist_items || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load checklist');
      } finally {
        setIsLoading(false);
      }
    };

    if (sessionId) {
      fetchChecklist();
    }
  }, [sessionId]);

  const handleToggle = (id: string) => {
    setChecklistItems((items) =>
      items.map((item) => {
        const itemId = item.id || item.item_id;
        return itemId === id ? { ...item, completed: !item.completed } : item;
      })
    );
  };

  const handleExport = () => {
    const content = checklistItems
      .map((item) => `${item.completed ? '[x]' : '[ ]'} ${item.text}`)
      .join('\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `checklist-${sessionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Group items by category
  const groupedItems = checklistItems.reduce((acc, item) => {
    const category = item.category || 'General';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(item);
    return acc;
  }, {} as Record<string, ChecklistItemType[]>);

  if (isLoading) {
    return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        <Header />
        <main className="flex-grow flex items-center justify-center">
          <div className="text-center">
            <Loading />
            <p className="mt-4 text-gray-600">Loading checklist...</p>
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        <Header />
        <main className="flex-grow flex items-center justify-center p-4">
          <div className="max-w-2xl w-full">
            <ErrorMessage message={error} onRetry={() => window.location.reload()} />
            <div className="mt-6 text-center">
              <Button onClick={() => router.push('/')}>Back to Upload</Button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  const completedCount = checklistItems.filter((item) => item.completed).length;
  const totalCount = checklistItems.length;
  const progressPercentage = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Header />

      <div className="border-b border-gray-200 bg-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <nav className="flex items-center gap-8 text-sm font-medium">
              <Link
                href={`/results/${sessionId}`}
                className="text-gray-600 hover:text-primary transition-colors pb-3"
              >
                Summary
              </Link>
              <Link
                href={`/chat/${sessionId}`}
                className="text-gray-600 hover:text-primary transition-colors pb-3"
              >
                Chat
              </Link>
              <span className="text-primary border-b-2 border-primary pb-3">Checklist</span>
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

      <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">
                Diligence Checklist
              </h2>
              <p className="text-sm text-gray-600">
                {completedCount} of {totalCount} items completed ({Math.round(progressPercentage)}%)
              </p>
            </div>
            <div className="flex gap-3 mt-4 md:mt-0">
              <Button
                onClick={handleExport}
                variant="outline"
                className="flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export
              </Button>
              <Link href={`/results/${sessionId}`}>
                <Button variant="outline" className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                  Back to Results
                </Button>
              </Link>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
          </div>

          {/* Checklist Items */}
          <div className="bg-white border border-gray-200 rounded-xl shadow-sm">
            {Object.entries(groupedItems).map(([category, items], categoryIdx) => (
              <div
                key={category}
                className={categoryIdx > 0 ? 'border-t border-gray-200' : ''}
              >
                <div className="p-4 sm:p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">{category}</h3>
                  <ul className="space-y-4">
                    {items.map((item) => (
                      <ChecklistItem key={item.id || item.item_id} item={item} onToggle={handleToggle} />
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>

          {checklistItems.length === 0 && (
            <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-12 text-center">
              <p className="text-gray-500">No checklist items available for this analysis.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
