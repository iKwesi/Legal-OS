import React from 'react';
import { Card } from '@/components/ui/card';
import { RedFlag } from '@/types/api';

interface RedFlagAlertProps {
  redFlag: RedFlag;
}

export function RedFlagAlert({ redFlag }: RedFlagAlertProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-50 border-red-300';
      case 'high':
        return 'bg-red-50 border-red-200';
      case 'medium':
        return 'bg-yellow-50 border-yellow-200';
      case 'low':
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const getSeverityIcon = (severity: string) => {
    if (severity === 'critical' || severity === 'high') {
      return (
        <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    }
    return (
      <svg className="w-5 h-5 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    );
  };

  return (
    <Card className={`p-4 border-l-4 ${getSeverityColor(redFlag.severity)}`}>
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          {getSeverityIcon(redFlag.severity)}
        </div>
        <div className="flex-1">
          <div className="flex items-start justify-between mb-2">
            <h4 className="font-semibold text-gray-900">{redFlag.title}</h4>
            <span className="text-xs font-medium text-gray-600 ml-2">
              Score: {redFlag.risk_score.toFixed(2)}
            </span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed mb-2">
            {redFlag.description}
          </p>
          {redFlag.recommendation && (
            <div className="mt-3 p-3 bg-white rounded-md border border-gray-200">
              <p className="text-xs font-medium text-gray-700 mb-1">Recommendation:</p>
              <p className="text-xs text-gray-600">{redFlag.recommendation}</p>
            </div>
          )}
          {redFlag.related_clauses && redFlag.related_clauses.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {redFlag.related_clauses.map((clause, idx) => (
                <span key={idx} className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-700">
                  {clause}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
