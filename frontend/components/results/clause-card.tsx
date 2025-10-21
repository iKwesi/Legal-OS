import React from 'react';
import { Card } from '@/components/ui/card';
import { ExtractedClause } from '@/types/api';

interface ClauseCardProps {
  clause: ExtractedClause;
}

export function ClauseCard({ clause }: ClauseCardProps) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskBadgeColor = (level: string) => {
    switch (level) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'low':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const riskLevel = clause.risk_level || 'medium';
  const riskScore = clause.risk_score || 0;
  const clauseText = clause.text || clause.clause_text || 'No text available';

  return (
    <Card className={`p-4 border-l-4 ${getRiskColor(riskLevel)}`}>
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-semibold text-gray-900">{clause.clause_type}</h4>
        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getRiskBadgeColor(riskLevel)}`}>
          {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk
        </span>
      </div>
      <p className="text-sm text-gray-700 leading-relaxed mb-2">{clauseText}</p>
      <div className="flex items-center gap-4 text-xs text-gray-500">
        {clause.page_number && (
          <span>Page {clause.page_number}</span>
        )}
        {clause.confidence && (
          <span>Confidence: {Math.round(clause.confidence * 100)}%</span>
        )}
        <span>Risk Score: {riskScore.toFixed(2)}</span>
      </div>
    </Card>
  );
}
