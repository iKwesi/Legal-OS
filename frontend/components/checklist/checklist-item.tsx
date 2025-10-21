import React from 'react';
import { ChecklistItem as ChecklistItemType } from '@/types/api';

interface ChecklistItemProps {
  item: ChecklistItemType;
  onToggle: (id: string) => void;
}

export function ChecklistItem({ item, onToggle }: ChecklistItemProps) {
  const getPriorityColor = (priority?: string) => {
    switch (priority) {
      case 'high':
        return 'text-red-600';
      case 'medium':
        return 'text-yellow-600';
      case 'low':
        return 'text-blue-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <li className="flex items-start group">
      <input
        type="checkbox"
        id={item.id}
        checked={item.completed}
        onChange={() => onToggle(item.id)}
        className="peer h-5 w-5 mt-0.5 shrink-0 rounded border-gray-300 text-primary focus:ring-primary/50 cursor-pointer"
      />
      <label
        htmlFor={item.id}
        className="ml-3 text-sm text-gray-700 peer-checked:line-through peer-checked:text-gray-500 cursor-pointer flex-1"
      >
        {item.text}
        {item.priority && (
          <span className={`ml-2 text-xs font-medium ${getPriorityColor(item.priority)}`}>
            ({item.priority})
          </span>
        )}
        {item.related_red_flags && item.related_red_flags.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {item.related_red_flags.map((flag, idx) => (
              <span
                key={idx}
                className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-red-50 text-red-700 border border-red-200"
              >
                ⚠ {flag}
              </span>
            ))}
          </div>
        )}
      </label>
    </li>
  );
}
