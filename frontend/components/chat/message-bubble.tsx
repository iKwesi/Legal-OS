import { cn } from '@/lib/utils';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
  provenance?: Array<{
    source: string;
    page?: number;
  }>;
}

export function MessageBubble({ role, content, provenance }: MessageBubbleProps) {
  return (
    <div
      className={cn(
        'flex items-start gap-3',
        role === 'user' ? 'justify-end' : 'justify-start'
      )}
    >
      {role === 'assistant' && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold text-sm">
          AI
        </div>
      )}

      <div
        className={cn(
          'flex flex-col gap-1.5 max-w-xl',
          role === 'user' ? 'items-end' : 'items-start'
        )}
      >
        <p className="text-xs font-medium text-muted-foreground">
          {role === 'user' ? 'You' : 'AI Assistant'}
        </p>
        <div
          className={cn(
            'rounded-lg p-3 text-sm transition-all',
            role === 'user'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted/60 text-foreground'
          )}
        >
          {content}
        </div>

        {/* Provenance sources */}
        {provenance && provenance.length > 0 && (
          <div className="flex flex-col gap-2 w-full mt-1">
            {provenance.map((source, idx) => (
              <div
                key={idx}
                className="flex items-center gap-3 rounded-lg border border-border bg-background p-3 cursor-pointer hover:bg-accent transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-primary">
                    {source.page || '?'}
                  </span>
                </div>
                <p className="text-sm font-medium text-foreground line-clamp-2 flex-1">
                  {source.source}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {role === 'user' && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-secondary flex items-center justify-center text-secondary-foreground font-semibold text-sm">
          U
        </div>
      )}
    </div>
  );
}
