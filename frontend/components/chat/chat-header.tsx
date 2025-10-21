'use client';

import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';

interface ChatHeaderProps {
  sessionId: string;
}

export function ChatHeader({ sessionId }: ChatHeaderProps) {
  const router = useRouter();

  return (
    <header className="flex items-center justify-between border-b border-border bg-background px-6 py-3">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => router.push(`/results/${sessionId}`)}
          className="mr-2"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <div className="text-sm text-muted-foreground mb-1">
            <span
              className="hover:text-primary cursor-pointer"
              onClick={() => router.push(`/results/${sessionId}`)}
            >
              Analysis Results
            </span>
            <span className="mx-1">/</span>
            <span className="text-foreground">Chat</span>
          </div>
          <h1 className="text-2xl font-bold">Chat with Documents</h1>
        </div>
      </div>
    </header>
  );
}
