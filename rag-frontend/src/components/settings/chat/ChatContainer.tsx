import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Trash2, Download, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import ChatMessage from './ChatMessage';
import { useStore } from '@/store/useStore';
import { cn } from '@/lib/utils';

const TypingIndicator: React.FC = () => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -10 }}
    className="flex gap-3 mb-4"
  >
    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 flex items-center justify-center">
      <MessageSquare className="h-4 w-4" />
    </div>
    <div className="bg-muted px-4 py-3 rounded-lg">
      <div className="typing-indicator">
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
      </div>
    </div>
  </motion.div>
);

const EmptyState: React.FC = () => {
  const { isIngestionComplete } = useStore();

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex-1 flex items-center justify-center p-8"
    >
      <div className="text-center space-y-4 max-w-md">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto">
          <MessageSquare className="h-8 w-8 text-muted-foreground" />
        </div>
        
        <div className="space-y-2">
          <h3 className="text-lg font-semibold">
            {isIngestionComplete ? 'Ready to Chat' : 'Welcome to RAG Pipeline'}
          </h3>
          <p className="text-muted-foreground text-sm leading-relaxed">
            {isIngestionComplete 
              ? 'Ask questions about your documents using the configured pipeline settings.'
              : 'Configure your pipeline settings and run ingestion to start chatting with your documents.'
            }
          </p>
        </div>

        {isIngestionComplete && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground font-medium">Try asking:</p>
            <div className="flex flex-wrap gap-2 justify-center">
              <Badge variant="outline" className="text-xs">
                "What are the main topics?"
              </Badge>
              <Badge variant="outline" className="text-xs">
                "Summarize the key findings"
              </Badge>
              <Badge variant="outline" className="text-xs">
                "Find information about..."
              </Badge>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const ChatContainer: React.FC = () => {
  const { messages, chatState, clearMessages } = useStore();
  const scrollAreaRef = React.useRef<HTMLDivElement>(null);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const exportChat = () => {
    const chatData = {
      timestamp: new Date().toISOString(),
      messages: messages.map(msg => ({
        type: msg.type,
        content: msg.content,
        timestamp: msg.timestamp.toISOString(),
        metadata: msg.metadata,
      })),
    };

    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
      type: 'application/json',
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rag-chat-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const hasMessages = messages.length > 0;
  const isLoading = chatState === 'loading';

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            <h1 className="text-lg font-semibold">RAG Pipeline Chat</h1>
          </div>
          
          {hasMessages && (
            <Badge variant="outline" className="text-xs">
              {messages.length} message{messages.length !== 1 ? 's' : ''}
            </Badge>
          )}
        </div>

        {hasMessages && (
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={exportChat}
                    className="gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Export
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Export chat history</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearMessages}
                    className="gap-2"
                  >
                    <Trash2 className="h-4 w-4" />
                    Clear
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Clear chat history</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>

      {/* Chat Messages */}
      <div className="flex-1 relative">
        {hasMessages ? (
          <ScrollArea ref={scrollAreaRef} className="h-full">
            <div className="p-4 space-y-4">
              <AnimatePresence initial={false}>
                {messages.map((message, index) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    isLast={index === messages.length - 1}
                  />
                ))}
              </AnimatePresence>

              {/* Typing Indicator */}
              <AnimatePresence>
                {isLoading && <TypingIndicator />}
              </AnimatePresence>

              {/* Scroll anchor */}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        ) : (
          <EmptyState />
        )}

        {/* Performance Indicator */}
        {chatState === 'loading' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute top-4 right-4"
          >
            <div className="bg-background/95 backdrop-blur border rounded-lg px-3 py-2 shadow-lg">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Zap className="h-3 w-3" />
                Processing query...
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default ChatContainer;