import React from 'react';
import { motion } from 'framer-motion';
import { User, Bot, Info, Clock, Zap } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Message } from '@/types';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  message: Message;
  isLast?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isLast = false }) => {
  const isUser = message.type === 'user';
  const isSystem = message.type === 'system';
  
  const formatTime = (timestamp: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(timestamp);
  };

  const getIcon = () => {
    if (isUser) return <User className="h-4 w-4" />;
    if (isSystem) return <Info className="h-4 w-4" />;
    return <Bot className="h-4 w-4" />;
  };

  const getMessageVariant = () => {
    if (isUser) return 'user';
    if (isSystem) return 'system';
    return 'assistant';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={cn(
        'flex gap-3 group',
        isUser && 'justify-end',
        isLast && 'mb-4'
      )}
    >
      {!isUser && (
        <div className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isSystem ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600',
          'dark:bg-gray-800 dark:text-gray-300'
        )}>
          {getIcon()}
        </div>
      )}

      <div className={cn(
        'flex flex-col gap-1 max-w-[80%]',
        isUser && 'items-end'
      )}>
        <Card className={cn(
          'px-4 py-3 relative',
          isUser && 'bg-primary text-primary-foreground',
          isSystem && 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800',
          !isUser && !isSystem && 'bg-muted'
        )}>
          <div className={cn(
            'text-sm leading-relaxed whitespace-pre-wrap',
            isUser && 'text-primary-foreground'
          )}>
            {message.content}
          </div>

          {/* Metadata badges for assistant responses */}
          {message.metadata && !isUser && !isSystem && (
            <div className="flex flex-wrap gap-1 mt-3 pt-3 border-t border-border/50">
              {message.metadata.queryTranslation && (
                <Badge variant="outline" className="text-xs">
                  {message.metadata.queryTranslation}
                </Badge>
              )}
              {message.metadata.reranking && (
                <Badge variant="outline" className="text-xs">
                  {message.metadata.reranking}
                </Badge>
              )}
              {message.metadata.embeddingModel && (
                <Badge variant="outline" className="text-xs">
                  {message.metadata.embeddingModel}
                </Badge>
              )}
              {message.metadata.topK && (
                <Badge variant="outline" className="text-xs">
                  Top {message.metadata.topK}
                </Badge>
              )}
              {message.metadata.processingTime && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" className="text-xs flex items-center gap-1">
                        <Zap className="h-3 w-3" />
                        {message.metadata.processingTime}ms
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>Processing time</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          )}
        </Card>

        {/* Timestamp */}
        <div className={cn(
          'flex items-center gap-1 text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity',
          isUser && 'flex-row-reverse'
        )}>
          <Clock className="h-3 w-3" />
          <span>{formatTime(message.timestamp)}</span>
        </div>
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center">
          {getIcon()}
        </div>
      )}
    </motion.div>
  );
};

export default ChatMessage;