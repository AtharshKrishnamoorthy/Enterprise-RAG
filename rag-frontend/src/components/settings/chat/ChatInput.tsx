import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Settings, Mic, Upload, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useStore } from '@/store/useStore';
import { api } from '@/lib/api';
import { useToast } from '@/components/ui/use-toast';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onToggleSettings: () => void;
  showSettings: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onToggleSettings, showSettings }) => {
  const {
    currentQuery,
    setCurrentQuery,
    chatState,
    setChatState,
    pipelineConfig,
    setPipelineConfig,
    isIngestionComplete,
    addMessage,
  } = useStore();

  const { toast } = useToast();
  const [showQuickSettings, setShowQuickSettings] = React.useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentQuery.trim()) return;
    
    if (!isIngestionComplete) {
      toast({
        title: "Ingestion Required",
        description: "Please run ingestion first to index your documents",
        variant: "destructive",
      });
      return;
    }

    const query = currentQuery.trim();
    setCurrentQuery('');
    setChatState('loading');

    // Add user message
    addMessage({
      type: 'user',
      content: query,
    });

    try {
      const startTime = Date.now();
      const response = await api.runRetrieval(query, pipelineConfig);
      const processingTime = Date.now() - startTime;

      // Add assistant response
      addMessage({
        type: 'assistant',
        content: response.answer,
        metadata: {
          queryTranslation: response.details.query_translation_strategy,
          reranking: response.details.reranking_strategy,
          embeddingModel: response.details.embedding_model,
          topK: response.details.top_k,
          processingTime,
        },
      });

      setChatState('idle');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
      addMessage({
        type: 'system',
        content: `❌ Failed to retrieve answer: ${errorMessage}`,
      });

      setChatState('error');
      
      toast({
        title: "Query Failed",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const isLoading = chatState === 'loading';

  return (
    <div className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="p-4 space-y-3">
        {/* Quick Pipeline Settings */}
        <AnimatePresence>
          {showQuickSettings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="bg-muted/50 rounded-lg p-3 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Quick Pipeline Settings</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowQuickSettings(false)}
                    className="h-6 w-6 p-0"
                  >
                    ×
                  </Button>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Query Translation</label>
                    <Select
                      value={pipelineConfig.queryTranslationStrategy}
                      onValueChange={(value: any) => setPipelineConfig({ queryTranslationStrategy: value })}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="multi_query">Multi Query</SelectItem>
                        <SelectItem value="decomposition">Decomposition</SelectItem>
                        <SelectItem value="hyde">HyDE</SelectItem>
                        <SelectItem value="step_back">Step Back</SelectItem>
                        <SelectItem value="none">None</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Reranking</label>
                    <Select
                      value={pipelineConfig.rerankingStrategy}
                      onValueChange={(value: any) => setPipelineConfig({ rerankingStrategy: value })}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="reciprocal_rank_fusion">RRF</SelectItem>
                        <SelectItem value="cross_encoder">Cross Encoder</SelectItem>
                        <SelectItem value="cohere">Cohere</SelectItem>
                        <SelectItem value="none">None</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Current Configuration Display */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Pipeline:</span>
            <div className="flex gap-1">
              <Badge variant="outline" className="text-xs px-1.5 py-0.5">
                {pipelineConfig.queryTranslationStrategy}
              </Badge>
              <Badge variant="outline" className="text-xs px-1.5 py-0.5">
                {pipelineConfig.rerankingStrategy}
              </Badge>
              <Badge variant="outline" className="text-xs px-1.5 py-0.5">
                {pipelineConfig.embeddingModel}
              </Badge>
            </div>
          </div>
          
          {!isIngestionComplete && (
            <Badge variant="destructive" className="text-xs">
              Ingestion Required
            </Badge>
          )}
        </div>

        <Separator />

        {/* Chat Input */}
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <div className="flex-1 relative">
            <Input
              value={currentQuery}
              onChange={(e) => setCurrentQuery(e.target.value)}
              placeholder={
                isIngestionComplete 
                  ? "Ask a question about your documents..." 
                  : "Run ingestion first to start chatting..."
              }
              disabled={isLoading || !isIngestionComplete}
              className="pr-32"
              autoComplete="off"
            />
            
            {/* Input Actions */}
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      onClick={() => setShowQuickSettings(!showQuickSettings)}
                    >
                      <Settings className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Quick settings</TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      disabled
                    >
                      <Mic className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Voice input (coming soon)</TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0"
                      disabled
                    >
                      <Upload className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Upload file (coming soon)</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          <div className="flex gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={onToggleSettings}
                    className={cn(
                      'transition-colors',
                      showSettings && 'bg-muted'
                    )}
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Open pipeline settings</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <Button
              type="submit"
              size="sm"
              disabled={isLoading || !currentQuery.trim() || !isIngestionComplete}
              className="min-w-[60px]"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInput;