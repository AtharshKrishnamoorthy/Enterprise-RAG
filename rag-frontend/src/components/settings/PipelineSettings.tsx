import React from 'react';
import { motion } from 'framer-motion';
import { Settings, Cpu, Database, Search, Shuffle, Layers } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useStore } from '@/store/useStore';
import { api } from '@/lib/api';
import { useToast } from '@/components/ui/use-toast';

const PipelineSettings: React.FC = () => {
  const {
    pipelineConfig,
    ingestionConfig,
    setPipelineConfig,
    setIngestionConfig,
    resetPipelineConfig,
    resetIngestionConfig,
    setIngestionComplete,
    addMessage,
  } = useStore();

  const { toast } = useToast();
  const [isRunningIngestion, setIsRunningIngestion] = React.useState(false);

  const handleRunIngestion = async () => {
    setIsRunningIngestion(true);
    
    try {
      addMessage({
        type: 'system',
        content: 'Starting ingestion process...',
      });

      const response = await api.runIngestion(ingestionConfig);
      
      setIngestionComplete(true);
      
      addMessage({
        type: 'system',
        content: `✅ Ingestion completed successfully! Processed ${response.details.files_processed} files using ${response.details.chunking_strategy} chunking and ${response.details.embedding_model} embeddings.`,
      });

      toast({
        title: "Ingestion Successful",
        description: `Processed ${response.details.files_processed} transcript files`,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
      addMessage({
        type: 'system',
        content: `❌ Ingestion failed: ${errorMessage}`,
      });

      toast({
        title: "Ingestion Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsRunningIngestion(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Settings className="h-6 w-6" />
        <h2 className="text-2xl font-semibold">Pipeline Configuration</h2>
      </div>

      {/* Ingestion Settings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Ingestion Pipeline
            </CardTitle>
            <CardDescription>
              Configure how documents are processed and stored in the vector database
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  Chunking Strategy
                </label>
                <Select
                  value={ingestionConfig.chunkingStrategy}
                  onValueChange={(value: any) => setIngestionConfig({ chunkingStrategy: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fixed">Fixed Size</SelectItem>
                    <SelectItem value="semantic">Semantic</SelectItem>
                    <SelectItem value="raptor">RAPTOR</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  Embedding Model
                </label>
                <Select
                  value={ingestionConfig.embeddingModel}
                  onValueChange={(value: any) => setIngestionConfig({ embeddingModel: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="huggingface">Hugging Face</SelectItem>
                    <SelectItem value="google">Google</SelectItem>
                    <SelectItem value="openai">OpenAI</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleRunIngestion}
                disabled={isRunningIngestion}
                className="flex-1"
              >
                {isRunningIngestion ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Running Ingestion...
                  </>
                ) : (
                  'Run Ingestion'
                )}
              </Button>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={resetIngestionConfig}
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Reset to defaults</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <Separator />

      {/* Retrieval Settings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Retrieval Pipeline
            </CardTitle>
            <CardDescription>
              Configure how queries are processed and relevant documents are retrieved
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Query Translation</label>
                <Select
                  value={pipelineConfig.queryTranslationStrategy}
                  onValueChange={(value: any) => setPipelineConfig({ queryTranslationStrategy: value })}
                >
                  <SelectTrigger>
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

              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Shuffle className="h-4 w-4" />
                  Reranking Strategy
                </label>
                <Select
                  value={pipelineConfig.rerankingStrategy}
                  onValueChange={(value: any) => setPipelineConfig({ rerankingStrategy: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="reciprocal_rank_fusion">Reciprocal Rank Fusion</SelectItem>
                    <SelectItem value="cross_encoder">Cross Encoder</SelectItem>
                    <SelectItem value="cohere">Cohere</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  Embedding Model
                </label>
                <Select
                  value={pipelineConfig.embeddingModel}
                  onValueChange={(value: any) => setPipelineConfig({ embeddingModel: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="huggingface">Hugging Face</SelectItem>
                    <SelectItem value="google">Google</SelectItem>
                    <SelectItem value="openai">OpenAI</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Top K Results</label>
                <Select
                  value={pipelineConfig.topK.toString()}
                  onValueChange={(value) => setPipelineConfig({ topK: parseInt(value) })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="3">3</SelectItem>
                    <SelectItem value="5">5</SelectItem>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="15">15</SelectItem>
                    <SelectItem value="20">20</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex gap-2 pt-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      onClick={resetPipelineConfig}
                      className="flex-1"
                    >
                      <Settings className="h-4 w-4 mr-2" />
                      Reset to Defaults
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Reset retrieval settings to defaults</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>

            {/* Current Configuration Summary */}
            <div className="space-y-2 pt-4 border-t">
              <p className="text-sm font-medium text-muted-foreground">Current Configuration:</p>
              <div className="flex flex-wrap gap-1">
                <Badge variant="secondary">{pipelineConfig.queryTranslationStrategy}</Badge>
                <Badge variant="secondary">{pipelineConfig.rerankingStrategy}</Badge>
                <Badge variant="secondary">{pipelineConfig.embeddingModel}</Badge>
                <Badge variant="secondary">Top {pipelineConfig.topK}</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default PipelineSettings;