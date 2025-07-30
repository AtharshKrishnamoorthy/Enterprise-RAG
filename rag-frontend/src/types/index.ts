// Types for the RAG Pipeline Frontend

export interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    queryTranslation?: string;
    reranking?: string;
    embeddingModel?: string;
    topK?: number;
    processingTime?: number;
  };
}

export interface PipelineConfig {
  queryTranslationStrategy: 'multi_query' | 'decomposition' | 'hyde' | 'step_back' | 'none';
  rerankingStrategy: 'reciprocal_rank_fusion' | 'cross_encoder' | 'cohere' | 'none';
  embeddingModel: 'huggingface' | 'google' | 'openai';
  topK: number;
}

export interface IngestionConfig {
  chunkingStrategy: 'fixed' | 'semantic' | 'raptor';
  embeddingModel: 'huggingface' | 'google' | 'openai';
  vectordb: 'faiss';
}

export interface IngestionResponse {
  status: string;
  message: string;
  details: {
    files_processed: number;
    file_paths: string[];
    chunking_strategy: string;
    embedding_model: string;
    vectordb: string;
  };
}

export interface RetrievalResponse {
  status: string;
  query: string;
  answer: string;
  details: {
    embedding_model: string;
    top_k: number;
    query_translation_strategy: string;
    reranking_strategy: string;
  };
}

export interface EvaluationInput {
  query: string;
  answer: string;
  retrieval_context: string;
  expected_output: string;
  eval_llm: string;
}

export interface EvaluationResponse {
  status: string;
  query: string;
  results: Record<string, any>;
  summary: {
    average_score: number;
    metrics_evaluated: number;
  };
  details: {
    eval_llm: string;
    answer_length: number;
    context_length: number;
  };
}

export type ChatState = 'idle' | 'loading' | 'error';

export interface AppState {
  messages: Message[];
  pipelineConfig: PipelineConfig;
  ingestionConfig: IngestionConfig;
  chatState: ChatState;
  isIngestionComplete: boolean;
  currentQuery: string;
}