import { IngestionConfig, PipelineConfig, IngestionResponse, RetrievalResponse, EvaluationInput, EvaluationResponse } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'APIError';
  }
}

async function fetchAPI<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new APIError(response.status, errorData.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export const api = {
  // Health check
  async healthCheck(): Promise<{ message: string; endpoints: string[] }> {
    return fetchAPI('/');
  },

  // Ingestion endpoint
  async runIngestion(config: IngestionConfig): Promise<IngestionResponse> {
    return fetchAPI('/ingestion', {
      method: 'POST',
      body: JSON.stringify({
        chunking_strategy: config.chunkingStrategy,
        embedding_model: config.embeddingModel,
        vectordb: config.vectordb,
      }),
    });
  },

  // Retrieval endpoint
  async runRetrieval(query: string, config: PipelineConfig): Promise<RetrievalResponse> {
    return fetchAPI('/retrieval', {
      method: 'POST',
      body: JSON.stringify({
        query,
        embedding_model: config.embeddingModel,
        top_k: config.topK,
      }),
    });
  },

  // Evaluation endpoint
  async runEvaluation(input: EvaluationInput): Promise<EvaluationResponse> {
    return fetchAPI('/evaluation', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  },
};

export { APIError };