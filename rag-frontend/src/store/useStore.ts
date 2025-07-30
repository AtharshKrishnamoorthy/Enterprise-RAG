import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { AppState, Message, PipelineConfig, IngestionConfig, ChatState } from '@/types';

interface StoreActions {
  // Message actions
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  clearMessages: () => void;
  
  // Pipeline config actions
  setPipelineConfig: (config: Partial<PipelineConfig>) => void;
  resetPipelineConfig: () => void;
  
  // Ingestion config actions
  setIngestionConfig: (config: Partial<IngestionConfig>) => void;
  resetIngestionConfig: () => void;
  
  // Chat state actions
  setChatState: (state: ChatState) => void;
  setIngestionComplete: (complete: boolean) => void;
  setCurrentQuery: (query: string) => void;
}

const defaultPipelineConfig: PipelineConfig = {
  queryTranslationStrategy: 'hyde',
  rerankingStrategy: 'cohere',
  embeddingModel: 'huggingface',
  topK: 5,
};

const defaultIngestionConfig: IngestionConfig = {
  chunkingStrategy: 'raptor',
  embeddingModel: 'huggingface',
  vectordb: 'faiss',
};

export const useStore = create<AppState & StoreActions>()(
  persist(
    (set, get) => ({
      // Initial state
      messages: [],
      pipelineConfig: defaultPipelineConfig,
      ingestionConfig: defaultIngestionConfig,
      chatState: 'idle',
      isIngestionComplete: false,
      currentQuery: '',

      // Message actions
      addMessage: (message) =>
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: crypto.randomUUID(),
              timestamp: new Date(),
            },
          ],
        })),

      clearMessages: () => set({ messages: [] }),

      // Pipeline config actions
      setPipelineConfig: (config) =>
        set((state) => ({
          pipelineConfig: { ...state.pipelineConfig, ...config },
        })),

      resetPipelineConfig: () => set({ pipelineConfig: defaultPipelineConfig }),

      // Ingestion config actions
      setIngestionConfig: (config) =>
        set((state) => ({
          ingestionConfig: { ...state.ingestionConfig, ...config },
        })),

      resetIngestionConfig: () => set({ ingestionConfig: defaultIngestionConfig }),

      // Chat state actions
      setChatState: (chatState) => set({ chatState }),
      setIngestionComplete: (isIngestionComplete) => set({ isIngestionComplete }),
      setCurrentQuery: (currentQuery) => set({ currentQuery }),
    }),
    {
      name: 'rag-pipeline-store',
      partialize: (state) => ({
        pipelineConfig: state.pipelineConfig,
        ingestionConfig: state.ingestionConfig,
        isIngestionComplete: state.isIngestionComplete,
      }),
    }
  )
);