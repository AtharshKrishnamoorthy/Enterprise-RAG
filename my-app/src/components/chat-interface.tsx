// components/chat-interface.tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Send, Mic, Upload, ChevronDown, Settings, Bot, MessageSquare, Database, Brain, Zap, CheckCircle2, AlertCircle, Copy, RefreshCw, Star } from 'lucide-react';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'framer-motion';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from './ui/chart';
import { RadialBar, RadialBarChart, PolarAngleAxis, PolarRadiusAxis } from "recharts";
import axios from 'axios';

// Proper type definitions based on your API response
interface EvaluationResult {
  score: number;
  reason?: string;
  [key: string]: any;
}

interface EvaluationResponse {
  status: string;
  query: string;
  results: Record<string, EvaluationResult>;
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

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  type: 'text' | 'config' | 'evaluation';
  timestamp: Date;
  config?: {
    query_translation_strategy?: string;
    reranking_strategy?: string;
    embedding_model?: string;
    chunking_strategy?: string;
    vectordb?: string;
  };
  // Fields for evaluation - updated to match API response
  query?: string;
  retrievalContext?: string; // Changed from string[] to string to match API
  finalDocuments?: string[]; // Added to store final documents
  evaluationResults?: EvaluationResponse;
  expectedOutput?: string;
}

const API_BASE_URL = 'http://localhost:8000';

// Safe score color class function with proper null checking
const getScoreColorClass = (score: number | string | null | undefined): string => {
  if (score === null || score === undefined) return 'text-gray-500';
  
  if (typeof score === 'number') {
    if (score >= 0.8) return 'text-green-500';
    else if (score >= 0.5) return 'text-yellow-500';
    else return 'text-red-500';
  } else if (score === 'pass') {
    return 'text-green-500';
  } else if (score === 'fail') {
    return 'text-red-500';
  }
  return 'text-gray-500';
};

// Safe score extraction function
const getScoreValue = (result: any): number | null => {
  if (result && typeof result === 'object' && 'score' in result && typeof result.score === 'number') {
    return result.score;
  }
  return null;
};

// Updated configuration options to match API parameters
const CONFIG_OPTIONS = {
  queryTranslation: {
    none: { label: 'None', description: 'Direct query without transformation' },
    hyde: { label: 'HyDE', description: 'Hypothetical Document Embeddings' },
    multi_query: { label: 'Multi-Query', description: 'Generate multiple query variations' },
    decomposition: { label: 'Decomposition', description: 'Break down complex queries' },
    step_back: { label: 'Step-Back', description: 'Abstract query reasoning' },
  },
  reranking: {
    none: { label: 'None', description: 'No reranking applied' },
    reciprocal_rank_fusion: { label: 'RRF', description: 'Reciprocal Rank Fusion' },
    cross_encoder: { label: 'Cross-Encoder', description: 'Advanced semantic reranking' },
    cohere: { label: 'Cohere', description: 'Cohere reranking API' },
  },
  embeddingModel: {
    huggingface: { label: 'HuggingFace', description: 'Open-source embeddings' },
    google: { label: 'Google', description: 'Google Universal Sentence Encoder' },
    openai: { label: 'OpenAI', description: 'OpenAI text-embedding models' },
  },
  chunkingStrategy: {
    recursive: { label: 'Recursive', description: 'Recursive text splitting' },
    sentence: { label: 'Sentence', description: 'Sentence-based chunking' },
    token: { label: 'Token', description: 'Token-based chunking' },
    raptor: { label: 'RAPTOR', description: 'Recursive clustering approach' },
  },
  vectordb: {
    faiss: { label: 'FAISS', description: 'Facebook AI Similarity Search' },
  },
};

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Configuration states - updated defaults to match API expectations
  const [queryTranslation, setQueryTranslation] = useState('hyde');
  const [reranking, setReranking] = useState('reciprocal_rank_fusion'); // Changed from 'cohere' to match API
  const [embeddingModel, setEmbeddingModel] = useState('huggingface');
  const [chunkingStrategy, setChunkingStrategy] = useState('raptor');
  const [vectordb, setVectordb] = useState('faiss');
  const [topK, setTopK] = useState(5); // Added topK state

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const newUserMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      type: 'text',
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, newUserMessage]);
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      if (currentInput.toLowerCase().includes('ingest files')) {
        const response = await axios.post(`${API_BASE_URL}/ingestion`, {
          chunking_strategy: chunkingStrategy,
          embedding_model: embeddingModel,
          vectordb: vectordb,
        });
        
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString() + '-bot',
            text: `✅ Ingestion completed successfully! ${response.data.message}`,
            sender: 'bot',
            type: 'text',
            timestamp: new Date(),
          },
        ]);
      } else {
        // Updated retrieval API call to include all required parameters
        const response = await axios.post(`${API_BASE_URL}/retrieval`, {
          query: currentInput,
          embedding_model: embeddingModel,
          top_k: topK,
          query_translation_strategy: queryTranslation,
          reranking_strategy: reranking,
        });
        
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString() + '-bot',
            text: response.data.answer,
            sender: 'bot',
            type: 'text',
            timestamp: new Date(),
            query: currentInput,
            retrievalContext: response.data.retrieval_context || '', // Handle as string
            finalDocuments: response.data.expected_output || [], // Store final documents
            expectedOutput: response.data.expected_output ? response.data.expected_output.join('\n') : '',
          },
        ]);
      }
    } catch (error) {
      console.error('API call failed:', error);
      let errorMessage = '❌ Unable to process your request. Please try again.';
      
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          errorMessage = '❌ Vector database not found. Please run ingestion first by typing "ingest files".';
        } else if (error.response?.data?.detail) {
          errorMessage = `❌ Error: ${error.response.data.detail}`;
        }
      }
      
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString() + '-error',
          text: errorMessage,
          sender: 'bot',
          type: 'text',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const sendConfigMessage = (config: Message['config']) => {
    const configDetails = Object.entries(config || {})
      .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
      .join(', ');
      
    const newConfigMessage: Message = {
      id: Date.now().toString(),
      text: `Configuration updated: ${configDetails}`,
      sender: 'user',
      type: 'config',
      timestamp: new Date(),
      config: config,
    };
    setMessages((prev) => [...prev, newConfigMessage]);
  };

  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Response copied to clipboard!');
  };

  const handleEvaluate = async (message: Message) => {
    if (!message.query || !message.retrievalContext) {
      toast.error('Missing data for evaluation (query or retrieval context).');
      return;
    }

    try {
      setEvaluating(true);
      
      // Updated evaluation API call to match the expected format
      const response = await axios.post(`${API_BASE_URL}/evaluation`, {
        query: message.query,
        answer: message.text,
        retrieval_context: message.retrievalContext, // Already a string
        expected_output: message.expectedOutput || message.query, // Use expectedOutput if available
      });

      // Debug logging to see the actual API response structure
      console.log('Full Evaluation API Response:', response.data);
      console.log('Results structure:', JSON.stringify(response.data.results, null, 2));
      
      // Validate the response structure
      if (!response.data.results) {
        throw new Error('No results found in evaluation response');
      }

      toast.success('Evaluation completed successfully!');

      // Update the message with evaluation results
      setMessages((prev) =>
        prev.map((m) =>
          m.id === message.id ? { ...m, evaluationResults: response.data } : m
        )
      );
    } catch (error) {
      console.error('Error during evaluation:', error);
      if (axios.isAxiosError(error)) {
        console.error('Response data:', error.response?.data);
        toast.error(`Evaluation failed: ${error.response?.data?.detail || error.message}`);
      } else {
        toast.error('Failed to run evaluation.');
      }
    } finally {
      setEvaluating(false);
    }
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <TooltipProvider>
      <div className="flex flex-col h-full bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 text-gray-900 dark:text-gray-100 rounded-xl shadow-2xl overflow-hidden border border-gray-200/50 dark:border-gray-700/50">
        {/* Enhanced Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200/80 dark:border-gray-700/80 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg">
              <Brain className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                RAG Assistant
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Intelligent Document Retrieval
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs px-2 py-1">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse"></div>
              Online
            </Badge>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setMessages([])}
                  className="text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-200"
                >
                  <RefreshCw className="h-5 w-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Clear Chat</p>
              </TooltipContent>
            </Tooltip>
            <Popover open={isConfigOpen} onOpenChange={setIsConfigOpen}>
              <PopoverTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-200"
                >
                  <Settings className="h-5 w-5" />
                </Button>
              </PopoverTrigger>
              <PopoverContent 
                className="w-96 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl p-0 overflow-hidden"
                align="end"
              >
                <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-700 border-b border-gray-200 dark:border-gray-600">
                  <div className="flex items-center gap-2">
                    <Settings className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                    <h3 className="font-semibold text-gray-900 dark:text-gray-100">Pipeline Configuration</h3>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Customize your RAG pipeline settings
                  </p>
                </div>
                
                <div className="p-4 space-y-6 max-h-96 overflow-y-auto">
                  {/* Top K Parameter */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Label className="font-medium">Top K Documents</Label>
                    </div>
                    <Input
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                      min="1"
                      max="20"
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Number of documents to retrieve (1-20)
                    </p>
                  </div>

                  <Separator />

                  {/* Query Translation */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <MessageSquare className="h-4 w-4 text-blue-500" />
                        </TooltipTrigger>
                        <TooltipContent side="right" align="center">
                          <p>Query Translation Strategy</p>
                        </TooltipContent>
                      </Tooltip>
                      <Label className="font-medium">Query Translation Strategy</Label>
                    </div>
                    <RadioGroup
                      value={queryTranslation}
                      onValueChange={(value) => {
                        setQueryTranslation(value);
                        sendConfigMessage({ query_translation_strategy: value });
                      }}
                      className="grid grid-cols-1 gap-2"
                    >
                      {Object.entries(CONFIG_OPTIONS.queryTranslation).map(([key, option]) => (
                        <div key={key} className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                          <RadioGroupItem value={key} id={`qt-${key}`} />
                          <div className="flex-1">
                            <Label htmlFor={`qt-${key}`} className="font-medium cursor-pointer">
                              {option.label}
                            </Label>
                            <p className="text-xs text-gray-500 dark:text-gray-400">{option.description}</p>
                          </div>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  <Separator />

                  {/* Reranking Strategy */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Zap className="h-4 w-4 text-orange-500" />
                        </TooltipTrigger>
                        <TooltipContent side="right" align="center">
                          <p>Reranking Strategy</p>
                        </TooltipContent>
                      </Tooltip>
                      <Label className="font-medium">Reranking Strategy</Label>
                    </div>
                    <RadioGroup
                      value={reranking}
                      onValueChange={(value) => {
                        setReranking(value);
                        sendConfigMessage({ reranking_strategy: value });
                      }}
                      className="grid grid-cols-1 gap-2"
                    >
                      {Object.entries(CONFIG_OPTIONS.reranking).map(([key, option]) => (
                        <div key={key} className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                          <RadioGroupItem value={key} id={`rr-${key}`} />
                          <div className="flex-1">
                            <Label htmlFor={`rr-${key}`} className="font-medium cursor-pointer">
                              {option.label}
                            </Label>
                            <p className="text-xs text-gray-500 dark:text-gray-400">{option.description}</p>
                          </div>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  <Separator />

                  {/* Embedding Model */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Brain className="h-4 w-4 text-purple-500" />
                        </TooltipTrigger>
                        <TooltipContent side="right" align="center">
                          <p>Embedding Model</p>
                        </TooltipContent>
                      </Tooltip>
                      <Label className="font-medium">Embedding Model</Label>
                    </div>
                    <RadioGroup
                      value={embeddingModel}
                      onValueChange={(value) => {
                        setEmbeddingModel(value);
                        sendConfigMessage({ embedding_model: value });
                      }}
                      className="grid grid-cols-1 gap-2"
                    >
                      {Object.entries(CONFIG_OPTIONS.embeddingModel).map(([key, option]) => (
                        <div key={key} className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                          <RadioGroupItem value={key} id={`em-${key}`} />
                          <div className="flex-1">
                            <Label htmlFor={`em-${key}`} className="font-medium cursor-pointer">
                              {option.label}
                            </Label>
                            <p className="text-xs text-gray-500 dark:text-gray-400">{option.description}</p>
                          </div>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  <Separator />

                  {/* Chunking Strategy */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Database className="h-4 w-4 text-green-500" />
                        </TooltipTrigger>
                        <TooltipContent side="right" align="center">
                          <p>Chunking Strategy</p>
                        </TooltipContent>
                      </Tooltip>
                      <Label className="font-medium">Chunking Strategy</Label>
                    </div>
                    <RadioGroup
                      value={chunkingStrategy}
                      onValueChange={(value) => {
                        setChunkingStrategy(value);
                        sendConfigMessage({ chunking_strategy: value });
                      }}
                      className="grid grid-cols-1 gap-2"
                    >
                      {Object.entries(CONFIG_OPTIONS.chunkingStrategy).map(([key, option]) => (
                        <div key={key} className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                          <RadioGroupItem value={key} id={`cs-${key}`} />
                          <div className="flex-1">
                            <Label htmlFor={`cs-${key}`} className="font-medium cursor-pointer">
                              {option.label}
                            </Label>
                            <p className="text-xs text-gray-500 dark:text-gray-400">{option.description}</p>
                          </div>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  <Separator />

                  {/* Vector Database */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Database className="h-4 w-4 text-indigo-500" />
                        </TooltipTrigger>
                        <TooltipContent side="right" align="center">
                          <p>Vector Database</p>
                        </TooltipContent>
                      </Tooltip>
                      <Label className="font-medium">Vector Database</Label>
                    </div>
                    <RadioGroup
                      value={vectordb}
                      onValueChange={(value) => {
                        setVectordb(value);
                        sendConfigMessage({ vectordb: value });
                      }}
                      className="grid grid-cols-1 gap-2"
                    >
                      {Object.entries(CONFIG_OPTIONS.vectordb).map(([key, option]) => (
                        <div key={key} className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                          <RadioGroupItem value={key} id={`vd-${key}`} />
                          <div className="flex-1">
                            <Label htmlFor={`vd-${key}`} className="font-medium cursor-pointer">
                              {option.label}
                            </Label>
                            <p className="text-xs text-gray-500 dark:text-gray-400">{option.description}</p>
                          </div>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>
                </div>
              </PopoverContent>
            </Popover>
          </div>
        </div>

        {/* Enhanced Chat Messages Area */}
        <ScrollArea ref={scrollAreaRef} className="flex-1 p-6 space-y-4 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-60">
              <div className="p-4 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30">
                <MessageSquare className="h-12 w-12 text-blue-500 dark:text-blue-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                  Welcome to RAG Assistant
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-md">
                  Ask questions about your documents or type "ingest files" to start document processing
                </p>
              </div>
            </div>
          ) : (
            <AnimatePresence>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  layout
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.95 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className={`flex items-start gap-4 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.sender === 'bot' && (
                    <Avatar className="w-10 h-10 shadow-lg ring-2 ring-blue-100 dark:ring-blue-900/30">
                      <AvatarFallback className="bg-gradient-to-br from-blue-500 to-purple-600 text-white">
                        <Bot className="w-6 h-6" />
                      </AvatarFallback>
                    </Avatar>
                  )}
                  
                  <div className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'} max-w-[75%]`}>
                    <div
                      className={`p-4 rounded-2xl shadow-lg backdrop-blur-sm border transition-all duration-200 hover:shadow-xl ${
                        msg.sender === 'user'
                          ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white border-blue-500/20 rounded-br-md'
                          : msg.type === 'config'
                          ? 'bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 text-amber-800 dark:text-amber-200 border-amber-200 dark:border-amber-700'
                          : 'bg-white/80 dark:bg-gray-800/80 text-gray-900 dark:text-gray-100 border-gray-200/50 dark:border-gray-700/50 rounded-bl-md'
                      }`}
                    >
                      {msg.type === 'text' ? (
                        <div className="space-y-2">
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                          {msg.sender === 'bot' && (
                            <div className="flex justify-end">
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => copyToClipboard(msg.text)}
                                    className="h-6 w-6 p-0 opacity-60 hover:opacity-100 transition-opacity"
                                  >
                                    <Copy className="h-3 w-3" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Copy message</TooltipContent>
                              </Tooltip>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="secondary"
                                    size="sm"
                                    onClick={() => handleEvaluate(msg)}
                                    className="h-6 px-2 py-1 text-xs transition-opacity ml-2"
                                    disabled={evaluating}
                                  >
                                    {evaluating ? 'Evaluating...' : 'Evaluate'}
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Evaluate response</TooltipContent>
                              </Tooltip>
                            </div>
                          )}
                          {msg.evaluationResults && (
                            <Tabs defaultValue="results" className="mt-2">
                              <TabsList>
                                <TabsTrigger value="results">Evaluation Results</TabsTrigger>
                                <TabsTrigger value="charts">Charts</TabsTrigger>
                              </TabsList>
                              <TabsContent value="results">
                                <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-md text-xs text-gray-700 dark:text-gray-300">
                                  <p className="font-semibold">Evaluation Results:</p>
                                  {msg.evaluationResults.summary && (
                                    <p className="font-bold text-sm mt-1">
                                      Average Score: 
                                      <span className={`ml-1 ${getScoreColorClass(msg.evaluationResults.summary.average_score)}`}>
                                        {msg.evaluationResults.summary.average_score}
                                      </span>
                                    </p>
                                  )}
                                  {msg.evaluationResults.results && Object.entries(msg.evaluationResults.results).map(([metric, data]) => {
                                    const score = getScoreValue(data);
                                    const reason = data && typeof data === 'object' && 'reason' in data ? data.reason : null;
                                    
                                    return (
                                      <div key={metric} className="mt-2">
                                        <p className="font-medium capitalize">
                                          - {metric.replace(/_/g, ' ')}: 
                                          <span className={`ml-1 ${getScoreColorClass(score)}`}>
                                            {score !== null ? score : 'N/A'}
                                          </span>
                                        </p>
                                        {reason && (
                                          <p className="ml-4 text-gray-600 dark:text-gray-400">Reason: {reason}</p>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </TabsContent>
                              <TabsContent value="charts" className="border border-black rounded-lg p-4">
                                {msg.evaluationResults?.results && Object.keys(msg.evaluationResults.results).length > 0 ? (
                                  <div className="space-y-4">
                                    {/* Radial Chart for All Metrics */}
                                    <ChartContainer
                                      config={Object.fromEntries(
                                        Object.entries(msg.evaluationResults.results).map(([metric, result], index) => {
                                          const score = getScoreValue(result);
                                          // Generate different colors for each metric
                                          const colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'pink'];
                                          const colorName = colors[index % colors.length];
                                          return [
                                            metric,
                                            {
                                              label: metric.replace(/_/g, " "),
                                              color: `hsl(var(--${colorName}))`,
                                            },
                                          ];
                                        })
                                      )}
                                      className="h-64 w-full"
                                    >
                                      <RadialBarChart
                                        accessibilityLayer
                                        data={Object.entries(msg.evaluationResults.results).map(([metric, result], index) => {
                                          const score = getScoreValue(result);
                                          const colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899'];
                                          return {
                                            metric: metric.replace(/_/g, " "),
                                            score: score !== null ? score * 100 : 0,
                                            fill: colors[index % colors.length],
                                          };
                                        })}
                                        innerRadius="20%"
                                        outerRadius="90%"
                                        startAngle={90}
                                        endAngle={450}
                                      >
                                        <PolarAngleAxis
                                          type="number"
                                          domain={[0, 100]}
                                          angleAxisId={0}
                                          tick={false}
                                        />
                                        <PolarRadiusAxis
                                          angle={90}
                                          domain={[0, 100]}
                                          tick={false}
                                          axisLine={false}
                                        />
                                        <ChartTooltip
                                          cursor={false}
                                          content={({ active, payload }) => {
                                            if (active && payload && payload.length) {
                                              const data = payload[0].payload;
                                              return (
                                                <div className="bg-white dark:bg-gray-800 p-2 border rounded shadow-lg">
                                                  <p className="font-medium capitalize">{data.metric}</p>
                                                  <p className="text-sm">Score: {(data.score / 100).toFixed(2)}</p>
                                                </div>
                                              );
                                            }
                                            return null;
                                          }}
                                        />
                                        {Object.entries(msg.evaluationResults.results).map(([metric, result], index) => (
                                          <RadialBar
                                            key={metric}
                                            dataKey="score"
                                            cornerRadius={5}
                                            fill={`var(--color-${metric})`}
                                            data={[{
                                              metric: metric.replace(/_/g, " "),
                                              score: getScoreValue(result) !== null ? getScoreValue(result)! * 100 : 0,
                                            }]}
                                          />
                                        ))}
                                      </RadialBarChart>
                                    </ChartContainer>

                                    {/* Legend */}
                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                      {Object.entries(msg.evaluationResults.results).map(([metric, result], index) => {
                                        const score = getScoreValue(result);
                                        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899'];
                                        return (
                                          <div key={metric} className="flex items-center gap-2">
                                            <div 
                                              className="w-3 h-3 rounded-full" 
                                              style={{ backgroundColor: colors[index % colors.length] }}
                                            ></div>
                                            <span className="capitalize truncate">
                                              {metric.replace(/_/g, " ")}: {score !== null ? score.toFixed(2) : 'N/A'}
                                            </span>
                                          </div>
                                        );
                                      })}
                                    </div>

                                    {/* Summary Statistics */}
                                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 space-y-2">
                                      <h4 className="font-medium text-sm">Summary</h4>
                                      <div className="grid grid-cols-2 gap-4 text-xs">
                                        <div>
                                          <span className="text-gray-600 dark:text-gray-400">Average Score:</span>
                                          <span className={`ml-2 font-medium ${getScoreColorClass(msg.evaluationResults.summary?.average_score)}`}>
                                            {msg.evaluationResults.summary?.average_score?.toFixed(2) || 'N/A'}
                                          </span>
                                        </div>
                                        <div>
                                          <span className="text-gray-600 dark:text-gray-400">Metrics Count:</span>
                                          <span className="ml-2 font-medium">
                                            {Object.keys(msg.evaluationResults.results).length}
                                          </span>
                                        </div>
                                        <div>
                                          <span className="text-gray-600 dark:text-gray-400">Highest Score:</span>
                                          <span className="ml-2 font-medium text-green-600">
                                            {Math.max(...Object.values(msg.evaluationResults.results)
                                              .map(result => getScoreValue(result))
                                              .filter(score => score !== null) as number[]).toFixed(2)}
                                          </span>
                                        </div>
                                        <div>
                                          <span className="text-gray-600 dark:text-gray-400">Lowest Score:</span>
                                          <span className="ml-2 font-medium text-red-600">
                                            {Math.min(...Object.values(msg.evaluationResults.results)
                                              .map(result => getScoreValue(result))
                                              .filter(score => score !== null) as number[]).toFixed(2)}
                                          </span>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="h-48 w-full rounded-md bg-gray-100 p-2 dark:bg-gray-800">
                                    <p className="text-center text-muted-foreground">
                                      No evaluation results available to display chart.
                                    </p>
                                  </div>
                                )}
                              </TabsContent>
                            </Tabs>
                          )}
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="h-4 w-4" />
                            <p className="text-sm font-medium">{msg.text}</p>
                          </div>
                          {msg.config && (
                            <div className="text-xs space-y-1 mt-3 p-3 bg-white/50 dark:bg-gray-700/50 rounded-lg border border-amber-200/50 dark:border-amber-700/50">
                              {Object.entries(msg.config).map(([key, value]) => (
                                <div key={key} className="flex justify-between items-center">
                                  <span className="font-medium capitalize">{key.replace(/_/g, ' ')}:</span>
                                  <Badge variant="outline" className="text-xs">
                                    {value}
                                  </Badge>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-2 mt-1 px-1">
                      <span className="text-xs text-gray-400 dark:text-gray-500">
                        {formatTime(msg.timestamp)}
                      </span>
                    </div>
                  </div>

                  {msg.sender === 'user' && (
                    <Avatar className="w-10 h-10 shadow-lg ring-2 ring-gray-200 dark:ring-gray-700">
                      <AvatarFallback className="bg-gradient-to-br from-gray-600 to-gray-800 text-white text-xs font-bold">
                        YOU
                      </AvatarFallback>
                    </Avatar>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          )}
          
          {/* Enhanced Loading State */}
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-4 justify-start"
            >
              <Avatar className="w-10 h-10 shadow-lg ring-2 ring-blue-100 dark:ring-blue-900/30">
                <AvatarFallback className="bg-gradient-to-br from-blue-500 to-purple-600 text-white">
                  <Bot className="w-6 w-6" />
                </AvatarFallback>
              </Avatar>
              <div className="p-4 rounded-2xl rounded-bl-md bg-white/80 dark:bg-gray-800/80 border border-gray-200/50 dark:border-gray-700/50 shadow-lg backdrop-blur-sm">
                <div className="flex items-center gap-3">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                  </div>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Processing your request...
                  </span>
                </div>
              </div>
            </motion.div>
          )}
        </ScrollArea>

        {/* Enhanced Input Area */}
        <div className="p-6 border-t border-gray-200/80 dark:border-gray-700/80 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
          <div className="flex items-end gap-3">
            <div className="flex-1 relative">
              <Input
                ref={inputRef}
                placeholder="Ask about your documents or type 'ingest files' to start..."
                className="min-h-[48px] pr-12 bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-blue-500 text-gray-900 dark:text-gray-100 rounded-xl shadow-sm transition-all duration-200 resize-none"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                disabled={loading}
              />
              
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-8 w-8 p-0 opacity-60 hover:opacity-100 transition-opacity"
                    >
                      <Upload className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" align="center">Upload files</TooltipContent>
                </Tooltip>
              </div>
            </div>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  onClick={sendMessage} 
                  disabled={loading || input.trim() === ''} 
                  className="h-12 w-12 bg-gradient-to-br from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <RefreshCw className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top" align="center">
                <p>Send Message</p>
              </TooltipContent>
            </Tooltip>
          </div>
          
          <div className="flex items-center justify-between mt-3 text-xs text-gray-400 dark:text-gray-500">
            <div className="flex items-center gap-4">
              <span>Press Enter to send</span>
              <Separator orientation="vertical" className="h-3" />
              <span>Shift + Enter for new line</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Connected</span>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}