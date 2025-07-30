# Imports 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
from ingestion import IngestionPipeline
from retrieval import RetrieverPipeline
from eval import EvaluationPipeline

# Instance

app = FastAPI(
    title = "APIs for RAG pipeline",
    description = "These APIs are for each workflows of the RAG pipeline including ingestion , retrieval and evaluation",
    version = "1.0"
)

# CORS Origin 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models

class IngestionInput(BaseModel):
    """Input model for the ingestion endpoint."""
    chunking_strategy: str = Field(..., description="Chunking strategy to use (e.g., 'recursive', 'sentence', 'token', 'raptor')")
    embedding_model: str = Field(..., description="Embedding model to use (e.g., 'huggingface', 'google', 'openai')")
    vectordb: str = Field("faiss", description="Vector database to use (currently only 'faiss' is supported)")

class RetrievalInput(BaseModel):
    """Input model for the retrieval endpoint."""
    query: str = Field(..., description="The query to retrieve information for")
    embedding_model: str = Field(..., description="Embedding model to use (e.g., 'huggingface', 'google', 'openai')")
    top_k: int = Field(5, description="Number of documents to retrieve and use for generating the answer")

class EvaluationInput(BaseModel):
    """Input model for the evaluation endpoint."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The answer to evaluate")
    retrieval_context: str = Field(..., description="The context used to generate the answer")
    expected_output: str = Field(..., description="The expected answer for comparison")
    eval_llm: str = Field("gemini-2.0-flash-lite", description="The LLM to use for evaluation")
    

# API routes 

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running", "endpoints": ["/ingestion", "/retrieval", "/evaluation"]}

@app.post("/ingestion", summary="Process transcript files for RAG", 
         description="Processes all transcript files in the Transcripts directory using the specified chunking strategy and embedding model")
async def run_ingestion(input_data: IngestionInput):
    """Process transcript files for RAG.
    
    This endpoint:
    1. Initializes the ingestion pipeline with the provided parameters
    2. Scans the Transcripts directory for .txt files
    3. Processes all found transcript files
    4. Creates or updates the vector database
    
    Returns:
        A JSON object with status, message, and processing details
    """
    try:
        # Initialize the ingestion pipeline with the provided parameters
        pipeline = IngestionPipeline(
            chunking_strategy=input_data.chunking_strategy,
            embedding_model=input_data.embedding_model,
            vectordb=input_data.vectordb
        )
        
        # Get all transcript files from the Transcripts directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        transcript_dir = os.path.join(BASE_DIR, "Transcripts")
        
        # Check if Transcripts directory exists
        if not os.path.exists(transcript_dir):
            raise HTTPException(
                status_code=404, 
                detail=f"Transcripts directory not found at {transcript_dir}"
            )
            
        transcript_files = []
        
        # Walk through all subdirectories in the Transcripts folder
        for root, dirs, files in os.walk(transcript_dir):
            for file in files:
                if file.endswith(".txt"):
                    transcript_files.append(os.path.join(root, file))
        
        # Check if any transcript files were found
        if not transcript_files:
            raise HTTPException(
                status_code=404, 
                detail="No transcript files found in the Transcripts directory. Please add .txt files before running ingestion."
            )
        
        # Run the ingestion process
        vectordb = pipeline.ingestion(transcript_files)
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(transcript_files)} transcript files",
            "details": {
                "files_processed": len(transcript_files),
                "file_paths": [os.path.relpath(f, BASE_DIR) for f in transcript_files],
                "chunking_strategy": input_data.chunking_strategy,
                "embedding_model": input_data.embedding_model,
                "vectordb": input_data.vectordb
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        # Log the error and return a 500 response
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/evaluation", summary="Evaluate RAG pipeline results",
         description="Evaluates the quality of RAG pipeline results using multiple metrics")
async def run_evaluation(input_data: EvaluationInput):
    """Evaluate RAG pipeline results.
    
    This endpoint:
    1. Initializes the evaluation pipeline with the provided parameters
    2. Runs multiple evaluation metrics (answer relevancy, faithfulness, contextual precision/recall/relevancy)
    3. Returns detailed evaluation results with scores and explanations
    
    Returns:
        A JSON object with status, query, evaluation results, and details
    """
    try:
        # Initialize the evaluation pipeline with the provided parameters
        pipeline = EvaluationPipeline(
            eval_llm=input_data.eval_llm,
            query=input_data.query,
            answer=input_data.answer,
            retrieval_context=input_data.retrieval_context,
            expected_output=input_data.expected_output
        )
        
        # Run the evaluation process
        results = pipeline.compiled_eval()
        
        # Calculate average score across all metrics
        total_score = 0
        metric_count = 0
        for metric, data in results.items():
            if 'score' in data:
                total_score += data['score']
                metric_count += 1
        
        avg_score = total_score / metric_count if metric_count > 0 else 0
        
        return {
            "status": "success",
            "query": input_data.query,
            "results": results,
            "summary": {
                "average_score": round(avg_score, 2),
                "metrics_evaluated": metric_count
            },
            "details": {
                "eval_llm": input_data.eval_llm,
                "answer_length": len(input_data.answer),
                "context_length": len(input_data.retrieval_context)
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        # Log the error and return a 500 response
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/retrieval", summary="Retrieve information based on a query",
         description="Retrieves information from the vector database based on the provided query")
async def run_retrieval(input_data: RetrievalInput):
    """Retrieve information based on a query.
    
    This endpoint:
    1. Initializes the retrieval pipeline with the provided parameters
    2. Processes the query to retrieve relevant information
    3. Returns the answer generated from the retrieved documents
    
    Returns:
        A JSON object with status, query, answer, and retrieval details
    """
    try:
        # Check if the FAISS index exists
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
        
        if not os.path.exists(FAISS_INDEX_PATH):
            raise HTTPException(
                status_code=404,
                detail="Vector database not found. Please run the ingestion endpoint first."
            )
        
        # Initialize the retrieval pipeline
        pipeline = RetrieverPipeline(
            query_translation_strategy="hyde",  # Using a default strategy
            reranking_strategy="reciprocal_rank_fusion",  # Using a default strategy
            embedding_model=input_data.embedding_model,
            top_k=input_data.top_k  # Use the top_k parameter from the input
        )
        
        # Run the retrieval process
        answer = pipeline.retrieve(input_data.query)
        
        return {
            "status": "success",
            "query": input_data.query,
            "answer": answer,
            "details": {
                "embedding_model": input_data.embedding_model,
                "top_k": input_data.top_k,
                "query_translation_strategy": "hyde",
                "reranking_strategy": "none"
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        # Log the error and return a 500 response
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
