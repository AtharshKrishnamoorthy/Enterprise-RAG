# RAG Evaluation Project

This repository hosts a comprehensive Retrieval-Augmented Generation (RAG) evaluation project, designed to process and analyze financial earnings call transcripts. It features a robust FastAPI-based API that provides endpoints for data ingestion, information retrieval, and RAG pipeline evaluation.

## Features

### 1. Data Ingestion (`/ingestion` endpoint)
- **Purpose**: Processes raw earnings call transcripts and prepares them for retrieval.
- **Functionality**:
  - Scans a designated `Transcripts` directory for company-specific earnings call files (e.g., AAPL, AMD, AMZN).
  - Chunks the text content of these transcripts using a configurable chunking strategy (e.g., `RecursiveCharacterTextSplitter`).
  - Generates embeddings for the chunks using a specified embedding model (e.g., `text-embedding-004`).
  - Stores the processed data in a FAISS vector database for efficient similarity search.
- **Error Handling**: Includes robust checks for directory existence, file availability, and provides detailed error messages with file paths.

### 2. Information Retrieval (`/retrieval` endpoint)
- **Purpose**: Facilitates querying the ingested data to retrieve relevant information.
- **Functionality**:
  - Accepts natural language queries.
  - Translates queries if necessary (e.g., using a query translation strategy).
  - Performs similarity search against the FAISS vector database.
  - Reranks retrieved documents based on relevance using a configurable reranking strategy.
  - Allows customization of the number of top-k results to retrieve.
- **Error Handling**: Verifies the existence of the FAISS index and provides informative error responses for missing indices or other issues.

### 3. RAG Pipeline Evaluation (`/evaluation` endpoint)
- **Purpose**: Assesses the performance of the RAG pipeline using various deep evaluation metrics.
- **Functionality**:
  - Takes `query`, `answer`, `retrieval_context`, and `expected_output` as input.
  - Utilizes a configurable evaluation LLM (default: `gemini-2.0-flash-lite`) to perform evaluations.
  - Calculates and returns scores for the following metrics:
    - **Answer Relevancy**: How relevant the generated answer is to the query.
    - **Faithfulness**: How well the generated answer is supported by the retrieval context.
    - **Contextual Precision**: The precision of the retrieved context with respect to the query.
    - **Contextual Recall**: The recall of the retrieved context with respect to the expected answer.
    - **Contextual Relevancy**: The overall relevancy of the retrieved context.
  - Provides a compiled evaluation result, including individual metric scores, reasoning, and an average overall score.
- **Error Handling**: Comprehensive error handling for evaluation failures, providing detailed status and traceback information.

## Project Structure

```
.gitignore
README.md
Transcripts/
├── AAPL/
├── AMD/
├── AMZN/
├── ASML/
├── CSCO/
├── GOOGL/
├── INTC/
├── MSFT/
├── MU/
├── NVDA/
api.py
eval.py
eval_demo.ipynb
ingestion.py
rag-frontend/
requirements.txt
retrieval.py
```

- `api.py`: Contains the FastAPI application with all defined endpoints.
- `eval.py`: Implements the `EvaluationPipeline` and individual evaluation metrics.
- `ingestion.py`: Handles the processing and storage of transcript data into the vector database.
- `retrieval.py`: Manages the retrieval process, including query handling and document reranking.
- `Transcripts/`: Directory containing earnings call transcripts organized by company ticker.
- `rag-frontend/`: (Optional) A placeholder for a potential frontend application.
- `requirements.txt`: Lists all Python dependencies.

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd RAG_EVAL
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Transcripts**: Ensure your earnings call transcript files are placed in the `Transcripts/` directory, organized by company ticker (e.g., `Transcripts/AAPL/2023-01-01-AAPL.txt`).
4.  **Run the FastAPI application**:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```
5.  **Interact with the API**:
    - **Ingestion**: Send a POST request to `/ingestion` to process your transcripts.
    - **Retrieval**: Send a POST request to `/retrieval` with your query.
    - **Evaluation**: Send a POST request to `/evaluation` with your query, answer, context, and expected output.

This project provides a robust framework for building, testing, and evaluating RAG pipelines for financial document analysis.