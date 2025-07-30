



# Imports 
import os
import logging
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any

import langchain 
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ingestion_pipeline')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

class IngestionPipeline:
    def __init__(self, chunking_strategy: str, embedding_model: str, vectordb: str = "faiss"):
        logger.info(f"Initializing IngestionPipeline with chunking_strategy='{chunking_strategy}', "
                   f"embedding_model='{embedding_model}', vectordb='{vectordb}'")
        
        # Validate parameters
        self._validate_parameters(chunking_strategy, embedding_model, vectordb)
        
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vectordb_type = vectordb
        self.vectordb = None
        self.embeddings = None
        
        # Initialize embeddings
        logger.info(f"Initializing embedding model: {embedding_model}")
        try:
            self._initialize_embeddings()
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def _validate_parameters(self, chunking_strategy: str, embedding_model: str, vectordb: str):
        """Validate input parameters."""
        logger.debug("Validating input parameters")
        
        valid_chunking = ["fixed", "semantic", "raptor"]
        valid_embeddings = ["huggingface", "google", "openai"]
        valid_vectordb = ["faiss"]
        
        if chunking_strategy not in valid_chunking:
            error_msg = (f"Chunking strategy '{chunking_strategy}' not supported. "
                        f"Valid options: {valid_chunking}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if embedding_model not in valid_embeddings:
            error_msg = (f"Embedding model '{embedding_model}' not supported. "
                        f"Valid options: {valid_embeddings}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if vectordb not in valid_vectordb:
            error_msg = (f"Vector database '{vectordb}' not supported. "
                        f"Valid options: {valid_vectordb}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("All parameters validated successfully")
   
    def _initialize_embeddings(self):
        """Initialize the embedding model based on the specified type."""
        logger.debug(f"Setting up {self.embedding_model} embeddings")
        
        if self.embedding_model == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("HuggingFace embeddings initialized with model: sentence-transformers/all-MiniLM-L6-v2")
        elif self.embedding_model == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            logger.info("Google Generative AI embeddings initialized")
        elif self.embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
            logger.info("OpenAI embeddings initialized")
        else:
            error_msg = f"Embedding model '{self.embedding_model}' not supported"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def load_docs(self, paths: List[str]) -> List[Document]:
        """Load documents from text files."""
        logger.info(f"Starting to load documents from {len(paths)} text files")
        docs = []
        successful_loads = 0
        
        for i, path in enumerate(paths, 1):
            logger.debug(f"Processing file {i}/{len(paths)}: {path}")
            
            if not os.path.exists(path):
                logger.warning(f"File {path} does not exist. Skipping.")
                continue
                
            try:
                loader = TextLoader(path)
                file_docs = loader.load()
                docs.extend(file_docs)
                successful_loads += 1
                logger.info(f"Successfully loaded {len(file_docs)} pages from {path}")
            except Exception as e:
                logger.error(f"Error loading {path}: {str(e)}")
                
        logger.info(f"Document loading completed. Successfully loaded {successful_loads}/{len(paths)} files, "
                   f"total pages: {len(docs)}")
        return docs
    
    def chunk_docs(self, docs: List[Document]) -> List[Document]:
        """Chunk documents based on the specified strategy."""
        logger.info(f"Starting document chunking with strategy: {self.chunking_strategy}")
        logger.debug(f"Input: {len(docs)} documents to be chunked")
        
        try:
            if self.chunking_strategy == "fixed":
                logger.debug("Initializing RecursiveCharacterTextSplitter for fixed chunking")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
            elif self.chunking_strategy == "semantic":
                logger.debug("Initializing SemanticChunker for semantic chunking")
                text_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95
                )
            elif self.chunking_strategy == "raptor":
                logger.debug("Starting RAPTOR chunking process")
                return self._raptor_chunking(docs)
            else:
                error_msg = f"Chunking strategy '{self.chunking_strategy}' not supported"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Chunking the docs 
            logger.debug("Splitting documents into chunks")
            chunked_docs = text_splitter.split_documents(docs)
            logger.info(f"Document chunking completed. Created {len(chunked_docs)} chunks from {len(docs)} documents")
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error during document chunking: {str(e)}")
            raise
    
    def _raptor_chunking(self, docs: List[Document]) -> List[Document]:
        """
        Implement RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) chunking.
        This creates a hierarchical tree structure by clustering and summarizing chunks.
        """
        logger.info("Starting RAPTOR chunking process")
        
        # First, create base chunks using recursive splitter
        logger.debug("Creating base chunks for RAPTOR")
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        base_chunks = base_splitter.split_documents(docs)
        logger.info(f"Created {len(base_chunks)} base chunks for RAPTOR processing")
        
        # Create embeddings for base chunks
        logger.debug("Generating embeddings for base chunks")
        chunk_texts = [chunk.page_content for chunk in base_chunks]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
        logger.debug(f"Generated embeddings for {len(chunk_embeddings)} base chunks")
        
        # Perform hierarchical clustering
        logger.debug("Starting hierarchical clustering and tree creation")
        hierarchical_chunks = self._create_raptor_tree(base_chunks, chunk_embeddings)
        
        logger.info(f"RAPTOR chunking completed. Total chunks (including summaries): {len(hierarchical_chunks)}")
        return hierarchical_chunks
    
    def _create_raptor_tree(self, chunks: List[Document], embeddings: List[List[float]]) -> List[Document]:
        """Create hierarchical tree structure for RAPTOR indexing."""
        logger.debug("Creating RAPTOR tree structure")
        all_chunks = chunks.copy()
        current_level_embeddings = np.array(embeddings)
        
        # Create multiple levels of the tree
        for level in range(3):  # Create 3 levels of hierarchy
            logger.debug(f"Processing RAPTOR level {level + 1}")
            
            if len(current_level_embeddings) <= 2:
                logger.debug(f"Stopping at level {level + 1}: insufficient chunks for clustering")
                break
                
            # Cluster chunks at current level
            n_clusters = max(2, len(current_level_embeddings) // 3)
            logger.debug(f"Creating {n_clusters} clusters for level {level + 1}")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(current_level_embeddings)
            
            # Group chunks by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_chunks[-(len(current_level_embeddings) - i)])
            
            # Create summary chunks for each cluster
            level_summaries = []
            level_embeddings = []
            summaries_created = 0
            
            for cluster_id, cluster_chunks in clusters.items():
                if len(cluster_chunks) > 1:  # Only summarize if there are multiple chunks
                    logger.debug(f"Creating summary for cluster {cluster_id} with {len(cluster_chunks)} chunks")
                    
                    # Create a summary document (simplified version - in practice, use LLM)
                    combined_text = "\n".join([chunk.page_content for chunk in cluster_chunks])
                    summary_text = self._create_summary(combined_text)
                    
                    summary_doc = Document(
                        page_content=summary_text,
                        metadata={
                            "level": level + 1,
                            "cluster_id": cluster_id,
                            "source": "raptor_summary",
                            "child_chunks": len(cluster_chunks)
                        }
                    )
                    level_summaries.append(summary_doc)
                    summaries_created += 1
                    
                    # Get embedding for summary
                    summary_embedding = self.embeddings.embed_documents([summary_text])[0]
                    level_embeddings.append(summary_embedding)
            
            logger.debug(f"Created {summaries_created} summaries for level {level + 1}")
            
            # Add summaries to all chunks
            all_chunks.extend(level_summaries)
            current_level_embeddings = np.array(level_embeddings) if level_embeddings else current_level_embeddings
        
        logger.debug(f"RAPTOR tree creation completed with {len(all_chunks)} total chunks")
        return all_chunks
    
    def _create_summary(self, text: str) -> str:
        """
        Create a summary of the given text. 
        In a production environment, this should use an LLM for better summarization.
        """
        logger.debug(f"Creating summary for text of length {len(text)}")
        # Simple extractive summary - take first and key sentences
        sentences = text.split('.')[:3]  # Take first 3 sentences as summary
        summary = '. '.join(sentences) + '.'
        summary = summary[:500]  # Limit summary length
        logger.debug(f"Created summary of length {len(summary)}")
        return summary
    
    def embed_docs(self, chunked_docs: List[Document]) -> List[List[float]]:
        """Create embeddings for the chunked documents."""
        logger.info(f"Starting embedding generation for {len(chunked_docs)} chunks")
        
        try:
            texts = [doc.page_content for doc in chunked_docs]
            logger.debug("Generating embeddings using the configured embedding model")
            chunked_docs_embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated embeddings. Shape: {len(chunked_docs_embeddings)} documents")
            return chunked_docs_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def ingestion(self, file_paths: List[str]) -> FAISS:
        """
        Main ingestion method that processes text files and creates/updates the vector database.
        """
        logger.info("="*60)
        logger.info("STARTING RAG INGESTION PIPELINE")
        logger.info("="*60)
        
        try:
            # Load documents
            logger.info(f"Step 1/4: Loading documents from {len(file_paths)} text files")
            docs = self.load_docs(file_paths)
            if not docs:
                error_msg = "No documents were loaded. Please check your text file paths."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"✓ Successfully loaded {len(docs)} document pages")
            
            # Chunk documents
            logger.info(f"Step 2/4: Chunking documents using '{self.chunking_strategy}' strategy")
            chunked_docs = self.chunk_docs(docs)
            logger.info(f"✓ Successfully created {len(chunked_docs)} chunks")
            
            # Create embeddings
            logger.info(f"Step 3/4: Generating embeddings using '{self.embedding_model}' model")
            chunked_docs_embeddings = self.embed_docs(chunked_docs)
            logger.info(f"✓ Successfully generated embeddings for {len(chunked_docs_embeddings)} chunks")
            
            # Indexing the docs 
            logger.info("Step 4/4: Creating/updating vector database")
            if os.path.exists(FAISS_INDEX_PATH):
                logger.info(f"Existing FAISS index found at: {FAISS_INDEX_PATH}")
                try:
                    self.vectordb = FAISS.load_local(
                        FAISS_INDEX_PATH, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.debug("Existing index loaded successfully")
                    
                    logger.debug("Adding new documents to existing index")
                    self.vectordb.add_documents(chunked_docs)
                    self.vectordb.save_local(FAISS_INDEX_PATH)
                    logger.info("✓ Successfully updated existing FAISS index")
                    
                except Exception as e:
                    logger.warning(f"Error loading existing index: {e}")
                    logger.info("Creating new index instead")
                    self.vectordb = FAISS.from_documents(chunked_docs, self.embeddings)
                    self.vectordb.save_local(FAISS_INDEX_PATH)
                    logger.info("✓ Successfully created new FAISS index")
            else:
                logger.info("No existing index found. Creating new FAISS index")
                self.vectordb = FAISS.from_documents(chunked_docs, self.embeddings)
                self.vectordb.save_local(FAISS_INDEX_PATH)
                logger.info("✓ Successfully created new FAISS index")
            
            logger.info("="*60)
            logger.info("RAG INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total transcript files processed: {len(file_paths)}")
            logger.info(f"Total documents processed: {len(docs)}")
            logger.info(f"Total chunks created: {len(chunked_docs)}")
            logger.info(f"Chunking strategy: {self.chunking_strategy}")
            logger.info(f"Embedding model: {self.embedding_model}")
            logger.info(f"Index saved at: {FAISS_INDEX_PATH}")
            logger.info("="*60)
            
            return self.vectordb
            
        except Exception as e:
            logger.error("="*60)
            logger.error("RAG INGESTION PIPELINE FAILED!")
            logger.error(f"Error: {str(e)}")
            logger.error("="*60)
            raise
    
    def get_vectordb(self) :
        """Return the vector database instance."""
        if self.vectordb is None:
            if os.path.exists(FAISS_INDEX_PATH):
                self.vectordb = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                raise ValueError("No vector database found. Please run ingestion first.")
        return self.vectordb


# Example usage
if __name__ == "__main__":
    try:
        print("Starting example usage of ingestion pipeline")
        
        # Initialize the pipeline with correct parameters
        pipeline = IngestionPipeline(
            chunking_strategy="raptor",  # Options: "fixed", "semantic", "raptor"
            embedding_model="huggingface",  # Options: "huggingface", "google", "openai"
            vectordb="faiss"
        )
        
        # Get all transcript files from the Transcripts directory
        transcript_dir = os.path.join(BASE_DIR, "Transcripts")
        transcript_files = []
        
        # Walk through all subdirectories in the Transcripts folder
        for root, dirs, files in os.walk(transcript_dir):
            for file in files:
                if file.endswith(".txt"):
                    transcript_files.append(os.path.join(root, file))
        
        print(f"Found {len(transcript_files)} transcript files for ingestion")
        
        # Check if any transcript files were found
        if not transcript_files:
            print("Warning: No transcript files found in the Transcripts directory.")
        else:
            vectordb = pipeline.ingestion(transcript_files)
            
            # Test retrieval
            query = "What is the main topic discussed in the documents?"
            results = vectordb.similarity_search(query, k=3)
            print(f"\nSearch results for query: '{query}'")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Content: {result.page_content[:200]}...")
                print(f"Metadata: {result.metadata}")
                
    except Exception as e:
        print(f"Error in example usage: {str(e)}")
        print("\nTo use this pipeline:")
        print("1. Ensure the Transcripts directory contains valid text files")
        print("2. Choose your preferred chunking strategy: 'fixed', 'semantic', or 'raptor'")
        print("3. Choose your preferred embedding model: 'huggingface', 'google', or 'openai'")
        print("4. For Google embeddings, set GOOGLE_API_KEY environment variable")
        print("5. For OpenAI embeddings, set OPENAI_API_KEY environment variable")