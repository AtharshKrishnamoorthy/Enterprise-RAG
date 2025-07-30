# RETRIEVAL PIPELINE 

# Imports 
import os
import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configure logging for retrieval pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('retrieval_pipeline')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

class RetrieverPipeline:
    def __init__(self, 
                 query_translation_strategy: str, 
                 reranking_strategy: str,  
                 embedding_model: str,
                 top_k: int = 10):
        
        logger.info(f"Initializing RetrieverPipeline with query_translation='{query_translation_strategy}', "
                   f"reranking='{reranking_strategy}', embedding_model='{embedding_model}', "
                   f"top_k={top_k}, llm='gemini-2.0-flash'")
        
        # Validate parameters
        self._validate_parameters(query_translation_strategy, reranking_strategy, embedding_model)
        
        self.query_translation_strategy = query_translation_strategy
        self.reranking_strategy = reranking_strategy
        self.llm_model = "gemini-2.0-flash"
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.vectordb = None  # Will be loaded later
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model}")
        try:
            self.embedding_model = self._initialize_embedding_model(embedding_model)
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {self.llm_model}")
        try:
            self.llm = ChatGoogleGenerativeAI(model=self.llm_model, temperature=0.2)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
        
        logger.info("RetrieverPipeline initialization completed successfully")
    
    def _validate_parameters(self, query_translation_strategy: str, reranking_strategy: str, embedding_model: str):
        """Validate input parameters."""
        logger.debug("Validating input parameters")
        
        valid_query_strategies = ["multi_query", "decomposition", "hyde", "step_back", "none"]
        valid_reranking_strategies = ["reciprocal_rank_fusion", "cross_encoder", "cohere", "none"]
        valid_embeddings = ["huggingface", "google", "openai"]
        
        if query_translation_strategy not in valid_query_strategies:
            error_msg = (f"Query translation strategy '{query_translation_strategy}' not supported. "
                        f"Valid options: {valid_query_strategies}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if reranking_strategy not in valid_reranking_strategies:
            error_msg = (f"Reranking strategy '{reranking_strategy}' not supported. "
                        f"Valid options: {valid_reranking_strategies}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if embedding_model not in valid_embeddings:
            error_msg = (f"Embedding model '{embedding_model}' not supported. "
                        f"Valid options: {valid_embeddings}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("All parameters validated successfully")
    
    def _initialize_embedding_model(self, embedding_model: str):
        """Initialize the embedding model based on the specified type."""
        if embedding_model == "huggingface":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        elif embedding_model == "google":
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        elif embedding_model == "openai":
            return OpenAIEmbeddings()
        else:
            raise ValueError(f"Embedding model '{embedding_model}' not supported")
    
    def get_retriever(self):
       """Get the retriever from the vector database."""
       logger.debug("Getting retriever from vector database")
       try:
         if self.vectordb is None:
             logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
             # Try the newer format first
             try:
                 self.vectordb = FAISS.load_local(
                     folder_path=FAISS_INDEX_PATH,
                     embeddings=self.embedding_model,
                     allow_dangerous_deserialization=True
                 )
             except Exception:
                 # Fallback to older format
                 self.vectordb = FAISS.load_local(
                     FAISS_INDEX_PATH,
                     self.embedding_model,
                     allow_dangerous_deserialization=True
                 )
             logger.info("FAISS index loaded successfully")
         
         retriever = self.vectordb.as_retriever(search_kwargs={"k": self.top_k})
         logger.debug("Retriever obtained successfully")
         return retriever
       except Exception as e:
         logger.error(f"Error getting retriever: {str(e)}")
         raise
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using the configured embedding model."""
        logger.debug(f"Embedding query: '{query[:50]}...'")
        try:
            embedding = self.embedding_model.embed_query(query)
            logger.debug(f"Query embedded successfully, dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def query_translation(self, query: str) -> List[str]:
        """Translate query based on the specified strategy."""
        logger.info(f"Starting query translation with strategy: {self.query_translation_strategy}")
        logger.debug(f"Original query: '{query}'")
        
        if self.query_translation_strategy == "none":
            logger.debug("No query translation applied")
            return [query]
        
        try:
            if self.query_translation_strategy == "multi_query":
                return self._multi_query_translation(query)
            elif self.query_translation_strategy == "decomposition":
                return self._decomposition_translation(query)
            elif self.query_translation_strategy == "hyde":
                return self._hyde_translation(query)
            elif self.query_translation_strategy == "step_back":
                return self._step_back_translation(query)
            else:
                logger.warning(f"Unknown query translation strategy: {self.query_translation_strategy}")
                return [query]
        except Exception as e:
            logger.error(f"Error in query translation: {str(e)}")
            logger.info("Falling back to original query")
            return [query]
    
    def _multi_query_translation(self, query: str) -> List[str]:
        """Generate multiple variations of the query."""
        logger.debug("Applying multi-query translation")
        
        prompt = ChatPromptTemplate.from_template("""
        Generate 4 different variations of this query that would help retrieve relevant information:
        
        Original Query: {query}
        
        Return only the variations, one per line, numbered 1-4:
        """)
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        
        # Parse the result to extract queries
        content = result.content if hasattr(result, 'content') else str(result)
        queries = self._parse_multiple_queries(content)
        queries.insert(0, query)  # Include original query
        
        logger.info(f"Generated {len(queries)} query variations")
        for i, q in enumerate(queries):
            logger.debug(f"Query {i+1}: {q}")
        
        return queries
    
    def _decomposition_translation(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-queries."""
        logger.debug("Applying query decomposition")
        
        prompt = ChatPromptTemplate.from_template("""
        Break down this complex query into 3-4 simpler sub-questions that together would help answer the original question:
        
        Original Query: {query}
        
        Return only the sub-questions, one per line, numbered 1-4:
        """)
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        
        content = result.content if hasattr(result, 'content') else str(result)
        sub_queries = self._parse_multiple_queries(content)
        sub_queries.insert(0, query)
        
        logger.info(f"Decomposed query into {len(sub_queries)} sub-questions")
        for i, q in enumerate(sub_queries):
            logger.debug(f"Sub-query {i+1}: {q}")
        
        return sub_queries
    
    def _hyde_translation(self, query: str) -> List[str]:
        """Generate hypothetical document for the query (HyDE approach)."""
        logger.debug("Applying HyDE translation")
        
        prompt = ChatPromptTemplate.from_template("""
        Write a hypothetical passage that would contain the answer to this question. 
        Make it detailed and factual as if it's from a real document:
        
        Question: {query}
        
        Hypothetical Passage:
        """)
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        
        content = result.content if hasattr(result, 'content') else str(result)
        
        logger.info("Generated hypothetical document for HyDE")
        logger.debug(f"HyDE content: {content[:100]}...")
        
        return [query, content]
    
    def _step_back_translation(self, query: str) -> List[str]:
        """Generate step-back abstracted query."""
        logger.debug("Applying step-back translation")
        
        prompt = ChatPromptTemplate.from_template("""
        Given the specific question below, generate a more general, high-level question that would help provide context for answering the specific question:
        
        Specific Question: {query}
        
        Step-back Question:
        """)
        
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        
        content = result.content if hasattr(result, 'content') else str(result)
        step_back_query = content.strip()
        
        logger.info("Generated step-back query")
        logger.debug(f"Original: {query}")
        logger.debug(f"Step-back: {step_back_query}")
        
        return [query, step_back_query]
    
    def _parse_multiple_queries(self, content: str) -> List[str]:
        """Parse multiple queries from LLM response."""
        queries = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering (1., 2., etc.)
                clean_line = re.sub(r'^\d+\.?\s*', '', line)
                if clean_line and len(clean_line) > 10:  # Filter out very short lines
                    queries.append(clean_line)
        
        return queries[:5]  # Limit to 5 queries max
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on the specified strategy."""
        logger.info(f"Starting reranking with strategy: {self.reranking_strategy}")
        logger.debug(f"Input: {len(documents)} documents to rerank")
        
        if self.reranking_strategy == "none" or len(documents) <= 1:
            logger.debug("No reranking applied")
            return documents
        
        try:
            if self.reranking_strategy == "reciprocal_rank_fusion":
                return self._reciprocal_rank_fusion(query, documents)
            elif self.reranking_strategy == "cross_encoder":
                return self._cross_encoder_rerank(query, documents)
            elif self.reranking_strategy == "cohere":
                return self._cohere_rerank(query, documents)
            else:
                logger.warning(f"Unknown reranking strategy: {self.reranking_strategy}")
                return documents
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            logger.info("Returning original document order")
            return documents
    
    def _reciprocal_rank_fusion(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply Reciprocal Rank Fusion (RRF) reranking."""
        logger.debug("Applying Reciprocal Rank Fusion")
        
        # For RRF, we need multiple rankings. Here we'll use semantic similarity and keyword matching
        k = 60  # RRF constant
        
        # Get semantic similarity scores
        query_embedding = self.embed_query(query)
        doc_embeddings = [self.embedding_model.embed_query(doc.page_content[:500]) for doc in documents]
        
        # Calculate cosine similarity
        import numpy as np
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        semantic_scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(doc_embeddings)]
        semantic_ranking = sorted(semantic_scores, key=lambda x: x[1], reverse=True)
        
        # Simple keyword matching scores
        query_words = set(query.lower().split())
        keyword_scores = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            keyword_scores.append((i, overlap))
        
        keyword_ranking = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
        
        # Apply RRF formula
        rrf_scores = defaultdict(float)
        
        for rank, (doc_idx, _) in enumerate(semantic_ranking):
            rrf_scores[doc_idx] += 1 / (k + rank + 1)
        
        for rank, (doc_idx, _) in enumerate(keyword_ranking):
            rrf_scores[doc_idx] += 1 / (k + rank + 1)
        
        # Sort by RRF scores
        final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_docs = [documents[doc_idx] for doc_idx, _ in final_ranking]
        
        logger.info(f"RRF reranking completed. Top document changed: {final_ranking[0][0] != 0}")
        return reranked_docs
    
    def _cross_encoder_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply cross-encoder reranking (simplified version)."""
        logger.debug("Applying cross-encoder reranking")
        
        # This is a simplified version. In production, you'd use a proper cross-encoder model
        # For now, we'll use the LLM to score relevance
        
        scored_docs = []
        for i, doc in enumerate(documents[:10]):  # Limit to top 10 for efficiency
            try:
                prompt = ChatPromptTemplate.from_template("""
                Rate the relevance of this document to the query on a scale of 1-10:
                
                Query: {query}
                Document: {document}
                
                Return only the number (1-10):
                """)
                
                chain = prompt | self.llm
                result = chain.invoke({
                    "query": query,
                    "document": doc.page_content[:500]
                })
                
                content = result.content if hasattr(result, 'content') else str(result)
                try:
                    score = float(re.findall(r'\d+', content)[0])
                except:
                    score = 5.0  # Default score
                
                scored_docs.append((doc, score))
                
            except Exception as e:
                logger.warning(f"Error scoring document {i}: {str(e)}")
                scored_docs.append((doc, 5.0))
        
        # Add remaining documents with default score
        for doc in documents[10:]:
            scored_docs.append((doc, 1.0))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs]
        
        logger.info("Cross-encoder reranking completed")
        return reranked_docs
    
    def _cohere_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply Cohere reranking (placeholder - requires Cohere API)."""
        logger.warning("Cohere reranking not implemented. Falling back to original order.")
        return documents
    
    def retrieve(self, query: str) -> str:
        """Main retrieval method that handles the complete pipeline."""
        logger.info("="*60)
        logger.info("STARTING RAG RETRIEVAL PIPELINE")
        logger.info("="*60)
        logger.info(f"Original query: '{query}'")
        
        try:
            # Step 1: Query Translation
            logger.info("Step 1/4: Query Translation")
            translated_queries = self.query_translation(query)
            logger.info(f"- Generated {len(translated_queries)} query variations")
            
            # Step 2: Retrieve documents for each query
            logger.info("Step 2/4: Document Retrieval")
            all_documents = []
            retriever = self.get_retriever()
            
            for i, tq in enumerate(translated_queries):
                logger.debug(f"Retrieving for query {i+1}: '{tq[:50]}...'")
                try:
                    docs = retriever.get_relevant_documents(tq)
                    all_documents.extend(docs)
                    logger.debug(f"Retrieved {len(docs)} documents for query {i+1}")
                except Exception as e:
                    logger.warning(f"Error retrieving for query {i+1}: {str(e)}")
            
            # Remove duplicates based on content
            unique_docs = []
            seen_content = set()
            for doc in all_documents:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
            
            logger.info(f"- Retrieved {len(unique_docs)} unique documents")
            
            # Step 3: Reranking
            logger.info("Step 3/4: Document Reranking")
            reranked_docs = self.rerank(query, unique_docs)
            logger.info(f"- Reranking completed, found {len(reranked_docs)} relevant documents")
            
            # Step 4: Generate Response
            logger.info("Step 4/4: Response Generation")
            final_docs = reranked_docs[:self.top_k]  # Use top_k documents
            logger.info(f"- Using top {min(self.top_k, len(reranked_docs))} documents for response generation")
            
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant who can answer questions based on the provided context. 
            If the context is not relevant to the query, please say "I don't have enough relevant information to answer this question."
            
            Context: {context}
            
            Question: {input}
            
            Answer:
            """)
            
            # Create document chain
            doc_chain = create_stuff_documents_chain(self.llm, prompt)
            
            
            retriever_chain = create_retrieval_chain(retriever, doc_chain)
            
            # Generate response directly without retrieval chain
            logger.debug("Generating final response")
            context = "\n\n".join([doc.page_content for doc in final_docs])

            result = doc_chain.invoke({
            "context": final_docs,
            "input": query
            })
 
            answer = result if isinstance(result, str) else result.get("answer", "No answer generated")
            
            
            
            logger.info("- Response generated successfully")
            logger.info("="*60)
            logger.info("RAG RETRIEVAL PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Final answer length: {len(answer)} characters")
            logger.info("="*60)
            
            return answer
            
        except Exception as e:
            logger.error("="*60)
            logger.error("RAG RETRIEVAL PIPELINE FAILED!")
            logger.error(f"Error: {str(e)}")
            logger.error("="*60)
            raise


# Example usage
if __name__ == "__main__":
    try:
        logger.info("Starting example usage of retrieval pipeline")
        
        # Initialize retrieval pipeline
        retriever = RetrieverPipeline(
            query_translation_strategy="hyde",  # Options: "multi_query", "decomposition", "hyde", "step_back", "none"
            reranking_strategy="cohere",  # Options: "reciprocal_rank_fusion", "cross_encoder", "cohere", "none"
            embedding_model="huggingface",  # Options: "huggingface", "google", "openai"
            top_k=5,  # Number of documents to retrieve
        )
        
        # Test query
        test_query = "What is the diffrence btw prompt injection and jailbreaking"
        logger.info(f"Testing with query: '{test_query}'")
        
        answer = retriever.retrieve(test_query)
        
        logger.info("\n" + "="*50)
        logger.info("FINAL ANSWER:")
        logger.info("="*50)
        logger.info(answer)
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        logger.info("\nTo use this pipeline:")
        logger.info("1. Make sure you have run the ingestion pipeline first")
        logger.info("2. Set up your API keys (GOOGLE_API_KEY for Gemini)")
        logger.info("3. Choose your preferred strategies for query translation and reranking")
        logger.info("4. The vectordb should be the same one used in ingestion")