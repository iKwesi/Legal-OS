"""
Simplified evaluation using LangChain's native Qdrant integration.

This module provides a cleaner evaluation approach that follows LangChain
patterns more closely for better RAGAS compatibility.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

from app.core.config import settings
from app.rag.chunking import get_chunker, ChunkingStrategy

logger = logging.getLogger(__name__)


class LangChainRAGEvaluator:
    """Simplified evaluator using LangChain's native patterns."""

    def __init__(self, sgd_path: str = "golden_dataset/sgd_benchmark.csv"):
        """
        Initialize evaluator.
        
        Args:
            sgd_path: Path to synthetic golden dataset
        """
        self.sgd_path = sgd_path
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant analyzing legal documents. 
Use the following context from the documents to answer the user's question. 
If you cannot find the answer in the context, say so clearly.

Context from documents:
{context}

User Question: {question}

Please provide a clear, accurate answer based on the context above."""
        )
        
        logger.info("LangChainRAGEvaluator initialized")
    
    def load_sgd(self) -> pd.DataFrame:
        """Load synthetic golden dataset."""
        sgd_df = pd.read_csv(self.sgd_path)
        logger.info(f"Loaded SGD with {len(sgd_df)} samples")
        return sgd_df
    
    def load_and_chunk_documents(
        self,
        chunking_strategy: ChunkingStrategy = "naive",
        chunking_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load documents from data directory and chunk them.
        
        Args:
            chunking_strategy: Chunking strategy to use
            chunking_params: Parameters for chunking
            
        Returns:
            List of LangChain Document objects (chunked)
        """
        from langchain_community.document_loaders import PyMuPDFLoader
        
        # Get chunker
        params = chunking_params or {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
        chunker = get_chunker(strategy=chunking_strategy, **params)
        
        # Load documents
        data_dir = Path("../data")
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in data directory")
            return []
        
        all_documents = []
        
        for pdf_file in pdf_files[:1]:  # Use first PDF for evaluation
            logger.info(f"Loading {pdf_file.name}")
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Combine pages into single text
            full_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Chunk the document
            chunks = chunker.chunk_document(
                document_id=pdf_file.stem,
                text=full_text,
                metadata={"file_name": pdf_file.name, "source": str(pdf_file)},
            )
            
            # Convert chunks to LangChain Documents
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
                all_documents.append(doc)
            
            logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
        
        logger.info(f"Total documents created: {len(all_documents)}")
        return all_documents
    
    def evaluate_naive_retrieval(
        self,
        chunking_strategy: ChunkingStrategy = "naive",
        chunking_params: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate naive (vector similarity) retrieval.
        
        Args:
            chunking_strategy: Chunking strategy to use
            chunking_params: Parameters for chunking
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        logger.info(f"Evaluating {chunking_strategy} chunking with naive retrieval (k={top_k})")
        
        # Load and chunk documents
        documents = self.load_and_chunk_documents(chunking_strategy, chunking_params)
        
        if not documents:
            raise ValueError("No documents loaded for evaluation")
        
        # Create vectorstore
        logger.info("Creating in-memory vectorstore")
        vectorstore = Qdrant.from_documents(
            documents,
            self.embeddings,
            location=":memory:",
            collection_name=f"eval_{chunking_strategy}",
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        # Create RAG chain
        rag_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.rag_prompt | self.llm, "context": itemgetter("context")}
        )
        
        # Load SGD
        sgd_df = self.load_sgd()
        
        # Run evaluation
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        logger.info(f"Running RAG chain on {len(sgd_df)} questions")
        
        for _, row in sgd_df.iterrows():
            question = row["user_input"]
            ground_truth = row["reference"]
            
            # Invoke RAG chain
            result = rag_chain.invoke({"question": question})
            
            # Extract results
            answer = result["response"].content if hasattr(result["response"], "content") else str(result["response"])
            context_docs = result["context"]
            
            questions.append(question)
            answers.append(answer)
            contexts.append([doc.page_content for doc in context_docs])
            ground_truths.append(ground_truth)
        
        # Create RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        
        # Evaluate with RAGAS
        logger.info("Running RAGAS evaluation")
        ragas_result = evaluate(
            eval_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        
        # Extract metrics
        metrics = {}
        for metric_name in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            value = ragas_result[metric_name]
            if isinstance(value, list):
                metrics[metric_name] = float(sum(value) / len(value)) if value else 0.0
            else:
                metrics[metric_name] = float(value)
        
        execution_time = time.time() - start_time
        
        result = {
            "chunking_strategy": chunking_strategy,
            "retriever_type": "naive",
            "top_k": top_k,
            "num_documents": len(documents),
            "num_samples": len(sgd_df),
            "metrics": metrics,
            "execution_time_seconds": execution_time,
        }
        
        logger.info(f"Evaluation complete in {execution_time:.2f}s")
        logger.info(f"Metrics: {metrics}")
        
        return result
    
    def evaluate_bm25_retrieval(
        self,
        chunking_strategy: ChunkingStrategy = "naive",
        chunking_params: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate BM25 (keyword-based) retrieval.
        
        Args:
            chunking_strategy: Chunking strategy to use
            chunking_params: Parameters for chunking
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        logger.info(f"Evaluating {chunking_strategy} chunking with BM25 retrieval (k={top_k})")
        
        # Load and chunk documents
        documents = self.load_and_chunk_documents(chunking_strategy, chunking_params)
        
        if not documents:
            raise ValueError("No documents loaded for evaluation")
        
        # Create BM25 retriever
        logger.info("Creating BM25 retriever")
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = top_k
        
        # Create RAG chain
        rag_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.rag_prompt | self.llm, "context": itemgetter("context")}
        )
        
        # Load SGD
        sgd_df = self.load_sgd()
        
        # Run evaluation
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        logger.info(f"Running RAG chain on {len(sgd_df)} questions")
        
        for _, row in sgd_df.iterrows():
            question = row["user_input"]
            ground_truth = row["reference"]
            
            # Invoke RAG chain
            result = rag_chain.invoke({"question": question})
            
            # Extract results
            answer = result["response"].content if hasattr(result["response"], "content") else str(result["response"])
            context_docs = result["context"]
            
            questions.append(question)
            answers.append(answer)
            contexts.append([doc.page_content for doc in context_docs])
            ground_truths.append(ground_truth)
        
        # Create RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        
        # Evaluate with RAGAS
        logger.info("Running RAGAS evaluation")
        ragas_result = evaluate(
            eval_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        
        # Extract metrics
        metrics = {}
        for metric_name in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            value = ragas_result[metric_name]
            if isinstance(value, list):
                metrics[metric_name] = float(sum(value) / len(value)) if value else 0.0
            else:
                metrics[metric_name] = float(value)
        
        execution_time = time.time() - start_time
        
        result = {
            "chunking_strategy": chunking_strategy,
            "retriever_type": "bm25",
            "top_k": top_k,
            "num_documents": len(documents),
            "num_samples": len(sgd_df),
            "metrics": metrics,
            "execution_time_seconds": execution_time,
        }
        
        logger.info(f"Evaluation complete in {execution_time:.2f}s")
        logger.info(f"Metrics: {metrics}")
        
        return result
