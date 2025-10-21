"""
RAG (Retrieval-Augmented Generation) pipeline using LangChain patterns.

This module provides a flexible RAG pipeline that supports swappable retrievers
through configuration-driven selection.
"""

import logging
from typing import Dict, Any, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from app.core.config import settings
from app.models.retriever import RetrieverConfig
from app.rag.retrievers import get_retriever
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for question answering over documents using LangChain patterns.
    
    Combines retrieval and generation using LangChain's retrieval chain pattern
    for better performance and compatibility.
    """

    # Default RAG prompt template
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful AI assistant analyzing legal documents. Use the following context from the documents to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context from documents:
{context}

User Question: {question}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, acknowledge this limitation."""

    def __init__(
        self,
        retriever: Optional[Any] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[ChatOpenAI] = None,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline with swappable retriever support.

        Args:
            retriever: LangChain retriever instance (legacy support)
            retriever_config: Configuration for retriever selection (new approach)
            vector_store: VectorStore instance (required if using retriever_config)
            llm: Optional LLM instance (creates new if not provided)
            prompt_template: Optional custom prompt template
            
        Note:
            Either provide a retriever directly OR provide retriever_config + vector_store.
            If both are provided, retriever takes precedence for backward compatibility.
        """
        # Handle retriever initialization
        if retriever is not None:
            # Legacy mode: use provided retriever directly
            self.retriever = retriever
            self.retriever_config = None
            logger.info("RAGPipeline initialized with provided retriever (legacy mode)")
        elif retriever_config is not None:
            # New mode: create retriever from config
            if not vector_store:
                raise ValueError("vector_store is required when using retriever_config")
            
            self.retriever_config = retriever_config
            self.vector_store = vector_store
            
            # Create retriever based on config
            if retriever_config.retriever_type == "naive":
                self.retriever = get_retriever(
                    retriever_type="naive",
                    vector_store=vector_store,
                    top_k=retriever_config.top_k,
                    **retriever_config.params
                )
            elif retriever_config.retriever_type == "bm25":
                # For BM25, we need to get all documents from vector store
                documents = vector_store.get_all_documents()
                self.retriever = get_retriever(
                    retriever_type="bm25",
                    documents=documents,
                    top_k=retriever_config.top_k,
                    **retriever_config.params
                )
            else:
                raise ValueError(f"Unknown retriever type: {retriever_config.retriever_type}")
            
            logger.info(
                f"RAGPipeline initialized with {retriever_config.get_description()}"
            )
        else:
            # Default: create naive retriever with default vector store
            self.vector_store = vector_store or VectorStore()
            self.retriever_config = RetrieverConfig(retriever_type="naive", top_k=10)
            self.retriever = get_retriever(
                retriever_type="naive",
                vector_store=self.vector_store,
                top_k=10
            )
            logger.info("RAGPipeline initialized with default naive retriever")

        # Initialize LLM
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )

        # Set up prompt template
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Build retrieval chain using LangChain pattern
        self.rag_chain = (
            {"context": itemgetter("question") | self.retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.prompt | self.llm, "context": itemgetter("context")}
        )

        logger.info(
            f"RAGPipeline initialized with model={settings.llm_model}, "
            f"temperature={settings.llm_temperature}"
        )

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            session_id: Optional session ID (not used in current implementation)
            top_k: Optional top_k (not used - set during retriever creation)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing RAG query: {question[:100]}...")

            # Invoke the RAG chain
            result = self.rag_chain.invoke({"question": question})
            
            # Extract answer
            answer = result["response"].content if hasattr(result["response"], "content") else str(result["response"])
            
            # Extract context documents
            context_docs = result["context"]
            
            if not context_docs:
                logger.warning("No relevant chunks found for query")
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "metadata": {
                        "chunks_retrieved": 0,
                        "model": settings.llm_model,
                    },
                }

            # Format sources
            sources = [
                {
                    "text": doc.page_content[:200] + "...",  # Truncate for brevity
                    "score": 1.0 - (idx / max(len(context_docs), 1)),  # Rank-based score
                    "source": doc.metadata.get("file_name", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", idx),
                }
                for idx, doc in enumerate(context_docs)
            ]

            logger.info(
                f"Successfully generated answer using {len(context_docs)} chunks"
            )

            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "chunks_retrieved": len(context_docs),
                    "model": settings.llm_model,
                    "temperature": settings.llm_temperature,
                },
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise

    def query_with_history(
        self,
        question: str,
        chat_history: list[Dict[str, str]],
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question with chat history context (for future use).

        Args:
            question: User's question
            chat_history: List of previous Q&A pairs
            session_id: Optional session ID to filter document context
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # For now, just use the basic query method
        # This can be enhanced in future stories to incorporate chat history
        logger.info("Processing query with history (using basic query for now)")
        return self.query(question=question, session_id=session_id, top_k=top_k)
