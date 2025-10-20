"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.rag.retrievers import NaiveRetriever
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for question answering over documents.

    Combines retrieval and generation to answer questions based on
    document context.
    """

    # Default RAG prompt template
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful AI assistant analyzing legal documents. Use the following context from the documents to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context from documents:
{context}

User Question: {question}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, acknowledge this limitation."""

    def __init__(
        self,
        retriever: Optional[NaiveRetriever] = None,
        llm: Optional[ChatOpenAI] = None,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: Optional retriever instance (creates new if not provided)
            llm: Optional LLM instance (creates new if not provided)
            prompt_template: Optional custom prompt template
        """
        # Initialize retriever
        self.retriever = retriever or NaiveRetriever()

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
            session_id: Optional session ID to filter document context
            top_k: Number of chunks to retrieve (defaults to settings.retrieval_top_k)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing RAG query: {question[:100]}...")

            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                session_id=session_id,
                top_k=top_k,
            )

            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "metadata": {
                        "chunks_retrieved": 0,
                        "model": settings.llm_model,
                    },
                }

            # Step 2: Format context
            context = self.retriever.format_context(retrieved_chunks)

            # Step 3: Generate answer using LLM
            logger.info(f"Generating answer using {settings.llm_model}")

            # Create the prompt
            messages = self.prompt.format_messages(
                context=context,
                question=question,
            )

            # Get LLM response
            response = self.llm.invoke(messages)
            answer = response.content

            # Step 4: Extract sources
            sources = [
                {
                    "text": chunk.get("text", "")[:200] + "...",  # Truncate for brevity
                    "score": chunk.get("score", 0.0),
                    "source": chunk.get("metadata", {}).get("file_name", "Unknown"),
                    "chunk_index": chunk.get("chunk_index", 0),
                }
                for chunk in retrieved_chunks
            ]

            logger.info(
                f"Successfully generated answer using {len(retrieved_chunks)} chunks"
            )

            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "chunks_retrieved": len(retrieved_chunks),
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
