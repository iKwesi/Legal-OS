"""
Configuration management for Legal-OS backend.

This module handles all configuration settings using pydantic-settings,
including LLM, embedding models, Qdrant, and application settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    app_name: str = Field(default="Legal-OS", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    data_dir: str = Field(default="./data", description="Directory for uploaded documents")

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key for LLM and embeddings")
    
    # LLM Configuration
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for text generation",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM generation (0.0 = deterministic)",
    )
    llm_max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for LLM responses",
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions",
    )

    # Qdrant Configuration
    qdrant_host: str = Field(
        default="qdrant",
        description="Qdrant service host (use 'qdrant' for docker-compose, 'localhost' for local)",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant HTTP API port",
    )
    qdrant_collection_name: str = Field(
        default="legal_documents",
        description="Qdrant collection name for document chunks",
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for document splitting",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks",
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve for RAG",
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for retrieval",
    )


# Global settings instance
settings = Settings()
