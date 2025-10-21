"""
Retriever configuration models for swappable RAG architecture.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


RetrieverType = Literal["naive", "bm25"]


class RetrieverConfig(BaseModel):
    """
    Configuration for retriever selection and parameters.
    
    This model enables runtime retriever selection via configuration,
    supporting the swappable retriever architecture.
    """
    
    retriever_type: RetrieverType = Field(
        default="naive",
        description="Type of retriever to use: 'naive' (vector similarity) or 'bm25' (keyword-based)"
    )
    
    top_k: int = Field(
        default=10,
        description="Number of documents to retrieve",
        ge=1,
        le=50
    )
    
    # Type-specific parameters stored as dict for flexibility
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Retriever-specific parameters (e.g., k1, b for BM25)"
    )
    
    @field_validator('params')
    @classmethod
    def validate_params(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Validate retriever-specific parameters."""
        # For now, just ensure it's a dict
        # Future: Add type-specific validation
        if not isinstance(v, dict):
            raise ValueError("params must be a dictionary")
        return v
    
    def get_description(self) -> str:
        """Get human-readable description of this configuration."""
        retriever_names = {
            "naive": "Naive (Vector Similarity)",
            "bm25": "BM25 (Keyword-based)"
        }
        
        desc = f"{retriever_names.get(self.retriever_type, self.retriever_type)} with top_k={self.top_k}"
        
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            desc += f" ({param_str})"
        
        return desc
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "retriever_type": "naive",
                    "top_k": 10,
                    "params": {}
                },
                {
                    "retriever_type": "bm25",
                    "top_k": 10,
                    "params": {"k1": 1.5, "b": 0.75}
                }
            ]
        }
    }
