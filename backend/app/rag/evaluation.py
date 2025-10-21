"""
Evaluation framework for RAG pipeline configurations.

This module provides tools for evaluating different chunking strategies
and retriever types using RAGAS metrics.
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

from app.rag.chunking import get_chunker, ChunkingStrategy
from app.rag.retrievers import get_retriever, RetrieverType
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for a RAG pipeline evaluation run."""

    chunking_strategy: ChunkingStrategy
    retriever_type: RetrieverType
    chunking_params: Dict[str, Any]
    retriever_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_description(self) -> str:
        """Get human-readable description of configuration."""
        return f"{self.chunking_strategy.capitalize()} Chunking + {self.retriever_type.upper()} Retrieval"


@dataclass
class EvaluationResult:
    """Results from a RAG pipeline evaluation."""

    config: EvaluationConfig
    config_hash: str
    timestamp: str
    metrics: Dict[str, float]
    num_samples: int
    execution_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "execution_time_seconds": self.execution_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create result from dictionary."""
        config_data = data["config"]
        config = EvaluationConfig(
            chunking_strategy=config_data["chunking_strategy"],
            retriever_type=config_data["retriever_type"],
            chunking_params=config_data["chunking_params"],
            retriever_params=config_data["retriever_params"],
        )
        
        return cls(
            config=config,
            config_hash=data["config_hash"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            num_samples=data["num_samples"],
            execution_time_seconds=data["execution_time_seconds"],
        )


class EvaluationCache:
    """Cache for evaluation results."""

    def __init__(self, cache_dir: str = "backend/.cache/evaluation_results"):
        """
        Initialize evaluation cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized EvaluationCache at {self.cache_dir}")

    def _get_cache_path(self, config_hash: str) -> Path:
        """Get cache file path for a configuration hash."""
        return self.cache_dir / f"{config_hash}.json"

    def get(self, config: EvaluationConfig) -> Optional[EvaluationResult]:
        """
        Get cached result for a configuration.

        Args:
            config: Evaluation configuration

        Returns:
            Cached result if exists, None otherwise
        """
        config_hash = config.get_hash()
        cache_path = self._get_cache_path(config_hash)

        if not cache_path.exists():
            logger.info(f"No cache found for config {config_hash}")
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            
            result = EvaluationResult.from_dict(data)
            logger.info(f"Loaded cached result for config {config_hash}")
            return result

        except Exception as e:
            logger.error(f"Error loading cache for config {config_hash}: {e}")
            return None

    def set(self, result: EvaluationResult):
        """
        Cache an evaluation result.

        Args:
            result: Evaluation result to cache
        """
        cache_path = self._get_cache_path(result.config_hash)

        try:
            with open(cache_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Cached result for config {result.config_hash}")

        except Exception as e:
            logger.error(f"Error caching result for config {result.config_hash}: {e}")

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cleared evaluation cache")

    def list_cached_configs(self) -> List[str]:
        """List all cached configuration hashes."""
        return [f.stem for f in self.cache_dir.glob("*.json")]


class RAGEvaluator:
    """Evaluator for RAG pipeline configurations using RAGAS."""

    def __init__(
        self,
        sgd_path: str = "golden_dataset/sgd_benchmark.csv",
        cache_dir: str = "backend/.cache/evaluation_results",
        use_cache: bool = True,
    ):
        """
        Initialize RAG evaluator.

        Args:
            sgd_path: Path to synthetic golden dataset CSV
            cache_dir: Directory for caching results
            use_cache: Whether to use cached results
        """
        self.sgd_path = sgd_path
        self.cache = EvaluationCache(cache_dir) if use_cache else None
        self.use_cache = use_cache
        
        logger.info(f"Initialized RAGEvaluator with SGD at {sgd_path}")

    def load_sgd(self) -> pd.DataFrame:
        """
        Load synthetic golden dataset.

        Returns:
            DataFrame with SGD data
        """
        try:
            sgd_df = pd.read_csv(self.sgd_path)
            logger.info(f"Loaded SGD with {len(sgd_df)} samples from {self.sgd_path}")
            return sgd_df

        except Exception as e:
            logger.error(f"Error loading SGD from {self.sgd_path}: {e}")
            raise

    def evaluate_configuration(
        self,
        config: EvaluationConfig,
        session_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate a RAG pipeline configuration using RAGAS.

        Args:
            config: Evaluation configuration
            session_id: Optional session ID for document filtering
            force_refresh: Force re-evaluation even if cached

        Returns:
            Evaluation result with RAGAS metrics
        """
        import time
        
        # Check cache first
        if self.use_cache and not force_refresh:
            cached_result = self.cache.get(config)
            if cached_result:
                logger.info(f"Using cached result for {config.get_description()}")
                return cached_result

        logger.info(f"Evaluating configuration: {config.get_description()}")
        start_time = time.time()

        try:
            # Load SGD
            sgd_df = self.load_sgd()

            # Build RAG pipeline with specified configuration
            pipeline = self._build_pipeline(config, session_id)

            # Run pipeline on SGD questions
            questions = []
            answers = []
            contexts = []
            ground_truths = []

            for _, row in sgd_df.iterrows():
                question = row["user_input"]
                ground_truth = row["reference"]

                # Query the pipeline
                result = pipeline.query(question, session_id=session_id)

                questions.append(question)
                answers.append(result["answer"])
                
                # Extract context texts from sources
                context_texts = [source["text"] for source in result["sources"]]
                contexts.append(context_texts)
                
                ground_truths.append(ground_truth)

            # Create dataset for RAGAS
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })

            # Evaluate with RAGAS
            ragas_result = evaluate(
                eval_dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
            )

            # Extract metrics (RAGAS returns dict with metric names as keys)
            # Values might be lists or scalars, so we handle both cases
            metrics = {}
            for metric_name in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                value = ragas_result[metric_name]
                # If it's a list, take the mean; otherwise use the value directly
                if isinstance(value, list):
                    metrics[metric_name] = float(sum(value) / len(value)) if value else 0.0
                else:
                    metrics[metric_name] = float(value)

            execution_time = time.time() - start_time

            # Create result
            result = EvaluationResult(
                config=config,
                config_hash=config.get_hash(),
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                num_samples=len(sgd_df),
                execution_time_seconds=execution_time,
            )

            # Cache result
            if self.use_cache:
                self.cache.set(result)

            logger.info(
                f"Evaluation complete for {config.get_description()} "
                f"in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating configuration: {e}")
            raise

    def _build_pipeline(
        self,
        config: EvaluationConfig,
        session_id: Optional[str] = None,
    ) -> RAGPipeline:
        """
        Build RAG pipeline with specified configuration and load documents.

        Args:
            config: Evaluation configuration
            session_id: Optional session ID

        Returns:
            Configured RAG pipeline with documents loaded
        """
        from pathlib import Path
        from app.pipelines.ingestion_pipeline import IngestionPipeline
        
        # Create chunker with config-specific strategy
        chunker = get_chunker(
            strategy=config.chunking_strategy,
            **config.chunking_params,
        )

        # Create in-memory vector store
        vector_store = VectorStore(use_memory=True)
        
        # Create ingestion pipeline with the in-memory vector store and chunker
        ingestion = IngestionPipeline(
            vector_store=vector_store,
            chunker=chunker,
        )
        
        # Load and ingest sample documents for evaluation
        logger.info(f"Loading documents for evaluation with {config.get_description()}")
        data_dir = Path("../data")  # Relative to backend/ directory
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if pdf_files:
            # Use first PDF for evaluation
            sample_doc = pdf_files[0]
            logger.info(f"Ingesting {sample_doc.name} for evaluation")
            
            # Ingest document using the pipeline
            result = ingestion.ingest_document(
                file_path=str(sample_doc),
                session_id=session_id or "evaluation",
            )
            
            logger.info(
                f"Ingested {result['file_name']}: {result['chunk_count']} chunks"
            )
        else:
            logger.warning("No PDF files found in data directory for evaluation")

        # Create retriever with the same vector store
        retriever = get_retriever(
            retriever_type=config.retriever_type,
            vector_store=vector_store,
            **config.retriever_params,
        )

        # Create pipeline
        pipeline = RAGPipeline(retriever=retriever)

        return pipeline

    def compare_configurations(
        self,
        configs: List[EvaluationConfig],
        session_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple configurations.

        Args:
            configs: List of configurations to compare
            session_id: Optional session ID

        Returns:
            DataFrame with comparison results
        """
        results = []

        for config in configs:
            result = self.evaluate_configuration(config, session_id)
            
            row = {
                "Configuration": config.get_description(),
                "Chunking": config.chunking_strategy,
                "Retriever": config.retriever_type,
                **result.metrics,
                "Execution Time (s)": result.execution_time_seconds,
            }
            results.append(row)

        comparison_df = pd.DataFrame(results)
        
        logger.info(f"Compared {len(configs)} configurations")
        
        return comparison_df


def create_default_configs() -> List[EvaluationConfig]:
    """
    Create the 4 default evaluation configurations.

    Returns:
        List of 4 evaluation configurations
    """
    configs = [
        # Config 1: Naive Chunking + Naive Retrieval
        EvaluationConfig(
            chunking_strategy="naive",
            retriever_type="naive",
            chunking_params={
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            retriever_params={},
        ),
        # Config 2: Naive Chunking + BM25 Retrieval
        EvaluationConfig(
            chunking_strategy="naive",
            retriever_type="bm25",
            chunking_params={
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            retriever_params={
                "k1": 1.5,
                "b": 0.75,
            },
        ),
        # Config 3: Semantic Chunking + Naive Retrieval
        EvaluationConfig(
            chunking_strategy="semantic",
            retriever_type="naive",
            chunking_params={
                "breakpoint_threshold_type": "percentile",
                "breakpoint_threshold_amount": 95.0,
            },
            retriever_params={},
        ),
        # Config 4: Semantic Chunking + BM25 Retrieval
        EvaluationConfig(
            chunking_strategy="semantic",
            retriever_type="bm25",
            chunking_params={
                "breakpoint_threshold_type": "percentile",
                "breakpoint_threshold_amount": 95.0,
            },
            retriever_params={
                "k1": 1.5,
                "b": 0.75,
            },
        ),
    ]

    return configs
