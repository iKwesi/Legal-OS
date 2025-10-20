#!/usr/bin/env python3
"""
Synthetic Golden Dataset (SGD) Generation Script

This script generates a synthetic golden dataset using RAGAS TestsetGenerator
from source documents in the data/ directory. The generated dataset is saved
as a CSV file in the golden_dataset/ directory.

Usage:
    python backend/app/scripts/generate_sgd.py [options]

Options:
    --test-size INT    Number of test samples to generate (default: 10)
    --output PATH      Output CSV file path (default: golden_dataset/sgd_benchmark.csv)
    --data-dir PATH    Directory containing source documents (default: data/)
    --verbose          Enable verbose logging

Environment Variables:
    OPENAI_API_KEY         Required: OpenAI API key for LLM and embeddings

Example:
    # Generate 10 questions with default distribution (uses first document only)
    python backend/app/scripts/generate_sgd.py

    # Generate 20 questions with custom output path
    python backend/app/scripts/generate_sgd.py --test-size 20 --output custom_sgd.csv

    # Generate with verbose logging
    python backend/app/scripts/generate_sgd.py --verbose
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from ragas.testset import TestsetGenerator
from openai import RateLimitError, APIConnectionError, APITimeoutError

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the script.

    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Suppress verbose logging from third-party libraries unless in verbose mode
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("ragas").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


def load_documents_from_directory(data_dir: str) -> List[Document]:
    """
    Load all supported documents from the specified directory.
    
    This function loads documents directly without requiring Qdrant/vector store,
    making it suitable for standalone script execution.

    Args:
        data_dir: Path to directory containing source documents

    Returns:
        List of LangChain Document objects

    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If no valid documents found in directory
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not data_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")

    logger.info(f"Loading documents from: {data_dir}")

    # Supported file extensions (matching IngestionPipeline)
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
    
    # Get all files with supported extensions
    supported_files = []
    for ext in SUPPORTED_EXTENSIONS:
        supported_files.extend(data_path.glob(f"*{ext}"))

    if not supported_files:
        raise ValueError(
            f"No supported documents found in {data_dir}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"Found {len(supported_files)} supported documents")

    # Load each document
    documents = []
    for file_path in supported_files:
        try:
            logger.info(f"Loading: {file_path.name}")
            
            # Select appropriate loader based on file extension
            extension = file_path.suffix.lower()
            if extension == ".pdf":
                loader = PyMuPDFLoader(str(file_path))
            elif extension == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif extension == ".txt":
                loader = TextLoader(str(file_path))
            else:
                logger.warning(f"Skipping unsupported file: {file_path.name}")
                continue
            
            # Load document
            loaded_docs = loader.load()
            
            # Combine all pages/sections into single document
            content = "\n\n".join([doc.page_content for doc in loaded_docs])
            
            # Create metadata
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": extension.lstrip("."),
                "file_size": file_path.stat().st_size,
            }
            
            # Create LangChain Document for RAGAS
            doc = Document(
                page_content=content,
                metadata=metadata,
            )
            documents.append(doc)

            logger.info(
                f"Loaded {file_path.name}: {len(doc.page_content)} characters"
            )

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            # Continue with other documents rather than failing completely
            continue

    if not documents:
        raise ValueError("Failed to load any documents successfully")

    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


def generate_testset(
    documents: List[Document],
    test_size: int,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Any:
    """
    Generate synthetic test dataset using RAGAS with rate limiting and retry logic.

    Args:
        documents: List of source documents
        test_size: Number of test samples to generate
        max_retries: Maximum number of retry attempts for rate limit errors
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        RAGAS Testset object

    Raises:
        Exception: If generation fails after all retries
    """
    logger.info("Initializing RAGAS TestsetGenerator with rate limiting")

    # Initialize LLM and embeddings with conservative rate limiting settings
    generator_llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        max_retries=5,  # Increased retries for rate limiting
        timeout=60.0,  # 60 second timeout
        request_timeout=60.0,  # Request-level timeout
    )

    critic_llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        max_retries=5,
        timeout=60.0,
        request_timeout=60.0,
    )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        max_retries=5,
        timeout=60.0,
    )

    # Create generator (RAGAS API uses 'llm' and 'embedding_model' parameter names)
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    logger.info(f"Generating {test_size} test samples with rate limiting")
    logger.info(f"Using first document only to minimize API costs")
    logger.info(f"Max retries: {max_retries}, Base delay: {base_delay}s")

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff: 2s, 4s, 8s, etc.
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay...")
                time.sleep(delay)

            # Generate testset (limit to first document to save API costs)
            logger.info("Starting RAGAS generation...")
            testset = generator.generate_with_langchain_docs(
                documents=documents[:1],
                testset_size=test_size,
                with_debugging_logs=logger.level == logging.DEBUG,
                raise_exceptions=False,  # Handle errors gracefully
            )

            logger.info(f"Successfully generated {test_size} test samples")
            return testset

        except RateLimitError as e:
            logger.warning(f"Rate limit error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for rate limit errors")
                raise Exception(
                    f"Rate limit exceeded after {max_retries} attempts. "
                    "Please wait a few minutes and try again with a smaller --test-size. "
                    f"Current test size: {test_size}"
                ) from e
            continue

        except (APIConnectionError, APITimeoutError) as e:
            logger.warning(f"Connection/timeout error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for connection errors")
                raise Exception(
                    f"Connection error after {max_retries} attempts. "
                    "Please check your internet connection and OpenAI API status. "
                    "You may also try reducing --test-size."
                ) from e
            continue

        except Exception as e:
            # For other errors, log details and raise immediately
            logger.error(f"Unexpected error during generation: {type(e).__name__}: {e}")
            logger.error(f"Error details: {str(e)}")
            raise Exception(
                f"Failed to generate testset: {type(e).__name__}: {e}. "
                "Try running with --verbose for more details."
            ) from e

    # Should never reach here, but just in case
    raise Exception(f"Failed to generate testset after {max_retries} attempts")


def save_testset_to_csv(testset: Any, output_path: str) -> None:
    """
    Save RAGAS testset to CSV file.

    Args:
        testset: RAGAS Testset object
        output_path: Path to output CSV file

    Raises:
        Exception: If saving fails
    """
    output_file = Path(output_path)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving testset to: {output_path}")

    try:
        # Convert to pandas DataFrame
        df = testset.to_pandas()

        # Verify required columns exist
        required_columns = ["user_input", "reference_contexts", "reference", "synthesizer_name"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(
                f"Missing expected columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        # Save to CSV
        df.to_csv(output_path, index=False)

        logger.info(
            f"Successfully saved {len(df)} samples to {output_path} "
            f"with columns: {list(df.columns)}"
        )

        # Log summary statistics
        logger.info(f"Dataset summary:")
        logger.info(f"  Total samples: {len(df)}")
        if "synthesizer_name" in df.columns:
            logger.info(f"  Question type distribution:")
            for qtype, count in df["synthesizer_name"].value_counts().items():
                logger.info(f"    {qtype}: {count} ({count/len(df):.1%})")

    except Exception as e:
        logger.error(f"Failed to save testset to CSV: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Golden Dataset using RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=3,
        help="Number of test samples to generate (default: 3, recommended to start small)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: golden_dataset/sgd_benchmark_YYYYMMDD_HHMMSS.csv with timestamp)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing source documents (default: data/)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main execution function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Generate default output filename with timestamp if not specified
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"golden_dataset/sgd_benchmark_{timestamp}.csv"

        # Setup logging
        setup_logging(args.verbose)

        logger.info("=" * 80)
        logger.info("Synthetic Golden Dataset Generation")
        logger.info("=" * 80)

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )

        logger.info(f"Configuration:")
        logger.info(f"  Test size: {args.test_size}")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output file: {args.output}")
        logger.info(f"  LLM model: {settings.llm_model}")
        logger.info(f"  Embedding model: {settings.embedding_model}")
        
        # Warn about API costs
        if args.test_size > 5:
            logger.warning(
                f"Generating {args.test_size} samples may consume significant API credits. "
                "Consider starting with --test-size 3 for testing."
            )

        # Load documents
        logger.info("\nStep 1: Loading documents...")
        documents = load_documents_from_directory(args.data_dir)

        # Generate testset
        logger.info("\nStep 2: Generating testset...")
        testset = generate_testset(
            documents=documents,
            test_size=args.test_size,
        )

        # Save to CSV
        logger.info("\nStep 3: Saving testset to CSV...")
        save_testset_to_csv(testset, args.output)

        logger.info("\n" + "=" * 80)
        logger.info("✓ Synthetic Golden Dataset generation completed successfully!")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 1

    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=args.verbose if 'args' in locals() else False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
