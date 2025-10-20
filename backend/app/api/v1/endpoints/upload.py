"""
Upload endpoint for document ingestion.
"""

import logging
import os
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.core.config import settings
from app.models.api import UploadResponse
from app.agents.ingestion import IngestionAgent

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Documents to upload and ingest"),
) -> UploadResponse:
    """
    Upload and ingest documents.

    This endpoint:
    1. Receives uploaded files
    2. Saves them to the data directory
    3. Triggers the ingestion pipeline
    4. Returns a session ID for tracking

    Args:
        files: List of uploaded files (PDF, DOCX, TXT)

    Returns:
        UploadResponse with session_id and file_names

    Raises:
        HTTPException: If upload or ingestion fails
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided",
            )

        # Generate session ID
        session_id = str(uuid4())

        logger.info(f"Processing upload for session {session_id}: {len(files)} files")

        # Ensure data directory exists
        data_dir = Path(settings.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        saved_files = []
        file_names = []

        for file in files:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in IngestionAgent.SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue

            # TODO: Post-MVP - Improve file handling to avoid saving copies with session IDs
            # Consider: process in memory, use temp directory, or implement file deduplication
            # Current approach: Save with session ID prefix to prevent filename collisions
            unique_filename = f"{session_id}_{file.filename}"
            file_path = data_dir / unique_filename

            # Save file
            try:
                contents = await file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)

                saved_files.append(str(file_path))
                file_names.append(file.filename)
                logger.info(f"Saved file: {file_path}")

            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                # Clean up any saved files on error
                for saved_file in saved_files:
                    try:
                        os.remove(saved_file)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error saving file {file.filename}: {str(e)}",
                )

        if not saved_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid files to process. Supported formats: PDF, DOCX, TXT",
            )

        # Initialize ingestion agent and process documents
        try:
            ingestion_agent = IngestionAgent()
            result = ingestion_agent.ingest_documents(
                file_paths=saved_files,
                session_id=session_id,
            )

            logger.info(
                f"Ingestion complete for session {session_id}: "
                f"{result['successful']} successful, {result['failed']} failed"
            )

            # Check if any documents failed
            if result["failed"] > 0:
                logger.warning(
                    f"Some documents failed ingestion: {result['failures']}"
                )

            return UploadResponse(
                session_id=session_id,
                file_names=file_names,
                message=f"Successfully ingested {result['successful']} of {result['total_documents']} documents",
            )

        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            # Clean up saved files on ingestion error
            for saved_file in saved_files:
                try:
                    os.remove(saved_file)
                except Exception:
                    pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during document ingestion: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )
