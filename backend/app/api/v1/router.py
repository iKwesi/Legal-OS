"""
API v1 router combining all endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import upload, query, analyze

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(upload.router, tags=["upload"])
api_router.include_router(query.router, tags=["query"])
api_router.include_router(analyze.router, tags=["analysis"])
