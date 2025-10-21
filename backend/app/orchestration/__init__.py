"""
Orchestration module for coordinating multi-agent workflows.

This module provides the supervisor agent and orchestration logic for
coordinating specialized agents in the Legal-OS system.
"""

from app.orchestration.pipeline import DocumentOrchestrator

__all__ = ["DocumentOrchestrator"]
