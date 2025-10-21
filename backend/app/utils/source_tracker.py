"""
Source tracking utility for Legal-OS.

This module provides utilities for tracking and embedding source information
throughout the analysis pipeline, enabling traceability from findings back to
source documents.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC

from app.models.agent import (
    SourceReference,
    SourceMetadata,
    SourceLink,
    SourcedItem,
    ExtractedClause,
    RedFlag,
    KeyFinding,
    Recommendation,
)

logger = logging.getLogger(__name__)


class SourceTracker:
    """
    Utility class for tracking and embedding source information.
    
    The SourceTracker helps maintain source attribution throughout the
    analysis pipeline, from document ingestion through clause extraction,
    risk scoring, and summary generation.
    
    Example:
        ```python
        tracker = SourceTracker(document_id="doc_123")
        
        # Create source reference from chunk
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="The purchase price shall be...",
            page=5,
            section="Section 2.1"
        )
        
        # Create source metadata
        metadata = tracker.create_source_metadata(
            sources=[source_ref],
            extraction_method="llm_extraction",
            confidence=0.95
        )
        
        # Generate frontend link
        link = tracker.generate_link(source_ref)
        ```
    """
    
    def __init__(self, document_id: str):
        """
        Initialize the source tracker.
        
        Args:
            document_id: Unique identifier for the document being analyzed
        """
        self.document_id = document_id
        self._source_chain: Dict[str, SourceMetadata] = {}
        logger.info(f"SourceTracker initialized for document_id={document_id}")
    
    def create_source_reference(
        self,
        chunk_id: str,
        text_snippet: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
        confidence: float = 1.0
    ) -> SourceReference:
        """
        Create a source reference for a specific location in the document.
        
        Args:
            chunk_id: Unique identifier for the document chunk
            text_snippet: Short excerpt (100-200 chars) from the source
            page: Optional page number in the document
            section: Optional section or heading reference
            confidence: Confidence score for this source attribution (0.0 to 1.0)
            
        Returns:
            SourceReference object
        """
        # Truncate text snippet to reasonable length
        if len(text_snippet) > 200:
            text_snippet = text_snippet[:197] + "..."
        
        source_ref = SourceReference(
            document_id=self.document_id,
            page=page,
            section=section,
            chunk_id=chunk_id,
            text_snippet=text_snippet,
            confidence=confidence
        )
        
        logger.debug(f"Created source reference: chunk_id={chunk_id}, page={page}, section={section}")
        return source_ref
    
    def create_source_metadata(
        self,
        sources: List[SourceReference],
        extraction_method: str = "llm_extraction",
        confidence: Optional[float] = None
    ) -> SourceMetadata:
        """
        Create provenance metadata from source references.
        
        Args:
            sources: List of source references supporting this item
            extraction_method: Method used to extract/generate the item
            confidence: Overall confidence score (defaults to average of source confidences)
            
        Returns:
            SourceMetadata object
        """
        # Calculate overall confidence if not provided
        if confidence is None and sources:
            confidence = sum(s.confidence for s in sources) / len(sources)
        elif confidence is None:
            confidence = 1.0
        
        provenance = SourceMetadata(
            sources=sources,
            confidence=confidence,
            extraction_method=extraction_method,
            timestamp=datetime.now(UTC)
        )
        
        logger.debug(f"Created provenance: {len(sources)} sources, confidence={confidence:.2f}")
        return provenance
    
    def track_item(
        self,
        item_id: str,
        provenance: SourceMetadata
    ) -> None:
        """
        Track provenance for a specific item.
        
        Args:
            item_id: Unique identifier for the item
            provenance: Provenance metadata to track
        """
        self._source_chain[item_id] = provenance
        logger.debug(f"Tracked provenance for item_id={item_id}")
    
    def get_source_metadata(self, item_id: str) -> Optional[SourceMetadata]:
        """
        Retrieve provenance metadata for a tracked item.
        
        Args:
            item_id: Unique identifier for the item
            
        Returns:
            SourceMetadata if found, None otherwise
        """
        return self._source_chain.get(item_id)
    
    def generate_link(
        self,
        source_ref: SourceReference,
        base_url: str = "/documents"
    ) -> SourceLink:
        """
        Generate a frontend-renderable link from a source reference.
        
        Args:
            source_ref: Source reference to create link from
            base_url: Base URL for document links
            
        Returns:
            SourceLink object
        """
        # Generate unique link ID
        link_id = f"link_{uuid.uuid4().hex[:8]}"
        
        # Build link text
        parts = []
        if source_ref.section:
            parts.append(source_ref.section)
        if source_ref.page:
            parts.append(f"Page {source_ref.page}")
        
        link_text = ", ".join(parts) if parts else f"Chunk {source_ref.chunk_id}"
        
        # Build link URL
        url_parts = [f"{base_url}/{source_ref.document_id}"]
        if source_ref.page:
            url_parts.append(f"page={source_ref.page}")
        if source_ref.section:
            url_parts.append(f"section={source_ref.section}")
        
        link_url = url_parts[0]
        if len(url_parts) > 1:
            link_url += "#" + "&".join(url_parts[1:])
        
        link = SourceLink(
            link_id=link_id,
            link_text=link_text,
            link_url=link_url,
            tooltip=source_ref.text_snippet,
            document_id=source_ref.document_id,
            page=source_ref.page,
            section=source_ref.section
        )
        
        logger.debug(f"Generated link: {link_text} -> {link_url}")
        return link
    
    def generate_links(
        self,
        provenance: SourceMetadata,
        base_url: str = "/documents"
    ) -> List[SourceLink]:
        """
        Generate frontend-renderable links from provenance metadata.
        
        Args:
            provenance: Provenance metadata containing source references
            base_url: Base URL for document links
            
        Returns:
            List of SourceLink objects
        """
        return [
            self.generate_link(source_ref, base_url)
            for source_ref in provenance.sources
        ]
    
    def trace_chain(self, item_id: str) -> List[SourceMetadata]:
        """
        Trace the full provenance chain for an item.
        
        Args:
            item_id: Unique identifier for the item
            
        Returns:
            List of SourceMetadata objects in the chain
        """
        chain = []
        current_provenance = self.get_source_metadata(item_id)
        
        if current_provenance:
            chain.append(current_provenance)
            # Note: In a more complex system, we might trace through
            # multiple levels of provenance (e.g., finding -> clause -> chunk)
            # For now, we return the direct provenance
        
        logger.debug(f"Traced provenance chain for item_id={item_id}: {len(chain)} levels")
        return chain
    
    def embed_provenance_in_clause(
        self,
        clause: ExtractedClause,
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractedClause:
        """
        Embed provenance metadata into an extracted clause.
        
        Args:
            clause: ExtractedClause to enrich with provenance
            chunk_metadata: Optional metadata about source chunks
            
        Returns:
            ExtractedClause with embedded provenance
        """
        sources = []
        
        # Create source references from chunk IDs
        for chunk_id in clause.source_chunk_ids:
            # Extract metadata if available
            page = None
            section = None
            text_snippet = clause.clause_text[:200]
            
            if chunk_metadata and chunk_id in chunk_metadata:
                metadata = chunk_metadata[chunk_id]
                page = metadata.get("page")
                section = metadata.get("section")
                text_snippet = metadata.get("text", clause.clause_text[:200])
            
            source_ref = self.create_source_reference(
                chunk_id=chunk_id,
                text_snippet=text_snippet,
                page=page,
                section=section,
                confidence=clause.confidence
            )
            sources.append(source_ref)
        
        # Create provenance metadata
        if sources:
            provenance = self.create_source_metadata(
                sources=sources,
                extraction_method="llm_extraction",
                confidence=clause.confidence
            )
            clause.provenance = provenance
        
        return clause
    
    def embed_provenance_in_finding(
        self,
        finding: KeyFinding,
        source_clauses: List[ExtractedClause]
    ) -> KeyFinding:
        """
        Embed provenance metadata into a key finding.
        
        Args:
            finding: KeyFinding to enrich with provenance
            source_clauses: List of clauses that support this finding
            
        Returns:
            KeyFinding with embedded provenance
        """
        sources = []
        
        # Aggregate sources from all supporting clauses
        for clause in source_clauses:
            if clause.provenance:
                sources.extend(clause.provenance.sources)
        
        # Create provenance metadata
        if sources:
            provenance = self.create_source_metadata(
                sources=sources,
                extraction_method="aggregation"
            )
            finding.provenance = provenance
        
        return finding
    
    def embed_provenance_in_recommendation(
        self,
        recommendation: Recommendation,
        source_findings: List[KeyFinding]
    ) -> Recommendation:
        """
        Embed provenance metadata into a recommendation.
        
        Args:
            recommendation: Recommendation to enrich with provenance
            source_findings: List of findings that support this recommendation
            
        Returns:
            Recommendation with embedded provenance
        """
        sources = []
        
        # Aggregate sources from all supporting findings
        for finding in source_findings:
            if finding.provenance:
                sources.extend(finding.provenance.sources)
        
        # Create provenance metadata
        if sources:
            provenance = self.create_source_metadata(
                sources=sources,
                extraction_method="aggregation"
            )
            recommendation.provenance = provenance
        
        return recommendation
    
    def get_sources_for_item(
        self,
        item_id: str
    ) -> List[SourceReference]:
        """
        Get all source references for a specific item.
        
        Args:
            item_id: Unique identifier for the item
            
        Returns:
            List of SourceReference objects
        """
        provenance = self.get_source_metadata(item_id)
        if provenance:
            return provenance.sources
        return []
    
    def get_items_from_source(
        self,
        chunk_id: str
    ) -> List[str]:
        """
        Get all items that reference a specific source chunk.
        
        Args:
            chunk_id: Chunk identifier to search for
            
        Returns:
            List of item IDs that reference this chunk
        """
        matching_items = []
        
        for item_id, provenance in self._source_chain.items():
            for source in provenance.sources:
                if source.chunk_id == chunk_id:
                    matching_items.append(item_id)
                    break
        
        logger.debug(f"Found {len(matching_items)} items from chunk_id={chunk_id}")
        return matching_items
    
    def generate_citation(
        self,
        provenance: SourceMetadata,
        format: str = "short"
    ) -> str:
        """
        Generate a citation string from provenance metadata.
        
        Args:
            provenance: Provenance metadata to cite
            format: Citation format ('short' or 'full')
            
        Returns:
            Citation string
        """
        if not provenance.sources:
            return "No source information available"
        
        if format == "short":
            # Short format: "Section X.Y, Page Z"
            source = provenance.sources[0]
            parts = []
            if source.section:
                parts.append(source.section)
            if source.page:
                parts.append(f"Page {source.page}")
            return ", ".join(parts) if parts else f"Document {source.document_id}"
        
        else:  # full format
            # Full format: List all sources
            citations = []
            for source in provenance.sources:
                parts = [f"Document {source.document_id}"]
                if source.section:
                    parts.append(source.section)
                if source.page:
                    parts.append(f"Page {source.page}")
                citations.append(", ".join(parts))
            return "; ".join(citations)
