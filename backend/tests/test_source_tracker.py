"""
Tests for the Provenance Tracker.

This module tests the provenance tracking functionality including source
reference creation, provenance metadata management, link generation, and
integration with agent outputs.
"""

import pytest
from datetime import datetime, UTC

from app.utils.source_tracker import SourceTracker
from app.models.agent import (
    SourceReference,
    SourceMetadata,
    SourceLink,
    ExtractedClause,
    KeyFinding,
    Recommendation,
)


class TestSourceTracker:
    """Tests for SourceTracker initialization and basic functionality."""
    
    def test_initialization(self):
        """Test SourceTracker initialization."""
        tracker = SourceTracker(document_id="doc_123")
        assert tracker.document_id == "doc_123"
        assert tracker._source_chain == {}
    
    def test_create_source_reference(self):
        """Test creating a source reference."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="The purchase price shall be $10,000,000.",
            page=5,
            section="Section 2.1",
            confidence=0.95
        )
        
        assert isinstance(source_ref, SourceReference)
        assert source_ref.document_id == "doc_123"
        assert source_ref.chunk_id == "chunk_456"
        assert source_ref.page == 5
        assert source_ref.section == "Section 2.1"
        assert source_ref.confidence == 0.95
        assert "purchase price" in source_ref.text_snippet
    
    def test_create_source_reference_truncates_long_text(self):
        """Test that long text snippets are truncated."""
        tracker = SourceTracker(document_id="doc_123")
        
        long_text = "A" * 300
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet=long_text
        )
        
        assert len(source_ref.text_snippet) == 200
        assert source_ref.text_snippet.endswith("...")
    
    def test_create_source_metadata(self):
        """Test creating provenance metadata."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text",
            confidence=0.9
        )
        
        provenance = tracker.create_source_metadata(
            sources=[source_ref],
            extraction_method="llm_extraction",
            confidence=0.9
        )
        
        assert isinstance(provenance, SourceMetadata)
        assert len(provenance.sources) == 1
        assert provenance.confidence == 0.9
        assert provenance.extraction_method == "llm_extraction"
        assert isinstance(provenance.timestamp, datetime)
    
    def test_create_source_metadata_calculates_average_confidence(self):
        """Test that provenance calculates average confidence from sources."""
        tracker = SourceTracker(document_id="doc_123")
        
        source1 = tracker.create_source_reference(
            chunk_id="chunk_1",
            text_snippet="Text 1",
            confidence=0.8
        )
        source2 = tracker.create_source_reference(
            chunk_id="chunk_2",
            text_snippet="Text 2",
            confidence=1.0
        )
        
        provenance = tracker.create_source_metadata(sources=[source1, source2])
        
        assert provenance.confidence == 0.9  # (0.8 + 1.0) / 2


class TestProvenanceTracking:
    """Tests for tracking and retrieving provenance."""
    
    def test_track_and_get_source_metadata(self):
        """Test tracking and retrieving provenance for an item."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text"
        )
        provenance = tracker.create_source_metadata(sources=[source_ref])
        
        tracker.track_item("item_1", provenance)
        
        retrieved = tracker.get_source_metadata("item_1")
        assert retrieved is not None
        assert retrieved.sources[0].chunk_id == "chunk_456"
    
    def test_get_source_metadata_returns_none_for_unknown_item(self):
        """Test that get_source_metadata returns None for unknown items."""
        tracker = SourceTracker(document_id="doc_123")
        
        result = tracker.get_source_metadata("unknown_item")
        assert result is None
    
    def test_trace_chain(self):
        """Test tracing provenance chain."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text"
        )
        provenance = tracker.create_source_metadata(sources=[source_ref])
        tracker.track_item("item_1", provenance)
        
        chain = tracker.trace_chain("item_1")
        
        assert len(chain) == 1
        assert chain[0] == provenance
    
    def test_trace_chain_empty_for_unknown_item(self):
        """Test that trace_chain returns empty list for unknown items."""
        tracker = SourceTracker(document_id="doc_123")
        
        chain = tracker.trace_chain("unknown_item")
        assert chain == []


class TestLinkGeneration:
    """Tests for generating frontend-renderable links."""
    
    def test_generate_link_with_page_and_section(self):
        """Test generating a link with page and section."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text",
            page=5,
            section="Section 2.1"
        )
        
        link = tracker.generate_link(source_ref)
        
        assert isinstance(link, SourceLink)
        assert "Section 2.1" in link.link_text
        assert "Page 5" in link.link_text
        assert "/documents/doc_123" in link.link_url
        assert "page=5" in link.link_url
        assert "section=Section 2.1" in link.link_url
        assert link.tooltip == "Test text"
    
    def test_generate_link_without_page_and_section(self):
        """Test generating a link without page and section."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text"
        )
        
        link = tracker.generate_link(source_ref)
        
        assert "Chunk chunk_456" in link.link_text
        assert link.link_url == "/documents/doc_123"
    
    def test_generate_links_from_provenance(self):
        """Test generating multiple links from provenance metadata."""
        tracker = SourceTracker(document_id="doc_123")
        
        source1 = tracker.create_source_reference(
            chunk_id="chunk_1",
            text_snippet="Text 1",
            page=1
        )
        source2 = tracker.create_source_reference(
            chunk_id="chunk_2",
            text_snippet="Text 2",
            page=2
        )
        
        provenance = tracker.create_source_metadata(sources=[source1, source2])
        links = tracker.generate_links(provenance)
        
        assert len(links) == 2
        assert all(isinstance(link, SourceLink) for link in links)


class TestProvenanceEmbedding:
    """Tests for embedding provenance in agent outputs."""
    
    def test_embed_provenance_in_clause(self):
        """Test embedding provenance in an extracted clause."""
        tracker = SourceTracker(document_id="doc_123")
        
        clause = ExtractedClause(
            clause_text="The purchase price shall be $10,000,000.",
            clause_type="payment_terms",
            location={"page": 5},
            confidence=0.95,
            source_chunk_ids=["chunk_456", "chunk_457"]
        )
        
        chunk_metadata = {
            "chunk_456": {
                "page": 5,
                "section": "Section 2.1",
                "text": "The purchase price shall be $10,000,000."
            },
            "chunk_457": {
                "page": 5,
                "section": "Section 2.1",
                "text": "payable in cash at closing."
            }
        }
        
        enriched_clause = tracker.embed_provenance_in_clause(clause, chunk_metadata)
        
        assert enriched_clause.provenance is not None
        assert len(enriched_clause.provenance.sources) == 2
        assert enriched_clause.provenance.sources[0].page == 5
        assert enriched_clause.provenance.sources[0].section == "Section 2.1"
    
    def test_embed_provenance_in_clause_without_metadata(self):
        """Test embedding provenance without chunk metadata."""
        tracker = SourceTracker(document_id="doc_123")
        
        clause = ExtractedClause(
            clause_text="The purchase price shall be $10,000,000.",
            clause_type="payment_terms",
            confidence=0.95,
            source_chunk_ids=["chunk_456"]
        )
        
        enriched_clause = tracker.embed_provenance_in_clause(clause)
        
        assert enriched_clause.provenance is not None
        assert len(enriched_clause.provenance.sources) == 1
        assert enriched_clause.provenance.sources[0].chunk_id == "chunk_456"
    
    def test_embed_provenance_in_finding(self):
        """Test embedding provenance in a key finding."""
        tracker = SourceTracker(document_id="doc_123")
        
        # Create clauses with provenance
        clause1 = ExtractedClause(
            clause_text="Clause 1",
            clause_type="indemnification",
            confidence=0.9,
            source_chunk_ids=["chunk_1"]
        )
        clause1 = tracker.embed_provenance_in_clause(clause1)
        
        clause2 = ExtractedClause(
            clause_text="Clause 2",
            clause_type="indemnification",
            confidence=0.95,
            source_chunk_ids=["chunk_2"]
        )
        clause2 = tracker.embed_provenance_in_clause(clause2)
        
        # Create finding
        finding = KeyFinding(
            finding="Unlimited indemnification liability",
            severity="Critical",
            impact="High financial risk"
        )
        
        enriched_finding = tracker.embed_provenance_in_finding(
            finding,
            [clause1, clause2]
        )
        
        assert enriched_finding.provenance is not None
        assert len(enriched_finding.provenance.sources) == 2
        assert enriched_finding.provenance.extraction_method == "aggregation"
    
    def test_embed_provenance_in_recommendation(self):
        """Test embedding provenance in a recommendation."""
        tracker = SourceTracker(document_id="doc_123")
        
        # Create finding with provenance
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_1",
            text_snippet="Test text"
        )
        provenance = tracker.create_source_metadata(sources=[source_ref])
        
        finding = KeyFinding(
            finding="Test finding",
            severity="High",
            impact="Test impact",
            provenance=provenance
        )
        
        # Create recommendation
        recommendation = Recommendation(
            recommendation="Negotiate better terms",
            priority="High",
            rationale="To reduce risk"
        )
        
        enriched_rec = tracker.embed_provenance_in_recommendation(
            recommendation,
            [finding]
        )
        
        assert enriched_rec.provenance is not None
        assert len(enriched_rec.provenance.sources) == 1
        assert enriched_rec.provenance.extraction_method == "aggregation"


class TestProvenanceQueries:
    """Tests for querying provenance information."""
    
    def test_get_sources_for_item(self):
        """Test getting sources for a specific item."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text"
        )
        provenance = tracker.create_source_metadata(sources=[source_ref])
        tracker.track_item("item_1", provenance)
        
        sources = tracker.get_sources_for_item("item_1")
        
        assert len(sources) == 1
        assert sources[0].chunk_id == "chunk_456"
    
    def test_get_sources_for_unknown_item(self):
        """Test getting sources for unknown item returns empty list."""
        tracker = SourceTracker(document_id="doc_123")
        
        sources = tracker.get_sources_for_item("unknown_item")
        assert sources == []
    
    def test_get_items_from_source(self):
        """Test getting all items from a specific source chunk."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text"
        )
        provenance = tracker.create_source_metadata(sources=[source_ref])
        
        tracker.track_item("item_1", provenance)
        tracker.track_item("item_2", provenance)
        
        items = tracker.get_items_from_source("chunk_456")
        
        assert len(items) == 2
        assert "item_1" in items
        assert "item_2" in items
    
    def test_get_items_from_unknown_source(self):
        """Test getting items from unknown source returns empty list."""
        tracker = SourceTracker(document_id="doc_123")
        
        items = tracker.get_items_from_source("unknown_chunk")
        assert items == []


class TestCitationGeneration:
    """Tests for generating citations from provenance."""
    
    def test_generate_short_citation(self):
        """Test generating a short citation."""
        tracker = SourceTracker(document_id="doc_123")
        
        source_ref = tracker.create_source_reference(
            chunk_id="chunk_456",
            text_snippet="Test text",
            page=5,
            section="Section 2.1"
        )
        provenance = tracker.create_source_metadata(sources=[source_ref])
        
        citation = tracker.generate_citation(provenance, format="short")
        
        assert "Section 2.1" in citation
        assert "Page 5" in citation
    
    def test_generate_full_citation(self):
        """Test generating a full citation with multiple sources."""
        tracker = SourceTracker(document_id="doc_123")
        
        source1 = tracker.create_source_reference(
            chunk_id="chunk_1",
            text_snippet="Text 1",
            page=5,
            section="Section 2.1"
        )
        source2 = tracker.create_source_reference(
            chunk_id="chunk_2",
            text_snippet="Text 2",
            page=6,
            section="Section 3.1"
        )
        
        provenance = tracker.create_source_metadata(sources=[source1, source2])
        citation = tracker.generate_citation(provenance, format="full")
        
        assert "Document doc_123" in citation
        assert "Section 2.1" in citation
        assert "Page 5" in citation
        assert "Section 3.1" in citation
        assert "Page 6" in citation
    
    def test_generate_citation_with_no_sources(self):
        """Test generating citation with no sources."""
        tracker = SourceTracker(document_id="doc_123")
        
        provenance = tracker.create_source_metadata(sources=[])
        citation = tracker.generate_citation(provenance)
        
        assert "No source information available" in citation
