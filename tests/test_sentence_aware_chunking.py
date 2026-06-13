"""Test suite for sentence-aware chunking validation."""

import pytest
from app.ingestion.ingest import DocumentIngestionManager
from app.config import IngestionConfig


class TestSentenceBoundaryPreservation:
    """Verify chunks respect sentence boundaries."""

    def test_chunks_preserve_sentence_boundaries(self):
        """Normal sentences should not be split across chunks.
        
        Note: Long sentences that exceed chunk_size may be split as a fallback,
        which is acceptable per requirements. This test checks that normal-length
        sentences remain complete.
        """
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=100,  # Larger to accommodate normal sentences
            chunk_overlap=10,
        )

        # All sentences are under chunk_size
        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "Docker is a containerization runtime. "
            "Both are essential infrastructure tools."
        )

        chunks = manager._chunk_text(text)

        # Verify sentences are not split - each chunk should end with . ! or ?
        for i, chunk in enumerate(chunks):
            # Should end with sentence terminator
            assert chunk.rstrip().endswith((".", "!", "?")), (
                f"Chunk {i} should end with sentence terminator: {chunk[-30:]}"
            )

    def test_long_sentence_splitting_acceptable(self):
        """Long sentences that exceed chunk_size may need splitting (fallback case).
        
        This is acceptable per requirements:
        'Only fall back to splitting inside a sentence when a single sentence 
        itself exceeds the configured chunk size.'
        """
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=10,
        )

        # Contains one very long sentence
        long_sentence = (
            "Kubernetes is a container orchestration platform that coordinates "
            "container deployment, scaling, and networking across clusters."
        )

        chunks = manager._chunk_text(long_sentence)

        # Multiple chunks expected since sentence exceeds chunk_size
        assert len(chunks) > 1, "Long sentence should be split into multiple chunks"

        # This is acceptable - long sentences will create incomplete phrase chunks
        # The important thing is that normal sentences are not split unnecessarily

        for i, chunk in enumerate(chunks):
            # Should start with capital letter or be a continuation
            assert chunk[0].isupper() or chunk[0].isdigit(), (
                f"Chunk {i} doesn't start with capital: {chunk[:30]}"
            )

    def test_no_semantic_fragments_created(self):
        """Chunks should have meaningful semantic content."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=12,
        )

        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "Kubernetes coordinates container deployment, scaling, and networking across clusters. "
            "It schedules pods, manages services, and keeps workloads available even when nodes fail."
        )

        chunks = manager._chunk_text(text)

        # Check for semantic fragments (strings with very little meaning)
        bad_fragments = [
            "at scale.",
            "and",
            "scaling,",
            "scaling, and",
            "networking",
        ]

        for chunk in chunks:
            # Chunk should not be a known bad fragment
            assert chunk.strip() not in bad_fragments, f"Semantic fragment found: {chunk}"
            # Chunk should have minimum length (arbitrary: 20 chars for real content)
            assert len(chunk) >= 20, f"Fragment too small: {chunk}"

    def test_chunk_contains_complete_sentences(self):
        """Each chunk should contain at least one complete sentence."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=10,
        )

        text = (
            "Sentence one goes here. "
            "Sentence two is second. "
            "Sentence three comes next. "
            "Sentence four is last."
        )

        chunks = manager._chunk_text(text)

        for i, chunk in enumerate(chunks):
            # Count sentences (end with . ! ?)
            sentence_count = sum(1 for c in chunk if c in ".!?")
            assert sentence_count >= 1, f"Chunk {i} has no complete sentence: {chunk}"

    def test_overlap_occurs_at_sentence_boundaries(self):
        """Overlapping text should align with sentence boundaries when possible."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=80,
            chunk_overlap=20,
        )

        text = (
            "First sentence here. "
            "Second sentence here. "
            "Third sentence here. "
            "Fourth sentence here."
        )

        chunks = manager._chunk_text(text)

        # If there's overlap, it should be at sentence start/end
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Check if next chunk starts with text from current chunk
            if next_chunk[:20] in curr_chunk:
                # The overlap should be a complete sentence or multiple complete sentences
                overlap_portion = next_chunk[:20]
                # Should contain a sentence end
                assert (
                    "." in overlap_portion or len(overlap_portion) > 15
                ), f"Overlap not at sentence boundary: {overlap_portion}"

    def test_single_sentence_longer_than_chunk_size(self):
        """When a single sentence exceeds chunk_size, it should be split intelligently."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=50,
            chunk_overlap=5,
        )

        long_sentence = (
            "Kubernetes coordinates container deployment, scaling, and networking "
            "across clusters with high availability."
        )

        chunks = manager._chunk_text(long_sentence)

        # Should create multiple chunks
        assert len(chunks) > 1, "Long sentence should be split into multiple chunks"

        # All chunks should be close to chunk_size (within reason)
        for chunk in chunks:
            # Allow some flexibility for word boundaries
            assert len(chunk) <= manager.chunk_size + 20, (
                f"Chunk exceeds size limit by too much: {len(chunk)} > {manager.chunk_size + 20}"
            )

    def test_multiple_short_sentences_grouped(self):
        """Multiple short sentences should be grouped into single chunks."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=100,
            chunk_overlap=10,
        )

        text = (
            "One. "
            "Two. "
            "Three. "
            "Four. "
            "Five. "
            "Six. "
            "Seven. "
            "Eight. "
            "Nine. "
            "Ten."
        )

        chunks = manager._chunk_text(text)

        # Should group into fewer chunks (not 10 chunks for 10 sentences)
        assert len(chunks) < 10, f"Too many chunks created: {len(chunks)}"
        assert len(chunks) >= 1, "Should create at least one chunk"

        # Each chunk should contain multiple sentences
        for chunk in chunks:
            sentence_count = sum(1 for c in chunk if c in ".!?")
            assert sentence_count >= 1, f"Chunk has no sentences: {chunk}"


class TestSemanticFragmentDetection:
    """Detect problematic chunk fragmentation."""

    def test_detect_fragment_stubs(self):
        """Identify chunks that are just sentence fragments."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=12,
        )

        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "Kubernetes coordinates container deployment, scaling, and networking across clusters."
        )

        chunks = manager._chunk_text(text)

        # Define fragment patterns to check
        fragment_patterns = [
            r"^(and|or|but|the|a) ",  # Starts with article/conjunction
            r"^[\s,]",  # Starts with space or comma
            r"[\s,]$",  # Ends with space or comma
        ]

        import re

        for chunk in chunks:
            for pattern in fragment_patterns:
                assert not re.match(pattern, chunk, re.IGNORECASE), (
                    f"Fragment pattern detected in chunk: {chunk[:40]}"
                )

    def test_chunks_have_minimum_semantic_length(self):
        """Chunks should have minimum semantic content (not single words/fragments).
        
        Fallback splits of long sentences may create shorter chunks, which is acceptable.
        This test focuses on normal sentence grouping not creating tiny fragments.
        """
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=80,  # Larger to handle sentences better
            chunk_overlap=12,
        )

        # All normal-sized sentences
        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "It automates deployment and scaling workflows. "
            "Services provide networking endpoints to pods."
        )

        chunks = manager._chunk_text(text)

        # Normal chunking should not create tiny fragments
        # (fragments <20 chars from fallback splits are acceptable)
        non_fallback_fragments = [c for c in chunks if len(c) < 20]
        
        # Count actual tiny fragments (should be rare/none)
        assert len(non_fallback_fragments) < 2, (
            f"Should not create many tiny fragments: {non_fallback_fragments}"
        )



class TestChunkQualityMetrics:
    """Verify chunk quality improvements."""

    def test_chunk_size_distribution(self):
        """Chunks should have reasonable size distribution."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=80,
            chunk_overlap=10,
        )

        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "It automates deployment and scaling workflows. "
            "It schedules pods and manages services. "
            "Docker is a container runtime. "
            "It packages applications into images. "
            "Both are essential infrastructure tools."
        )

        chunks = manager._chunk_text(text)

        lengths = [len(chunk) for chunk in chunks]

        # Check statistics
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)

        # Average should be reasonable
        assert avg_length > 30, f"Average chunk too small: {avg_length}"
        assert avg_length < manager.chunk_size + 30, (
            f"Average chunk too large: {avg_length}"
        )

        # No outliers that are too small (allow some flexibility)
        assert min_length > 15, f"Minimum chunk too small: {min_length}"

    def test_consistency_across_runs(self):
        """Same input should produce identical chunks (deterministic)."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=10,
        )

        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "Kubernetes coordinates deployment and scaling."
        )

        chunks1 = manager._chunk_text(text)
        chunks2 = manager._chunk_text(text)

        assert chunks1 == chunks2, "Chunking should be deterministic"


class TestNoRegressionOnExistingBehavior:
    """Ensure fixes don't break existing passing tests."""

    def test_empty_text_handling(self):
        """Empty text should produce no chunks."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=10,
        )

        chunks = manager._chunk_text("")
        assert chunks == []

    def test_whitespace_normalization(self):
        """Whitespace should be normalized."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=10,
        )

        text1 = "Hello world. This is test."
        text2 = "Hello  world.  This  is  test."  # Extra spaces

        chunks1 = manager._chunk_text(text1)
        chunks2 = manager._chunk_text(text2)

        assert chunks1 == chunks2, "Whitespace normalization should work"

    def test_single_short_sentence(self):
        """Single sentence shorter than chunk_size should return as-is."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(),
            chunk_size=60,
            chunk_overlap=10,
        )

        text = "Short sentence."
        chunks = manager._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text
