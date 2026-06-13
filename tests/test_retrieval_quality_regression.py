"""Retrieval quality regression tests - verify chunking improvements don't hurt retrieval."""

import pytest
from app.ingestion.ingest import DocumentIngestionManager
from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager
from app.retrieval.retriever import RetrievalManager
from app.retrieval.pipeline import RetrievalPipeline
from app.config import IngestionConfig, RetrievalConfig
import tempfile


class TestRetrievalQualityWithSentenceAwareChunking:
    """Verify that sentence-aware chunking maintains or improves retrieval quality."""

    @pytest.fixture
    def setup_pipeline(self):
        """Set up test pipeline with sentence-aware chunking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use temp directory for ChromaDB
            chroma = ChromaDBManager(persist_directory=temp_dir)
            ingestion_config = IngestionConfig(
                chunk_size=80,
                chunk_overlap=15,
            )
            retrieval_config = RetrievalConfig(
                top_k=5,
                min_score_threshold=0.65,
            )

            manager = DocumentIngestionManager(
                chroma=chroma,
                embedding_manager=EmbeddingManager,
                config=ingestion_config,
                collection_name="test_retrieval",
            )

            retrieval_mgr = RetrievalManager(
                chroma=chroma,
                config=retrieval_config,
                collection_name="test_retrieval",
            )

            yield manager, retrieval_mgr, chroma

    def test_related_query_retrieves_relevant_chunks(self, setup_pipeline):
        """Related queries should return high-scoring results."""
        manager, retrieval_mgr, _ = setup_pipeline

        # Ingest test documents
        docs = [
            {
                "id": "k8s_guide",
                "text": (
                    "Kubernetes is a container orchestration platform. "
                    "It helps manage containers at scale. "
                    "Kubernetes coordinates container deployment, scaling, and networking across clusters. "
                    "It schedules pods, manages services, and keeps workloads available even when nodes fail."
                ),
                "metadata": {
                    "title": "Kubernetes Guide",
                    "category": "orchestration",
                },
            },
            {
                "id": "docker_guide",
                "text": (
                    "Docker is a container runtime that packages applications into portable images. "
                    "It simplifies local development, packaging, and deployment. "
                    "Docker images share the host OS kernel while isolating runtime environments."
                ),
                "metadata": {
                    "title": "Docker Guide",
                    "category": "runtime",
                },
            },
        ]

        for doc in docs:
            manager.ingest_text(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc["metadata"],
            )

        # Query for Kubernetes-related content
        query = "How does Kubernetes manage containers?"
        results = retrieval_mgr.retrieve(query=query)

        # Should return Kubernetes document as top result
        assert len(results) > 0, "Query should return results"

        top_result = results[0]
        assert top_result["source_id"] == "k8s_guide", (
            f"Top result should be from Kubernetes document, got {top_result['source_id']}"
        )

        # Top result should have good score
        assert top_result["score"] > 0.70, (
            f"Top result score should be high, got {top_result['score']}"
        )

    def test_unrelated_query_filtered_by_threshold(self, setup_pipeline):
        """Unrelated queries should be filtered by score threshold."""
        manager, retrieval_mgr, _ = setup_pipeline

        # Ingest test document
        manager.ingest_text(
            doc_id="containers",
            text=(
                "Kubernetes is a container orchestration platform. "
                "It helps manage containers at scale."
            ),
            metadata={"title": "Container Basics"},
        )

        # Query with unrelated content
        query = "What is quantum mechanics?"
        results = retrieval_mgr.retrieve(query=query)

        # Should return no results or very low scoring
        if results:
            for result in results:
                assert result["score"] < 0.65, (
                    f"Unrelated query should not score above threshold, got {result['score']}"
                )
        else:
            # No results is also acceptable for unrelated queries
            assert len(results) == 0

    def test_chunk_quality_improves_disambiguation(self, setup_pipeline):
        """Better chunks should improve query disambiguation."""
        manager, retrieval_mgr, _ = setup_pipeline

        # Ingest documents with distinct content
        manager.ingest_text(
            doc_id="k8s",
            text=(
                "Kubernetes orchestration manages container deployment. "
                "Pods are the smallest deployable units. "
                "Services provide stable endpoints for pod access."
            ),
            metadata={"title": "Kubernetes", "service": "orchestration"},
        )

        manager.ingest_text(
            doc_id="docker",
            text=(
                "Docker is a containerization tool. "
                "Images are pre-built containers. "
                "Containers run isolated application environments."
            ),
            metadata={"title": "Docker", "service": "runtime"},
        )

        # Query specific to Kubernetes
        query = "How do pods work?"
        results = retrieval_mgr.retrieve(query=query)

        # Should prioritize Kubernetes results
        top_5_sources = [r["source_id"] for r in results[:3]]
        k8s_count = sum(1 for s in top_5_sources if s == "k8s")

        assert k8s_count > 0, "Kubernetes document should be in top results for pod query"

    def test_long_text_chunking_preserves_meaning(self, setup_pipeline):
        """Chunking long text should preserve semantic meaning."""
        manager, retrieval_mgr, _ = setup_pipeline

        long_text = (
            "Kubernetes is a production-grade container orchestration platform. "
            "It automates many of the manual processes involved in deploying, managing, and scaling containerized applications. "
            "Kubernetes lets you treat groups of machines as a single unit. "
            "It handles scheduling, load balancing, and replication. "
            "The system is designed to give you the benefits of Platform as a Service (PaaS), "
            "Infrastructure as a Service (IaaS), and the flexibility of running applications on your own infrastructure."
        )

        manager.ingest_text(
            doc_id="k8s_long",
            text=long_text,
            metadata={"title": "Kubernetes Overview"},
        )

        # Query about specific capability
        query = "What does Kubernetes do with load balancing?"
        results = retrieval_mgr.retrieve(query=query)

        # Should find relevant chunk
        assert len(results) > 0, "Should find chunks about load balancing"

        # Results should score reasonably (not perfect but relevant)
        top_score = results[0]["score"] if results else 0
        assert top_score > 0.60, (
            f"Long text chunking should preserve retrievability, got score {top_score}"
        )

    def test_overlap_improves_boundary_queries(self, setup_pipeline):
        """Overlap should help queries near chunk boundaries."""
        manager, retrieval_mgr, _ = setup_pipeline

        text = (
            "Deployment specifies desired state. "
            "Kubernetes maintains this state continuously. "
            "ReplicaSets ensure correct number of replicas. "
            "Services expose applications to network traffic."
        )

        manager.ingest_text(
            doc_id="deploy",
            text=text,
            metadata={"title": "Kubernetes Deployment"},
        )

        # Query that might span chunk boundary
        query = "How does Kubernetes maintain state with replicas?"
        results = retrieval_mgr.retrieve(query=query)

        assert len(results) > 0, "Boundary-spanning query should still retrieve relevant chunks"

    def test_metadata_preserved_through_chunking(self, setup_pipeline):
        """Metadata should be properly propagated to all chunks."""
        manager, retrieval_mgr, _ = setup_pipeline

        manager.ingest_text(
            doc_id="test_meta",
            text=(
                "First sentence of document. "
                "Second sentence of document. "
                "Third sentence of document. "
                "Fourth sentence of document."
            ),
            metadata={
                "title": "Test Document",
                "category": "test",
                "author": "tester",
            },
        )

        query = "sentence of document"
        results = retrieval_mgr.retrieve(query=query)

        # All results should have metadata
        for result in results:
            assert "metadata" in result or "source_id" in result, "Result should have metadata"
            if isinstance(result.get("metadata"), dict):
                assert result["metadata"].get("title") == "Test Document", (
                    "Metadata should be preserved"
                )


class TestChunkingConsistencyMetrics:
    """Verify chunking consistency and quality improvements."""

    def test_chunk_count_reasonable(self):
        """Chunk count should be reasonable relative to text size."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(
                chunk_size=60,
                chunk_overlap=10,
            ),
        )

        # For 261 chars with 60 char chunks, we'd expect roughly 4-5 chunks
        text = (
            "Kubernetes is a container orchestration platform. "
            "It helps manage containers at scale. "
            "Kubernetes coordinates container deployment, scaling, and networking across clusters. "
            "It schedules pods, manages services, and keeps workloads available even when nodes fail."
        )

        chunks = manager._chunk_text(text)

        # Rough calculation: text_len / chunk_size should give us a ballpark
        min_expected = len(text) // (manager.chunk_size + manager.chunk_overlap) + 1
        max_expected = len(text) // (manager.chunk_size // 2)

        assert len(chunks) >= 2, "Should create at least 2 chunks"
        assert len(chunks) <= max_expected, f"Too many chunks created: {len(chunks)}"

    def test_no_identical_consecutive_chunks(self):
        """No two consecutive chunks should be identical."""
        manager = DocumentIngestionManager(
            config=IngestionConfig(
                chunk_size=60,
                chunk_overlap=10,
            ),
        )

        text = (
            "Sentence one. "
            "Sentence two. "
            "Sentence three. "
            "Sentence four. "
            "Sentence five. "
            "Sentence six."
        )

        chunks = manager._chunk_text(text)

        for i in range(len(chunks) - 1):
            assert chunks[i] != chunks[i + 1], (
                f"Consecutive chunks should not be identical: {chunks[i]}"
            )
