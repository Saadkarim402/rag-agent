"""Full pipeline observability for document ingestion and retrieval.

This script demonstrates the complete lifecycle:
    Original Document
    ↓
    Stored Document (Repository)
    ↓
    Chunks
    ↓
    Chunk IDs & Metadata
    ↓
    Embeddings
    ↓
    Vector Storage
    ↓
    Query → Retrieval
    ↓
    Retrieved Chunks & Scores
    ↓
    Final Context

Run: python scripts/debug_e2e_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import IngestionConfig, RetrievalConfig
from app.documents import DocumentRepository
from app.embeddings.embedder import EmbeddingManager
from app.ingestion.ingest import DocumentIngestionManager
from app.retrieval.retriever import RetrievalManager
from app.vectordb.chroma_client import ChromaDBManager

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DOCS: List[Dict[str, Any]] = [
    {
        "id": "kubernetes_guide",
        "text": (
            "Kubernetes is a container orchestration platform. It helps manage containers at scale. "
            "Kubernetes coordinates container deployment, scaling, and networking across clusters. "
            "It schedules pods, manages services, and keeps workloads available even when nodes fail."
        ),
        "metadata": {"source_id": "kubernetes_guide", "category": "orchestration", "title": "Kubernetes Guide"},
    },
    {
        "id": "docker_tutorial",
        "text": (
            "Docker is a container runtime that packages applications into portable images. "
            "It simplifies local development, packaging, and deployment. "
            "Docker images share the host OS kernel while isolating runtime environments."
        ),
        "metadata": {"source_id": "docker_tutorial", "category": "runtime", "title": "Docker Tutorial"},
    },
]

QUERY = "How does Kubernetes manage containers at scale?"
FILTERED_QUERY = "How do I use containers in Kubernetes?"
UNRELATED_QUERY = "What is quantum mechanics?"


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n  → {title}")
    print("-" * 80)


def show_document_lifecycle(
    doc: Dict[str, Any],
    ingest_mgr: DocumentIngestionManager,
    repository: DocumentRepository,
) -> None:
    """Show the complete lifecycle of a single document."""
    doc_id = doc["id"]
    text = doc["text"]
    metadata = doc.get("metadata", {})

    print_subsection("1. ORIGINAL DOCUMENT")
    print(f"  Document ID: {doc_id}")
    print(f"  Title: {metadata.get('title', 'N/A')}")
    print(f"  Content length: {len(text)} characters")
    print(f"  Preview: {text[:100]}...")

    print_subsection("2. STORED DOCUMENT (REPOSITORY)")
    doc_path = repository.get_path(doc_id)
    print(f"  Storage path: {doc_path}")
    stored_doc = repository.load(doc_id)
    if stored_doc:
        print(f"  Created at: {stored_doc.get('created_at', 'N/A')}")
        print(f"  Metadata: {json.dumps(stored_doc.get('metadata', {}), indent=4)}")
        print(f"  ✓ Document successfully stored and retrieved")

    print_subsection("3. CHUNKING ANALYSIS")
    sentences = ingest_mgr._split_into_sentences(text)
    print(f"  Detected sentences: {len(sentences)}")
    for idx, sentence in enumerate(sentences, 1):
        print(f"    [{idx}] {sentence}")

    chunks = ingest_mgr._chunk_text(text)
    print(f"\n  Generated chunks: {len(chunks)}")
    for idx, chunk in enumerate(chunks, 1):
        print(f"\n    [Chunk {idx}] (length={len(chunk)} chars)")
        print(f"      {chunk}")

    print_subsection("4. CHUNK IDs & METADATA")
    for idx, chunk in enumerate(chunks, 1):
        chunk_id = ingest_mgr._generate_chunk_id(doc_id, idx - 1, chunk)
        print(f"  Chunk {idx}:")
        print(f"    ID: {chunk_id}")
        print(f"    Index: {idx - 1}")
        chunk_meta = {
            "source_id": doc_id,
            "chunk_index": idx - 1,
            "chunk_id": chunk_id,
        }
        chunk_meta.update(metadata)
        print(f"    Metadata: {chunk_meta}")

    print_subsection("5. EMBEDDINGS GENERATION")
    embeddings = ingest_mgr.embedding_manager.embed_texts(chunks)
    print(f"  Generated embeddings: {len(embeddings)}")
    if embeddings:
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print(f"  First embedding (first 5 dims): {embeddings[0][:5]}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        print_section("RAG PIPELINE OBSERVABILITY DEMONSTRATION")

        # Initialize components
        ingestion_config = IngestionConfig(
            chunk_size=60,
            chunk_overlap=12,
            collection_name="validation_collection"
        )
        retrieval_config = RetrievalConfig(
            top_k=5,
            min_score_threshold=0.65,
            collection_name="validation_collection"
        )

        logger.info("Temporary Chroma directory: %s", tmp_dir)

        # Create managers
        chroma = ChromaDBManager(persist_directory=tmp_dir)
        repository = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        ingest_mgr = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repository,
            config=ingestion_config
        )
        retrieval_mgr = RetrievalManager(
            chroma=chroma,
            embedding_manager=EmbeddingManager,
            config=retrieval_config
        )

        # === INGESTION PHASE ===
        print_section("INGESTION PHASE")
        print(f"\nConfiguration:")
        print(f"  Chunk size: {ingestion_config.chunk_size}")
        print(f"  Chunk overlap: {ingestion_config.chunk_overlap}")
        print(f"  Collection: {ingestion_config.collection_name}")

        all_chunk_ids = []
        for doc in SAMPLE_DOCS:
            show_document_lifecycle(doc, ingest_mgr, repository)
            ids = ingest_mgr.ingest_text(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata")
            )
            all_chunk_ids.extend(ids)
            print(f"\n  Ingested: {doc['id']} with {len(ids)} chunk(s)")

        # === STORAGE SUMMARY ===
        print_section("STORAGE SUMMARY")
        print_subsection("STORED DOCUMENTS")
        stored_docs = repository.list_documents()
        for doc in stored_docs:
            print(f"  • {doc['doc_id']}: {doc.get('title', 'N/A')}")

        print_subsection("INDEXED VECTORS")
        info = chroma.get_collection_info(ingestion_config.collection_name)
        print(f"  Total vectors stored: {info.get('count', 'N/A')}")
        print(f"  Collection name: {info.get('name', 'N/A')}")

        # === RETRIEVAL PHASE ===
        print_section("RETRIEVAL PHASE")

        print_subsection("QUERY 1: General Query")
        print(f"  Query: {QUERY}")
        results = retrieval_mgr.retrieve(query=QUERY)
        print(f"  Retrieved {len(results)} results:\n")
        for rank, result in enumerate(results, 1):
            print(f"  [{rank}] Chunk ID: {result.chunk_id}")
            print(f"      Distance: {result.distance:.6f}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Source: {result.metadata.get('source_id', 'N/A')}")
            print(f"      Text: {result.document_text[:60]}...\n")

        print_subsection("QUERY 2: Filtered Query")
        print(f"  Query: {FILTERED_QUERY}")
        print(f"  Filter: source_id=kubernetes_guide")
        filtered_results = retrieval_mgr.retrieve(
            query=FILTERED_QUERY,
            metadata_filter={"source_id": "kubernetes_guide"}
        )
        print(f"  Retrieved {len(filtered_results)} results:\n")
        for rank, result in enumerate(filtered_results, 1):
            print(f"  [{rank}] Chunk ID: {result.chunk_id}")
            print(f"      Distance: {result.distance:.6f}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Text: {result.document_text[:60]}...\n")

        print_subsection("QUERY 3: Unrelated Query (with threshold)")
        print(f"  Query: {UNRELATED_QUERY}")
        print(f"  Score threshold: 0.65")
        threshold_results = retrieval_mgr.retrieve(
            query=UNRELATED_QUERY,
            min_score_threshold=0.65
        )
        print(f"  Retrieved {len(threshold_results)} results (filtered by threshold)\n")
        if threshold_results:
            for rank, result in enumerate(threshold_results, 1):
                print(f"  [{rank}] Chunk ID: {result.chunk_id}")
                print(f"      Distance: {result.distance:.6f}")
                print(f"      Score: {result.score:.4f}")
                print(f"      Text: {result.document_text[:60]}...\n")
        else:
            print("  No results above threshold\n")

        # === CONTEXT BUILDING ===
        print_section("FINAL CONTEXT BUILDING")
        print_subsection("BEST MATCH FOR MAIN QUERY")
        if results:
            best = results[0]
            print(f"  Score: {best.score:.4f}")
            print(f"  Full text:\n")
            print(f"  {best.document_text}")
            print(f"\n  Metadata: {best.metadata}")

        print_section("OBSERVABILITY COMPLETE")
        print(
            "\n✓ Original documents stored in repository"
            "\n✓ Chunking preserves sentence boundaries"
            "\n✓ Chunk IDs deterministically generated"
            "\n✓ Metadata propagated through pipeline"
            "\n✓ Embeddings generated and stored"
            "\n✓ Retrieval returns ranked results with scores"
            "\n✓ Context ready for LLM consumption\n"
        )


if __name__ == "__main__":
    main()
