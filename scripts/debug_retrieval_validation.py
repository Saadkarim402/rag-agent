from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import IngestionConfig, RetrievalConfig
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
        "metadata": {"source_id": "kubernetes_guide", "category": "orchestration"},
    },
    {
        "id": "docker_tutorial",
        "text": (
            "Docker is a container runtime that packages applications into portable images. "
            "It simplifies local development, packaging, and deployment. "
            "Docker images share the host OS kernel while isolating runtime environments."
        ),
        "metadata": {"source_id": "docker_tutorial", "category": "runtime"},
    },
]

QUERY = "How does Kubernetes manage containers at scale?"
FILTERED_QUERY = "How do I use containers in Kubernetes?"
UNRELATED_QUERY = "What is quantum mechanics?"


def print_results(results: List[Any], label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"returned {len(results)} results")
    for index, item in enumerate(results, start=1):
        print(f"[{index}] chunk_id={item.chunk_id}, score={item.score:.4f}, source_id={item.metadata.get('source_id')}")
        print(f"      text={item.document_text[:120]}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Using temporary Chroma directory: %s", tmp_dir)

        ingestion_config = IngestionConfig(chunk_size=60, chunk_overlap=12, collection_name="validation_collection")
        retrieval_config = RetrievalConfig(top_k=5, min_score_threshold=0.65, collection_name="validation_collection")

        logger.info("[CONFIG] ingestion=%s", ingestion_config)
        logger.info("[CONFIG] retrieval=%s", retrieval_config)

        chroma = ChromaDBManager(persist_directory=tmp_dir)
        ingest_mgr = DocumentIngestionManager(chroma=chroma, config=ingestion_config)
        retrieval_mgr = RetrievalManager(chroma=chroma, embedding_manager=EmbeddingManager, config=retrieval_config)

        def print_chunk_debug(doc: Dict[str, Any]) -> None:
            print("\n=== CHUNKING DEBUG ===")
            print("Original document:")
            print(doc["text"])
            sentences = ingest_mgr._split_into_sentences(doc["text"])
            print(f"Detected sentences={len(sentences)}")
            for idx, sentence in enumerate(sentences):
                print(f"  sentence {idx}: {sentence}")
            chunks = ingest_mgr._chunk_text(doc["text"])
            print(f"Generated chunks={len(chunks)}")
            for idx, chunk in enumerate(chunks):
                print(f"\n## Chunk {idx}:\n{chunk}")
                print(f"length={len(chunk)}")
                if idx + 1 < len(chunks):
                    overlap = set(chunks[idx].split()) & set(chunks[idx + 1].split())
                    print(f"overlap preview={sorted(list(overlap))[:10]}")

        print("\n=== INGESTION ===")
        for doc in SAMPLE_DOCS:
            print_chunk_debug(doc)
            ids = ingest_mgr.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])
            print(f"Ingested {doc['id']} with {len(ids)} chunk(s)")
            print("  chunk ids:", ids)

        info = chroma.get_collection_info(ingestion_config.collection_name)
        print("Collection info:", info)

        results = retrieval_mgr.retrieve(query=QUERY)
        print_results(results, "RETRIEVAL")

        filtered_results = retrieval_mgr.retrieve(query=FILTERED_QUERY, metadata_filter={"source_id": "kubernetes_guide"})
        print_results(filtered_results, "FILTERED RETRIEVAL")

        threshold_results = retrieval_mgr.retrieve(query=UNRELATED_QUERY, min_score_threshold=0.65)
        print_results(threshold_results, "THRESHOLD RETRIEVAL")

        print("\n=== DONE ===")


if __name__ == "__main__":
    main()
