import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIRECTORY))

from app.ingestion.ingest import DocumentIngestionManager
from app.vectordb.chroma_client import ChromaDBManager


def main() -> None:
    collection_name = "test_ingestion_collection"
    sample_doc = {
        "id": "sample-doc-001",
        "text": (
            "OpenAI released ChatGPT in 2022, which popularized conversational AI. "
            "This sample document is used to verify ingestion, chunking, embedding, and storage." 
            "The pipeline should create deterministic chunks with metadata and persist them into ChromaDB."
        ),
        "metadata": {
            "source": "verification_script",
            "category": "test",
        },
    }

    print("=== Ingestion verification script ===")
    print("Creating ingestion manager...")

    ingestion_manager = DocumentIngestionManager(collection_name=collection_name)
    print("Ingesting sample document...")
    chunk_ids = ingestion_manager.ingest_text(
        doc_id=sample_doc["id"],
        text=sample_doc["text"],
        metadata=sample_doc["metadata"],
    )

    print(f"Generated {len(chunk_ids)} chunk IDs:")
    for chunk_id in chunk_ids:
        print(f"  - {chunk_id}")

    print("\nChecking collection info from a new manager instance...")
    chroma_manager = ChromaDBManager()
    info = chroma_manager.get_collection_info(collection_name)
    print(f"Collection name: {info['name']}")
    print(f"Stored vector count: {info['count']}")

    print("\nRunning a quick debug query using the original sample text...")
    if len(chunk_ids) == 0:
        print("No chunks created; aborting debug query.")
        return

    query_embedding = ingestion_manager.embedding_manager.embed_text(sample_doc["text"])
    query_results = chroma_manager.query_embeddings(
        collection_name=collection_name,
        query_embeddings=[query_embedding],
        n_results=1,
        include_metadata=True,
    )

    retrieved_ids = query_results.get("ids", [[]])[0]
    retrieved_documents = query_results.get("documents", [[]])[0]
    retrieved_metadatas = query_results.get("metadatas", [[]])[0]

    print("\nSample stored record:")
    if retrieved_ids:
        print(f"Chunk id: {retrieved_ids[0]}")
        print(f"Chunk text: {retrieved_documents[0]}")
        print(f"Metadata: {retrieved_metadatas[0]}")
    else:
        print("No results returned from the debug query.")

    print("\nIngestion verification complete.")


if __name__ == "__main__":
    main()
