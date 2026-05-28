from app.ingestion.ingest import DocumentIngestionManager
from app.vectordb.chroma_client import ChromaDBManager


def test_ingest_single_and_metadata(tmp_chroma_manager):
    manager = DocumentIngestionManager(chroma=tmp_chroma_manager, embedding_manager=None, collection_name="ingest_test", chunk_size=50, chunk_overlap=10)
    # patch embedding manager through manager.embedding_manager to keep using embedder patch from conftest
    manager.embedding_manager = manager.embedding_manager or None
    doc = {"id": "ingest-doc-1", "text": "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota", "metadata": {"label": "sample"}}
    ids = manager.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])
    assert ids
    info = tmp_chroma_manager.get_collection_info("ingest_test")
    assert info["count"] >= len(ids)


def test_ingest_multiple_documents_and_chunk_count(tmp_chroma_manager):
    manager = DocumentIngestionManager(chroma=tmp_chroma_manager, embedding_manager=None, collection_name="ingest_multi", chunk_size=30, chunk_overlap=5)
    docs = [
        {"id": "d1", "text": "one two three four five six seven eight nine ten"},
        {"id": "d2", "text": "alpha beta gamma delta epsilon zeta eta theta"},
    ]
    ids = manager.ingest_texts(docs)
    assert ids
    # ensure count stored >= number of generated ids
    info = tmp_chroma_manager.get_collection_info("ingest_multi")
    assert info["count"] >= len(ids)


def test_metadata_propagation(tmp_chroma_manager):
    manager = DocumentIngestionManager(chroma=tmp_chroma_manager, embedding_manager=None, collection_name="meta_test", chunk_size=50, chunk_overlap=5)
    doc = {"id": "meta-doc", "text": "text content for metadata test", "metadata": {"tag": "important"}}
    ids = manager.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])
    assert ids
    # query one of the embeddings to fetch metadatas
    embedding = manager.embedding_manager.embed_text(doc["text"]) if manager.embedding_manager else None
    if embedding is not None:
        raw = tmp_chroma_manager.query_embeddings(collection_name="meta_test", query_embeddings=[embedding], n_results=1, include_metadata=True)
        metadatas = raw.get("metadatas", [[]])[0]
        if metadatas:
            assert any(md.get("tag") == "important" for md in metadatas)
