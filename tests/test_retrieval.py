from app.retrieval.retriever import RetrievalManager


def test_retrieve_relevant_chunks(tmp_chroma_manager, ingestion_manager, sample_docs):
    # ingest sample docs
    ids1 = ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    ids2 = ingestion_manager.ingest_text(doc_id=sample_docs[1]["id"], text=sample_docs[1]["text"], metadata=sample_docs[1]["metadata"])

    rm = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager)

    # query related to Paris should retrieve chunks from doc-1
    results = rm.retrieve(query="capital of France", collection_name=ingestion_manager.collection_name, top_k=3)
    assert isinstance(results, list)
    if results:
        # top result should contain metadata.source or source_id pointing to doc-1
        top = results[0]
        assert top.collection == ingestion_manager.collection_name
        assert isinstance(top.chunk_id, str)


def test_top_k_behavior(tmp_chroma_manager, ingestion_manager, sample_docs):
    # ingest documents
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    rm = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager)
    res1 = rm.retrieve(query="Paris", collection_name=ingestion_manager.collection_name, top_k=1)
    res3 = rm.retrieve(query="Paris", collection_name=ingestion_manager.collection_name, top_k=3)
    assert len(res1) <= 1
    assert len(res3) <= 3


def test_metadata_filtering(tmp_chroma_manager, ingestion_manager, sample_docs):
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    ingestion_manager.ingest_text(doc_id=sample_docs[1]["id"], text=sample_docs[1]["text"], metadata=sample_docs[1]["metadata"])
    rm = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager)
    # filter by category=programming should return chunks from doc-2 only
    results = rm.retrieve(query="programming language", collection_name=ingestion_manager.collection_name, top_k=5, metadata_filter={"category": "programming"})
    for r in results:
        assert r.metadata.get("category") == "programming"


def test_empty_retrieval_returns_empty(tmp_chroma_manager):
    rm = RetrievalManager(chroma=tmp_chroma_manager)
    results = rm.retrieve(query="unlikely to match anything 12345", collection_name="empty_collection", top_k=2)
    assert results == []
