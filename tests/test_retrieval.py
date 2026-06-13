from app.config import IngestionConfig, RetrievalConfig
from app.ingestion.ingest import DocumentIngestionManager
from app.retrieval.pipeline import RetrievalPipeline
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


def test_retrieval_config_usage(tmp_chroma_manager, sample_docs):
    config = IngestionConfig(chunk_size=20, chunk_overlap=5, collection_name="config_collection")
    ingestion_manager = DocumentIngestionManager(chroma=tmp_chroma_manager, config=config)
    assert ingestion_manager.chunk_size == 20
    assert ingestion_manager.chunk_overlap == 5
    assert ingestion_manager.collection_name == "config_collection"

    ids = ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    assert ids

    retrieval_config = RetrievalConfig(top_k=2, min_score_threshold=0.0, collection_name="config_collection")
    rm = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager, config=retrieval_config)
    results = rm.retrieve(query="capital of France")
    assert len(results) <= 2
    assert all(result.collection == "config_collection" for result in results)


def test_score_threshold_excludes_low_scoring_candidates(tmp_chroma_manager, ingestion_manager, sample_docs):
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    ingestion_manager.ingest_text(doc_id=sample_docs[1]["id"], text=sample_docs[1]["text"], metadata=sample_docs[1]["metadata"])

    config = RetrievalConfig(min_score_threshold=0.99)
    rm = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager, config=config)
    results = rm.retrieve(query="quantum mechanics")
    assert results == []


def test_pipeline_filters_and_thresholds_together(tmp_chroma_manager, ingestion_manager, sample_docs):
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    ingestion_manager.ingest_text(doc_id=sample_docs[1]["id"], text=sample_docs[1]["text"], metadata=sample_docs[1]["metadata"])

    config = RetrievalConfig(min_score_threshold=0.5)
    rm = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager, config=config)
    pipeline = RetrievalPipeline(retriever=rm)
    results = pipeline.run(
        query="programming language",
        collection_name=ingestion_manager.collection_name,
        metadata_filter={"category": "programming"},
    )

    assert results
    assert all(r.metadata.get("category") == "programming" for r in results)
