from app.retrieval.pipeline import MockReranker, RetrievalPipeline
from app.retrieval.retriever import RetrievalManager, RetrievalResult


def test_pipeline_retrieval_flow(tmp_chroma_manager, ingestion_manager, sample_docs):
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    ingestion_manager.ingest_text(doc_id=sample_docs[1]["id"], text=sample_docs[1]["text"], metadata=sample_docs[1]["metadata"])

    retriever = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager)
    pipeline = RetrievalPipeline(retriever=retriever)

    pipeline_results = pipeline.run(query="capital of France", collection_name=ingestion_manager.collection_name, top_k=3)
    direct_results = retriever.retrieve(query="capital of France", collection_name=ingestion_manager.collection_name, top_k=3)

    assert isinstance(pipeline_results, list)
    assert pipeline_results == direct_results
    assert all(isinstance(result, RetrievalResult) for result in pipeline_results)


def test_pipeline_with_reverse_reranker(tmp_chroma_manager, ingestion_manager, sample_docs):
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    ingestion_manager.ingest_text(doc_id=sample_docs[1]["id"], text=sample_docs[1]["text"], metadata=sample_docs[1]["metadata"])

    retriever = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager)
    reranker = MockReranker(mode="reverse")
    pipeline = RetrievalPipeline(retriever=retriever, reranker=reranker)

    direct_results = retriever.retrieve(query="Paris", collection_name=ingestion_manager.collection_name, top_k=4)
    pipeline_results = pipeline.run(query="Paris", collection_name=ingestion_manager.collection_name, top_k=4)

    assert pipeline_results == list(reversed(direct_results))


def test_pipeline_deterministic_ordering(tmp_chroma_manager, ingestion_manager, sample_docs):
    ingestion_manager.ingest_text(doc_id=sample_docs[0]["id"], text=sample_docs[0]["text"], metadata=sample_docs[0]["metadata"])
    retriever = RetrievalManager(chroma=tmp_chroma_manager, embedding_manager=ingestion_manager.embedding_manager)
    pipeline = RetrievalPipeline(retriever=retriever)

    first_results = pipeline.run(query="Paris", collection_name=ingestion_manager.collection_name, top_k=3)
    second_results = pipeline.run(query="Paris", collection_name=ingestion_manager.collection_name, top_k=3)

    assert first_results == second_results


def test_pipeline_empty_retrieval_returns_empty(tmp_chroma_manager):
    retriever = RetrievalManager(chroma=tmp_chroma_manager)
    pipeline = RetrievalPipeline(retriever=retriever)

    results = pipeline.run(query="unlikely to match anything 12345", collection_name="empty_collection", top_k=2)
    assert results == []
