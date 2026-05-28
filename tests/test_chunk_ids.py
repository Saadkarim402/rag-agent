from app.ingestion.ingest import DocumentIngestionManager


def test_chunk_id_stability():
    manager = DocumentIngestionManager(chunk_size=50, chunk_overlap=10, collection_name="c")
    text = "stable chunk content for id test"
    chunks = manager._chunk_text(text)
    ids1 = [manager._generate_chunk_id("docA", i, c) for i, c in enumerate(chunks)]
    ids2 = [manager._generate_chunk_id("docA", i, c) for i, c in enumerate(chunks)]
    assert ids1 == ids2


def test_chunk_id_changes_with_content_or_doc():
    manager = DocumentIngestionManager(chunk_size=50, chunk_overlap=10, collection_name="c")
    chunks = manager._chunk_text("a b c d e f g")
    id_orig = manager._generate_chunk_id("doc1", 0, chunks[0])
    id_changed_content = manager._generate_chunk_id("doc1", 0, chunks[0] + " x")
    id_changed_doc = manager._generate_chunk_id("doc2", 0, chunks[0])
    assert id_orig != id_changed_content
    assert id_orig != id_changed_doc
