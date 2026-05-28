from app.ingestion.ingest import DocumentIngestionManager


def test_chunking_deterministic():
    manager = DocumentIngestionManager(chunk_size=20, chunk_overlap=5, collection_name="c")
    text = "This   is   a    test    of   chunking    behavior.  Repeat words repeat."
    a = manager._chunk_text(text)
    b = manager._chunk_text(text)
    assert a == b


def test_chunking_overlap_and_size():
    manager = DocumentIngestionManager(chunk_size=20, chunk_overlap=5, collection_name="c")
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = manager._chunk_text(text)
    assert chunks, "Expected at least one chunk"
    # each chunk length <= chunk_size
    for c in chunks:
        assert len(c) <= manager.chunk_size
    # overlap: adjacent chunks should share some words when overlap>0
    if len(chunks) > 1:
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        assert first_words & second_words, "Expected overlap between adjacent chunks"


def test_chunking_edge_cases_empty_and_small():
    manager = DocumentIngestionManager(chunk_size=50, chunk_overlap=10, collection_name="c")
    assert manager._chunk_text("") == []
    small = "tiny"
    chunks = manager._chunk_text(small)
    assert len(chunks) == 1
    assert chunks[0] == "tiny"


def test_chunking_repeated_whitespace_normalization():
    manager = DocumentIngestionManager(chunk_size=50, chunk_overlap=5, collection_name="c")
    text = "   lots\n\n of   whitespace\t and   newlines   "
    chunks = manager._chunk_text(text)
    assert "  " not in chunks[0]
