from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pprint import pprint
from pathlib import Path

from app.ingestion.ingest import DocumentIngestionManager
from app.vectordb.chroma_client import ChromaDBManager
from app.retrieval.retriever import RetrievalManager
from app.retrieval.pipeline import RetrievalPipeline, MockReranker
from app.embeddings.embedder import EmbeddingManager


def _summarize(obj):
    """Return a short, human-readable summary for common objects."""
    try:
        if obj is None:
            return "None"
        if isinstance(obj, (list, tuple)):
            return f"list(len={len(obj)})"
        if isinstance(obj, dict):
            return f"dict(len={len(obj)})"
        return f"{type(obj).__name__}"
    except Exception:
        return str(type(obj))


def call_and_print(name: str, fn, *args, summarize_result: bool = True, **kwargs):
    """Call `fn(*args, **kwargs)`, printing a concise pre/post summary."""
    print(f"\nCALL -> {name}")
    for i, a in enumerate(args):
        s = repr(a)
        print(f"  arg[{i}]: {s[:200]}")
    for k, v in kwargs.items():
        sv = repr(v)
        print(f"  kw[{k}]: {sv[:200]}")
    res = fn(*args, **kwargs)
    if summarize_result:
        print(f"RETURN <- {name}: {_summarize(res)}")
    else:
        print(f"RETURN <- {name}: {type(res).__name__}")
    return res


# ---- CONFIG ----
PERSIST_DIR = "./chroma_db_debug"
COL = "debug_collection"
DOC_ID_1 = "debug-doc-1"
TEXT_1 = "Kubernetes is a container orchestration platform. It helps manage containers at scale. " \
         "Kubernetes coordinates container deployment, scaling, and networking across clusters. " \
         "It schedules pods, manages services, and keeps workloads available even when nodes fail. " \
         "Developers use it to automate rollouts, monitor health, and manage storage for containerized apps."
DOC_ID_2 = "debug-doc-2"
TEXT_2 = "Docker is a platform that packages applications into containers. It simplifies local development, packaging, and deployment. " \
         "Containers share the host OS kernel while isolating runtime environments. " \
         "Docker images are built from layers, and registries store those images for reuse."
QUERY = "How does Kubernetes manage containers at scale?"
TOP_K = 5

# ---- SETUP ----
cm = ChromaDBManager(persist_directory=PERSIST_DIR)
ingest = DocumentIngestionManager(chroma=cm, collection_name=COL, chunk_size=50, chunk_overlap=10)
emb = EmbeddingManager

print("\n=== Normalization and Chunking ===")
doc_chunks = []
for doc_id, text in [(DOC_ID_1, TEXT_1), (DOC_ID_2, TEXT_2)]:
    print(f"\n--- Document {doc_id} ---")
    norm = call_and_print("DocumentIngestionManager._normalize_text", ingest._normalize_text, text)
    print("Normalized text:", norm)
    chunks = call_and_print("DocumentIngestionManager._chunk_text", ingest._chunk_text, text)
    print(f"Chunks (count={len(chunks)}):")
    for i, c in enumerate(chunks):
        print(f"  [{i}] len={len(c)}: {c[:120]}")
        overlap = call_and_print("DocumentIngestionManager._get_overlap_words", ingest._get_overlap_words, c.split())
        print("    overlap-preview:", overlap)
    doc_chunks.append((doc_id, chunks))

print("\n=== Chunk IDs ===")
ids = []
documents = []
metadatas = []
for doc_id, chunks in doc_chunks:
    for i, c in enumerate(chunks):
        cid = call_and_print("DocumentIngestionManager._generate_chunk_id", ingest._generate_chunk_id, doc_id, i, c)
        ids.append(cid)
        documents.append(c)
        metadatas.append({"source_id": doc_id, "chunk_index": i})
pprint(ids)

print("\n=== Embeddings (all chunks) ===")
if documents:
    emb_vecs = call_and_print("EmbeddingManager.embed_texts", emb.embed_texts, documents)
    # emb_vecs is a list of vectors; print a short sample
    print("Embedding vector length:", len(emb_vecs[0]))
    print("First vector sample (first 10 values):", emb_vecs[0][:10])

print("\n=== Store chunks to Chroma ===")
call_and_print("ChromaDBManager.add_documents", cm.add_documents, COL, ids, documents, emb_vecs, metadatas, summarize_result=False)
info = call_and_print("ChromaDBManager.get_collection_info", cm.get_collection_info, COL)
print("Collection info:", info)

print("\n=== Raw vector-store query (showing raw response) ===")
query_vec = call_and_print("EmbeddingManager.embed_text", emb.embed_text, QUERY)
raw = call_and_print("ChromaDBManager.query_embeddings", cm.query_embeddings, COL, [query_vec], TOP_K, True)
pprint(raw)

print("\n=== RetrievalManager.retrieve() results ===")
rm = RetrievalManager(chroma=cm, embedding_manager=emb)
results = call_and_print("RetrievalManager.retrieve", rm.retrieve, QUERY, COL, TOP_K)
for r in results:
    print("---")
    print("chunk_id:", r.chunk_id)
    print("distance:", r.distance)
    print("score:", r.score)
    print("metadata:", r.metadata)
    print("text preview:", r.document_text[:160])

print("\n=== RetrievalPipeline (no reranker) ===")
pipeline = RetrievalPipeline(retriever=rm)
pipe_results = call_and_print("RetrievalPipeline.run", pipeline.run, QUERY, COL, TOP_K)
print("pipeline returned", len(pipe_results), "results")

print("\n=== RetrievalPipeline (reverse reranker) ===")
rr = MockReranker(mode="reverse")
pipeline_r = RetrievalPipeline(retriever=rm, reranker=rr)
reversed_results = call_and_print("RetrievalPipeline.run (with reranker)", pipeline_r.run, QUERY, COL, TOP_K)
print("First ids (original):", [r.chunk_id for r in pipe_results[:5]])
print("First ids (reversed):", [r.chunk_id for r in reversed_results[:5]])

print("\n=== Persistence check ===")
cm2 = ChromaDBManager(persist_directory=PERSIST_DIR)
print("Count after reopen:", cm2.get_collection_info(COL)["count"])
