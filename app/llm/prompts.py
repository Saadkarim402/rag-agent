from typing import List
from app.retrieval.retriever import RetrievalResult

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a precise technical RAG assistant. Answer the user query using ONLY "
    "the facts from the provided context. If the context does not contain the answer, "
    "say 'I cannot answer this based on the provided context.' Do not use external knowledge."
)

DEFAULT_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)

def format_retrieved_context(results: List[RetrievalResult]) -> str:
    """Format list of RetrievalResult objects into a structured text context block."""
    context_parts = []
    for i, res in enumerate(results, 1):
        source_id = res.metadata.get("source_id") or res.metadata.get("source") or "unknown"
        context_parts.append(f"[{i}] (Source: {source_id}) {res.document_text}")
    return "\n".join(context_parts)
