"""End-to-End RAG Demonstration using Local Ollama.

This script runs the complete RAG loop:
    1. Ingestion of raw documents into ChromaDB.
    2. Querying the Vector DB to retrieve the most relevant chunks.
    3. Constructing the augmented prompt.
    4. Passing the prompt to the Ollama Client (running simulated responses if Ollama is offline).

Run: python scripts/debug_e2e_llm.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import IngestionConfig, RetrievalConfig
from app.documents import DocumentRepository
from app.embeddings.embedder import EmbeddingManager
from app.ingestion.ingest import DocumentIngestionManager
from app.retrieval.retriever import RetrievalManager
from app.vectordb.chroma_client import ChromaDBManager
from app.llm.ollama_client import OllamaClient

SAMPLE_DOCS: List[Dict[str, Any]] = [
    {
        "id": "kubernetes_guide",
        "text": (
            "Kubernetes is a container orchestration platform. It helps manage containers at scale. "
            "Kubernetes coordinates container deployment, scaling, and networking across clusters. "
            "It schedules pods, manages services, and keeps workloads available even when nodes fail."
        ),
        "metadata": {"source_id": "kubernetes_guide", "category": "orchestration", "title": "Kubernetes Guide"},
    },
    {
        "id": "docker_tutorial",
        "text": (
            "Docker is a container runtime that packages applications into portable images. "
            "It simplifies local development, packaging, and deployment. "
            "Docker images share the host OS kernel while isolating runtime environments."
        ),
        "metadata": {"source_id": "docker_tutorial", "category": "runtime", "title": "Docker Tutorial"},
    },
]

QUERY = "How does Kubernetes manage containers at scale?"

def main() -> None:
    # Use standard UTF-8 stdout encoding to avoid Windows Unicode errors
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 80)
    print("  STARTING END-TO-END RAG PROCESS")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Initialize isolated DB and Repository
        chroma = ChromaDBManager(persist_directory=tmp_dir)
        repository = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        
        ingest_mgr = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repository,
            config=IngestionConfig(chunk_size=80, chunk_overlap=15, collection_name="demo_collection")
        )
        
        retrieval_mgr = RetrievalManager(
            chroma=chroma,
            config=RetrievalConfig(top_k=3, min_score_threshold=0.60, collection_name="demo_collection")
        )
        
        # ----------------------------------------------------------------------
        # PHASE 1: DOCUMENT INGESTION
        # ----------------------------------------------------------------------
        print("\n--- PHASE 1: INGESTING DOCUMENTS ---")
        for doc in SAMPLE_DOCS:
            ids = ingest_mgr.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])
            print(f"✓ Ingested document '{doc['id']}' -> split into {len(ids)} chunks.")

        # ----------------------------------------------------------------------
        # PHASE 2: SEMANTIC RETRIEVAL
        # ----------------------------------------------------------------------
        print("\n--- PHASE 2: SEMANTIC RETRIEVAL ---")
        print(f"Query: '{QUERY}'")
        results = retrieval_mgr.retrieve(query=QUERY)
        
        print(f"Retrieved {len(results)} relevant context chunks:")
        for rank, res in enumerate(results, 1):
            print(f"  [{rank}] Score: {res.score:.4f} | Source: {res.metadata['source_id']}")
            print(f"      Text: \"{res.document_text}\"")

        # ----------------------------------------------------------------------
        # PHASE 3: PROMPT CONTEXT ASSEMBLY
        # ----------------------------------------------------------------------
        print("\n--- PHASE 3: AUGMENTED CONTEXT ASSEMBLY ---")
        
        # Format the retrieved chunks into a context block
        context_parts = []
        for i, res in enumerate(results, 1):
            context_parts.append(f"[{i}] (Source: {res.metadata['source_id']}) {res.document_text}")
        context_str = "\n".join(context_parts)
        
        system_instruction = (
            "You are a precise technical RAG assistant. Answer the user query using ONLY "
            "the facts from the provided context. If the context does not contain the answer, "
            "say 'I cannot answer this based on the provided context.' Do not use external knowledge."
        )
        
        prompt = f"Context:\n{context_str}\n\nQuestion: {QUERY}\nAnswer:"
        
        print("\n[System Instruction]:")
        print(system_instruction)
        print("\n[Compiled Prompt]:")
        print(prompt)

        # ----------------------------------------------------------------------
        # PHASE 4: LLM ANSWER GENERATION
        # ----------------------------------------------------------------------
        print("\n--- PHASE 4: LLM ANSWER GENERATION ---")
        llm = OllamaClient(model="llama3")
        
        print("Sending request to local Ollama server (http://localhost:11434)...")
        try:
            answer = llm.generate(prompt=prompt, system_instruction=system_instruction)
            print("\n[Ollama Response]:")
            print(answer)
        except RuntimeError as e:
            print("\n⚠️  [Ollama Server is Offline / Connection Refused]")
            print(f"   Original Error: {e}")
            print("\n🤖 [Simulating Ollama Response (Offline Demo Mode)]:")
            
            # Simulate the response based on context
            simulated_response = (
                "Based on the provided context, Kubernetes manages containers at scale by "
                "orchestrating and coordinating container deployment, scaling, and networking across clusters "
                "[1]. It helps manage these containers continuously [2] and acts as a container orchestration "
                "platform [3]."
            )
            print(simulated_response)

    print("\n" + "=" * 80)
    print("  END-TO-END DEMONSTRATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
