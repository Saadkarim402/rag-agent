import pytest
from typing import Optional

from app.llm.base import BaseLLMClient
from app.llm.chains import RAGChain, RAGResponse
from app.llm.prompts import DEFAULT_SYSTEM_INSTRUCTION
from app.retrieval.retriever import RetrievalManager
from app.retrieval.pipeline import RetrievalPipeline


class MockLLMClient(BaseLLMClient):
    """Simple mock LLM client to intercept prompts and return deterministic responses."""

    def __init__(self, response_text: str = "Mock LLM Response") -> None:
        self.response_text = response_text
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        return self.response_text


def test_rag_chain_initialization():
    mock_retrieval = object()
    mock_llm = MockLLMClient()
    chain = RAGChain(
        retriever=mock_retrieval,
        llm_client=mock_llm,
        default_collection_name="test_col",
        system_instruction="Custom Instruction",
        prompt_template="Prompt: {context} -> {query}"
    )

    assert chain.retriever is mock_retrieval
    assert chain.llm_client is mock_llm
    assert chain.default_collection_name == "test_col"
    assert chain.system_instruction == "Custom Instruction"
    assert chain.prompt_template == "Prompt: {context} -> {query}"


def test_rag_chain_query_with_retrieval_manager(tmp_chroma_manager, ingestion_manager, sample_docs):
    # Ingest documents
    for doc in sample_docs:
        ingestion_manager.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])

    retriever = RetrievalManager(chroma=tmp_chroma_manager, collection_name="test_collection")
    llm = MockLLMClient("Eiffel Tower is in Paris.")
    chain = RAGChain(retriever=retriever, llm_client=llm)

    response = chain.query(query="Where is Eiffel Tower?", metadata_filter={"category": "geography"})
    assert isinstance(response, RAGResponse)
    assert response.answer == "Eiffel Tower is in Paris."
    assert len(response.source_nodes) >= 1
    assert any("Paris" in node.document_text for node in response.source_nodes)
    
    # Verify the correct prompt was sent to the LLM
    assert "Where is Eiffel Tower?" in response.prompt
    assert "[1] (Source: doc-1)" in response.prompt
    assert llm.last_prompt == response.prompt
    assert llm.last_system_instruction == DEFAULT_SYSTEM_INSTRUCTION


def test_rag_chain_query_with_retrieval_pipeline(tmp_chroma_manager, ingestion_manager, sample_docs):
    # Ingest documents
    for doc in sample_docs:
        ingestion_manager.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])

    retriever = RetrievalManager(chroma=tmp_chroma_manager)
    pipeline = RetrievalPipeline(retriever=retriever)
    llm = MockLLMClient()
    chain = RAGChain(retriever=pipeline, llm_client=llm, default_collection_name="test_collection")

    # Use metadata filter to ensure programming document is selected
    response = chain.query(
        query="Python programming language", 
        top_k=2,
        metadata_filter={"category": "programming"}
    )
    assert isinstance(response, RAGResponse)
    assert len(response.source_nodes) >= 1
    assert any("programming" in node.metadata.get("category", "") for node in response.source_nodes)


def test_rag_chain_collection_resolution_checks(tmp_chroma_manager):
    retriever = RetrievalManager(chroma=tmp_chroma_manager, collection_name="configured_col")
    llm = MockLLMClient()
    
    # 1. Resolve from retriever
    chain = RAGChain(retriever=retriever, llm_client=llm)
    response = chain.query(query="test query")
    assert len(response.source_nodes) == 0  # Empty collection but query should run on 'configured_col'
    
    # 2. Raise ValueError when collection cannot be resolved
    retriever_no_col = RetrievalManager(chroma=tmp_chroma_manager, collection_name="")
    chain_no_col = RAGChain(retriever=retriever_no_col, llm_client=llm)
    with pytest.raises(ValueError, match="collection_name must be specified"):
        chain_no_col.query(query="test query")


def test_rag_chain_metadata_filtering(tmp_chroma_manager, ingestion_manager, sample_docs):
    # Ingest documents
    for doc in sample_docs:
        ingestion_manager.ingest_text(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])

    retriever = RetrievalManager(chroma=tmp_chroma_manager)
    llm = MockLLMClient()
    chain = RAGChain(retriever=retriever, llm_client=llm, default_collection_name="test_collection")

    # Filter only programming docs and limit to 1 result
    response = chain.query(
        query="Tell me about Python",
        metadata_filter={"category": "programming"},
        top_k=1
    )
    assert len(response.source_nodes) == 1
    assert response.source_nodes[0].metadata.get("category") == "programming"
    assert response.source_nodes[0].metadata.get("source_id") == "doc-2"
    assert "Paris" not in response.prompt
