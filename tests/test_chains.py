import pytest
from typing import Optional
from unittest.mock import MagicMock, patch

from app.llm.base import BaseLLMClient
from app.llm.chains import RAGChain, RAGResponse
from app.llm.prompts import DEFAULT_SYSTEM_INSTRUCTION
from app.retrieval.retriever import RetrievalManager, RetrievalResult
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


def test_rag_chain_build_history_context_short():
    llm = MockLLMClient("Eiffel summary")
    chain = RAGChain(retriever=object(), llm_client=llm)
    
    # 2 turns (4 messages) - under the verbatim limit of 6
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Who built Eiffel Tower?"},
        {"role": "assistant", "content": "Gustave Eiffel."}
    ]
    context = chain._build_history_context(history)
    assert "Conversation Summary" not in context
    assert "User: Hello" in context
    assert "Assistant: Hi there!" in context
    assert "User: Who built Eiffel Tower?" in context
    assert "Assistant: Gustave Eiffel." in context
    assert llm.last_prompt is None  # No summarizer was called


def test_rag_chain_build_history_context_long():
    llm = MockLLMClient("Mock summary of older turns.")
    chain = RAGChain(retriever=object(), llm_client=llm)
    
    # 4 turns (8 messages) - over the verbatim limit of 6
    history = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Reply 3"},
        {"role": "user", "content": "Message 4"},
        {"role": "assistant", "content": "Reply 4"},
    ]
    context = chain._build_history_context(history)
    assert "Conversation Summary (older turns): Mock summary of older turns." in context
    # Should contain the last 3 turns verbatim (messages 3 and 4; wait, limit 6 is messages 3, 4, and 2 assistant reply?)
    # Since valid_msgs[-6:] takes the last 6 messages: Message 3 user/assistant, Message 4 user/assistant, Message 2 user/assistant
    # So Message 1 user/assistant are older messages.
    assert "User: Message 1" not in context
    assert "Assistant: Reply 1" not in context
    assert "User: Message 3" in context
    assert "Assistant: Reply 3" in context
    assert "User: Message 4" in context
    assert "Assistant: Reply 4" in context
    assert "conversation summarizer" in llm.last_prompt.lower()


def test_rag_chain_corrective_agent_loop_no_rewrite():
    # Setup mock retriever returning high score
    mock_retriever = MagicMock()
    mock_result = RetrievalResult(
        chunk_id="c1", document_text="Eiffel tower is in Paris.", metadata={}, distance=None, score=0.9, collection="col"
    )
    mock_retriever.run.return_value = [mock_result]
    
    llm = MockLLMClient("Final Answer")
    chain = RAGChain(retriever=mock_retriever, llm_client=llm, confidence_threshold=0.5)
    
    response = chain.query("Where is Eiffel Tower?", collection_name="col")
    assert response.answer == "Final Answer"
    assert len(response.source_nodes) == 1
    assert "Attempt 1" in response.agent_loop_logs[0]
    assert "quality is sufficient" in response.agent_loop_logs[1]
    # Verify no rewrite or web search took place
    assert mock_retriever.run.call_count == 1
    assert not response.web_search_triggered


def test_rag_chain_corrective_agent_loop_rewrite_flow():
    # Setup mock retriever returning low score on first call, high score on rewritten call
    mock_retriever = MagicMock()
    low_score_result = RetrievalResult(
        chunk_id="c1", document_text="Vague document about towers.", metadata={}, distance=None, score=0.2, collection="col"
    )
    high_score_result = RetrievalResult(
        chunk_id="c2", document_text="Gustave Eiffel built Eiffel Tower.", metadata={}, distance=None, score=0.8, collection="col"
    )
    
    # Return low_score on first call, high_score on second call
    mock_retriever.run.side_effect = [[low_score_result], [high_score_result]]
    
    # Mock LLM client to return "optimized query" for query optimizer, then return "Final Answer"
    class DynamicMockLLMClient(BaseLLMClient):
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, system_instruction=None):
            self.calls += 1
            if "query optimizer" in prompt.lower() or "rewrite" in prompt.lower():
                return "optimized query"
            return f"Answer with {self.calls}"
            
    llm = DynamicMockLLMClient()
    chain = RAGChain(retriever=mock_retriever, llm_client=llm, confidence_threshold=0.5)
    
    response = chain.query("Eiffel?", collection_name="col")
    
    # Check that retriever was called twice (once with original, once with rewritten)
    assert mock_retriever.run.call_count == 2
    mock_retriever.run.assert_any_call(query="Eiffel?", collection_name="col")
    mock_retriever.run.assert_any_call(query="optimized query", collection_name="col")
    
    assert any("Self-Correction: Rewriting query" in log for log in response.agent_loop_logs)
    assert any("Query rewritten to: 'optimized query'" in log for log in response.agent_loop_logs)
    assert not response.web_search_triggered


def test_rag_chain_web_search_escalation():
    # Setup mock retriever returning low score consistently
    mock_retriever = MagicMock()
    low_score_result = RetrievalResult(
        chunk_id="c1", document_text="Irrelevant facts.", metadata={}, distance=None, score=0.1, collection="col"
    )
    mock_retriever.run.return_value = [low_score_result]
    
    # Mock LLM client
    class DynamicMockLLMClient(BaseLLMClient):
        def generate(self, prompt, system_instruction=None):
            if "query optimizer" in prompt.lower() or "rewrite" in prompt.lower():
                return "optimized query"
            return "Answer based on search"
            
    llm = DynamicMockLLMClient()
    chain = RAGChain(retriever=mock_retriever, llm_client=llm, confidence_threshold=0.5)
    
    # Mock DuckDuckGoSearchTool
    mock_search = MagicMock()
    mock_search.search.return_value = [
        {"title": "Web Page", "url": "http://web.com", "snippet": "Web search result snippet."}
    ]
    
    with patch("app.tools.web_search.DuckDuckGoSearchTool", return_value=mock_search):
        response = chain.query("Web Query?", collection_name="col")
        
    assert response.web_search_triggered is True
    assert len(response.source_nodes) >= 2  # Web result + original low-score result
    assert response.source_nodes[0].collection == "web_search"
    assert response.source_nodes[0].document_text == "Web search result snippet."
    assert any("Escalating to Web Search Fallback..." in log for log in response.agent_loop_logs)

