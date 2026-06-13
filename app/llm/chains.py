from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.llm.base import BaseLLMClient
from app.llm.prompts import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_INSTRUCTION,
    format_retrieved_context,
)
from app.retrieval.retriever import RetrievalResult


@dataclass
class RAGResponse:
    """Structured response from the RAG Chain containing the answer, source nodes, and prompt."""
    answer: str
    source_nodes: List[RetrievalResult]
    prompt: str


class RAGChain:
    """The orchestrator chain that combines document retrieval and LLM response generation."""

    def __init__(
        self,
        retriever: Any,
        llm_client: BaseLLMClient,
        default_collection_name: Optional[str] = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        """Initialize the RAG Chain.

        Args:
            retriever: RetrievalManager or RetrievalPipeline instance.
            llm_client: BaseLLMClient instance (e.g. OllamaClient, GeminiClient).
            default_collection_name: Optional default collection to query if none is provided.
            system_instruction: System prompt guiding the LLM behavior.
            prompt_template: Template used to assemble the query and context block.
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.default_collection_name = default_collection_name
        self.system_instruction = system_instruction
        self.prompt_template = prompt_template

    def query(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Executes the complete RAG pipeline.

        Args:
            query: The user query string.
            collection_name: Optional name of the collection. Falls back to default config if None.
            top_k: Optional number of context chunks to retrieve.
            min_score_threshold: Optional minimum similarity score threshold to accept chunks.
            metadata_filter: Optional dictionary filter for metadata fields.

        Returns:
            RAGResponse object containing the answer, sources, and compiled prompt.
        """
        # Resolve collection name
        col_name = collection_name or self.default_collection_name
        if not col_name:
            if hasattr(self.retriever, "default_collection_name") and getattr(self.retriever, "default_collection_name"):
                col_name = self.retriever.default_collection_name
            elif hasattr(self.retriever, "_retriever") and hasattr(self.retriever._retriever, "default_collection_name"):
                col_name = self.retriever._retriever.default_collection_name

        if not col_name:
            raise ValueError(
                "collection_name must be specified either in query(), during RAGChain initialization, "
                "or configured in the retriever."
            )

        # Retrieve context chunks
        kwargs = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_score_threshold is not None:
            kwargs["min_score_threshold"] = min_score_threshold
        if metadata_filter is not None:
            kwargs["metadata_filter"] = metadata_filter

        if hasattr(self.retriever, "run"):
            # RetrievalPipeline style
            results = self.retriever.run(
                query=query,
                collection_name=col_name,
                **kwargs,
            )
        elif hasattr(self.retriever, "retrieve"):
            # RetrievalManager style
            results = self.retriever.retrieve(
                query=query,
                collection_name=col_name,
                **kwargs,
            )
        else:
            raise AttributeError("The provided retriever does not have a run() or retrieve() method.")

        # Format context and compile prompt
        context_str = format_retrieved_context(results)
        compiled_prompt = self.prompt_template.format(context=context_str, query=query)

        # Call LLM client
        answer = self.llm_client.generate(
            prompt=compiled_prompt,
            system_instruction=self.system_instruction,
        )

        return RAGResponse(
            answer=answer,
            source_nodes=results,
            prompt=compiled_prompt,
        )
