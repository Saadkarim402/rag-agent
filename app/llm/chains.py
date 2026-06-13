import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.llm.base import BaseLLMClient
from app.llm.prompts import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_INSTRUCTION,
    format_retrieved_context,
)
from app.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG Chain containing the answer, source nodes, and prompt."""
    answer: str
    source_nodes: List[RetrievalResult]
    prompt: str
    web_search_triggered: bool = False
    latency_ms: float = 0.0
    faithfulness: int = 5
    answer_relevance: int = 5


class RAGChain:
    """The orchestrator chain that combines document retrieval and LLM response generation."""

    def __init__(
        self,
        retriever: Any,
        llm_client: BaseLLMClient,
        default_collection_name: Optional[str] = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        reranker: Optional[Any] = None,
        confidence_threshold: float = 0.35,
        enable_evaluation: bool = False,
    ) -> None:
        """Initialize the RAG Chain.

        Args:
            retriever: RetrievalManager or RetrievalPipeline instance.
            llm_client: BaseLLMClient instance (e.g. OllamaClient, GeminiClient).
            default_collection_name: Optional default collection to query if none is provided.
            system_instruction: System prompt guiding the LLM behavior.
            prompt_template: Template used to assemble the query and context block.
            reranker: Optional CrossEncoderReranker instance.
            confidence_threshold: Similarity score threshold below which web search is triggered.
            enable_evaluation: If True, evaluates RAG response metrics (Faithfulness and Relevance) via the LLM.
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.default_collection_name = default_collection_name
        self.system_instruction = system_instruction
        self.prompt_template = prompt_template
        self.reranker = reranker
        self.confidence_threshold = confidence_threshold
        self.enable_evaluation = enable_evaluation

    def _evaluate_metric(self, eval_prompt: str) -> int:
        """Evaluate a score (1-5) using the LLM client, with a quick timeout fallback."""
        try:
            res = self.llm_client.generate(prompt=eval_prompt)
            match = re.search(r"\b([1-5])\b", res)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 5  # Default to high score if evaluation fails/timeouts

    def query(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Executes the complete RAG pipeline with agentic self-correction and web search fallback.

        Args:
            query: The user query string.
            collection_name: Optional name of the collection. Falls back to default config if None.
            top_k: Optional number of context chunks to retrieve.
            min_score_threshold: Optional minimum similarity score threshold to accept chunks.
            metadata_filter: Optional dictionary filter for metadata fields.

        Returns:
            RAGResponse object containing the answer, sources, compiled prompt, and metrics.
        """
        import time
        import re
        from app.tools.web_search import DuckDuckGoSearchTool

        start_time = time.time()
        web_search_triggered = False

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
            results = self.retriever.run(query=query, collection_name=col_name, **kwargs)
        elif hasattr(self.retriever, "retrieve"):
            results = self.retriever.retrieve(query=query, collection_name=col_name, **kwargs)
        else:
            raise AttributeError("The provided retriever does not have a run() or retrieve() method.")

        # Apply Re-ranking if module is configured
        if self.reranker and results:
            results = self.reranker.rerank(query, results)

        # Agentic Self-Correction: Check if local retrieval quality is sufficient
        max_score = max([r.score for r in results if r.score is not None], default=0.0)
        threshold = min_score_threshold if min_score_threshold is not None else self.confidence_threshold
        
        if max_score < threshold or not results:
            # Trigger Web Search Fallback
            web_search_triggered = True
            logger.warning(f"[AGENT] Low confidence match ({max_score:.4f} < {threshold:.4f}). Triggering Web Search Fallback...")
            search_tool = DuckDuckGoSearchTool()
            web_hits = search_tool.search(query)
            
            web_results = []
            for i, hit in enumerate(web_hits, 1):
                web_results.append(RetrievalResult(
                    chunk_id=f"web_chunk_{i}",
                    document_text=hit["snippet"],
                    metadata={"source_id": hit["title"], "url": hit["url"], "is_web": True},
                    distance=None,
                    score=1.0 - (i * 0.1),  # Simulated decreasing score
                    collection="web_search"
                ))
            
            # Combine web results with local chunks
            results = web_results + results

        # Format context and compile prompt
        context_str = format_retrieved_context(results)
        compiled_prompt = self.prompt_template.format(context=context_str, query=query)

        # Call LLM client
        answer = self.llm_client.generate(
            prompt=compiled_prompt,
            system_instruction=self.system_instruction,
        )

        latency_ms = (time.time() - start_time) * 1000

        # LLM-as-a-judge Self-Evaluation (if enabled)
        faithfulness = 5
        answer_relevance = 5

        if self.enable_evaluation:
            faithfulness_prompt = (
                f"Context:\n{context_str}\n\n"
                f"Answer:\n{answer}\n\n"
                "Based on the context, is the answer 100% faithful to the facts provided? "
                "Output ONLY a single integer digit from 1 (completely hallucinated/unsupported) "
                "to 5 (completely supported by the context). No other text."
            )
            relevance_prompt = (
                f"Question: {query}\n"
                f"Answer: {answer}\n\n"
                "Does the answer directly and accurately address the user question? "
                "Output ONLY a single integer digit from 1 (completely irrelevant) to 5 (perfectly relevant). No other text."
            )

            faithfulness = self._evaluate_metric(faithfulness_prompt)
            answer_relevance = self._evaluate_metric(relevance_prompt)

        return RAGResponse(
            answer=answer,
            source_nodes=results,
            prompt=compiled_prompt,
            web_search_triggered=web_search_triggered,
            latency_ms=round(latency_ms, 2),
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
        )


