import logging
from dataclasses import dataclass, field
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
    agent_loop_logs: List[str] = field(default_factory=list)


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
        confidence_threshold: float = 0.0,
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
            import re
            match = re.search(r"\b([1-5])\b", res)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 5  # Default to high score if evaluation fails/timeouts

    def _build_history_context(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Builds a conversation summary buffer + recent verbatim turns from history."""
        if not chat_history:
            return ""

        # Filter only user and assistant messages
        valid_msgs = [m for m in chat_history if m.get("role") in ("user", "assistant")]
        if not valid_msgs:
            return ""

        # Take the last 6 messages (3 turns) verbatim
        verbatim_limit = 6
        verbatim_msgs = valid_msgs[-verbatim_limit:]
        older_msgs = valid_msgs[:-verbatim_limit]

        summary_buffer = ""
        if older_msgs:
            # Format older messages for summarization
            older_text_parts = []
            for msg in older_msgs:
                role = "User" if msg["role"] == "user" else "Assistant"
                older_text_parts.append(f"{role}: {msg['content']}")
            older_text = "\n".join(older_text_parts)

            summary_prompt = (
                "You are a conversation summarizer. Briefly summarize the following older chat turns "
                f"in 2-3 sentences to preserve conversation context:\n\n{older_text}\n\nSummary:"
            )
            try:
                summary_buffer = self.llm_client.generate(prompt=summary_prompt).strip()
            except Exception as e:
                logger.warning(f"Failed to generate conversation summary: {e}")
                summary_buffer = "Continued conversation history."

        # Compile history string
        history_parts = []
        if summary_buffer:
            history_parts.append(f"Conversation Summary (older turns): {summary_buffer}")

        history_parts.append("Recent conversation turns:")
        for msg in verbatim_msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")

        return "\n".join(history_parts)

    def query(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """Executes the complete RAG pipeline with agentic self-correction and web search fallback.

        Args:
            query: The user query string.
            collection_name: Optional name of the collection. Falls back to default config if None.
            top_k: Optional number of context chunks to retrieve.
            min_score_threshold: Optional minimum similarity score threshold to accept chunks.
            metadata_filter: Optional dictionary filter for metadata fields.
            chat_history: Optional list of previous chat messages.

        Returns:
            RAGResponse object containing the answer, sources, compiled prompt, and metrics.
        """
        import time
        import re
        from app.tools.web_search import DuckDuckGoSearchTool

        start_time = time.time()
        web_search_triggered = False
        agent_loop_logs = []
        current_query = query
        results = []
        answer = ""

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

        # Build conversation memory context
        history_context = self._build_history_context(chat_history)
        system_instruction = self.system_instruction
        if history_context:
            system_instruction += f"\n\n[CONVERSATION HISTORY CONTEXT]\n{history_context}"

        # Setup retrieval arguments
        kwargs = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_score_threshold is not None:
            kwargs["min_score_threshold"] = min_score_threshold
        if metadata_filter is not None:
            kwargs["metadata_filter"] = metadata_filter

        threshold = min_score_threshold if min_score_threshold is not None else self.confidence_threshold

        # Corrective Agent Loop (Max 3 attempts)
        for attempt in range(1, 4):
            log_msg = f"Attempt {attempt}: Querying vector database for '{current_query}'"
            agent_loop_logs.append(log_msg)
            logger.info(f"[AGENT] {log_msg}")

            # 1. Retrieve
            if hasattr(self.retriever, "run"):
                results = self.retriever.run(query=current_query, collection_name=col_name, **kwargs)
            elif hasattr(self.retriever, "retrieve"):
                results = self.retriever.retrieve(query=current_query, collection_name=col_name, **kwargs)
            else:
                raise AttributeError("The retriever does not have a run() or retrieve() method.")

            # 2. Re-rank
            if self.reranker and results:
                results = self.reranker.rerank(current_query, results)

            # 3. Context sufficiency critique
            max_score = max([r.score for r in results if r.score is not None], default=0.0)
            context_is_sufficient = len(results) > 0 and max_score >= threshold

            if context_is_sufficient or threshold <= 0.0:
                log_msg = f"Retrieved {len(results)} chunks. Context quality is sufficient (max score {max_score:.4f} >= {threshold:.4f})."
                agent_loop_logs.append(log_msg)
                logger.info(f"[AGENT] {log_msg}")
            else:
                log_msg = f"Context quality is insufficient (max score {max_score:.4f} < {threshold:.4f} or no local matches)."
                agent_loop_logs.append(log_msg)
                logger.info(f"[AGENT] {log_msg}")

                if attempt < 3:
                    # Query Rewrite Action
                    log_msg = "Self-Correction: Rewriting query for better retrieval..."
                    agent_loop_logs.append(log_msg)
                    logger.info(f"[AGENT] {log_msg}")

                    rewrite_prompt = (
                        "You are an AI search query optimizer. Rewrite the following user query to "
                        "help find relevant technical documentation. Output ONLY the optimized search keywords/phrase. "
                        f"No extra text.\nQuery: '{current_query}'\nOptimized Query:"
                    )
                    try:
                        rewritten = self.llm_client.generate(prompt=rewrite_prompt).strip().strip("'\"")
                        if rewritten and rewritten != current_query:
                            current_query = rewritten
                            log_msg = f"Query rewritten to: '{current_query}'"
                            agent_loop_logs.append(log_msg)
                            logger.info(f"[AGENT] {log_msg}")
                            continue  # Rerun loop with rewritten query
                    except Exception as e:
                        logger.error(f"Query rewrite failed: {e}")

                # Escalate to Web Search Fallback
                log_msg = "Escalating to Web Search Fallback..."
                agent_loop_logs.append(log_msg)
                logger.info(f"[AGENT] {log_msg}")
                web_search_triggered = True

                search_tool = DuckDuckGoSearchTool()
                web_hits = search_tool.search(current_query)
                web_results = []
                for idx_h, hit in enumerate(web_hits, 1):
                    web_results.append(RetrievalResult(
                        chunk_id=f"web_chunk_{idx_h}",
                        document_text=hit["snippet"],
                        metadata={"source_id": hit["title"], "url": hit["url"], "is_web": True},
                        distance=None,
                        score=1.0 - (idx_h * 0.1),
                        collection="web_search"
                    ))
                results = web_results + results
                break  # Complete retrieval phase with web results

            # 4. Generate LLM Answer
            context_str = format_retrieved_context(results)
            compiled_prompt = self.prompt_template.format(context=context_str, query=current_query)

            answer = self.llm_client.generate(
                prompt=compiled_prompt,
                system_instruction=system_instruction,
            )

            # 5. Answer Critique (Self-Evaluation / Faithfulness check)
            if self.enable_evaluation:
                faithfulness_prompt = (
                    f"Context:\n{context_str}\n\n"
                    f"Answer:\n{answer}\n\n"
                    "Based on the context, is the answer 100% faithful to the facts provided? "
                    "Output ONLY a single integer digit from 1 (completely hallucinated/unsupported) "
                    "to 5 (completely supported by the context). No other text."
                )
                faithfulness = self._evaluate_metric(faithfulness_prompt)

                log_msg = f"Critique: Faithfulness Score evaluated as {faithfulness}/5"
                agent_loop_logs.append(log_msg)
                logger.info(f"[AGENT] {log_msg}")

                if faithfulness < 3 and attempt < 3:
                    log_msg = f"Critique Check Failed (Faithfulness {faithfulness} < 3). Triggering rewrite loop."
                    agent_loop_logs.append(log_msg)
                    logger.info(f"[AGENT] {log_msg}")

                    rewrite_prompt = (
                        "The previous answer contained hallucinations. Rewrite the query to find better, "
                        f"more precise facts for: '{current_query}'\nOptimized Query:"
                    )
                    try:
                        rewritten = self.llm_client.generate(prompt=rewrite_prompt).strip().strip("'\"")
                        if rewritten:
                            current_query = rewritten
                            continue
                    except Exception:
                        pass

            break  # Finished loop successfully

        # Final generation if loop completed via web fallback escalation
        if not answer:
            context_str = format_retrieved_context(results)
            compiled_prompt = self.prompt_template.format(context=context_str, query=current_query)
            answer = self.llm_client.generate(
                prompt=compiled_prompt,
                system_instruction=system_instruction,
            )

        latency_ms = (time.time() - start_time) * 1000

        # LLM-as-a-judge Self-Evaluation final calculations (if enabled)
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
            agent_loop_logs=agent_loop_logs,
        )



