import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import RetrievalConfig
from app.embeddings.embedder import EmbeddingManager
from app.retrieval.retriever import RetrievalManager, RetrievalResult
from app.vectordb.chroma_client import ChromaDBManager

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Lightweight, pure-Python BM25 keyword search retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric word tokens
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve_scores(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Scores all chunks based on BM25 relevance to the query.

        Args:
            query: User search query.
            chunks: List of chunk dictionaries, each having keys: 'id', 'text', 'metadata'.
            top_k: Number of scored results to return.

        Returns:
            Sorted list of tuples (chunk, score) with score > 0.
        """
        if not query or not chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Tokenize corpus documents
        corpus_tokens = [self._tokenize(c["text"]) for c in chunks]
        doc_lengths = [len(tokens) for tokens in corpus_tokens]
        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

        # Document Frequency calculation
        N = len(chunks)
        df: Dict[str, int] = {}
        for q_tok in set(query_tokens):
            df[q_tok] = sum(1 for tokens in corpus_tokens if q_tok in tokens)

        # IDF calculation with smoothing
        idf: Dict[str, float] = {}
        for q_tok, f_q in df.items():
            idf[q_tok] = math.log(1.0 + (N - f_q + 0.5) / (f_q + 0.5))

        # Score chunks
        scored_chunks: List[Tuple[Dict[str, Any], float]] = []
        for i, chunk in enumerate(chunks):
            tokens = corpus_tokens[i]
            doc_len = doc_lengths[i]

            # Term frequency in doc
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1

            score = 0.0
            for q_tok in query_tokens:
                if q_tok in tf:
                    f_q_d = tf[q_tok]
                    numerator = f_q_d * (self.k1 + 1)
                    denominator = f_q_d + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                    score += idf[q_tok] * (numerator / denominator)

            if score > 0.0:
                scored_chunks.append((chunk, score))

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]


class HybridRetriever(RetrievalManager):
    """Advanced Hybrid Retriever combining dense vector search and sparse keyword search (BM25)

    using Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(
        self,
        chroma: Optional[ChromaDBManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[RetrievalConfig] = None,
        collection_name: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_constant: int = 60,
    ) -> None:
        super().__init__(
            chroma=chroma,
            embedding_manager=embedding_manager,
            config=config,
            collection_name=collection_name,
        )
        self.bm25_retriever = BM25Retriever(k1=k1, b=b)
        self.rrf_constant = rrf_constant

    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval: dense vector search + sparse keyword search

        and merge via Reciprocal Rank Fusion (RRF).
        """
        col_name = collection_name or self.default_collection_name
        if not col_name:
            raise ValueError("collection_name must be a non-empty string")

        req_top_k = top_k if top_k is not None else self.default_top_k

        # 1. Fetch Dense Results
        try:
            dense_results = super().retrieve(
                query=query,
                collection_name=col_name,
                top_k=req_top_k * 2,  # Fetch extra for hybrid fusion
                metadata_filter=metadata_filter,
                min_score_threshold=0.0,  # Filter threshold is applied after re-ranking/fusion
            )
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}. Falling back to sparse search.")
            dense_results = []

        # 2. Fetch Sparse BM25 Results
        # Retrieve all chunks from collection for the local BM25 index
        try:
            collection = self.chroma.get_collection(col_name)
            # ChromaDB supports filtering directly in `.get()` using 'where'
            get_kwargs = {}
            if metadata_filter:
                get_kwargs["where"] = metadata_filter

            raw_data = collection.get(include=["documents", "metadatas"], **get_kwargs)
            
            all_chunks = []
            if raw_data and "documents" in raw_data and raw_data["documents"]:
                for i in range(len(raw_data["documents"])):
                    all_chunks.append({
                        "id": raw_data["ids"][i],
                        "text": raw_data["documents"][i],
                        "metadata": raw_data["metadatas"][i] if raw_data["metadatas"] else {}
                    })
            
            sparse_scores = self.bm25_retriever.retrieve_scores(
                query=query,
                chunks=all_chunks,
                top_k=req_top_k * 2
            )
            
            sparse_results = []
            for chunk, score in sparse_scores:
                sparse_results.append(RetrievalResult(
                    chunk_id=chunk["id"],
                    document_text=chunk["text"],
                    metadata=chunk["metadata"],
                    distance=None,
                    score=score,
                    collection=col_name
                ))
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}. Falling back to dense-only results.")
            sparse_results = []

        # 3. Merge results using Reciprocal Rank Fusion (RRF)
        # Combine dense and sparse lists
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        # Dense ranks
        for rank, res in enumerate(dense_results, 1):
            chunk_map[res.chunk_id] = res
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_constant + rank))

        # Sparse ranks
        for rank, res in enumerate(sparse_results, 1):
            if res.chunk_id not in chunk_map:
                chunk_map[res.chunk_id] = res
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_constant + rank))

        # Sort unique chunks by RRF score descending
        merged_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_results = []
        for cid in merged_chunk_ids:
            res = chunk_map[cid]
            # Calculate a normalized hybrid score based on RRF value
            # Max possible RRF score is (1/60 + 1/60) = 0.0333 for rank 1 in both retrievers.
            # We scale it to be in 0.0 - 1.0 range
            raw_rrf = rrf_scores[cid]
            max_possible_rrf = (1.0 / (self.rrf_constant + 1)) * 2
            normalized_score = raw_rrf / max_possible_rrf
            
            # If the dense similarity score is present, we can blend them
            final_score = res.score if res.score is not None else normalized_score
            
            # Enforce threshold (if provided)
            threshold = min_score_threshold if min_score_threshold is not None else self.default_min_score_threshold
            if final_score >= threshold:
                res.score = final_score
                final_results.append(res)

        logger.info(f"[HYBRID] Retrieved {len(final_results)} merged chunks.")
        return final_results[:req_top_k]
