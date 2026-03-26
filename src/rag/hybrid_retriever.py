from __future__ import annotations
from typing import List, Tuple

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, Fusion, FusionQuery
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "thai_laws_collection"
VECTOR_DENSE = "dense"
VECTOR_SPARSE = "sparse"


class HybridRetriever:
    """
    LangChain-compatible retriever using BGE-M3 (dense + sparse) + RRF fusion
    + BGE-Reranker-v2-M3 for Thai legal documents stored in Qdrant.
    """

    def __init__(
        self,
        embed_model: BGEM3FlagModel,
        reranker: FlagReranker,
        client: QdrantClient,
        retrieval_limit: int ,
        final_limit: int ,
    ) -> None:
        self.embed_model = embed_model
        self.reranker = reranker
        self.client = client
        self.retrieval_limit = retrieval_limit
        self.final_limit = final_limit

    # ------------------------------------------------------------------
    # Core search logic
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> Tuple[list, SparseVector]:
        output = self.embed_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            max_length=512,
        )
        dense_vec = output["dense_vecs"][0].tolist()
        sparse_dict = output["lexical_weights"][0]
        sparse_vec = SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()],
        )
        return dense_vec, sparse_vec

    def _hybrid_search(self, dense_vec: list, sparse_vec: SparseVector) -> list:
        return self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec, using=VECTOR_DENSE, limit=self.retrieval_limit
                ),
                Prefetch(
                    query=sparse_vec, using=VECTOR_SPARSE, limit=self.retrieval_limit
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.retrieval_limit,
            with_payload=True,
        ).points

    def _rerank(self, query: str, candidates: list) -> list:
        if not candidates:
            return []
        pairs = [[query, r.payload["text"]] for r in candidates]
        scores = self.reranker.compute_score(pairs, batch_size=32)
        for i, r in enumerate(candidates):
            r.score = float(scores[i])
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[: self.final_limit]

    # ------------------------------------------------------------------
    # Public API — returns LangChain Documents
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        dense_vec, sparse_vec = self._encode_query(query)
        candidates = self._hybrid_search(dense_vec, sparse_vec)
        ranked = self._rerank(query, candidates)

        docs = []
        for r in ranked:
            p = r.payload
            docs.append(
                Document(
                    page_content=p.get("text", ""),
                    metadata={
                        "law_name": p.get("law_name", ""),
                        "section_num": p.get("section_num", ""),
                        "score": r.score,
                    },
                )
            )
        return docs

    def retrieve_with_scores(self, query: str):
        dense_vec, sparse_vec = self._encode_query(query)
        candidates = self._hybrid_search(dense_vec, sparse_vec)
        reranked = self._rerank(query, candidates)
        return candidates, reranked

    # LangChain runnable interface
    def __call__(self, query: str) -> List[Document]:
        return self.retrieve(query)