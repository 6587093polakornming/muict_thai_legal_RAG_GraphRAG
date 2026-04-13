from __future__ import annotations
from typing import List, Tuple

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import SparseVector, Prefetch, Fusion, FusionQuery
from langchain_core.documents import Document
from typing import Optional
from .config import RAGConfig  # Configuration


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
        config: RAGConfig,
    ) -> None:
        self.embed_model = embed_model
        self.reranker = reranker
        self.client = client
        self.config = config
        self.retrieval_limit = config.retrieval_limit
        self.reranking_limit = config.reranking_limit
        self.final_limit = config.final_limit

    # Core search logic
    def _encode_query(
        self, query: str, is_return_dense=True, is_return_sparse=True
    ) -> Tuple[list, SparseVector]:
        output = self.embed_model.encode(
            [query],
            return_dense=is_return_dense,
            return_sparse=is_return_sparse,
            max_length=512,
        )
        dense_vec, sparse_vec = [], SparseVector(indices=[], values=[])
        if is_return_dense:
            dense_vec = output["dense_vecs"][0].tolist()
        if is_return_sparse:
            sparse_dict = output["lexical_weights"][0]
            sparse_vec = SparseVector(
                indices=[int(k) for k in sparse_dict.keys()],
                values=[float(v) for v in sparse_dict.values()],
            )
        return dense_vec, sparse_vec

    def _hybrid_search(self, dense_vec: list, sparse_vec: SparseVector) -> list:
        return self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=self.config.vector_dense,
                    limit=self.retrieval_limit,
                ),
                Prefetch(
                    query=sparse_vec,
                    using=self.config.vector_sparse,
                    limit=self.retrieval_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.retrieval_limit,
            with_payload=True,
        ).points

    def _vector_search(self, dense_vec: list) -> list:
        return self.client.query_points(
            collection_name=self.config.collection_name,
            query=dense_vec,
            using=self.config.vector_dense,
            limit=self.retrieval_limit,
            with_payload=True,
        ).points

    def _keyword_search(self, sparse_vec: SparseVector) -> list:
        return self.client.query_points(
            collection_name=self.config.collection_name,
            query=sparse_vec,
            using=self.config.vector_sparse,
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
        return candidates[: self.reranking_limit]

    # TODO add logic and parameter to _link_ref_law
    # new parameter
    # expansion_mode: str = top-1 (default) , top-n
    # top-1 Expand เฉพาะมาตราอ้างอิงที่พบในผลลัพธ์อันดับที่ 1, top-n Expand มาตราอ้างอิงที่พบจากผลลัพธ์ทั้ง N อันดับ
    # reorder_mode: str  =  parent-first (default), append-last
    # parent-first เรียงแบบ มาตราหลัก (Parent) → มาตราอ้างอิง (Child) คู่ ดั้งเดิมทำแบบนี้, append-last นำมาตราอ้างอิงทั้งหมดไปต่อท้าย Context หลักทั้งหมด
    def _link_ref_law(
        self, list_pts: list, expansion_mode="top-1", reorder_mode="parent-first"
    ) -> list:
        """
        Augment retrieved contexts for Hybrid RAG by injecting related law references.

        Parameters:
            list_pts (list): List of ScoredPoint results from hybrid search.
            expansion_mode (str):
                - "top-1" : Expand reference laws only from the 1st ranked result (default).
                - "top-n" : Expand reference laws from ALL N ranked results.
            reorder_mode (str):
                - "parent-first" : [parent] → [ref laws ของ parent] → [remaining] (default).
                - "append-last"  : [all original results] → [all ref laws ต่อท้าย].

        Returns:
            list: Reordered list of ScoredPoint with reference law contexts injected.
        """
        if not list_pts:
            return []

        # expansion_mode: เลือก context ที่จะนำไปหา reference_laws
        if expansion_mode == "top-n":
            lst_context = list_pts
        else:  # default: top-1
            lst_context = [list_pts[0]]

        # query_lst: flatten reference_laws จากทุก context ที่เลือก
        query_lst = [
            ref for context in lst_context
            for ref in context.payload.get("reference_laws", [])
        ]

        if not query_lst:
            return list_pts

        # ดึง reference law documents จาก Qdrant
        should_conditions = [
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_name",
                        match=models.MatchValue(value=ref["law_name"]),
                    ),
                    models.FieldCondition(
                        key="section_num",
                        match=models.MatchValue(value=ref["section_num"]),
                    ),
                ]
            )
            for ref in query_lst
        ]

        batch_results, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=models.Filter(should=should_conditions),
            limit=len(query_lst),
            with_payload=True,
            with_vectors=False,
        )

        # lookup (law_name, section_num) → res  แทน triple-nested loop O(N×M×R) → O(R)
        batch_key_map = {
            (res.payload["law_name"], res.payload["section_num"]): res
            for res in batch_results
        }
        # lookup id → res สำหรับ append-last
        batch_map = {res.id: res for res in batch_results}

        # สร้าง ref_to_parent: res.id → parent_context ที่ถูกต้อง (ใช้ร่วมกันทั้ง 2 mode)
        # O(N × M) แทน O(N × M × R) เดิม
        ref_to_parent: dict = {}
        for parent_context in lst_context:
            for ref in parent_context.payload.get("reference_laws", []):
                key = (ref["law_name"], ref["section_num"])
                res = batch_key_map.get(key)
                if res and res.id not in ref_to_parent:
                    ref_to_parent[res.id] = parent_context

        if reorder_mode == "parent-first":
            # เรียง: [parent] → [ref laws ของ parent นั้น] → [remaining]
            seen_ids = set()
            final_results = []

            for parent_context in lst_context:
                if parent_context.id not in seen_ids:
                    final_results.append(parent_context)
                    seen_ids.add(parent_context.id)

                # กรองเฉพาะ ref ที่เป็นของ parent นี้ — ถูก score และถูก ordering
                for res_id, owner in ref_to_parent.items():
                    if owner is parent_context and res_id not in seen_ids:
                        res = batch_map[res_id]
                        final_results.append(
                            models.ScoredPoint(
                                id=res.id, payload=res.payload,
                                score=parent_context.score, version=0
                            )
                        )
                        seen_ids.add(res_id)

            for pt in list_pts:
                if pt.id not in seen_ids:
                    final_results.append(pt)
                    seen_ids.add(pt.id)

            return final_results

        elif reorder_mode == "append-last":
            # เรียง: [original results ทั้งหมด] → [ref laws ต่อท้าย]
            seen_ids = {pt.id for pt in list_pts}
            final_results = list(list_pts)

            for res_id, parent_context in ref_to_parent.items():
                if res_id not in seen_ids:
                    res = batch_map[res_id]
                    final_results.append(
                        models.ScoredPoint(
                            id=res.id, payload=res.payload,
                            score=parent_context.score, version=0
                        )
                    )
                    seen_ids.add(res_id)

            return final_results

    # Public API — returns LangChain Documents
    def retrieve(self, query: str) -> List[Document]:
        dense_vec, sparse_vec = self._encode_query(query)
        candidates = self._hybrid_search(dense_vec, sparse_vec)
        ranked = self._rerank(query, candidates)
        augmented_context = self._link_ref_law(ranked)

        docs = []
        for i, r in enumerate(augmented_context, start=1):
            p = r.payload
            docs.append(
                Document(
                    page_content=p.get("text", ""),
                    metadata={
                        "law_name": r.payload.get("law_name", ""),
                        "section_num": r.payload.get("section_num", ""),
                        "reference_laws": r.payload.get("reference_laws", []),
                        "rank": i,
                        "score": r.score,
                    },
                )
            )

        # Final Limit
        if self.final_limit and self.final_limit > 0:
            return docs[: self.final_limit]
        return docs

    # LangChain runnable interface
    def __call__(self, query: str) -> List[Document]:
        return self.retrieve(query)
