from __future__ import annotations
from typing import List, Tuple

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import SparseVector, Prefetch, Fusion, FusionQuery
from langchain_core.documents import Document
from typing import Optional

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
        retrieval_limit: int,
        reranking_limit: int,
        final_limit: Optional[int] = None, # กำหนดค่า Default เป็น None ตรงนี้,
    ) -> None:
        self.embed_model = embed_model
        self.reranker = reranker
        self.client = client
        self.retrieval_limit = retrieval_limit
        self.reranking_limit = reranking_limit
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
        return candidates[: self.reranking_limit]

    def _ref_law_link(self, list_pts: list) -> list:
        if not list_pts:
            return []

        first_context = list_pts[0]
        query_lst = first_context.payload.get("reference_laws", [])

        # ถ้าไม่มีกฎหมายอ้างอิง ให้จัดการเรื่อง limit แล้วคืนค่าเลย
        if not query_lst:
            return list_pts[: self.final_limit] if self.final_limit else list_pts

        # 1. เตรียม Filter สำหรับ Batch Query
        should_conditions = [
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_name", match=models.MatchValue(value=ref["law_name"])
                    ),
                    models.FieldCondition(
                        key="section_num",
                        match=models.MatchValue(value=ref["section_num"]),
                    ),
                ]
            )
            for ref in query_lst
        ]

        # 2. ดึงข้อมูลกฎหมายอ้างอิง
        batch_results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(should=should_conditions),
            limit=len(query_lst),
            with_payload=True,
            with_vectors=False,
        )

        # แปลงเป็น ScoredPoint
        ref_scored_pts = [
            models.ScoredPoint(
                id=res.id, payload=res.payload, score=first_context.score, version=0
            )
            for res in batch_results
        ]

        # 3. รวมผลลัพธ์และลบตัวซ้ำ (Deduplicate) โดยใช้ ID เป็นเกณฑ์
        seen_ids = {first_context.id}  # เริ่มต้นด้วย ID ของตัวแรก
        final_results = [first_context]

        # เพิ่มกฎหมายอ้างอิง (ถ้ายังไม่มีใน list)
        for pt in ref_scored_pts:
            if pt.id not in seen_ids:
                final_results.append(pt)
                seen_ids.add(pt.id)

        # เพิ่มมาตราอื่น ๆ จากการ Search เดิม (ถ้ายังไม่มีใน list)
        for pt in list_pts[1:]:
            if pt.id not in seen_ids:
                final_results.append(pt)
                seen_ids.add(pt.id)
                
        return final_results

    # ------------------------------------------------------------------
    # Public API — returns LangChain Documents
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        dense_vec, sparse_vec = self._encode_query(query)
        candidates = self._hybrid_search(dense_vec, sparse_vec)
        ranked = self._rerank(query, candidates)
        augmented_context = self._ref_law_link(ranked)

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
