from __future__ import annotations
import torch
from typing import List, Tuple

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from .hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

QDRANT_URL       = "http://localhost:6333"
EMBED_MODEL_NAME = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3"
RERANK_MODEL     = "BAAI/bge-reranker-v2-m3"

# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """คุณคือผู้ช่วยด้านกฎหมายไทย (Thai Legal AI Assistant) ที่มีความเชี่ยวชาญด้านกฎหมายแพ่งและพาณิชย์ (CCL)
คุณจะตอบคำถามโดยอิงจากข้อความกฎหมายที่ถูกดึงมาเท่านั้น ห้ามอนุมานหรือแต่งเติมข้อมูลที่ไม่มีในบริบท

## กฎการตอบ
1. อ้างอิง **ชื่อกฎหมาย** และ **มาตรา** ที่เกี่ยวข้องทุกครั้ง
2. หากบริบทไม่มีข้อมูลเพียงพอ ให้แจ้งว่า "ไม่พบข้อมูลที่ตรงกับคำถามในฐานข้อมูลกฎหมายที่มีอยู่"
3. ตอบเป็นภาษาไทยที่ชัดเจน กระชับ และเข้าใจง่าย
4. หากมีโทษทางอาญา ให้ระบุอัตราโทษอย่างครบถ้วน (จำคุก / ปรับ / ทั้งจำทั้งปรับ)

## รูปแบบการตอบ
**สรุปคำตอบ:** [ตอบโดยตรง 1-2 ประโยค]

---
## บริบทจากฐานข้อมูลกฎหมาย
{context}
"""

# ---------------------------------------------------------------------------
# 3. Model Initialization
# ---------------------------------------------------------------------------

def _init_device() -> Tuple[str, bool]:
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[RAG] Running on: {device.upper()}")
    return device, use_fp16


def build_models() -> Tuple[BGEM3FlagModel, FlagReranker, QdrantClient]:
    device, use_fp16 = _init_device()

    embed_model = BGEM3FlagModel(
        EMBED_MODEL_NAME,
        use_fp16=use_fp16,
        batch_size=16,
        device=device,
    )
    reranker = FlagReranker(
        RERANK_MODEL,
        use_fp16=True,
        devices=device,
    )
    qdrant_client = QdrantClient(url=QDRANT_URL)

    return embed_model, reranker, qdrant_client

# ---------------------------------------------------------------------------
# 4. Context Formatter
# ---------------------------------------------------------------------------

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "ไม่พบข้อมูลกฎหมายที่เกี่ยวข้อง"
    parts = [
        f"[{i}] {doc.metadata.get('law_name', 'ไม่ทราบชื่อกฎหมาย')} มาตรา {doc.metadata.get('section_num', '-')}\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    ]
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# 5. Core LLM Call  (ไม่ใช้ LCEL chain — เรียก LLM โดยตรง)
# ---------------------------------------------------------------------------

def _call_llm(llm, query: str, docs: List[Document]) -> str:
    """
    สร้าง prompt จาก docs ที่ retrieve มาแล้ว แล้วเรียก LLM โดยตรง
    ไม่มี overhead ของ LCEL chain / RunnableLambda
    """
    context        = format_context(docs)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages       = [SystemMessage(content=system_content), HumanMessage(content=query)]
    response       = llm.invoke(messages)
    return response.content

# ---------------------------------------------------------------------------
# 6. ThaiLegalRAG — Main Interface
# ---------------------------------------------------------------------------

class ThaiLegalRAG:
    """High-level chat interface for Thai Legal RAG."""

    def __init__(self, llm, retrieval_limit: int, final_limit: int) -> None:
        embed_model, reranker, client = build_models()

        self.llm = llm
        self.retriever = HybridRetriever(
            embed_model=embed_model,
            reranker=reranker,
            client=client,
            retrieval_limit=retrieval_limit,
            final_limit=final_limit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, query: str) -> str:
        """รับ query → คืน answer (ไม่คืน sources)"""
        docs = self.retriever.retrieve(query)
        return _call_llm(self.llm, query, docs)

    def chat_with_sources(self, query: str) -> Tuple[str, List[Document]]:
        """
        รับ query → คืน (answer, docs)
        retrieve เพียงครั้งเดียว ไม่เรียก reranker ซ้ำ
        """
        docs   = self.retriever.retrieve(query)       # retrieve ครั้งเดียว
        answer = _call_llm(self.llm, query, docs)     # ส่ง docs ที่มีอยู่แล้วให้ LLM
        return answer, docs

    def debug(self, query: str) -> dict:
        """แสดง intermediate results ทุก step สำหรับ debugging"""
        dense_vec, sparse_vec = self.retriever._encode_query(query)
        candidates            = self.retriever._hybrid_search(dense_vec, sparse_vec)
        reranked_pts          = self.retriever._rerank(query, candidates)

        docs = [
            Document(
                page_content=r.payload.get("text", ""),
                metadata={
                    "law_name":    r.payload.get("law_name", ""),
                    "section_num": r.payload.get("section_num", ""),
                    "score":       r.score,
                },
            )
            for r in reranked_pts
        ]

        context = format_context(docs)
        answer  = _call_llm(self.llm, query, docs)

        return {
            "query":          query,
            "num_candidates": len(candidates),
            "num_final":      len(reranked_pts),
            "docs":           docs,
            "context":        context,
            "answer":         answer,
        }