from __future__ import annotations
import torch
import time
from typing import List, Tuple, Optional, Dict, Any

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from .hybrid_retriever import HybridRetriever


# 1. Configuration
from .config import RAGConfig


# 2. System Prompt
SYSTEM_PROMPT_TEMPLATE = """คุณคือผู้ช่วยด้านกฎหมายไทย (Thai Legal AI Assistant) ที่มีความเชี่ยวชาญด้านกฎหมายแพ่งและพาณิชย์ (CCL)
คุณจะตอบคำถามโดยอิงจากข้อความกฎหมายที่ถูกดึงมาเท่านั้น ห้ามอนุมานหรือแต่งเติมข้อมูลที่ไม่มีในบริบท

## กฎการตอบ
1. อ้างอิง **ชื่อกฎหมาย** และ **มาตรา** ที่เกี่ยวข้องทุกครั้ง
2. บริบทถูกจัดเรียงตาม score จากสูงไปต่ำ — กฎหมายที่มี score สูงสุดคือที่เกี่ยวข้องมากที่สุด ให้ใช้เป็นหลักในการตอบ
3. บริบทอาจมีกฎหมายหลายมาตราจากหลายฉบับ หากกฎหมายเหล่านั้นมี score สูงเท่ากัน หรืออยู่ในลำดับต้นๆ ของบริบท ให้พิจารณาร่วมกัน เพราะอาจเป็นกฎหมายที่อ้างอิงถึงกัน โดยอาจเป็นข้อยกเว้น เงื่อนไข หรือบทบัญญัติที่เสริมกัน
4. บริบทไม่อ้างอิงกฎหมายฉบับเก่า หรือ เมื่อพบบริบทกฎหมายฉบับเดียวกันแต่ต่างปี ให้เลือกกฎหมายฉบับล่าสุด สังเกตจาก law_name หรือ metadata
5. หากบริบทไม่มีข้อมูลเพียงพอ ให้แจ้งว่า "ไม่พบข้อมูลที่ตรงกับคำถามในฐานข้อมูลกฎหมายที่มีอยู่"
6. ตอบเป็นภาษาไทยที่ชัดเจน กระชับ และเข้าใจง่าย
7. หากมีโทษทางอาญา ให้ระบุอัตราโทษอย่างครบถ้วน (จำคุก / ปรับ / ทั้งจำทั้งปรับ)

## รูปแบบการตอบ
[ตอบโดยตรง 1-2 ประโยค เป็นการสรุปคำตอบ]

---
## บริบทจากฐานข้อมูลกฎหมาย
{context}
"""


# 3. Model Initialization
def _init_device() -> Tuple[str, bool]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[RAG] Running on: {device.upper()}")
    return device, use_fp16


def build_models(config: RAGConfig):
    device, use_fp16 = _init_device()

    embed_model = BGEM3FlagModel(
        config.embed_model_name,
        use_fp16=use_fp16,
        batch_size=16,
        device=device,
    )
    reranker = FlagReranker(
        config.rerank_model_name,
        use_fp16=True,
        devices=device,
    )
    qdrant_client = QdrantClient(url=config.qdrant_url)

    return embed_model, reranker, qdrant_client


# 4. Context Formatter
def format_context(docs: List[Document]) -> str:
    if not docs:
        return "ไม่พบข้อมูลกฎหมายที่เกี่ยวข้อง"
    parts = [
        f"Rank [{i}] Score:{doc.metadata.get('score')} ชื่อกฎหมาย:{doc.metadata.get('law_name', 'ไม่ทราบชื่อกฎหมาย')} มาตรา {doc.metadata.get('section_num', '-')}\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    ]
    return "\n\n".join(parts)


# Helper Convert Qdrant Point to Doc Langchain Stardard for Evaluation
def convert_to_context_doc(context: List):
    docs = [
        Document(
            page_content=r.payload.get("text", ""),
            metadata={
                "law_name": r.payload.get("law_name", ""),
                "section_num": r.payload.get("section_num", ""),
                "reference_laws": r.payload.get("reference_laws", []),
                "rank": i,  # Optional
                "score": r.score,  # Optional
            },
        )
        for i, r in enumerate(context, start=1)
    ]
    return docs


# 5. Core LLM Call
def _call_llm(llm, query: str, docs: List[Document]) -> Tuple[str, Dict[str, Any]]:
    """
    สร้าง prompt จาก docs ที่ retrieve มาแล้ว แล้วเรียก LLM โดยตรง
    """
    context = format_context(docs)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [SystemMessage(content=system_content), HumanMessage(content=query)]
    response = llm.invoke(messages)
    token_usage = response.response_metadata.get("token_usage", {})

    return response.content, token_usage


# 6. ThaiLegalRAG — Main Interface
class ThaiLegalRAG:
    """High-level chat interface for Thai Legal RAG."""

    def __init__(self, llm, config: RAGConfig = None) -> None:
        if config is None:
            config = RAGConfig()  # ใช้ค่า default ถ้าไม่ส่งมา

        embed_model, reranker, client = build_models(config)
        self.llm = llm
        self.retriever = HybridRetriever(
            embed_model=embed_model,
            reranker=reranker,
            client=client,
            config=config,  # ส่ง config object ตัวเดียว
        )

    # Public API
    def chat(self, query: str) -> str:
        """รับ query → คืน answer (ไม่คืน sources)"""
        docs = self.retriever.retrieve(query)
        answer, _ = _call_llm(self.llm, query, docs)
        return answer

    def chat_with_sources(self, query: str) -> Tuple[str, List[Document]]:
        """
        รับ query → คืน (answer, docs)
        retrieve เพียงครั้งเดียว ไม่เรียก reranker ซ้ำ
        """
        docs = self.retriever.retrieve(query)
        answer, _ = _call_llm(self.llm, query, docs)
        return answer, docs

    def debug(self, query: str) -> dict:
        """แสดง intermediate results ทุก step สำหรับ debugging"""
        start_retrieve = time.perf_counter()
        dense_vec, sparse_vec = self.retriever._encode_query(query)
        candidates = self.retriever._hybrid_search(dense_vec, sparse_vec)
        reranked_pts = self.retriever._rerank(query, candidates)
        augmented_context = self.retriever._link_ref_law(reranked_pts)

        docs = convert_to_context_doc(augmented_context)

        # Final Limits
        if self.retriever.final_limit and self.retriever.final_limit > 0:
            docs = docs[: self.retriever.final_limit]
        retrieve_time = time.perf_counter() - start_retrieve

        context = format_context(docs)
        start_llm = time.perf_counter()
        answer, token_usage = _call_llm(self.llm, query, docs)
        llm_time = time.perf_counter() - start_llm

        total_elapsed = retrieve_time + llm_time
        time_elapsed = {
            "retrieve_time": retrieve_time,
            "llm_time": llm_time,
            "total_elapsed": total_elapsed,
        }

        return {
            "query": query,
            "num_candidates": len(candidates),
            "rerank_candidates": len(reranked_pts),
            "final": len(docs),
            "docs_candidates": docs,
            "context": context,
            "answer": answer,
            "token": token_usage,
            "time_elapsed": time_elapsed,
        }

    def debug_custom_pipeline(self, pipeline_name: str, query: str) -> dict:
        # Pipeline
        start_retrieve = time.perf_counter()
        # vector search
        print(f"Calling {pipeline_name} pipeline")
        dense_vec, _ = self.retriever._encode_query(
            query, is_return_dense=True, is_return_sparse=False
        )
        candidates = self.retriever._vector_search(dense_vec)

        len_candidates = len(candidates)

        # convert qdrant points to langchain doc
        docs = [
            Document(
                page_content=r.payload.get("text", ""),
                metadata={
                    "law_name": r.payload.get("law_name", ""),
                    "section_num": r.payload.get("section_num", ""),
                    "reference_laws": r.payload.get("reference_laws", []),
                    "rank": i + 1,
                    "score": len_candidates - i,  # Manual Score with length element
                },
            )
            for i, r in enumerate(candidates, start=0)
        ]

        # Final Limits
        if self.retriever.final_limit and self.retriever.final_limit > 0:
            docs = docs[: self.retriever.final_limit]
        retrieve_time = time.perf_counter() - start_retrieve

        # Format Context
        context = format_context(docs)

        # Call LLM
        start_llm = time.perf_counter()
        answer, token_usage = _call_llm(self.llm, query, docs)
        llm_time = time.perf_counter() - start_llm
        total_elapsed = retrieve_time + llm_time
        time_elapsed = {
            "retrieve_time": retrieve_time,
            "llm_time": llm_time,
            "total_elapsed": total_elapsed,
        }

        return {
            "query": query,
            "num_candidates": len(candidates),
            "final": len(docs),
            "docs_candidates": docs,
            "context": context,
            "answer": answer,
            "token": token_usage,
            "time_elapsed": time_elapsed,
        }
