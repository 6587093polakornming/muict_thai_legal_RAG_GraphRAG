from __future__ import annotations
import torch
from typing import List, Tuple

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# from langchain_openai import ChatOpenAI  # swap to langchain_anthropic, langchain_google_genai, etc.
# from langchain_community.chat_models import ChatOllama
from .hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
EMBED_MODEL_NAME = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """คุณคือผู้ช่วยด้านกฎหมายไทย (Thai Legal AI Assistant) ที่มีความเชี่ยวชาญด้านกฎหมายแพ่งและพาณิชย์ (CCL)
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
# 3. Model & Client Initialization
# ---------------------------------------------------------------------------


def _init_device() -> Tuple[str, bool]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[RAG] Running on: {device.upper()}")
    return device, use_fp16


def build_models() -> Tuple[BGEM3FlagModel, FlagReranker, QdrantClient]:
    """Initialize embedding model, reranker, and Qdrant client."""
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
# 5. Context Formatter
# ---------------------------------------------------------------------------


def format_context(docs: List[Document]) -> str:
    """Convert retrieved Documents into a structured context string for the prompt."""
    if not docs:
        return "ไม่พบข้อมูลกฎหมายที่เกี่ยวข้อง"

    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{i}] {meta.get('law_name', 'ไม่ทราบชื่อกฎหมาย')} มาตรา {meta.get('section_num', '-')}"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# 6. RAG Chain Builder
# ---------------------------------------------------------------------------


def build_rag_chain(
    retriever: HybridRetriever,
    llm,
):
    """
    Build a LangChain LCEL chain:
        query -> retrieve -> format context -> prompt -> LLM -> answer
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    # Sub-chain: query -> retrieved docs -> formatted string
    retrieval_chain = (
        RunnableLambda(lambda x: x["question"])
        | RunnableLambda(retriever.retrieve)
        | RunnableLambda(format_context)
    )

    # Full chain with parallel context + question passthrough
    rag_chain = (
        RunnablePassthrough.assign(context=retrieval_chain)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ---------------------------------------------------------------------------
# 7. Main Chat Interface
# ---------------------------------------------------------------------------


class ThaiLegalRAG:
    """High-level chat interface for Thai Legal RAG."""

    def __init__(
        self,
        llm,
        retrieval_limit: int,
        final_limit: int,
    ) -> None:
        embed_model, reranker, client = build_models()

        self.retriever = HybridRetriever(
            embed_model=embed_model,
            reranker=reranker,
            client=client,
            retrieval_limit=retrieval_limit,
            final_limit=final_limit,
        )
        self.chain = build_rag_chain(self.retriever, llm)

    def chat(self, query: str) -> str:
        """
        Accept a user query and return the LLM-generated answer.

        Args:
            query: คำถามภาษาไทยเกี่ยวกับกฎหมาย

        Returns:
            คำตอบที่อ้างอิงจากฐานข้อมูลกฎหมาย
        """
        return self.chain.invoke({"question": query})

    def chat_with_sources(self, query: str) -> Tuple[str, List[Document]]:
            """
            แก้ไขให้เรียกใช้ self.chain หลัก เพื่อรักษาค่า max_tokens ที่ตั้งไว้
            """
            # 1. ดึงเอกสารจาก Retriever (ทำงานเหมือนเดิม)
            docs = self.retriever.retrieve(query)
            
            # 2. เรียกใช้ chain หลักที่ประกาศไว้ใน __init__ 
            # ตัวนี้จะวิ่งผ่าน retrieval_chain -> prompt -> llm (ที่ตั้ง max_tokens ไว้แล้ว)
            answer = self.chain.invoke({"question": query})
            
            return answer, docs

    def debug(self, query: str):
        # 1. Encode
        dense_vec, sparse_vec = self.retriever._encode(query)

        # 2. Retrieve (before rerank)
        candidates = self.retriever._search(dense_vec, sparse_vec)

        # 3. Rerank
        reranked = self.retriever._rerank(query, candidates)

        # 4. Convert to readable docs
        docs = []
        for r in reranked:
            p = r.payload
            docs.append(
                {
                    "text": p.get("text", ""),
                    "law_name": p.get("law_name", ""),
                    "section": p.get("section_num", ""),
                    "score": r.score,
                }
            )

        # 5. Build context
        context = format_context(
            [Document(page_content=d["text"], metadata=d) for d in docs]
        )

        # ❗ IMPORTANT: อย่าเรียก self.chat() (จะ retrieve ซ้ำ)
        answer = self.chain.invoke({"question": query, "context": context})

        return {
            "query": query,
            "num_candidates": len(candidates),
            "num_final": len(reranked),
            "reranked": docs,
            "context": context,
            "answer": answer,
        }