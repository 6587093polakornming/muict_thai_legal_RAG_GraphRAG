"""
Thai Legal CCL - RAG Querying Pipeline
======================================
Hybrid Search (Dense + Sparse + RRF Fusion + Reranker) with LangChain
"""

from __future__ import annotations
import time
import os
import torch
import os
from typing import List, Tuple

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, Fusion, FusionQuery

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# from langchain_openai import ChatOpenAI  # swap to langchain_anthropic, langchain_google_genai, etc.
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "thai_laws_collection"
EMBED_MODEL_NAME = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

VECTOR_DENSE = "dense"
VECTOR_SPARSE = "sparse"

RETRIEVAL_LIMIT = 3  # candidates sent to reranker
FINAL_LIMIT = 3  # top-k returned to LLM


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
# 4. Hybrid Retriever
# ---------------------------------------------------------------------------


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
        retrieval_limit: int = RETRIEVAL_LIMIT,
        final_limit: int = FINAL_LIMIT,
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
        retrieval_limit: int = RETRIEVAL_LIMIT,
        final_limit: int = FINAL_LIMIT,
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


# ---------------------------------------------------------------------------
# 8. Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("test-456")
    # --- Configure your LLM here ---
    # OpenAI
    # llm = ChatOllama(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(
        model_name="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นที่ท่านต้องการใช้
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.opentyphoon.ai/v1",  # สำคัญ: ใส่แทน base_url เดิม
        temperature=0,
        max_tokens=4096,
    )

    # Anthropic (uncomment to use)
    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # Google (uncomment to use)
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # --- Build RAG ---
    rag = ThaiLegalRAG(llm=llm, retrieval_limit=RETRIEVAL_LIMIT, final_limit=FINAL_LIMIT)

    # # --- Simple chat ---
    # question = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"
    # answer = rag.chat(question)
    # print(answer)

    # # --- Chat with sources ---
    # answer, sources = rag.chat_with_sources(question)
    # print("\n=== Sources ===")
    # for doc in sources:
    #     m = doc.metadata
    #     print(f"  [{m['law_name']} มาตรา {m['section_num']}] score={m['score']:.4f}")

    ### Create Loop input user and Conuting Time
    ### Use
    # answer = rag.chat(question)
    # print(answer)
    while True:
        # 1. รับ input จาก user
        question = input("\nคำถามของคุณ: ").strip()

        # เช็คเงื่อนไขเพื่อออกจาก Loop
        if question.lower() in ["exit", "quit", "ออก"]:
            print("ปิดระบบ... สวัสดีครับ")
            break

        if not question:
            continue

        # 2. เริ่มจับเวลา
        print("กำลังค้นหาและประมวลผลคำตอบ...")
        start_time = time.time()

        # 3. เรียกใช้งาน RAG (ใช้ chat_with_sources เพื่อให้เห็นคะแนนด้วย)
        answer, sources = rag.chat_with_sources(question)

        # 4. คำนวณเวลาที่ใช้
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 5. แสดงผลลัพธ์
        print("\n=== คำตอบจาก AI ===")
        print(answer)

        print("\n=== อ้างอิงจากกฎหมาย ===")
        for doc in sources:
            # print(f"Score: {doc.score:.4f} | {doc.payload['law_name']} มาตรา {doc.payload['section_num']}")
            # print(f"Text: {doc.payload['text'][:150]}...\n")
            print(f"Score: {doc.metadata["score"]:.4f} | {doc.metadata["law_name"]} มาตรา {doc.metadata["section_num"]}")
            print(doc.page_content)
            

        print(f"\n⏱️ ใช้เวลาประมวลผลทั้งสิ้น: {elapsed_time:.2f} วินาที")
        print("-" * 50)
