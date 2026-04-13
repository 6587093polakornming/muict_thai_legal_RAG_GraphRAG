"""
eval_runner.py
==============
Step 1 — Run RAG system against test dataset and save results to JSONL.

Usage:
    # Retrieval only (no LLM) — experiment expansion_mode
    python eval_runner.py --system retrieval_top1
    python eval_runner.py --system retrieval_topn

    # Full pipeline experiments
    python eval_runner.py --system vector_rag
    python eval_runner.py --system sparse_rag
    python eval_runner.py --system hybrid_rag
    python eval_runner.py --system hybrid_rerank
    python eval_runner.py --system hybrid_ref_top1
    python eval_runner.py --system hybrid_ref_topn
    python eval_runner.py --system graph

    # Custom output path
    python eval_runner.py --system hybrid_ref_top1 --output data/evaluation/my_results.jsonl

Features:
    - Sequential execution (safe for rate-limited APIs e.g. Typhoon)
    - Checkpoint/resume: skips already-completed rows
    - Saves one JSON object per line (append mode)
    - Stores only essential metadata from retrieved docs (not full page_content)

Experiment Variants:
    System Name         Pipeline
    ------------------- ----------------------------------------------------------
    retrieval_top1      hybrid + rerank + ref(top-1)  — retrieval only, no LLM
    retrieval_topn      hybrid + rerank + ref(top-n)  — retrieval only, no LLM
    vector_rag          dense vector search            — full pipeline
    sparse_rag          sparse keyword search          — full pipeline
    hybrid_rag          hybrid RRF fusion              — full pipeline
    hybrid_rerank       hybrid + reranker              — full pipeline
    hybrid_ref_top1     hybrid + rerank + ref(top-1)  — full pipeline
    hybrid_ref_topn     hybrid + rerank + ref(top-n)  — full pipeline
    graph               GraphRAG                       — full pipeline

    # TODO [] add reorder_mode experiment (parent-first vs append-last) ถ้าต้องการ
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from tqdm import tqdm

from src.rag.hybridrag_langhchain import ThaiLegalRAG, build_models
from src.rag.config import RAGConfig
from src.rag.hybrid_retriever import HybridRetriever

load_dotenv()

# ---------------------------------------------------------------------------
# LLM / Model factory — shared across adapters เพื่อไม่โหลด model ซ้ำ
# ---------------------------------------------------------------------------


def _build_llm() -> ChatOpenAI:
    """สร้าง LLM instance สำหรับ full pipeline adapters"""
    return ChatOpenAI(
        model_name="typhoon-v2.5-30b-a3b-instruct",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.opentyphoon.ai/v1",
        temperature=0,
        max_tokens=16384,
    )


def _build_config() -> RAGConfig:
    return RAGConfig(retrieval_limit=3, reranking_limit=3)


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------


class RAGAdapter:
    """
    Base class — subclass ต้อง implement debug()
    Returns: (answer, docs, token, time_elapsed)
    """

    name: str = "base"

    def debug(self, query: str):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Retrieval-only Adapters (ไม่มี LLM) — ใช้วัด retrieval metrics เท่านั้น
# ---------------------------------------------------------------------------


class RetrievalOnlyHybridAdapter(RAGAdapter):
    """
    Hybrid + rerank + ref expansion — ข้าม LLM
    ใช้เปรียบเทียบ expansion_mode: top-1 vs top-n

    # TODO [] เพิ่ม reorder_mode parameter ถ้าต้องการ experiment parent-first vs append-last
    """

    def __init__(
        self, expansion_mode: str = "top-1", reorder_mode: str = "parent-first"
    ):
        self.name = f"retrieval_{expansion_mode.replace('-', '')}"  # retrieval_top1 / retrieval_topn
        self.expansion_mode = expansion_mode
        self.reorder_mode = reorder_mode

        config = _build_config()
        embed_model, reranker, client = build_models(config)
        self.retriever = HybridRetriever(
            embed_model=embed_model,
            reranker=reranker,
            client=client,
            config=config,
        )

    def debug(self, query: str):
        t0 = time.perf_counter()
        docs = self.retriever.retrieve(
            query=query,
            expansion_mode=self.expansion_mode,
            reorder_mode=self.reorder_mode,
        )
        retrieve_time = time.perf_counter() - t0

        return (
            "",  # answer — ไม่มี LLM
            docs,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            {
                "retrieve_time": retrieve_time,
                "llm_time": 0.0,
                "total_elapsed": retrieve_time,
            },
        )


# ---------------------------------------------------------------------------
# Full Pipeline Adapters
# ---------------------------------------------------------------------------


class _BaseThaiLegalRAGAdapter(RAGAdapter):
    """
    Base สำหรับ adapter ที่ใช้ ThaiLegalRAG
    subclass กำหนด name และ override _call_pipeline()
    """

    def __init__(self):
        llm = _build_llm()
        config = _build_config()
        self.rag = ThaiLegalRAG(llm=llm, config=config)
        self._warmup()

    def _warmup(self):
        """Cold start — โหลด model ก่อนรัน eval จริง"""
        test_query = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"
        self._call_pipeline(test_query)

    def _call_pipeline(self, query: str) -> dict:
        raise NotImplementedError

    def debug(self, query: str):
        results = self._call_pipeline(query)
        return (
            results.get("answer"),
            results.get("docs_candidates"),
            results.get("token"),
            results.get("time_elapsed"),
        )


class VectorRAGAdapter(_BaseThaiLegalRAGAdapter):
    """Dense vector search — full pipeline"""

    name = "vector_rag"

    def _call_pipeline(self, query: str) -> dict:
        return self.rag.dense_rag(query)


class SparseRAGAdapter(_BaseThaiLegalRAGAdapter):
    """Sparse keyword search — full pipeline
    # TODO [] ตรวจสอบ sparse_rag ใน hybridrag_langhchain ว่า stable แล้ว (marked TODO Test & Review)
    """

    name = "sparse_rag"

    def _call_pipeline(self, query: str) -> dict:
        return self.rag.sparse_rag(query)


class HybridRAGAdapter(_BaseThaiLegalRAGAdapter):
    """Hybrid RRF fusion (ไม่มี reranker) — full pipeline"""

    name = "hybrid_rag"

    def _call_pipeline(self, query: str) -> dict:
        return self.rag.hybrid_rag(query)


class HybridRerankAdapter(_BaseThaiLegalRAGAdapter):
    """Hybrid + reranker (ไม่มี ref expansion) — full pipeline"""

    name = "hybrid_rerank"

    def _call_pipeline(self, query: str) -> dict:
        return self.rag.hybrid_rerank_rag(query)


class HybridRefAdapter(_BaseThaiLegalRAGAdapter):
    """
    Hybrid + rerank + ref expansion — full pipeline
    expansion_mode: top-1 (default) หรือ top-n

    # TODO [] เพิ่ม reorder_mode parameter ถ้าต้องการ experiment parent-first vs append-last
    """

    def __init__(
        self, expansion_mode: str = "top-1", reorder_mode: str = "parent-first"
    ):
        self.expansion_mode = expansion_mode
        self.reorder_mode = reorder_mode
        self.name = f"hybrid_ref_{expansion_mode.replace('-', '')}_{reorder_mode.replace('-', '')}"  #
        super().__init__()

    def _call_pipeline(self, query: str) -> dict:
        return self.rag.hybrid_ref_rag(
            query=query,
            expansion_mode=self.expansion_mode,
            reorder_mode=self.reorder_mode,
        )


class GraphRAGAdapter(RAGAdapter):
    """GraphRAG — full pipeline
    # TODO [] ตรวจสอบ GraphRAGRetriever interface ให้ตรงกับ debug() ก่อนรัน
    """

    name = "graph"

    def __init__(self):
        from src.graph_rag.graphrag_retriever import GraphRAGRetriever

        llm = ChatOpenAI(
            model_name="typhoon-v2.5-30b-a3b-instruct",
            openai_api_key=os.getenv("thai_llm_API_key"),
            openai_api_base="https://api.opentyphoon.ai/v1",
            temperature=0,
            max_tokens=16384,
        )
        self.retriever = GraphRAGRetriever(llm=llm, top_k=3)

    def debug(self, query: str):
        results = self.retriever.debug(query=query)
        return (
            results.get("answer"),
            results.get("docs_candidates"),
            results.get("token"),
            results.get("time_elapsed"),
        )


# ---------------------------------------------------------------------------
# Adapter registry — map system name → adapter instance
# ---------------------------------------------------------------------------

# TODO [] เพิ่ม entry ใหม่ตรงนี้เมื่อมี adapter เพิ่ม
ADAPTER_REGISTRY: dict[str, RAGAdapter] = {
    # Retrieval only
    "retrieval_top1": lambda: RetrievalOnlyHybridAdapter(expansion_mode="top-1"),
    "retrieval_topn": lambda: RetrievalOnlyHybridAdapter(expansion_mode="top-n"),
    # Full pipeline
    "vector_rag": lambda: VectorRAGAdapter(),
    "sparse_rag": lambda: SparseRAGAdapter(),
    "hybrid_rag": lambda: HybridRAGAdapter(),
    "hybrid_rerank": lambda: HybridRerankAdapter(),
    "hybrid_ref_top1_parent": lambda: HybridRefAdapter(
        expansion_mode="top-1", reorder_mode="parent-first"
    ),
    "hybrid_ref_top1_append": lambda: HybridRefAdapter(
        expansion_mode="top-1", reorder_mode="append-last"
    ),
    "hybrid_ref_topn_parent": lambda: HybridRefAdapter(
        expansion_mode="top-n", reorder_mode="parent-first"
    ),
    "hybrid_ref_topn_append": lambda: HybridRefAdapter(
        expansion_mode="top-n", reorder_mode="append-last"
    ),
    "graph": lambda: GraphRAGAdapter(),
}

# ---------------------------------------------------------------------------
# Helper: serialise retrieved docs
# ---------------------------------------------------------------------------


def _serialise_docs(docs) -> list[dict]:
    """
    Extract only metadata fields needed for metric computation.
    Avoids storing full page_content.
    """
    result = []
    seen_law = set()

    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}

        law_name = meta.get("law_name", "")
        section_num = str(meta.get("section_num", ""))
        key = (law_name, section_num)

        if key in seen_law:
            continue
        seen_law.add(key)
        result.append({"law_name": law_name, "section_num": section_num})

        for ref in meta.get("reference_laws", []):
            ref_key = (ref.get("law_name", ""), str(ref.get("section_num", "")))
            if ref_key in seen_law:
                continue
            seen_law.add(ref_key)
            result.append(
                {
                    "law_name": ref.get("law_name", ""),
                    "section_num": str(ref.get("section_num", "")),
                }
            )

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluation(
    adapter: RAGAdapter,
    dataset_path: str,
    output_path: str,
    sleep_sec: float = 0.0,
):
    """
    Iterate over dataset, call RAG system, append results to JSONL.
    Checkpoint/resume: skips rows whose id already exists in output file.
    """
    df = pd.read_parquet(dataset_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- checkpoint ---
    done_ids: set[int] = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass
        print(f"[resume] found {len(done_ids)} completed rows, skipping them.")

    # --- main loop ---
    with output_file.open("a", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=adapter.name):
            if idx in done_ids:
                continue

            question = row["question"]
            try:
                answer, docs, token, time_elapsed = adapter.debug(question)
            except Exception as e:
                print(f"\n[ERROR] row {idx}: {e}")
                answer = "__ERROR__"
                docs = []
                token = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time_elapsed = {
                    "retrieve_time": 0.0,
                    "llm_time": 0.0,
                    "total_elapsed": 0.0,
                }

            record = {
                "id": idx,
                "system": adapter.name,
                "question": question,
                "gt_answer": row["answer"],
                "gt_reference_answer": row["reference_answer"],
                "gt_relevant_laws": [
                    {"law_name": r["law_name"], "section_num": str(r["section_num"])}
                    for r in row["relevant_laws"]
                ],
                "gt_reference_laws": [
                    {"law_name": r["law_name"], "section_num": str(r["section_num"])}
                    for r in row["reference_laws"]
                ],
                "rag_answer": answer,
                "retrieved_docs": _serialise_docs(docs),
                "prompt_tokens": token.get("prompt_tokens", 0),
                "completion_tokens": token.get("completion_tokens", 0),
                "total_tokens": token.get("total_tokens", 0),
                "retrieve_time": time_elapsed.get("retrieve_time", 0.0),
                "llm_time": time_elapsed.get("llm_time", 0.0),
                "total_elapsed": time_elapsed.get("total_elapsed", 0.0),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(sleep_sec)

    print(f"\n[done] results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--system",
        choices=list(ADAPTER_REGISTRY.keys()),
        required=True,
        help=(
            "Experiment system to run:\n"
            "  retrieval_top1  — hybrid+rerank+ref(top-1), retrieval only\n"
            "  retrieval_topn  — hybrid+rerank+ref(top-n), retrieval only\n"
            "  vector_rag      — dense vector, full pipeline\n"
            "  sparse_rag      — sparse keyword, full pipeline\n"
            "  hybrid_rag      — hybrid RRF, full pipeline\n"
            "  hybrid_rerank   — hybrid+rerank, full pipeline\n"
            "  hybrid_ref_top1_parent — hybrid+rerank+ref(top-1)+reorder(parent), full pipeline\n"
            "  hybrid_ref_top1_append — hybrid+rerank+ref(top-1)+reorder(append), full pipeline\n"
            "  hybrid_ref_topn_parent — hybrid+rerank+ref(top-n)+reorder(parent), full pipeline\n"
            "  hybrid_ref_topn_append — hybrid+rerank+ref(top-n)+reorder(append), full pipeline\n"
            "  graph           — GraphRAG, full pipeline\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="data/tests/test_dataset_2026-04-01_filter.parquet",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: data/evaluation/results_{system}.jsonl)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between rows (rate-limit buffer)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/evaluation/results_{args.system}.jsonl"

    adapter = ADAPTER_REGISTRY[args.system]()

    run_evaluation(
        adapter=adapter,
        dataset_path=args.dataset,
        output_path=args.output,
        sleep_sec=args.sleep,
    )

    # # Retrieval only
    # python eval_runner.py --system retrieval_top1
    # python eval_runner.py --system retrieval_topn

    # # Full pipeline ทั้งหมด
    # python eval_runner.py --system vector_rag
    # python eval_runner.py --system hybrid_ref_top1
