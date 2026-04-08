"""
eval_runner.py
==============
Step 1 — Run RAG system against test dataset and save results to JSONL.

Usage:
    python eval_runner.py --system hybrid --output results_hybrid.jsonl
    python eval_runner.py --system graph  --output results_graph.jsonl

Features:
    - Sequential execution (safe for rate-limited APIs e.g. Typhoon)
    - Checkpoint/resume: skips already-completed rows
    - Saves one JSON object per line (append mode)
    - Stores only essential metadata from retrieved docs (not full page_content)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Adapter interface — implement one for each RAG system
# ---------------------------------------------------------------------------

class RAGAdapter:
    """
    Base class. Subclass this for HybridRAG and GraphRAG.
    Must implement: chat_with_sources(query) -> (answer: str, docs: list)
    """
    name: str = "base"

    def chat_with_sources(self, query: str):
        """
        Returns:
            answer (str)  — LLM-generated answer
            docs   (list) — list of LangChain Document objects with metadata:
                            { law_name, section_num, rank, score }
        """
        raise NotImplementedError


class HybridRAGAdapter(RAGAdapter):
    """Adapter for HybridRAG (hybridrag_langhchain.py)"""
    name = "hybrid"

    def __init__(self):
        import os
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        # ---- import from your project ----
        from src.rag.config import RAGConfig
        from src.rag.hybridrag_langhchain import ThaiLegalRAG

        load_dotenv()

        llm = ChatOpenAI(
            model_name="typhoon-v2.5-30b-a3b-instruct",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://api.opentyphoon.ai/v1",
            temperature=0,
            max_tokens=23113,
        )
        config = RAGConfig(retrieval_limit=3, reranking_limit=3)
        self.rag = ThaiLegalRAG(llm=llm, config=config)

    def chat_with_sources(self, query: str):
        return self.rag.chat_with_sources(query)


class GraphRAGAdapter(RAGAdapter):
    """Adapter for GraphRAG — fill in your own import/init"""
    name = "graph"

    def __init__(self):
        # TODO: import and initialise your GraphRAG here
        # from src.rag.graphrag_langchain import ThaiLegalGraphRAG
        # self.rag = ThaiLegalGraphRAG(...)
        raise NotImplementedError("GraphRAG adapter not yet wired up")

    def chat_with_sources(self, query: str):
        return self.rag.chat_with_sources(query)


# ---------------------------------------------------------------------------
# Helper: serialise retrieved docs (only metadata, not full page_content)
# ---------------------------------------------------------------------------

def _serialise_docs(docs) -> list[dict]:
    """
    Extract only the metadata fields needed for metric computation.
    Avoids storing full page_content (which is already in the dataset).
    """
    result = []
    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        result.append({
            "law_name":   meta.get("law_name", ""),
            "section_num": str(meta.get("section_num", "")),
            "rank":        meta.get("rank", -1),
            "score":       float(meta.get("score", 0.0)),
        })
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    adapter: RAGAdapter,
    dataset_path: str,
    output_path: str,
    sleep_sec: float = 0.5,
):
    """
    Iterate over every row in the test dataset, call the RAG system,
    and append results to a JSONL file.

    Checkpointing: rows whose `id` already exist in the output file
    are skipped so the run can be safely interrupted and resumed.
    """
    df = pd.read_parquet(dataset_path)
    output_file = Path(output_path)

    # --- load already-completed ids (checkpoint) ---
    done_ids: set[int] = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add(obj["id"])
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
                answer, docs = adapter.chat_with_sources(question)
            except Exception as e:
                # log error but continue so one bad row doesn't kill the run
                print(f"\n[ERROR] row {idx}: {e}")
                answer = "__ERROR__"
                docs = []

            record = {
                "id":               idx,
                "system":           adapter.name,
                "question":         question,
                # Ground truth — generation
                "gt_answer":        row["answer"],
                "gt_reference_answer": row["reference_answer"],
                # Ground truth — retrieval
                "gt_relevant_laws": [
                    {"law_name": r["law_name"], "section_num": str(r["section_num"])}
                    for r in row["relevant_laws"]
                ],
                "gt_reference_laws": [
                    {"law_name": r["law_name"], "section_num": str(r["section_num"])}
                    for r in row["reference_laws"]
                ],
                # RAG output
                "rag_answer":       answer,
                "retrieved_docs":   _serialise_docs(docs),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # write to disk immediately

            time.sleep(sleep_sec)

    print(f"\n[done] results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--system",   choices=["hybrid", "graph"], required=True)
    parser.add_argument("--dataset",  default="data/tests/test_dataset_2026-04-01_filter.parquet")
    parser.add_argument("--output",   default=None)
    parser.add_argument("--sleep",    type=float, default=0.0,
                        help="Seconds to sleep between API calls (rate-limit buffer)")
    args = parser.parse_args()

    if args.output is None:
        args.output = "data/evaluation/"+f"results_{args.system}.jsonl"

    if args.system == "hybrid":
        adapter = HybridRAGAdapter()
    else:
        adapter = GraphRAGAdapter()

    run_evaluation(
        adapter=adapter,
        dataset_path=args.dataset,
        output_path=args.output,
        sleep_sec=args.sleep,
    )
