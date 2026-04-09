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
import tiktoken
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Adapter interface — implement one for each RAG system
# ---------------------------------------------------------------------------

class RAGAdapter:
    """
    Base class. Subclass this for HybridRAG and GraphRAG.
    Must implement: debug(query) -> (answer: str, docs: list, token: dict, time_elapsed: dict)
    """
    name: str = "base"

    # TODO change method to debug()
    def debug(self, query: str):
        """
        Returns:
            answer (str)  — LLM-generated answer
            docs   (list) — list of LangChain Document objects with metadata:
                            { law_name, section_num } 
            token (dict) — token usage from LLM callabck metadata:
                            {prompt_tokens, completion_tokens, total_tokens}
            time_elapsed (dict) -time Latency metadata:
                            {retrieve_time, llm_time, total_elapsed}
            
        """
        # TODO Exclude rank and score (because score is from reranking and rank is not use.)
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

        # TODO simple handle cold start
        test_query = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"
        self.rag.debug(test_query)

    # TODO change method to debug()
    def debug(self, query: str):
        results = self.rag.debug(query)
        answer:str = results.get("answer")
        docs:list = results.get("docs_candidates")
        token:dict = results.get("token")
        time_elapsed:dict = results.get("time_elapsed")
        return answer, docs, token, time_elapsed


class GraphRAGAdapter(RAGAdapter):
    """Adapter for GraphRAG — fill in your own import/init"""
    name = "graph"

    def __init__(self):
        from src.graph_rag.graphrag_retriever import LegalRetriever
        self.retriever = LegalRetriever()

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback if model name isn't recognized
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))


    def debug(self, query: str):
        from langchain_core.documents import Document
        retriever = self.retriever.get_retriever()

        retrieve_start = time.perf_counter()
        retrieved_docs = retriever.search(query_text=query, top_k=3)
        retrieve_end = time.perf_counter()
        retrieve_time = retrieve_end - retrieve_start

        start_llm = time.perf_counter()
        rag_response = self.retriever.get_answer(query)
        llm_time = time.perf_counter() - start_llm

        total_elapsed = retrieve_time + llm_time
        time_elapsed = {
            "retrieve_time": retrieve_time,
            "llm_time": llm_time,
            "total_elapsed": total_elapsed,
        }
        
        context_text = "\n".join([item.content for item in retrieved_docs.items])
        full_prompt = f"Context: {context_text}\n\nQuestion: {query}"
        p_tokens = self.count_tokens(full_prompt)
        c_tokens = self.count_tokens(rag_response.answer)

        token = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,  
            "total_tokens": p_tokens + c_tokens,       
        }

        docs = []
        rank = 1
        for item in retrieved_docs.items:
            meta = item.metadata or {}

            # Parent doc — score is a top-level attribute on RetrieverResultItem
            docs.append(Document(
                page_content=item.content,
                metadata={
                    "law_name":    meta.get("parent_law_name", ""),
                    "section_num": str(meta.get("parent_section_num", "")),
                }
            ))
            rank += 1

            # Children docs
            for child in meta.get("children", []):
                if not child.get("law_name") or not child.get("section_num"):
                    continue
                docs.append(Document(
                    page_content=child.get("text", ""),
                    metadata={
                        "law_name":    child["law_name"],
                        "section_num": str(child["section_num"]),
                    }
                ))
                rank += 1
        answer = rag_response.answer
        return answer, docs, token, time_elapsed


# ---------------------------------------------------------------------------
# Helper: serialise retrieved docs (only metadata, not full page_content)
# ---------------------------------------------------------------------------

def _serialise_docs(docs) -> list[dict]:
    """
    Extract only the metadata fields needed for metric computation.
    Avoids storing full page_content (which is already in the dataset).
    """
    result = []
    # Avoid duplicates (e.g. same section retrieved as parent and child in GraphRAG)
    seen_law = set()

    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        
        law_name = meta.get("law_name", "")
        section_num = str(meta.get("section_num", ""))
        key = (law_name, section_num)
        
        if key in seen_law:
            continue
        seen_law.add(key)

        result.append({
            "law_name":   meta.get("law_name", ""),
            "section_num": str(meta.get("section_num", "")),
            # TODO remove rank and score
            # "rank":        meta.get("rank", -1),
            # "score":       float(meta.get("score", 0.0)),
        })
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
                answer, docs, token, time_elapsed = adapter.debug(question)
            except Exception as e:
                # log error but continue so one bad row doesn't kill the run
                print(f"\n[ERROR] row {idx}: {e}")
                answer = "__ERROR__"
                docs = []
                token = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                time_elapsed = {"retrieve_time": 0.0, "llm_time": 0.0, "total_elapsed": 0.0}

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
                # TODO add token and time_elapsed metadata 
                "prompt_tokens":    token.get("prompt_tokens",0),
                "completion_tokens":    token.get("completion_tokens",0),
                "total_tokens":    token.get("total_tokens",0),

                "retrieve_time": time_elapsed.get("retrieve_time", 0.0),
                "llm_time": time_elapsed.get("llm_time", 0.0),
                "total_elapsed": time_elapsed.get("total_elapsed", 0.0),
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
