"""
eval_metrics.py
===============
Step 2 — Load JSONL result files and compute all evaluation metrics.

Usage:
    # Single system
    python eval_metrics.py --input results_hybrid.jsonl

    # Compare two systems
    python eval_metrics.py --input results_hybrid.jsonl results_graph.jsonl --compare

Metrics computed:
    Retrieval Layer
        - Top-1 Accuracy      : relevant_laws[0] == retrieved_docs[0]
        - Hit Rate (Recall@K) : relevant_laws[0] in retrieved_docs (any rank)
        - Multi-Hit Rate      : fraction of reference_laws found in retrieved_docs ???
        - MRR                 : 1 / rank of first relevant_laws[0] found 

    Generation Layer
        - Exact Match         : rag_answer == gt_reference_answer (normalised)
        - ROUGE-1 Recall      : unigram recall vs gt_reference_answer
        - ROUGE-L Recall      : LCS recall vs gt_reference_answer
        - BERTScore Recall    : semantic recall vs gt_answer (expert version)

Requirements:
    pip install rouge-score bert-score pandas tqdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Lazy imports — only loaded when needed
# ---------------------------------------------------------------------------

def _get_rouge():
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer
    except ImportError:
        raise ImportError("pip install rouge-score")


def _get_bertscore():
    try:
        from bert_score import BERTScorer
        return BERTScorer
    except ImportError:
        raise ImportError("pip install bert-score")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Strip whitespace and collapse multiple spaces."""
    return " ".join(text.strip().split())


# ---------------------------------------------------------------------------
# Retrieval metrics (per row)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(record: dict) -> dict:
    """
    Returns a dict with retrieval metric values for a single record.

    Key definitions
    ---------------
    gt_main   : (law_name, section_num) of the single relevant_laws entry
    retrieved : ordered list of (law_name, section_num) from retrieved_docs
    gt_refs   : set of (law_name, section_num) from reference_laws
    """
    retrieved = [
        (d["law_name"], str(d["section_num"]))
        for d in record["retrieved_docs"]
    ]
    retrieved_set = set(retrieved)

    gt_relevant = record["gt_relevant_laws"]
    gt_refs     = record["gt_reference_laws"]

    # --- Top-1 Accuracy ---
    # relevant_laws always has exactly 1 entry (confirmed from dataset analysis)
    gt_main = (gt_relevant[0]["law_name"], str(gt_relevant[0]["section_num"]))
    top1_acc = int(bool(retrieved) and retrieved[0] == gt_main)

    # --- Hit Rate (Recall@K) ---
    # Is the main relevant law anywhere in the retrieved set?
    hit_rate = int(gt_main in retrieved_set)

    # --- Multi-Hit Rate ---
    # What fraction of reference_laws appear in retrieved_docs?
    # Rows with no reference_laws are excluded from the average (return None)
    if gt_refs:
        gt_ref_set = {(r["law_name"], str(r["section_num"])) for r in gt_refs}
        multi_hit = len(gt_ref_set & retrieved_set) / len(gt_ref_set)
    else:
        multi_hit = None  # excluded from aggregate

    # TODO เพิ่ม MRR 
    # --- MRR ---
    # --- MRR ---
    # Reciprocal Rank of the FIRST correct main relevant law
    mrr = 0.0
    for i, doc in enumerate(retrieved):
        if doc == gt_main:
            mrr = 1.0 / (i + 1)
            break

    return {
        "top1_acc":   top1_acc,
        "hit_rate":   hit_rate,
        "multi_hit":  multi_hit,
        "mrr":        mrr,
    }


# ---------------------------------------------------------------------------
# Generation metrics (batch, for efficiency)
# ---------------------------------------------------------------------------

def compute_exact_match(predictions: list[str], references: list[str]) -> list[int]:
    """Exact match after normalisation."""
    return [
        int(_normalise(p) == _normalise(r))
        for p, r in zip(predictions, references)
    ]


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> tuple[list[float], list[float]]:
    """
    Returns (rouge1_recall_scores, rougeL_recall_scores).
    Uses Recall only — because RAG answers are longer than GT.
    """
    rouge_scorer = _get_rouge()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_scores, rL_scores = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        r1_scores.append(scores["rouge1"].recall)
        rL_scores.append(scores["rougeL"].recall)
    return r1_scores, rL_scores


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/mdeberta-v3-base",
    batch_size: int = 16,
) -> list[float]:
    """
    BERTScore Recall — semantic coverage of expert answer by RAG answer.

    Default model: mdeberta-v3-base (multilingual, good Thai support).
    Alternative:   VISAI-AI/nitibench-ccl-human-finetuned-bge-m3
                   (same embed model as the RAG pipeline, domain-tuned on CCL)

    Returns list of per-sample recall scores.
    """
    BERTScorer = _get_bertscore()
    scorer = BERTScorer(
        model_type=model_type,
        lang="th",
        rescale_with_baseline=False,
        batch_size=batch_size,
    )
    # bert_score returns (P, R, F1) tensors
    _, R, _ = scorer.score(cands=predictions, refs=references)
    return R.tolist()


# ---------------------------------------------------------------------------
# Aggregate a list of scores (skip None values)
# ---------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


# ---------------------------------------------------------------------------
# Full pipeline: load JSONL → compute all metrics → return DataFrame
# ---------------------------------------------------------------------------

def evaluate(
    jsonl_path: str,
    bertscore_model: str = "microsoft/mdeberta-v3-base",
    bertscore_batch: int = 16,
    skip_bertscore: bool = False,
) -> pd.DataFrame:
    """
    Load a JSONL result file and compute all metrics.
    Returns a per-row DataFrame with all metric columns.
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    print(f"[evaluate] loaded {len(records)} records from {jsonl_path}")

    # ---- Retrieval metrics (per-row, no batching needed) ----
    rows = []
    for rec in tqdm(records, desc="retrieval metrics"):
        ret = compute_retrieval_metrics(rec)
        rows.append({
            "id":       rec["id"],
            "system":   rec["system"],
            "question": rec["question"],
            "rag_answer": rec["rag_answer"],
            "gt_reference_answer": rec["gt_reference_answer"],
            "gt_answer": rec["gt_answer"],
            **ret,
        })
    df = pd.DataFrame(rows)

    # ---- Generation metrics (batched) ----
    predictions_rag   = df["rag_answer"].tolist()
    refs_short        = df["gt_reference_answer"].tolist()  # for EM + ROUGE
    refs_expert       = df["gt_answer"].tolist()            # for BERTScore

    # Exact Match vs reference_answer
    print("[evaluate] computing Exact Match...")
    df["exact_match"] = compute_exact_match(predictions_rag, refs_short)

    # ROUGE vs reference_answer
    print("[evaluate] computing ROUGE...")
    r1, rL = compute_rouge(predictions_rag, refs_short)
    df["rouge1_recall"] = r1
    df["rougeL_recall"] = rL

    # BERTScore vs answer (expert version)
    if not skip_bertscore:
        print(f"[evaluate] computing BERTScore (model={bertscore_model})...")
        bs_recall = compute_bertscore(
            predictions=predictions_rag,
            references=refs_expert,
            model_type=bertscore_model,
            batch_size=bertscore_batch,
        )
        df["bertscore_recall"] = bs_recall
    else:
        df["bertscore_recall"] = None
        print("[evaluate] BERTScore skipped (--skip-bertscore)")

    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-row metrics into a single summary row per system.
    multi_hit excludes rows where reference_laws is empty (None values).
    """
    metric_cols = [
        "top1_acc",
        "hit_rate",
        "multi_hit",       # None rows excluded from mean
        "exact_match",
        "rouge1_recall",
        "rougeL_recall",
        "bertscore_recall",
    ]

    summary_rows = []
    for system, group in df.groupby("system"):
        row = {"system": system, "n": len(group)}
        for col in metric_cols:
            if col in group.columns:
                row[col] = _mean(group[col].tolist())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).set_index("system")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RAG evaluation metrics")
    parser.add_argument("--input",   nargs="+", required=True,
                        help="One or more JSONL result files")
    parser.add_argument("--output",  default="eval_results.csv",
                        help="Path to save per-row results CSV")
    parser.add_argument("--summary", default="eval_summary.csv",
                        help="Path to save aggregated summary CSV")
    parser.add_argument("--bertscore-model",
                        default="microsoft/mdeberta-v3-base",
                        help="HuggingFace model for BERTScore")
    parser.add_argument("--bertscore-batch", type=int, default=16)
    parser.add_argument("--skip-bertscore",  action="store_true",
                        help="Skip BERTScore (faster, no GPU needed)")
    args = parser.parse_args()

    all_dfs = []
    for path in args.input:
        df = evaluate(
            jsonl_path=path,
            bertscore_model=args.bertscore_model,
            bertscore_batch=args.bertscore_batch,
            skip_bertscore=args.skip_bertscore,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save per-row results
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n[saved] per-row results → {args.output}")

    # Save + print summary
    summary = summarise(combined)
    summary.to_csv(args.summary, encoding="utf-8-sig")
    print(f"[saved] summary → {args.summary}")

    print("\n========== Evaluation Summary ==========")
    print(summary.to_string(float_format="{:.4f}".format))
    print("=========================================")
