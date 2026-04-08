"""
eval_metrics.py
===============
Step 2 — Load JSONL result files and compute all evaluation metrics.

Usage:
    # Single system
    python eval_metrics.py --input results_hybrid.jsonl

    # Compare two systems
    python eval_metrics.py --input results_hybrid.jsonl results_graph.jsonl

Metrics computed:
    Retrieval Layer
        T_main = relevant_laws only (1 entry per QA)
        T_full = relevant_laws + reference_laws

        - HitRate@k       (T_full) : มี doc ถูกต้องอย่างน้อย 1 ตัวใน retrieved ไหม
        - Multi-HitRate@k (T_full) : ทุก doc ใน T_full อยู่ใน retrieved ไหม (all-or-nothing)
        - Recall@k        (T_full) : สัดส่วนของ T_full ที่ดึงมาได้
        - MRR@k           (T_main) : 1 / rank ของมาตราหลัก (relevant_laws)

    Generation Layer
        - ROUGE-1 Recall   : unigram recall vs gt_reference_answer
        - ROUGE-L Recall   : LCS recall vs gt_reference_answer
        - BERTScore Recall : semantic recall vs gt_answer (expert version)

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
# Retrieval metrics (per row)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(record: dict) -> dict:
    """
    Compute all retrieval metrics for a single record.

    Ground Truth definitions
    ------------------------
    T_main : (law_name, section_num) จาก relevant_laws[0]
             → ใช้กับ MRR เท่านั้น (single-label)

    T_full : set ของ (law_name, section_num) จาก relevant_laws + reference_laws
             → ใช้กับ HitRate, Multi-HitRate, Recall (multi-label)

    Metric definitions
    ------------------
    HitRate       : I(T_full ∩ retrieved ≠ ∅)          → 0 หรือ 1
    Multi-HitRate : I(T_full ⊆ retrieved)               → 0 หรือ 1  (all-or-nothing)
    Recall        : |T_full ∩ retrieved| / |T_full|     → 0.0–1.0   (partial credit)
    MRR           : 1 / rank(T_main in retrieved)       → 0.0–1.0

    Note: rows ที่ไม่มี reference_laws จะมี T_full = T_main (1 ตัว)
          ซึ่งทำให้ HitRate = Multi-HitRate = Recall สำหรับ row นั้น
    """
    # retrieved: ordered list ตาม rank จาก RAG
    retrieved = [
        (d["law_name"], str(d["section_num"]))
        for d in record["retrieved_docs"]
    ]
    retrieved_set = set(retrieved)

    # T_main — single label (relevant_laws เสมอมีแค่ 1 ตัว)
    gt_relevant = record["gt_relevant_laws"]
    gt_main = (gt_relevant[0]["law_name"], str(gt_relevant[0]["section_num"]))

    # T_full — multi label (relevant + reference)
    gt_refs = record["gt_reference_laws"]
    gt_full = {(r["law_name"], str(r["section_num"])) for r in gt_relevant + gt_refs}

    # ------------------------------------------------------------------
    # HitRate (T_full) — มี doc ถูกต้องอย่างน้อย 1 ตัวใน retrieved ไหม
    # I(T_full ∩ retrieved ≠ ∅)
    # ------------------------------------------------------------------
    hit_rate = int(bool(gt_full & retrieved_set))

    # ------------------------------------------------------------------
    # Multi-HitRate (T_full) — ทุก doc ใน T_full อยู่ใน retrieved ไหม
    # I(T_full ⊆ retrieved)  → all-or-nothing, ขาดแม้แค่ตัวเดียว = 0
    # ------------------------------------------------------------------
    multi_hit_rate = int(gt_full.issubset(retrieved_set))

    # ------------------------------------------------------------------
    # Recall (T_full) — สัดส่วนของ T_full ที่ดึงมาได้
    # |T_full ∩ retrieved| / |T_full|  → partial credit
    # ------------------------------------------------------------------
    recall = len(gt_full & retrieved_set) / len(gt_full)

    # ------------------------------------------------------------------
    # MRR (T_main) — คุณภาพ ranking ของมาตราหลัก
    # 1 / rank ของ T_main ใน retrieved (rank เริ่มที่ 1)
    # ถ้าไม่พบเลย → 0.0
    # ------------------------------------------------------------------
    mrr = 0.0
    for i, doc in enumerate(retrieved):
        if doc == gt_main:
            mrr = 1.0 / (i + 1)
            break

    return {
        "hit_rate":       hit_rate,
        "multi_hit_rate": multi_hit_rate,
        "recall":         recall,
        "mrr":            mrr,
    }


# ---------------------------------------------------------------------------
# Generation metrics (batch, for efficiency)
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> tuple[list[float], list[float]]:
    """
    Compute ROUGE-1 Recall and ROUGE-L Recall.

    ใช้ Recall ไม่ใช่ F1 เพราะ RAG answer ยาวกว่า GT เสมอ
    Recall วัดว่า GT ถูก "ครอบคลุม" โดย answer มากแค่ไหน

    Returns: (rouge1_recall_list, rougeL_recall_list)
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
    model_type: str = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3",
    batch_size: int = 16,
) -> list[float]:
    """
    Compute BERTScore Recall.

    วัด semantic coverage ของ expert answer (gt_answer) โดย RAG answer
    ใช้ Recall เพราะต้องการวัดว่า expert answer ถูกครอบคลุมแค่ไหน

    Model แนะนำ: VISAI-AI/nitibench-ccl-human-finetuned-bge-m3
                 (domain-tuned บน CCL, ตรง domain กับโปรเจกต์)
    Fallback:    microsoft/mdeberta-v3-base (multilingual)

    Returns: list of per-sample BERTScore Recall
    """
    BERTScorer = _get_bertscore()
    scorer = BERTScorer(
        model_type=model_type,
        lang="th",
        rescale_with_baseline=False,
        batch_size=batch_size,
    )
    # bert_score returns (Precision, Recall, F1) tensors
    _, R, _ = scorer.score(cands=predictions, refs=references)
    return R.tolist()


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    """Mean ของ list โดย skip ค่า None"""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


# ---------------------------------------------------------------------------
# Full pipeline: load JSONL → compute metrics → return DataFrame
# ---------------------------------------------------------------------------

def evaluate(
    jsonl_path: str,
    bertscore_model: str = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3",
    bertscore_batch: int = 16,
    skip_bertscore: bool = False,
) -> pd.DataFrame:
    """
    Load JSONL result file และ compute metrics ทุกตัว
    Returns per-row DataFrame
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

    # ---- Retrieval metrics (per-row) ----
    rows = []
    for rec in tqdm(records, desc="retrieval metrics"):
        ret = compute_retrieval_metrics(rec)
        rows.append({
            "id":                  rec["id"],
            "system":              rec["system"],
            "question":            rec["question"],
            "rag_answer":          rec["rag_answer"],
            "gt_reference_answer": rec["gt_reference_answer"],
            "gt_answer":           rec["gt_answer"],
            **ret,
        })
    df = pd.DataFrame(rows)

    # ---- Generation metrics (batched) ----
    predictions_rag = df["rag_answer"].tolist()
    refs_short      = df["gt_reference_answer"].tolist()  # ROUGE — vs reference_answer
    refs_expert     = df["gt_answer"].tolist()             # BERTScore — vs expert answer

    # ROUGE vs reference_answer (short, no reasoning)
    print("[evaluate] computing ROUGE...")
    r1, rL = compute_rouge(predictions_rag, refs_short)
    df["rouge1_recall"] = r1
    df["rougeL_recall"] = rL

    # BERTScore vs answer (expert-revised, has reasoning + citation)
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
    Aggregate per-row metrics เป็น summary ต่อระบบ

    Retrieval metrics: mean ของทุก row
    Generation metrics: mean ของทุก row
    """
    metric_cols = [
        # Retrieval — T_full
        "hit_rate",
        "multi_hit_rate",
        "recall",
        # Retrieval — T_main
        "mrr",
        # Generation
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
                        default="VISAI-AI/nitibench-ccl-human-finetuned-bge-m3",
                        help="HuggingFace model for BERTScore "
                             "(แนะนำ: VISAI-AI/nitibench-ccl-human-finetuned-bge-m3)")
    parser.add_argument("--bertscore-batch", type=int, default=16)
    parser.add_argument("--skip-bertscore",  action="store_true",
                        help="Skip BERTScore computation (faster, no GPU needed)")
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
