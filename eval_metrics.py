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


def build_bertscore_scorer(
    model_type: str = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3",
    batch_size: int = 4,
    device: str | None = None,
):
    """
    Init BERTScorer ครั้งเดียว แล้วเก็บ instance ไว้ใช้ซ้ำได้หลาย batch
    เรียกใน evaluate() ก่อน loop แล้วส่งเข้า compute_bertscore(scorer=...)

    Parameters
    ----------
    model_type  : HuggingFace model ID
    batch_size  : จำนวน sample ต่อ forward pass (ปรับตาม VRAM)
    device      : "cuda" / "cpu" / None (auto-detect)
    """
    BERTScorer = _get_bertscore()
    scorer = BERTScorer(
        model_type=model_type,
        num_layers=24,
        lang="th",
        rescale_with_baseline=False,
        batch_size=batch_size,
        device=device,
    )
    return scorer


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3",
    batch_size: int = 4,
    scorer=None,
) -> list[float]:
    """
    Compute BERTScore Recall สำหรับ predictions/references ที่ส่งเข้ามา

    วัด semantic coverage ของ expert answer (gt_answer) โดย RAG answer
    ใช้ Recall เพราะต้องการวัดว่า expert answer ถูกครอบคลุมแค่ไหน

    Parameters
    ----------
    predictions : list ของ RAG answer
    references  : list ของ ground-truth answer
    model_type  : ใช้เมื่อ scorer=None เท่านั้น (fallback สร้างใหม่)
    batch_size  : ใช้เมื่อ scorer=None เท่านั้น
    scorer      : BERTScorer instance ที่ init ไว้แล้วจาก build_bertscore_scorer()
                  ถ้าส่งมา จะ reuse โดยไม่โหลด model ใหม่ (แนะนำใช้กับ manual batch loop)

    Returns
    -------
    list of per-sample BERTScore Recall
    """
    # ถ้ามี scorer ที่ init ไว้แล้ว — ใช้เลย ไม่โหลด model ใหม่
    if scorer is not None:
        _, R, _ = scorer.score(cands=predictions, refs=references)
        return R.tolist()

    # fallback: สร้าง scorer ชั่วคราว (กรณีเรียก standalone ไม่ผ่าน evaluate)
    import gc
    tmp_scorer = build_bertscore_scorer(model_type=model_type, batch_size=batch_size)
    try:
        _, R, _ = tmp_scorer.score(cands=predictions, refs=references)
        return R.tolist()
    finally:
        del tmp_scorer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


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
    bertscore_batch: int = 32,
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
            "prompt_tokens":       rec["prompt_tokens"],
            "completion_tokens":   rec["completion_tokens"],
            "total_tokens":        rec["total_tokens"],
            "retrieve_time":       rec["retrieve_time"],
            "llm_time":            rec["llm_time"],
            "total_elapsed":       rec["total_elapsed"],
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
        # 1. กรองเฉพาะ Row ที่เป็น String จริงๆ, ไม่ว่าง และไม่ใช่ __ERROR__ (รองรับทุกเคส)
        valid_indices = []
        for i, (ans, ref) in enumerate(zip(predictions_rag, refs_expert)):
            # ตรวจสอบ Prediction: ต้องเป็น str, ไม่ใช่ None, ไม่ใช่ค่าว่าง และไม่ใช่ __ERROR__
            is_valid_ans = (isinstance(ans, str) and 
                           ans.strip() != "" and 
                           ans.strip() != "__ERROR__")
            
            # ตรวจสอบ Reference (Ground Truth): ต้องเป็น str และไม่ว่าง
            is_valid_ref = isinstance(ref, str) and ref.strip() != ""
            
            if is_valid_ans and is_valid_ref:
                valid_indices.append(i)
        
        # 2. เตรียม List ผลลัพธ์เริ่มต้นเป็น 0.0 ทั้งหมด
        bs_recall = [0.0] * len(predictions_rag)

        if valid_indices:
            print(f"[evaluate] computing BERTScore for {len(valid_indices)} records (Manual Batching)...")

            sub_preds = [predictions_rag[i] for i in valid_indices]
            sub_refs  = [refs_expert[i]     for i in valid_indices]

            manual_batch_size = bertscore_batch
            all_sub_scores    = []

            # --- Init model ครั้งเดียวก่อน loop — ไม่โหลดซ้ำทุก batch ---
            import gc
            bert_scorer = build_bertscore_scorer(
                model_type=bertscore_model,
                batch_size=manual_batch_size,
            )
            try:
                for i in tqdm(range(0, len(sub_preds), manual_batch_size), desc="BERTScore Sub-batches"):
                    batch_p = sub_preds[i : i + manual_batch_size]
                    batch_r = sub_refs [i : i + manual_batch_size]

                    batch_scores = compute_bertscore(
                        predictions=batch_p,
                        references=batch_r,
                        scorer=bert_scorer,   # reuse — ไม่โหลด model ใหม่
                    )
                    all_sub_scores.extend(batch_scores)
            finally:
                # คืน VRAM หลัง loop เสร็จทั้งหมด
                del bert_scorer
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()
            # ---------------------------------------------------------------

            for idx, score in zip(valid_indices, all_sub_scores):
                bs_recall[idx] = score
        
        df["bertscore_recall"] = bs_recall
        print("[evaluate] BERTScore computation completed (Errors/Empty rows set to 0.0)")
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
        # Token metadata
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        # Time metadata
        "retrieve_time",
        "llm_time",
        "total_elapsed",
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
    parser.add_argument("--output",  default=None,
                        help="Path to save per-row results CSV")
    parser.add_argument("--summary", default=None,
                        help="Path to save aggregated summary CSV")
    parser.add_argument("--bertscore-model",
                        default="VISAI-AI/nitibench-ccl-human-finetuned-bge-m3",
                        help="HuggingFace model for BERTScore "
                             "(แนะนำ: VISAI-AI/nitibench-ccl-human-finetuned-bge-m3)")
    parser.add_argument("--bertscore-batch", type=int, default=32,
                        help="BERTScore batch size (default=4 สำหรับ GPU 4 GB; ลดลงถ้ายัง OOM)")
    parser.add_argument("--skip-bertscore",  action="store_true",
                        help="Skip BERTScore computation (faster, no GPU needed)")
    args = parser.parse_args()

    # Dynamic Output File
    # data/evaluation/eval_<filename>results.csv
    input_names = "_".join([Path(p).stem for p in args.input])
    # ถ้าผู้ใช้ไม่ระบุ --output ให้ใช้ชื่อตาม input
    if args.output is None:
        args.output = f"data/evaluation/{input_names}_results.csv"
        
    # ถ้าผู้ใช้ไม่ระบุ --summary ให้ใช้ชื่อตาม input
    if args.summary is None:
        args.summary = f"data/evaluation/{input_names}_summary.csv"

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
    
    # Handle Backlist for test_dataset_2026-04-01.parquet (typhoon-v2.5-30b context window is not enough)
    blacklist_ids = [99, 115, 119, 136, 180, 181, 208, 218, 403]
    combined = combined[~combined["id"].isin(blacklist_ids)]

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