# Thai Legal RAG — Evaluation Plan & Implementation Guide

## 1. ภาพรวมโปรเจกต์

เปรียบเทียบประสิทธิภาพของ RAG สองรูปแบบสำหรับการตอบคำถามกฎหมายไทย (Civil and Commercial Law)
โดยใช้ชุดข้อมูลทดสอบที่ filter มาจาก NitiBench

| ระบบ | บทบาท | รายละเอียด |
|---|---|---|
| **HybridRAG** | Baseline | BGE-M3 (dense+sparse) → RRF Fusion → BGE-Reranker → `_link_ref_law()` |
| **GraphRAG** | Treatment | Graph-based retrieval pipeline |
| **Naive Vector Search** | Optional Baseline | Dense-only ถ้ามีเวลา |

---

## 2. Test Dataset

**ไฟล์:** `test_dataset_2026-04-01.parquet`
**จำนวน:** 497 QA pairs (filter จาก NitiBench ~3,700 ข้อ)

### โครงสร้าง Columns

| Column | Type | บทบาทใน Evaluation |
|---|---|---|
| `question` | str | Input ให้ RAG system |
| `answer` | str | Expert-revised answer (reasoning + citation) | Ground Truth สำหรับ BERTScore |
| `reference_answer` | str | คำตอบสั้น ไม่มี reasoning | Ground Truth สำหรับ ROUGE |
| `relevant_laws` | List[Dict] | Context หลัก (1 มาตราต่อ QA เสมอ) | T_main สำหรับ MRR |
| `reference_laws` | List[Dict] | Context รอง / Cross-reference (91.1% ของ QA, max 39 มาตรา) | ส่วนหนึ่งของ T_full |
| `law_name` | str | ชื่อกฎหมาย (metadata) | — |

---

## 3. Evaluation Metrics

### นิยาม Ground Truth สำหรับ Retrieval

```
T_main = { relevant_laws }
         → มีแค่ 1 มาตราต่อ QA เสมอ
         → ใช้กับ MRR เท่านั้น

T_full = { relevant_laws } ∪ { reference_laws }
         → multi-label, มีได้หลายมาตรา
         → ใช้กับ HitRate, Multi-HitRate, Recall
```

---

### 3.1 Retrieval Layer

อ้างอิงจาก NitiBench Paper §3.2.1

#### HitRate@k — T_full
**วัดอะไร:** ใน top-k ที่ดึงมา มีมาตราถูกต้องอย่างน้อย 1 ตัวไหม

```
HitRate@k = (1/N) Σ I(T_full ∩ Rᵏ ≠ ∅)
```
- ค่า 0 หรือ 1 ต่อ QA → aggregate เป็น mean
- ตัวอย่าง: T_full = {1609, 1608}, Retrieved = [500, 1609, 200] → 1 (เจอ 1609)

#### Multi-HitRate@k — T_full
**วัดอะไร:** ใน top-k ที่ดึงมา มี **ทุกมาตรา** ใน T_full ครบหมดไหม (all-or-nothing)

```
Multi-HitRate@k = (1/N) Σ I(T_full ⊆ Rᵏ)
```
- ค่า 0 หรือ 1 ต่อ QA → strict กว่า HitRate มาก
- ตัวอย่าง: T_full = {1609, 1608}, Retrieved = [1609, 500, 200] → 0 (ขาด 1608)

#### Recall@k — T_full
**วัดอะไร:** สัดส่วนของมาตราใน T_full ที่ดึงขึ้นมาได้ (partial credit)

```
Recall@k = (1/N) Σ |T_full ∩ Rᵏ| / |T_full|
```
- ค่า 0.0–1.0 ต่อ QA
- ต่างจาก Multi-HitRate: ให้ partial credit ถ้าดึงมาได้บางส่วน
- ตัวอย่าง: T_full = {1609, 1608, 1607}, Retrieved ได้ 2 จาก 3 → Recall = 0.67

#### MRR@k — T_main
**วัดอะไร:** คุณภาพการจัดลำดับของมาตราหลัก — ยิ่งอยู่ rank ต้นยิ่งดี

```
MRR@k = (1/N) Σ 1 / rank(T_main ใน Rᵏ)
        ถ้าไม่พบ → 0
```
- ค่า 0.0–1.0 ต่อ QA
- ใช้ T_main (ไม่ใช่ T_full) เพราะมาตราหลักคือมาตราที่ตอบคำถามโดยตรง
- ตัวอย่าง: 1609 อยู่ rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, ไม่พบ → 0.0

---

### 3.2 Generation Layer

#### ROUGE-1 Recall และ ROUGE-L Recall
- **เทียบกับ:** `reference_answer` (สั้น ไม่มี reasoning)
- **ใช้ Recall ไม่ใช่ F1:** RAG answer ยาวกว่า GT เสมอ F1 จะ penalize โดยไม่จำเป็น
- ROUGE-1: unigram overlap (ไม่สนลำดับ)
- ROUGE-L: Longest Common Subsequence (รักษาลำดับ แต่ไม่ต้องติดกัน)

#### BERTScore Recall
- **เทียบกับ:** `answer` (expert-revised มี reasoning + citation)
- **วัดอะไร:** semantic coverage — token ใน GT แต่ละตัวจับคู่กับ token ใกล้เคียงที่สุดใน prediction
- **Model:** `VISAI-AI/nitibench-ccl-human-finetuned-bge-m3` (domain-tuned บน CCL)
- **Fallback:** `microsoft/mdeberta-v3-base`

---

### 3.3 สรุป Metrics ทั้งหมด

```
RETRIEVAL LAYER
┌──────────────────┬──────────┬────────────────────────────────────────┐
│ Metric           │ T        │ วัดอะไร                                │
├──────────────────┼──────────┼────────────────────────────────────────┤
│ HitRate@k        │ T_full   │ เจอมาตราถูกต้องอย่างน้อย 1 ตัวไหม    │
│ Multi-HitRate@k  │ T_full   │ ครบทุกมาตราไหม (all-or-nothing)       │
│ Recall@k         │ T_full   │ ดึงมาได้กี่ % ของทุกมาตรา             │
│ MRR@k            │ T_main   │ มาตราหลักอยู่ rank ไหน                │
└──────────────────┴──────────┴────────────────────────────────────────┘

GENERATION LAYER
┌──────────────────┬─────────┬────────────────────────────────────────┐
│ Metric           │ GT      │ วัดอะไร                                │
├──────────────────┼─────────┼────────────────────────────────────────┤
│ ROUGE-1 Recall   │ y_ref   │ คำเดี่ยว (unigram) ครอบคลุมแค่ไหน    │
│ ROUGE-L Recall   │ y_ref   │ ลำดับคำ (LCS) ครอบคลุมแค่ไหน         │
│ BERTScore Recall │ y_exp   │ ความหมายใกล้เคียง expert answer แค่ไหน│
└──────────────────┴─────────┴────────────────────────────────────────┘
```

---

## 4. Engineering Plan

### 4.1 Step 1 — Run RAG และเก็บผล (`eval_runner.py`)

**Pattern:** Sequential + JSONL checkpoint (ไม่ใช้ ThreadPool เพราะ Rate Limit)

```bash
# Run HybridRAG
python eval_runner.py --system hybrid --output results_hybrid.jsonl --sleep 0.5

# Run GraphRAG
python eval_runner.py --system graph  --output results_graph.jsonl  --sleep 0.5
```

**Data Structure ที่เก็บต่อ row:**
```json
{
  "id": 0,
  "system": "hybrid",
  "question": "...",
  "gt_answer": "...",
  "gt_reference_answer": "...",
  "gt_relevant_laws":  [{"law_name": "...", "section_num": "1609"}],
  "gt_reference_laws": [{"law_name": "...", "section_num": "1608"}],
  "rag_answer": "...",
  "retrieved_docs": [
    {"law_name": "...", "section_num": "1609", "rank": 1, "score": 0.92}
  ]
}
```

### 4.2 Step 2 — Compute Metrics (`eval_metrics.py`)

```bash
# Single system
python eval_metrics.py --input results_hybrid.jsonl

# Compare สองระบบ
python eval_metrics.py \
  --input results_hybrid.jsonl results_graph.jsonl \
  --output eval_results.csv \
  --summary eval_summary.csv \
  --bertscore-model VISAI-AI/nitibench-ccl-human-finetuned-bge-m3

# ถ้าไม่มี GPU หรืออยากทำเร็ว
python eval_metrics.py --input results_hybrid.jsonl --skip-bertscore
```

---

## 5. Dependencies

```bash
pip install pandas pyarrow tqdm rouge-score bert-score
```

---

## 6. Checklist ก่อน Run

- [ ] ตั้งค่า `OPENAI_API_KEY` ใน `.env` สำหรับ Typhoon
- [ ] Qdrant service รันอยู่ที่ `http://localhost:6333`
- [ ] Collection `thai_laws_collection` มีข้อมูลครบ
- [ ] GraphRAG adapter ถูก wire up ใน `eval_runner.py` แล้ว
- [ ] ถ้าใช้ BERTScore บน CPU ให้เพิ่ม `--bertscore-batch 4`

---

## 7. การแปลผล

| Metric | ค่าสูง หมายถึง |
|---|---|
| HitRate | Retriever ดึงมาตราที่เกี่ยวข้องได้อย่างน้อย 1 ตัวเสมอ |
| Multi-HitRate | Retriever ดึง cross-reference ได้ครบทุกตัว |
| Recall | Retriever ดึง cross-reference ได้ครอบคลุม (partial credit) |
| MRR | มาตราหลักอยู่ rank ต้นๆ เสมอ |
| ROUGE-1/L Recall | LLM ตอบครอบคลุม keyword และ sequence ของ GT |
| BERTScore Recall | LLM ตอบมีความหมายใกล้เคียง expert answer |

**จุดสำคัญในการเปรียบเทียบ HybridRAG vs GraphRAG:**
- ถ้า **Multi-HitRate** ต่างกันมาก → Cross-reference handling ต่างกัน
- ถ้า **MRR สูงแต่ Multi-HitRate ต่ำ** → ดึงมาตราหลักได้ดีแต่ cross-reference ยังพลาด
- ถ้า **Recall สูงแต่ Generation ไม่ดีขึ้น** → Context รองไม่ได้ช่วย LLM จริง
