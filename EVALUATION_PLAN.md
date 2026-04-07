# Thai Legal RAG — Evaluation Plan & Implementation Guide
!!! TODO ขาด MultiRate ยังไม่ถูกต้อง ต้องแก้ไข
MRR เพิ่ม Metrics ตัวนี้
อันนี้เป็น Draft

## 1. ภาพรวมโปรเจกต์

โปรเจกต์นี้เปรียบเทียบประสิทธิภาพของ RAG สองรูปแบบสำหรับการตอบคำถามกฎหมายไทย (Civil and Commercial Law) โดยใช้ชุดข้อมูลทดสอบที่ filter มาจาก NitiBench

| ระบบ | บทบาท | รายละเอียด |
|---|---|---|
| **HybridRAG** | Baseline | BGE-M3 (dense+sparse) → RRF Fusion → BGE-Reranker → `_link_ref_law()` |
| **GraphRAG** | Treatment | Graph-based retrieval pipeline |
| **Naive Vector Search** | Optional Baseline | Dense-only, ถ้ามีเวลา |

---

## 2. Test Dataset

**ไฟล์:** `test_dataset_2026-04-01.parquet`  
**จำนวน:** 497 QA pairs (filter จาก NitiBench ~3,700 ข้อ)

### โครงสร้าง Columns

| Column | Type | บทบาทใน Evaluation |
|---|---|---|
| `question` | str | Input ให้ RAG system |
| `answer` | str | Expert-revised answer (reasoning + citation) | Ground Truth สำหรับ BERTScore |
| `reference_answer` | str | คำตอบสั้น ไม่มี reasoning | Ground Truth สำหรับ ROUGE + Exact Match |
| `relevant_laws` | List[Dict] | Context หลัก (1 มาตราต่อ QA) | Ground Truth สำหรับ Retrieval |
| `reference_laws` | List[Dict] | Context รอง / Cross-reference (มีใน 91.1% ของ QA, max 39 มาตรา) | Ground Truth สำหรับ Multi-Hit Rate |
| `law_name` | str | ชื่อกฎหมาย (metadata) | — |

---

## 3. Evaluation Metrics

### 3.1 Retrieval Layer

วัดที่ **retrieved_docs** (output จาก RAG retriever) เทียบกับ Ground Truth ใน dataset

#### Top-1 Accuracy
- **วัดอะไร:** document อันดับ 1 ที่ดึงมาตรงกับ `relevant_laws[0]` หรือไม่
- **เหตุผล:** `relevant_laws` มี 1 มาตราเสมอ การที่ rank 1 ถูกต้องสำคัญที่สุดในการตอบคำถาม
- **สูตร:** `1 if retrieved[0] == (law_name, section_num) of relevant_laws[0] else 0`

#### Hit Rate (Recall@K)
- **วัดอะไร:** `relevant_laws[0]` อยู่ใน retrieved context ทั้งหมดไหม (ไม่สนใจลำดับ)
- **เหตุผล:** วัด coverage แบบกว้างกว่า Top-1 Acc
- **สูตร:** `1 if (law_name, section_num) in retrieved_set else 0`

#### Multi-Hit Rate
- **วัดอะไร:** สัดส่วนของ `reference_laws` ที่ดึงขึ้นมาได้
- **เหตุผล:** วัดความสามารถ Cross-reference ซึ่งเป็นจุดแข็งของ HybridRAG (`_link_ref_law`) และ GraphRAG
- **สูตร:** `|gt_refs ∩ retrieved_set| / |gt_refs|`
- **หมายเหตุ:** rows ที่ไม่มี `reference_laws` จะถูก exclude ออกจากการ average

---

### 3.2 Generation Layer

วัดที่ **rag_answer** (LLM output) เทียบกับ Ground Truth สองระดับ

#### Exact Match
- **เทียบกับ:** `reference_answer` (short, clean)
- **วัดอะไร:** คำตอบตรงกันทุกตัวอักษร (หลัง normalise whitespace)
- **บทบาท:** Lower bound / sanity check

#### ROUGE-1 Recall และ ROUGE-L Recall
- **เทียบกับ:** `reference_answer`
- **วัดอะไร:** ROUGE-1 = unigram overlap, ROUGE-L = Longest Common Subsequence
- **ใช้ Recall ไม่ใช่ F1:** เพราะ RAG answer ยาวกว่า GT เสมอ, Recall วัดว่า GT ถูก "ครอบคลุม" โดย answer มากแค่ไหน
- **ไม่ใช้ ROUGE-2:** Thai tokenization ไม่ชัดเจน + penalize ความยาวรุนแรงเกิน

#### BERTScore Recall
- **เทียบกับ:** `answer` (expert-revised, มี reasoning + citation)
- **วัดอะไร:** Semantic coverage ของ expert answer โดย RAG answer
- **Model ที่แนะนำ:** `VISAI-AI/nitibench-ccl-human-finetuned-bge-m3` (domain-tuned บน CCL, ใช้ model เดียวกับ embedding pipeline)
- **Alternative:** `microsoft/mdeberta-v3-base` (multilingual, ถ้า nitibench model ใช้ไม่ได้)

---

### 3.3 สรุป Metrics ทั้งหมด

```
Retrieval Layer (vs Ground Truth laws)
├── Top-1 Accuracy      → relevant_laws rank 1 ถูกไหม
├── Hit Rate            → relevant_laws อยู่ใน context ไหม
└── Multi-Hit Rate      → reference_laws ครอบคลุมแค่ไหน (91.1% ของ QA)

Generation Layer (vs Ground Truth answers)
├── vs reference_answer (short)
│   ├── Exact Match
│   ├── ROUGE-1 Recall
│   └── ROUGE-L Recall
└── vs answer (expert+reasoning)
    └── BERTScore Recall
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

**ทำไมไม่ใช้ ThreadPool:**
- Typhoon มี Rate Limit → parallel จะโดน HTTP 429
- JSONL append ทีละ row ทำให้ resume ได้ถ้า crash กลางทาง
- ง่ายกว่า debug

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
    {"law_name": "...", "section_num": "1609", "rank": 1, "score": 0.92},
    ...
  ]
}
```

**ไม่เก็บ `page_content`** เพราะ section_content อยู่ใน dataset แล้ว ประหยัด disk และ memory

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

**Output:**
- `eval_results.csv` — metric ทุก column ต่อ row (ใช้ drill-down วิเคราะห์)
- `eval_summary.csv` — aggregate mean ต่อระบบ (ใช้รายงานผล)

---

## 5. Dependencies

```bash
pip install pandas pyarrow tqdm rouge-score bert-score
```

---

## 6. Checklist ก่อน Run

- [ ] `results_hybrid.jsonl` และ `results_graph.jsonl` อยู่ใน directory เดียวกัน
- [ ] ตั้งค่า `OPENAI_API_KEY` ใน `.env` สำหรับ Typhoon
- [ ] Qdrant service รันอยู่ที่ `http://localhost:6333`
- [ ] Collection `thai_laws_collection` มีข้อมูลครบ
- [ ] GraphRAG adapter ถูก wire up ใน `eval_runner.py` แล้ว
- [ ] ถ้าใช้ BERTScore บน CPU ให้เพิ่ม `--bertscore-batch 4` เพื่อไม่ให้ memory เต็ม

---

## 7. การแปลผล

| Metric | ค่าสูง หมายถึง |
|---|---|
| Top-1 Accuracy | Retriever ดึง context หลักขึ้นมาอันดับ 1 ได้บ่อย |
| Hit Rate | Retriever ไม่พลาด context หลัก แม้ไม่ได้ rank 1 |
| Multi-Hit Rate | Retriever จับ cross-reference ได้ครอบคลุม |
| Exact Match | LLM ตอบตรงเป๊ะกับ reference_answer สั้นๆ |
| ROUGE-1/L Recall | LLM ตอบครอบคลุม keyword และ sequence ของ GT |
| BERTScore Recall | LLM ตอบมีความหมายใกล้เคียง expert answer |

**จุดสำคัญในการเปรียบเทียบ HybridRAG vs GraphRAG:**
- ถ้า **Multi-Hit Rate** ต่างกันมาก → แสดงว่า Cross-reference handling ต่างกัน
- ถ้า **Top-1 Acc สูงแต่ BERTScore ต่ำ** → Retrieval ดี แต่ LLM ยังตอบไม่ครบ
- ถ้า **Multi-Hit สูงแต่ Generation ไม่ดีขึ้น** → Context รองไม่ได้ช่วย LLM จริง
