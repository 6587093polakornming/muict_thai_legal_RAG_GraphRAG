# Experimental Guideline for Thai Legal RAG Project (2026) - Revised 16 April 2026

---

# Objectives
การปรับปรุง Retrieval-Augmented Generation (RAG) สำหรับกฎหมายไทย โดยเฉพาะในกรณีที่มีการอ้างอิงข้ามมาตรา (Cross-reference) โดยจะเปรียบเทียบประสิทธิภาพของ:
- Hybrid RAG (Baseline)
- Hybrid RAG + Reranking + Reference Expansion (Proposed)
- GraphRAG (VectorCypher)

---

# SECTION A: Dataset Preparation

## A1. เตรียม Dataset (Update Change)
ใช้ **NitiBench-CCL** ซึ่งประกอบด้วย 488 QA
- มี Reference: 444 ข้อ
- ไม่มี Reference: 44 ข้อ

## A2. Mapping Label สำหรับ Evaluation
เพื่อไม่ให้กระทบต่อ Evaluation Script แต่ให้สอดคล้องกับโครงสร้าง NitiBench จะใช้การ Mapping แทนการสร้าง Column ใหม่ ดังนี้:

| Field | Source (NitiBench) | Description |
| :--- | :--- | :--- |
| **query** | `question` | คำถามจาก Dataset |
| **gold_main_section** | `relevant_laws` | มาตราหลักที่ใช้ตอบ |
| **gold_reference_sections** | `reference_laws` | มาตราที่ถูกอ้างอิงจากมาตราหลัก |
| **is_reference_needed** | - | Inferred: `true` ถ้า `reference_laws` ไม่ว่าง |
| **hop_count** | - | Fixed: 1-hop (ตามข้อจำกัดของ Dataset) |

### ⚠️ Justification for 1-hop Constraint in NitiBench Dataset

ในการทดลองนี้ เรากำหนดขอบเขตการดึงข้อมูลอ้างอิงไว้ที่ **1-hop (Direct Reference)** เท่านั้น โดยมีเหตุผลสำคัญจากข้อจำกัดและลักษณะของชุดข้อมูลดังนี้:

1. **Dataset Characteristics (NitiBench QA Nature):** ชุดข้อมูล NitiBench-CCL ถูกออกแบบมาโดยเน้นไปที่การตอบคำถามกฎหมายเชิงสถานการณ์ ซึ่งโดยส่วนใหญ่ต้องการมาตราหลักที่เกี่ยวข้องและมาตราที่ถูกอ้างอิงโดยตรงจากมาตราหลักนั้น (Directly cited sections) เพื่อความสมบูรณ์ของคำตอบ การขยายขอบเขตไปมากกว่า 1-hop อาจส่งผลให้ได้เนื้อหาที่ไม่เกี่ยวข้องกับประเด็นคำถาม (Irrelevant Context) เข้ามามากเกินไป

2. **Structural Limitations (Nitibench provides 1-depth):** จากการตรวจสอบโครงสร้างข้อมูลใน NitiBench พบว่าการระบุ `reference_laws` ในชุดข้อมูลถูกจัดเตรียมไว้สำหรับความสัมพันธ์ชั้นเดียวเป็นหลัก การพยายามพัฒนาหรือขยายเป็น Multi-hop (Depth >= 2) ในสภาวะที่ Dataset ไม่ได้รองรับอย่างเป็นทางการ จะทำให้กระบวนการพัฒนามีความล่าช้าสูง (High Development Latency) และอาจไม่คุ้มค่าในเชิงประสิทธิภาพเมื่อเทียบกับระยะเวลาที่จำกัดของโครงการ

3. **Evaluation Script Handling:** เนื่องจาก Script สำหรับการวัดผล (Evaluation Script) ถูกออกแบบมาเพื่อรองรับโครงสร้างแบบ 1-hop การฝืนเพิ่มความลึกของความสัมพันธ์ (Hop count) จะส่งผลกระทบโดยตรงต่อความถูกต้องแม่นยำของ Metric อย่าง Multi-HitRate และ Recall ทำให้การเปรียบเทียบประสิทธิภาพระหว่างโมเดลขาดความชัดเจน

4. **Focus on Quality over Depth:** การทำ 1-hop ที่แม่นยำ (High Precision) มีความสำคัญต่อความน่าเชื่อถือในโดเมนกฎหมายมากกว่าการทำ Multi-hop ที่ซับซ้อนแต่เสี่ยงต่อการเกิด Hallucination หรือการดึงข้อมูลที่หลุดออกจากประเด็นหลักของตัวบทกฎหมายนั้นๆ

## A3. แบ่ง Query เป็น 2 กลุ่ม (Simplified)

| Group | N | Description |
| :--- | :---: | :--- |
| **Simple** | 44 | ไม่มีการอ้างอิงมาตราอื่น (`reference_laws` ว่าง) |
| **Reference** | 444 | มีการอ้างอิงมาตราอื่น 1 ระดับ (1-hop) — ส่วนใหญ่ (~72%) ต้องการเพียง 1 Reference Law |

---

# SECTION B: Baseline Experiments (คงเดิม)
- **B1. Dense Retrieval Only:** Embedding only.
- **B2. Sparse Retrieval Only:** BM25 / Keyword search.
- **B3. Hybrid Retrieval:** Dense + Sparse + Fusion (RRF).

---

# SECTION C: Proposed Method

## C1. Hybrid + Reference Expansion
ขั้นตอนการทำงาน:
1. Retrieve Top-K sections (K = 1, 2, 3)
2. Extract Metadata `reference_laws` จากผลลัพธ์ที่ได้
3. ดึงเนื้อหาของ Section ที่ถูก Reference มาเพิ่มเข้าไปใน Context

## C2. Experiment Variants

> **Note:** K=1 มี Expansion Variant เดียว เนื่องจาก Top-1 และ Top-N ให้ผลเหมือนกันเมื่อ K=1

| Top-K | Expansion Variant | Description |
| :---: | :--- | :--- |
| 1 | Expansion | Expand มาตราอ้างอิงจาก rank-1 (Top-1 = Top-N ในกรณีนี้) |
| 2 | Top-1 Expansion | Expand เฉพาะมาตราอ้างอิงที่พบในผลลัพธ์อันดับที่ 1 |
| 2 | Top-N Expansion | Expand มาตราอ้างอิงจากผลลัพธ์ทั้ง 2 อันดับ |
| 3 | Top-1 Expansion | Expand เฉพาะมาตราอ้างอิงที่พบในผลลัพธ์อันดับที่ 1 |
| 3 | Top-N Expansion | Expand มาตราอ้างอิงจากผลลัพธ์ทั้ง 3 อันดับ |

## C3. Context Ordering Experiment

> **Note:** K=1 ไม่มี Ordering Variant เนื่องจากมี retrieved section เดียว ordering จึง fixed อยู่แล้ว จึงทดสอบเฉพาะ K=2 และ K=3

| Top-K | Ordering Method | Description |
| :---: | :--- | :--- |
| 2 | Parent-first | มาตราหลัก (Parent) → มาตราอ้างอิง (Child) |
| 2 | Append-last | Parent ทั้งหมด → Child ทั้งหมดต่อท้าย |
| 3 | Parent-first | มาตราหลัก (Parent) → มาตราอ้างอิง (Child) |
| 3 | Append-last | Parent ทั้งหมด → Child ทั้งหมดต่อท้าย |

---

# SECTION D: GraphRAG Experiments
- **D1. Build Knowledge Graph:** Node (Section), Edge (REFERENCES_TO).
- **D2. VectorCypher Retrieval:** Vector Search → Graph Traversal (Depth=1) → Merge Context.
- **D3. Graph Depth:** ฟิกซ์ที่ **Depth = 1 (Direct Reference)** เท่านั้น.

---

# SECTION E: Evaluation (Updated Metrics)

## E1. Retrieval Layer
วัดผลที่ตัวเนื้อหามาตราที่ดึงมาได้:

| Metric | Target | Description |
| :--- | :--- | :--- |
| **HitRate@k** | $T_{full}$ | ดึงเจอมาตราที่ถูกต้องอย่างน้อย 1 ตัวใน Top-k หรือไม่ |
| **Multi-HitRate@k** | $T_{full}$ | ดึงมาตราที่เกี่ยวข้องมาได้ "ครบทุกตัว" (All-or-nothing) |
| **Recall@k** | $T_{full}$ | สัดส่วนมาตราที่ดึงได้จริงเทียบกับมาตราทั้งหมดในคำตอบ |
| **MRR@k** | $T_{main}$ | อันดับของ "มาตราหลัก" (Main Section) อยู่ที่เท่าไหร่ |

## E2. Generation Layer (typhoon-v2.5-30b-a3b-instrauct)
ใช้การเปรียบเทียบเชิง Textual Similarity เนื่องจาก Multi-HitRate มีความสัมพันธ์ (Correlation) กับ E2E Performance สูงถึง **0.989** จึงไม่จำเป็นต้องใช้ LLM-as-a-judge เพื่อลดค่าใช้จ่าย.

| Metric | Target | Description |
| :--- | :--- | :--- |
| **ROUGE-1 Recall** | $y_{ref}$ | ความครอบคลุมของคำเดี่ยว (Unigram) |
| **ROUGE-L Recall** | $y_{ref}$ | ความครอบคลุมของลำดับคำ (Longest Common Subsequence) |
| **BERTScore Recall** | $y_{exp}$ | ความถูกต้องเชิงความหมายเมื่อเทียบกับ Expert Answer |

> **Note:** กำหนด Temperature = 0 เพื่อให้ผลลัพธ์เป็น Deterministic.

### 💡 Justification for Metric Selection & E2E Evaluation Strategy

ในการทดลองนี้ เราตัดสินใจมุ่งเน้นไปที่ **Retrieval-level Metrics** และ **Textual Similarity** แทนการใช้ **LLM-as-a-judge (E2E Metrics)** ด้วยเหตุผลเชิงประจักษ์ดังนี้:

1. **High Correlation Proxy (0.989):** จากการอ้างอิงข้อมูลใน NitiBench Paper (Table 8) ผลการทดสอบแสดงให้เห็นว่า Retrieval Metrics โดยเฉพาะ **Multi-HitRate** มีค่าสหสัมพันธ์ (Correlation) กับคะแนนจากการตัดสินของ LLM (E2E Metrics) สูงถึง **0.989** ซึ่งหมายความว่าประสิทธิภาพในการดึงข้อมูลมาตราที่ครบถ้วน (Reference-aware Retrieval) สามารถใช้เป็นตัวบ่งชี้ความแม่นยำของคำตอบในขั้นตอนสุดท้ายได้เกือบสมบูรณ์

2. **Metric Redundancy (Citation Accuracy):** เนื่องจากในโดเมนกฎหมาย "ความแม่นยำในการอ้างอิงมาตรา" (Citation Accuracy) เป็นปัจจัยชี้ขาดคุณภาพคำตอบ ซึ่งตัวชี้วัดนี้มีความซ้ำซ้อนกับ Retrieval Metrics (HitRate และ Multi-HitRate) อยู่แล้ว การดึงมาตราที่ถูกต้องมาได้จึงเป็นเงื่อนไขจำเป็น (Prerequisite) ที่สำคัญที่สุดก่อนการสร้างคำตอบ

3. **Deterministic Generation Control:** เรากำหนดค่า **LLM Temperature เป็น 0** เพื่อให้กระบวนการสร้างคำตอบเป็นแบบ Deterministic ลดความผันแปรของภาษา (Variance) ทำให้ผลลัพธ์ของคำตอบขึ้นอยู่กับคุณภาพของ Context ที่ดึงมาเป็นหลัก การวัดผลที่ Retrieval Layer จึงมีความแม่นยำและเสถียรมากกว่า

4. **Resource Efficiency & Scalability:** การตัด LLM-as-a-judge ช่วยลดภาระด้าน LLM API Cost และเวลาในการประมวลผล โดยยังคงรักษามาตรฐานการวัดผลที่น่าเชื่อถือตามแนวทางที่ NitiBench แนะนำ ว่าการวัดผลที่ Retrieval Performance นั้น "เพียงพอและเหมาะสม" (Sufficient & Robust) สำหรับการเปรียบเทียบประสิทธิภาพระหว่างระบบ RAG ในเชิงโครงสร้าง

---

# RESULT TABLE FORMAT (Updated)

## Table 1: Experiment Variants (C2)

| Top-K | Expansion Variant | HitRate@k | Multi-HitRate@k | Recall@k | MRR 
| :---: | :--- | :--- | :--- | :--- | :--- | 
| 1 | Expansion |  |
| 2 | Top-1 Expansion |  |
| 2 | Top-N Expansion |  |
| 3 | Top-1 Expansion |  |
| 3 | Top-N Expansion |  |

## Table 2: Context Ordering (C3)

> **Note:** K=1 ไม่มี Ordering Variant — ทดสอบเฉพาะ K=2 และ K=3

| Top-K | Ordering Method | BERTScore | ROUGE-1 | ROUGE-L |
| :---: | :--- | :---: | :---: | :---: |
| 2 | Parent-first | | | |
| 2 | Append-last | | | |
| 3 | Parent-first | | | |
| 3 | Append-last | | | |

## Table 3: Overall Performance

| Top-K | Method | HitRate@k | Multi-HitRate@k | Recall@k | MRR | BERTScore | ROUGE-1 | ROUGE-L |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Dense | | | | | | | |
| 1 | Sparse | | | | | | | |
| 1 | Hybrid | | | | | | | |
| 1 | Hybrid+Rerank | | | | | | | |
| 1 | Hybrid+Rerank+Ref (Proposed)| | | | | | | |
| 1 | GraphRAG | | | | | | | |
| 2 | Dense | | | | | | | |
| 2 | Sparse | | | | | | | |
| 2 | Hybrid | | | | | | | |
| 2 | Hybrid+Rerank | | | | | | | |
| 2 | Hybrid+Rerank+Ref (Proposed)| | | | | | | |
| 2 | GraphRAG | | | | | | | |
| 3 | Dense | | | | | | | |
| 3 | Sparse | | | | | | | |
| 3 | Hybrid | | | | | | | |
| 3 | Hybrid+Rerank | | | | | | | |
| 3 | Hybrid+Rerank+Ref (Proposed)| | | | | | | |
| 3 | GraphRAG | | | | | | | |

## Table 4: Performance by Query Type

> **Note:** แสดงเฉพาะ Method ที่ออกแบบมาสำหรับ Reference queries (Proposed + GraphRAG) เปรียบเทียบกับ Baseline (Hybrid)

| Top-K | Method | Simple QA: HitRate@k | Simple QA: MRR | Reference QA: Multi-HitRate@k | Reference QA: Recall@k |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Hybrid | | | | |
| 1 | Hybrid+Rerank+Ref (Proposed)| | | | |
| 1 | GraphRAG | | | | |
| 2 | Hybrid | | | | |
| 2 | Hybrid+Rerank+Ref (Proposed)| | | | |
| 2 | GraphRAG | | | | |
| 3 | Hybrid | | | | |
| 3 | Hybrid+Rerank+Ref (Proposed)| | | | |
| 3 | GraphRAG | | | | |

## Table 5: Ablation Study

> แสดง Incremental improvement ของแต่ละ component ที่เพิ่มเข้ามา

| Top-K | Config | Recall@k | Multi-HitRate@k | BERTScore |
| :---: | :--- | :---: | :---: | :---: |
| 1 | Hybrid RAG | | | |
| 1 | + Reranking | | | |
| 1 | + Ref Expansion (Proposed) | | | |
| 1 | GraphRAG | | | |
| 2 | Hybrid RAG | | | |
| 2 | + Reranking | | | |
| 2 | + Ref Expansion (Proposed) | | | |
| 2 | GraphRAG | | | |
| 3 | Hybrid RAG | | | |
| 3 | + Reranking | | | |
| 3 | + Ref Expansion (Proposed)| | | |
| 3 | GraphRAG | | | |

## Table 6: Efficiency

| Top-K | Method | Latency: Retrieve (s) | Latency: LLM (s) | Prompt Tokens | Completion Tokens | Total Tokens |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Hybrid | | | | | |
| 1 | Hybrid+Rerank+Ref (Proposed) | | | | | |
| 1 | GraphRAG | | | | | |
| 2 | Hybrid | | | | | |
| 2 | Hybrid+Rerank+Ref (Proposed) | | | | | |
| 2 | GraphRAG | | | | | |
| 3 | Hybrid | | | | | |
| 3 | Hybrid+Rerank+Ref (Proposed) | | | | | |
| 3 | GraphRAG | | | | | |

---

# SECTION F: Analysis

## F1. Error Analysis
- **Wrong Law:** ดึงมาตราผิด หรือมาตราไม่เกี่ยวข้องขาด Main Context
- **Missing Reference:** ดึงมาตราหลักได้แต่ขาดมาตราอ้างอิง — ดูจาก QA ที่ Multi-HitRate = 0 และมี Main Context แต่ไม่มี Reference
- **Partial Answer:** ข้อมูลไม่ครบ — ดูจาก QA ที่ Multi-HitRate = 0 และ Recall < 1
- **Hallucination:** วิเคราะห์ผ่าน BERTScore ที่ต่ำผิดปกติ และสุ่มตรวจโดยมนุษย์ (Human Check)

## F2. Case Study (3 Scenarios)
1. **Model All Success:** ทุกวิธีตอบได้ถูกต้อง
2. **Hybrid Fail Only:** Hybrid ตกหล่นเรื่อง Reference แต่ Proposed/Graph ทำได้
3. **Model All Fail:** กรณีที่โจทย์มีความซับซ้อนสูงเกินกว่าทุก Model จะรับมือได้

---

# Note

### Main Contributions (Revised)
- **Problem Formulation:** นิยามปัญหาการดึงข้อมูลกฎหมายไทยที่ต้องอาศัยมาตราอ้างอิง (Reference-aware retrieval).
- **Reference Expansion Mechanism:** นำเสนอการขยาย Context ผ่าน Metadata เพื่อจำลองการทำ Graph Traversal ในระบบ RAG ปกติ.
- **Empirical Comparison:** เปรียบเทียบประสิทธิภาพระหว่าง Hybrid RAG และ GraphRAG บนชุดข้อมูลกฎหมายไทยอย่างเป็นระบบ.

### Research Questions (Revised)
- **RQ1:** การทำ Reference Expansion ช่วยเพิ่มความครบถ้วน (Multi-HitRate) ในการดึงมาตรากฎหมายได้ดีกว่า Hybrid RAG ปกติหรือไม่?
- **RQ2:** ในบริบท 1-hop ของกฎหมายไทย GraphRAG (VectorCypher) ให้ประสิทธิภาพที่เหนือกว่า Metadata-driven Expansion อย่างมีนัยสำคัญหรือไม่?
- **RQ3:** ความแม่นยำในระดับ Retrieval (Multi-HitRate) สามารถใช้เป็นตัวชี้วัดแทนคุณภาพของคำตอบ (Generation) ในโดเมนกฎหมายไทยได้ดีเพียงใด?

### Main Contributions (Original)
- **Problem Formulation**
  - We formalize reference-aware legal retrieval for Thai law, where correct answers require retrieving not only the primary statute but also cross-referenced legal provisions.
  - We show that standard semantic retrieval fails in this setting due to missing cited sections.
- **Reference-Aware Retrieval Method (CORE NOVELTY)**
  - We propose a metadata-driven reference expansion mechanism that augments hybrid retrieval with cited legal sections.
  - This simulates graph traversal without requiring full graph query complexity.
- **VectorCypher Graph Retrieval for Legal QA**
  - We implement a VectorCypher retrieval framework that combines vector similarity search with graph traversal over statutory references.
  - This enables structured multi-hop retrieval in Thai legal texts.
- **Comprehensive Comparative Study**
  - We conduct a controlled comparison of: Hybrid RAG, Hybrid RAG + Reference Expansion, and GraphRAG (VectorCypher) under identical datasets, metrics, and evaluation settings.

### Research Questions (Original)
- **RQ1 — Retrieval Effectiveness:** How does reference-aware retrieval affect the ability of RAG systems to retrieve legally relevant and cited sections in Thai law?
- **RQ2 — Graph vs Non-Graph:** Does graph-based retrieval (GraphRAG / VectorCypher) outperform hybrid vector retrieval when handling cross-referenced legal structures?
- **RQ3 — When Do Graphs Help?:** Under what conditions (e.g., number of references, multi-hop reasoning) does graph-based retrieval provide significant improvements?
- **RQ4 — Impact on Answer Quality:** How does improved retrieval (especially reference retrieval) impact answer correctness, citation accuracy, and hallucination rate?
