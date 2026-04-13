# Experimental Guideline for Thai Legal RAG Project (2026)

---

# Objectives

โปรเจคนี้มีเป้าหมายเพื่อศึกษา:
> Improving Retrieval-Augmented Generation (RAG) for Thai legal domain, especially in cases with cross-reference which is a unique characteristic of Thai laws.

> การปรับปรุง Retrieval-Augmented Generation (RAG) สำหรับกฎหมายไทย โดยเฉพาะในกรณีที่มีการอ้างอิงข้ามมาตรา (cross-reference) ซึ่งเป็นลักษณะเฉพาะของกฎหมายไทย

โดยจะเปรียบเทียบประสิทธิภาพของหลายๆ วิธีการ:
- Hybrid RAG
- Hybrid + Reference Expansion
- GraphRAG (VectorCypher)

---

# Concept สำคัญ

- ปัญหาหลัก:
  → Hybrid RAG ดึง context หลักได้ แต่ “ดึง reference laws ไม่ครบ” 

- วิธีแก้:
  → เพิ่ม “Link Reference Laws” (simulate graph traversal)   

- GraphRAG:
  → ใช้ Node = Section และ Edge = Reference  

---

# SECTION A: Dataset Preparation

## A1. เตรียม Dataset

ใช้:
- NitiBench-CCL


- มีทั้งหมด 497 QA
  - มี reference: 453
  - ไม่มี reference: 44   
(Check ให้ดีว่าแต่ละ QA มี metadata `reference_laws` ที่ถูกต้อง)
---

## A2. สร้าง Label สำหรับ Evaluation

ต้องเพิ่ม column:

| field | description |
|------|------------|
| query | คำถาม |
| gold_main_section | มาตราหลัก |
| gold_reference_sections | มาตราที่ถูกอ้างอิง |
| is_reference_needed | true/false |
| hop_count | จำนวน step ของ reference |

---

## A3. แบ่ง Query เป็น 3 กลุ่ม

| group | description |
|------|------------|
| simple | ไม่ต้องใช้ reference |
| reference | ต้องใช้ 1 reference |
| multi-hop | ต้องใช้หลาย reference |

---

# SECTION B: Baseline Experiments

## B1. Dense Retrieval Only

- ใช้ embedding only
- ไม่ใช้ keyword search
- ไม่ใช้ reference expansion

---

## B2. Sparse Retrieval Only

- ใช้ BM25 หรือ keyword search
- ไม่ใช้ embedding

---

## B3. Hybrid Retrieval (Baseline)

ตาม pipeline:

- Dense + Sparse
- Fusion (RRF)
- Reranking

---

# SECTION C: Proposed Method ***

## C1. Hybrid + Reference Expansion 

ขั้นตอน:

1. Retrieve Top-K (k=3)
2. ดู metadata:
   - `reference_laws`
3. ดึง section ที่ถูก reference มาเพิ่ม
4. รวม context

---

## C2. Experiment Variants

ต้องรันหลายแบบ:

| config | description |
|-------|------------|
| Top1 expansion | expand จาก context อันดับ 1 |
| Top3 expansion | expand จาก 3 context |
| depth=1 | reference ชั้นเดียว |
| depth=2 | reference ต่อ reference |

---

## C3. Context Ordering Experiment

ทดสอบ:

| method | description |
|-------|------------|
| original | ไม่ reorder |
| parent-first | parent → child |
| append-last | เอา reference ไปท้าย |

---

# SECTION D: GraphRAG Experiments

## D1. Build Knowledge Graph

Node:
- Section

Edge:
- REFERENCES_TO

---

## D2. VectorCypher Retrieval

ขั้นตอน:

1. Vector search → หา node
2. Graph traversal → หา neighbor
3. รวม context

---

## D3. Graph Depth Experiment

| depth | description |
|------|------------|
| 1 | direct reference |
| 2 | reference chain |
| 3 | deeper |

---

# SECTION E: Evaluation

## E1. Retrieval Metrics

จาก slide:
- Recall
- MRR
- Hit Rate
- Precision@k

เพิ่ม:

| metric | description |
|-------|------------|
| Reference Recall | ดึง reference ได้หรือไม่ |
| Multi-hop Recall | ดึงครบ chain หรือไม่ |

---

## E2. Generation Metrics

จาก slide:

- BERTScore
- ROUGE 

เพิ่ม:

| metric | description |
|-------|------------|
| Citation Accuracy | อ้างมาตราถูกไหม |
| Faithfulness | hallucination |

---

# RESULT TABLE FORMAT

## Table 1: Overall Performance

| Method | Recall@5 | MRR | Ref Recall | BERTScore | Citation Acc |
|--------|--------|-----|------------|-----------|--------------|
| Dense | | | | | |
| Sparse | | | | | |
| Hybrid | | | | | |
| Hybrid+Ref | | | | | |
| GraphRAG | | | | | |

---

## Table 2: By Query Type

| Method | Simple | Reference | Multi-hop |
|--------|--------|-----------|-----------|
| Hybrid | | | |
| Hybrid+Ref | | | |
| GraphRAG | | | |

---

## Table 3: Ablation Study

| Config | Recall | Ref Recall | Citation Acc |
|--------|--------|------------|--------------|
| Hybrid | | | |
| +Ref | | | |
| +Graph | | | |

---

## Table 4: Efficiency

| Method | Latency (sec) | Tokens | Cost |
|--------|--------------|--------|------|
| Hybrid | | | |
| Hybrid+Ref | | | |
| GraphRAG | | | |

---

# SECTION F: Analysis

## F1. Error Analysis

classify error:

| type | description |
|------|------------|
| Missing Reference | ไม่ดึง reference |
| Wrong Law | ดึงผิดมาตรา |
| Hallucination | สร้างมั่ว |
| Partial Answer | ข้อมูลไม่ครบ |

---

## F2. Case Study

ใช้ example เช่น:

- มาตรา 132 → อ้างอิง 54 

ต้องแสดง:

| Method | Result |
|--------|--------|
| Hybrid | fail |
| Hybrid+Ref | success |
| GraphRAG | success |

---
---

# Note

### Main Contributions
- Problem Formulation
  - We formalize reference-aware legal retrieval for Thai law, where correct answers require retrieving not only the primary statute but also cross-referenced legal provisions.
  - We show that standard semantic retrieval fails in this setting due to missing cited sections.
- Reference-Aware Retrieval Method (CORE NOVELTY)
  - We propose a metadata-driven reference expansion mechanism that augments hybrid retrieval with cited legal sections.
  - This simulates graph traversal without requiring full graph query complexity.
- VectorCypher Graph Retrieval for Legal QA
  - We implement a VectorCypher retrieval framework that combines vector similarity search with graph traversal over statutory references.
  - This enables structured multi-hop retrieval in Thai legal texts.
- Comprehensive Comparative Study
  - We conduct a controlled comparison of:
    - Hybrid RAG
    - Hybrid RAG + Reference Expansion
    - GraphRAG (VectorCypher)
  Under identical datasets, metrics, and evaluation settings.

### Research Questions
- RQ1: Retrieval Effectiveness

  How does reference-aware retrieval affect the ability of RAG systems to retrieve legally relevant and cited sections in Thai law?

- RQ2: Graph vs Non-Graph

  Does graph-based retrieval (GraphRAG / VectorCypher) outperform hybrid vector retrieval when handling cross-referenced legal structures?

- RQ3: When Do Graphs Help?

  Under what conditions (e.g., number of references, multi-hop reasoning) does graph-based retrieval provide significant improvements?

- RQ4: Impact on Answer Quality

  How does improved retrieval (especially reference retrieval) impact:

  - answer correctness
  - citation accuracy
  - hallucination rate