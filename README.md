# Project description
...

# Repository structure
```
root-repo/
├── data/                    
│   ├── raw/                # ไฟล์ Dataset กฎหมายไทยต้นฉบับ 
│   ├── processed/          # ข้อมูลที่ผ่านการ Transform, Cleansing และ Chunking แล้ว
│   └── tests/              # ชุดคำถาม-คำตอบ (Ground Truth) สำหรับทำ Evaluation
├── docker/                 # Infrastructure Stack
│   └── docker-compose.yml  # สำหรับรัน Qdrant, Neo4j และ Tool อื่นๆ
├── notebooks/              # Sandbox สำหรับ EDA, ทดลอง Prompt และ Tutorial
│   ├── data_prep_nititbench.ipynb
│   ├── indexing_rag_tutorial.ipynb
│   └── graph_extraction_test.ipynb
├── src/                    # หัวใจหลักของ Logic (Focus on Querying)
│   ├── common/             # Helpers/Plugins เช่น Thai Tokenizer, Custom Embedding Class
│   ├── rag/                # Standard Vector RAG Pipeline
│   │   └── retriever.py    # Logic การค้นหาและดึงข้อมูลจาก Vector DB
│   ├── graph_rag/          # Graph-based RAG Pipeline
│   │   └── retriever.py    # Logic การค้นหาแบบ Traverse ผ่าน Nodes/Edges
│   └── evaluation/         # เก็บผลลัพธ์การทดลอง (CSV, JSONLs)
├── .env                    # Environment Variables (Secrets & Configs)
├── pyproject.toml          # จัดการ Library ด้วย uv (รองรับการทำ Reproducible Build)
├── main.py                 # Entry point สำหรับ Web API (FastAPI) หรือ App หลัก
├── embedding_rag.py        # Script หลักสำหรับสร้าง Vector Index (Ingestion)
└── hybridrag_query_cli.py  # Interface สำหรับทดสอบ Query ผ่าน Command Line
...
```

### Guide Setup Project
require uv package manager


```
$ git clone https://github.com/6587093polakornming/muict_thai_legal_RAG_GraphRAG.git

$ uv venv
$ uv sync --frozen (need uv.lock)

[active .venv]
$ .\.venv\Scripts\activate

[setup dir]
$ python setup_project_dir.py

[download raw dataset]
$ python clone_dataset.py

[run docker compose]
$ cd docker
$ docker compose -f <filename.yaml> up -d

[update dependencies]
$ uv add <dependencies>
$ uv lock (generate new uv lock)

$ uv sync --frozen

[Evaluation Metric]

eval_runner.py — Step 1 รัน RAG แล้วเก็บผลเป็น JSONL ต่อ row มี checkpoint/resume ในตัว วิธีใช้คือ implement GraphRAGAdapter ให้ครบก่อน แล้วรัน

$ python eval_runner.py --system hybrid --output results_hybrid.jsonl
$ python eval_runner.py --system graph  --output results_graph.jsonl

eval_metrics.py — Step 2 โหลด JSONL แล้วคำนวณ metrics ทั้ง Retrieval + Generation Layer ได้ทีเดียว

$ python eval_metrics.py \
   --input results_hybrid.jsonl results_graph.jsonl \
   --bertscore-model VISAI-AI/nitibench-ccl-human-finetuned-bge-m3

```