# Project description
...

# Repository structure
```
root-repo/
├── data/                   
│   ├── raw/                # ไฟล์ Dataset กฎหมายไทยต้นฉบับ 
│   └── processed/          # ไฟล์ Markdown/JSON ที่ผ่านการ Chunking แล้ว 
├── docker/                 # สำหรับรัน Qdrant, Neo4j, Langfuse 
│   └── docker-compose.yml  
├── notebooks/              # สำหรับ EDA และทดลอง Prompt
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_prototype.ipynb
│   └── 03_graph_extraction_test.ipynb
├── src/                    # Source Code หลัก
│   ├── common/             # โค้ดที่ใช้ร่วมกัน เช่น Embedding Model (BGE-M3/E5), Logging
│   ├── rag/                # RAG Pipeline (Baseline) 
│   │   ├── ingest.py       # นำข้อมูลเข้า Qdrant 
│   │   └── retriever.py    # ระบบค้นหาแบบ Vector Search
│   ├── graph_rag/          # GraphRAG Pipeline (Proposed) 
│   │   ├── extractor.py    # สกัด Entity/Relation เข้า Neo4j 
│   │   └── search.py       # Local/Global Search Engine 
│   └── evaluation/         # ระบบวัดผล RAGAS / MLflow  
├── tests/                  # สำหรับเก็บชุดคำถามทดสอบ (Test Sets)
├── .env                    # เก็บ API Keys (OpenAI, Langfuse, Neo4j)
├── pyproject.toml      	# รายชื่อ Library (LangChain, Qdrant-client, etc.) pytome
└── main.py                 # 
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