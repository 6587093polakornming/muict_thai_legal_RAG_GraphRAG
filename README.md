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
$ uv sync

[active .venv]
$ .\.venv\Scripts\activate

[setup dir]
$ python setup_project_dir.py

[download raw dataset]
$ python clone_dataset.py
```