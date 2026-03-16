# Project description
...

# Repository structure
```
root-repo/
├── data/                   # [🔴 MUST DO]
│   ├── raw/                # ไฟล์ PDF กฎหมายไทยต้นฉบับ [cite: 1]
│   └── processed/          # ไฟล์ Markdown/JSON ที่ผ่านการ Chunking แล้ว [cite: 3]
├── docker/                 # สำหรับรัน Qdrant, Neo4j, Langfuse [cite: 4]
│   └── docker-compose.yml  
├── notebooks/              # [🟡 SHOULD DO] สำหรับ EDA และทดลอง Prompt
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_prototype.ipynb
│   └── 03_graph_extraction_test.ipynb
├── src/                    # [🔴 MUST DO] Source Code หลัก
│   ├── common/             # โค้ดที่ใช้ร่วมกัน เช่น Embedding Model (BGE-M3/E5), Logging
│   ├── rag/                # RAG Pipeline (Baseline) 
│   │   ├── ingest.py       # นำข้อมูลเข้า Qdrant 
│   │   └── retriever.py    # ระบบค้นหาแบบ Vector Search
│   ├── graph_rag/          # GraphRAG Pipeline (Proposed) [cite: 1, 7]
│   │   ├── extractor.py    # สกัด Entity/Relation เข้า Neo4j 
│   │   └── search.py       # Local/Global Search Engine [cite: 7]
│   └── evaluation/         # ระบบวัดผล RAGAS / MLflow  [cite: 7, 87-90]
├── tests/                  # สำหรับเก็บชุดคำถามทดสอบ (Test Sets)
├── .env                    # เก็บ API Keys (OpenAI, Langfuse, Neo4j)
├── pyproject.toml      	# รายชื่อ Library (LangChain, Qdrant-client, etc.) pytome
└── main.py                 # Entry point สำหรับรันเปรียบเทียบผล
```