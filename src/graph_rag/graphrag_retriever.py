from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from .config import Neo4jCompatibleEmbedder, get_neo4j_credentials
from neo4j import GraphDatabase
import os

SYSTEM_PROMPT_TEMPLATE = """คุณคือผู้ช่วยด้านกฎหมายไทย (Thai Legal AI Assistant) ที่มีความเชี่ยวชาญด้านกฎหมายแพ่งและพาณิชย์ (CCL)
คุณจะตอบคำถามโดยอิงจากข้อความกฎหมายที่ถูกดึงมาเท่านั้น ห้ามอนุมานหรือแต่งเติมข้อมูลที่ไม่มีในบริบท

## กฎการตอบ
1. อ้างอิง **ชื่อกฎหมาย** และ **มาตรา** ที่เกี่ยวข้องทุกครั้ง
2. บริบทถูกจัดเรียงตาม score จากสูงไปต่ำ — กฎหมายที่มี score สูงสุดคือที่เกี่ยวข้องมากที่สุด ให้ใช้เป็นหลักในการตอบ
3. บริบทอาจมีกฎหมายหลายมาตราจากหลายฉบับ หากกฎหมายเหล่านั้นมี score สูงเท่ากัน หรืออยู่ในลำดับต้นๆ ของบริบท ให้พิจารณาร่วมกัน เพราะอาจเป็นกฎหมายที่อ้างอิงถึงกัน โดยอาจเป็นข้อยกเว้น เงื่อนไข หรือบทบัญญัติที่เสริมกัน
4. บริบทไม่อ้างอิงกฎหมายฉบับเก่า หรือ เมื่อพบบริบทกฎหมายฉบับเดียวกันแต่ต่างปี ให้เลือกกฎหมายฉบับล่าสุด สังเกตจาก law_name หรือ metadata
5. หากบริบทไม่มีข้อมูลเพียงพอ ให้แจ้งว่า "ไม่พบข้อมูลที่ตรงกับคำถามในฐานข้อมูลกฎหมายที่มีอยู่"
6. ตอบเป็นภาษาไทยที่ชัดเจน กระชับ และเข้าใจง่าย
7. หากมีโทษทางอาญา ให้ระบุอัตราโทษอย่างครบถ้วน (จำคุก / ปรับ / ทั้งจำทั้งปรับ)

## รูปแบบการตอบ
[ตอบโดยตรง 1-2 ประโยค เป็นการสรุปคำตอบ]

---
## บริบทจากฐานข้อมูลกฎหมาย
{context}
"""

class LegalRetriever:
    def __init__(self):
        creds = get_neo4j_credentials()
        self.driver = GraphDatabase.driver(creds["uri"], auth=(creds["user"], creds["pwd"]))
        self.db_name = creds["db"]
        self.embedder = Neo4jCompatibleEmbedder()
        
        # Initialize Typhoon LLM via OpenRouter or direct API
        os.environ["OPENAI_API_KEY"] = os.getenv("thai_llm_API_key")
        self.llm = OpenAILLM(
            model_name="typhoon-v2.5-30b-a3b-instruct",
            base_url="https://api.opentyphoon.ai/v1",
            model_params={"temperature": 0.1, "max_tokens": 23113}
        )

    def get_retriever(self, cypher_query: str = None):
        # Cypher query designed to pull the parent section and its related law texts
        if cypher_query is None:
            cypher_query = """
            OPTIONAL MATCH (node)-[:REFERENCES_TO]->(referencedSection)
            WITH node, score, 
                collect({
                    law_name: referencedSection.law_name, 
                    section_num: referencedSection.section_num,
                    text: referencedSection.text
                }) AS children_nodes
            RETURN 
                node.text AS content, 
                score, 
                {
                    parent_law_name: node.law_name,
                    parent_section_num: node.section_num,
                    children: [c in children_nodes WHERE c.law_name IS NOT NULL]
                } AS metadata
            """
        return VectorCypherRetriever(
            driver=self.driver,
            index_name="text_embeddings", # Ensure this vector index exists in Neo4j
            embedder=self.embedder,
            retrieval_query=cypher_query,
            neo4j_database=self.db_name
        )

    def get_answer(self, query_text: str):
        retriever = self.get_retriever()
        rag = GraphRAG(llm=self.llm, retriever=retriever)
        return rag.search(query_text=query_text, retriever_config={"top_k": 5})