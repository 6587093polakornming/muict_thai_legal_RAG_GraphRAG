from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from .config import Neo4jCompatibleEmbedder, get_neo4j_credentials
from neo4j import GraphDatabase
import os, time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

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

CYPHER_QUERY = """
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


class GraphRAGRetriever:
    def __init__(self, llm):
        creds = get_neo4j_credentials()
        self.driver = GraphDatabase.driver(creds["uri"], auth=(creds["user"], creds["pwd"]))
        self.db_name = creds["db"]
        self.embedder = Neo4jCompatibleEmbedder()
        # Initialize Typhoon LLM via OpenRouter or direct API
        # os.environ["OPENAI_API_KEY"] = os.getenv("thai_llm_API_key")
        # self.llm = OpenAILLM(
        #     model_name="typhoon-v2.5-30b-a3b-instruct",
        #     base_url="https://api.opentyphoon.ai/v1",
        #     model_params={"temperature": 0, "max_tokens": 8196}
        # )

        self.llm = llm
        # ChatOpenAI(
        #     model_name="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นที่ท่านต้องการใช้
        #     # model_name="openai/gpt-4o-mini",  # หรือรุ่นที่ท่านต้องการใช้
        #     openai_api_key=os.getenv("thai_llm_API_key"),
        #     openai_api_base="https://api.opentyphoon.ai/v1",  # สำคัญ: ใส่แทน base_url เดิม
        #     # openai_api_base="https://openrouter.ai/api/v1",  # สำคัญ: ใส่แทน base_url เดิม
        #     temperature=0,
        #     max_tokens=16384,
        # )

    def get_retriever(self):
        return VectorCypherRetriever(
            driver=self.driver,
            index_name="text_embeddings", # Ensure this vector index exists in Neo4j
            embedder=self.embedder,
            retrieval_query=CYPHER_QUERY,
            neo4j_database=self.db_name
        )

    def get_retrieve(self, query: str) -> List[Document]:
        docs = []

        retriever = self.get_retriever()
        context = retriever.get_search_results(query_text=query, top_k=3)
        for i in range(len(context.records)):
            record = context.records[i]
            content = record.data().get("content", "")
            metadata = record.data().get("metadata", {})
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "law_name": metadata.get("parent_law_name", ""),
                        "section_num": metadata.get("parent_section_num", ""),
                        "reference_laws": metadata.get("children", []),
                        "score": record.data().get("score", 0)
                    }
                )
            )
            # print(docs)
        return docs

    def format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "ไม่พบข้อมูลกฎหมายที่เกี่ยวข้อง"
        parts = []
        for i, doc in enumerate(docs, start=1):
            m = doc.metadata
            header = (
                f"Rank [{i}] Score:{m.get('score')} "
                f"ชื่อกฎหมาย:{m.get('law_name', 'ไม่ทราบชื่อกฎหมาย')} "
                f"มาตรา {m.get('section_num', '-')}"
            )
            body = doc.page_content

            ref_lines = ""
            reference_laws = m.get('reference_laws', [])
            if reference_laws:
                refs = [
                    f"  - {r.get('law_name', '')} มาตรา {r.get('section_num', '')} : {r.get('text', '')}"
                    for r in reference_laws
                ]
                ref_lines = "\n[กฎหมายที่อ้างถึง]\n" + "\n".join(refs)

            parts.append(f"{header}\n{body}{ref_lines}")

        return "\n\n".join(parts)

    def debug(self, query: str) -> dict:
        """แสดง intermediate results ทุก step สำหรับ debugging"""
        start_retrieve = time.perf_counter()

        docs = self.get_retrieve(query=query)
        retrieve_time = time.perf_counter() - start_retrieve

        context = self.format_context(docs)
        start_llm = time.perf_counter()

        answer, token_usage = self._call_llm(query)
        llm_time = time.perf_counter() - start_llm

        total_elapsed = retrieve_time + llm_time
        time_elapsed = {
            "retrieve_time": retrieve_time,
            "llm_time": llm_time,
            "total_elapsed": total_elapsed,
        }

        return {
            "query": query,
            "final": len(docs),
            "docs_candidates": docs,
            "context": context,
            "answer": answer,
            "token": token_usage,
            "time_elapsed": time_elapsed
        }

    def _call_llm(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        สร้าง prompt จาก docs ที่ retrieve มาแล้ว แล้วเรียก LLM โดยตรง
        """
        
        docs = self.get_retrieve(query=query)
        context = self.format_context(docs)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        messages = [SystemMessage(content=system_content), HumanMessage(content=query)]
        response = self.llm.invoke(messages)
        token_usage = response.response_metadata.get("token_usage", {})

        return response.content, token_usage
