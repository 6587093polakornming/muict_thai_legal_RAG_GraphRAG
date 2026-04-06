from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from .config import Neo4jCompatibleEmbedder, get_neo4j_credentials
from neo4j import GraphDatabase
import os

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
            model_params={"temperature": 0.1, "max_tokens": 8192}
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

    def search(self, query_text: str):
        retriever = self.get_retriever()
        rag = GraphRAG(llm=self.llm, retriever=retriever)
        return rag.search(query_text=query_text, retriever_config={"top_k": 5})