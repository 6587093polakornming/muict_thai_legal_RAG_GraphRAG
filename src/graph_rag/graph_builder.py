import ast
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from .config import get_neo4j_credentials

class LegalKGManager:
    def __init__(self):
        creds = get_neo4j_credentials()
        self.driver = GraphDatabase.driver(creds["uri"], auth=(creds["user"], creds["pwd"]))
        self.db_name = creds["db"]

    def _process_embedding(self, val):
        """Standardizes embedding format for Neo4j."""
        if isinstance(val, (list, np.ndarray)):
            return [float(x) for x in val]
        return []

    def _process_reference_laws(self, val):
        """Cleans and parses the reference_laws column."""
        if pd.isna(val):
            return []
        if isinstance(val, (list, np.ndarray)):
            return list(val) if len(val) > 0 else []
        if isinstance(val, str):
            val = val.strip()
            if val in ["", "[]"]:
                return []
            try:
                res = ast.literal_eval(val)
                return res if isinstance(res, list) else []
            except:
                return []
        return []

    def import_legal_data(self, df):
        """Manual indexing logic for Section nodes and REFERENCES_TO links."""
        # Ensure database constraints
        self.driver.execute_query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.law_id IS UNIQUE",
            database_=self.db_name
        )

        for _, row in df.iterrows():
            law_id = f"{row['law_name']}_{row['section_num']}"
            embedding = self._process_embedding(row['dense_vector'])
            
            # 1. Insert/Update Main Section Node
            self.driver.execute_query(
                """
                MERGE (s:Section {law_id: $law_id})
                SET s.id = $id,
                    s.law_name = $law_name,
                    s.section_num = $section_num,
                    s.text = $text,
                    s.embedding = $embedding
                """,
                law_id=law_id,
                id=row['id'],
                law_name=row['law_name'],
                section_num=row['section_num'],
                text=row['text'],
                embedding=embedding,
                database_=self.db_name,
            )

            # 2. Process Relationships
            ref_laws = self._process_reference_laws(row['reference_laws'])
            for ref in ref_laws:
                if isinstance(ref, dict) and 'law_name' in ref and 'section_num' in ref:
                    target_id = f"{ref['law_name']}_{ref['section_num']}"
                    
                    self.driver.execute_query(
                        """
                        MATCH (source:Section {law_id: $source_id})
                        MERGE (target:Section {law_id: $target_id})
                        ON CREATE SET target.law_name = $t_name, 
                                      target.section_num = $t_sec
                        MERGE (source)-[:REFERENCES_TO]->(target)
                        """,
                        source_id=law_id,
                        target_id=target_id,
                        t_name=ref['law_name'],
                        t_sec=ref['section_num'],
                        database_=self.db_name,
                    )

    def close(self):
        self.driver.close()