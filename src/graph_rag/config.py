import os
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from neo4j_graphrag.embeddings import Embedder

load_dotenv()

class Neo4jCompatibleEmbedder(Embedder):
    def __init__(self, model_name='VISAI-AI/nitibench-ccl-human-finetuned-bge-m3'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_fp16 = True if device == 'cuda' else False
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    def embed_query(self, text: str) -> list[float]:
        result = self.model.encode(text)
        embedding = result.get('dense_vecs') if isinstance(result, dict) else result
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(x) for x in embedding]

    def embed_nodes(self, nodes):
        for node in nodes:
            node.embedding = self.embed_query(node.text)

def get_neo4j_credentials():
    return {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USERNAME"),
        "pwd": os.getenv("NEO4J_PASSWORD"),
        "db": os.getenv("NEO4J_DATABASE")
    }