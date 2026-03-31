from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class RAGConfig(BaseModel):
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "thai_laws_collection"
    vector_dense: str = "dense"
    vector_sparse: str = "sparse"

    # Models
    embed_model_name: str = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3"
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"

    # Limits
    retrieval_limit: int = 3
    reranking_limit: int = 3
    final_limit: Optional[int] = None