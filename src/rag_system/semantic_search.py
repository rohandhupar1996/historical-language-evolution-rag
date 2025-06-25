# ==========================================
# FILE: rag_system/semantic_search.py
# ==========================================
"""Semantic search functionality."""

from typing import List, Dict, Optional
from .vector_store import VectorStoreManager
from .embeddings import EmbeddingManager


class SemanticSearcher:
    """Handles semantic search operations."""
    
    def __init__(self, vector_manager: VectorStoreManager, embedding_manager: EmbeddingManager):
        self.vector_manager = vector_manager
        self.embedding_manager = embedding_manager
    
    def search(self, query: str, k: int = 5, period_filter: Optional[str] = None) -> List[Dict]:
        """Perform semantic search."""
        where_clause = None
        if period_filter:
            where_clause = {"period": period_filter}
        
        results = self.vector_manager.query(query, k, where_clause)
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            result = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'chunk_id': results['ids'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results