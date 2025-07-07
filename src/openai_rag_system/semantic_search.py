# ==========================================
# FILE: src/openai_rag_system/semantic_search.py
# ==========================================

from typing import List, Dict, Optional
from .vector_store import OpenAIVectorStore
from .embeddings import OpenAIEmbeddingManager

class OpenAISemanticSearcher:
    def __init__(self, vector_store: OpenAIVectorStore, embedding_manager: OpenAIEmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def search(self, query: str, k: int = 5, period_filter: Optional[str] = None,
               include_similarity_scores: bool = True) -> List[Dict]:
        """Advanced semantic search using OpenAI embeddings."""
        search_params = {
            'k': k,
            'period_filter': period_filter,
            'include_context': True,
            'rerank': True
        }
        
        results = self.vector_store.advanced_search(query, search_params, self.embedding_manager)
        
        formatted_results = []
        for result in results.get('results', []):
            formatted_result = {
                'text': result['document'],
                'metadata': result['metadata'],
                'similarity_score': result['similarity_score'],
                'chunk_id': result['metadata'].get('chunk_id', ''),
                'period': result['metadata'].get('period', ''),
                'genre': result['metadata'].get('genre', ''),
                'snippet': result.get('snippet', '')
            }
            formatted_results.append(formatted_result)
        
        return formatted_results