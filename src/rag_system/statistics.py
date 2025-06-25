# ==========================================
# FILE: rag_system/statistics.py
# ==========================================
"""Statistics calculation for RAG system."""

from sqlalchemy import create_engine, text
from typing import Dict, Any
import chromadb
from .config import COLLECTION_NAME


class StatisticsCalculator:
    """Calculates system statistics."""
    
    def __init__(self, db_config: Dict, vector_db_path: str):
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.vector_db_path = vector_db_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # PostgreSQL stats
        with self.engine.connect() as conn:
            db_stats = {}
            
            chunk_stats = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(token_count) as avg_word_count,
                    MIN(period) as earliest_period,
                    MAX(period) as latest_period,
                    COUNT(DISTINCT period) as period_count,
                    COUNT(DISTINCT genre) as genre_count
                FROM chunks
            """)).fetchone()
            
            db_stats['chunks'] = dict(chunk_stats._mapping)
            
            period_dist = conn.execute(text("""
                SELECT period, COUNT(*) as chunk_count
                FROM chunks
                GROUP BY period
                ORDER BY period
            """)).fetchall()
            
            db_stats['period_distribution'] = [dict(row._mapping) for row in period_dist]
            
            genre_dist = conn.execute(text("""
                SELECT genre, COUNT(*) as chunk_count
                FROM chunks
                GROUP BY genre
                ORDER BY chunk_count DESC
            """)).fetchall()
            
            db_stats['genre_distribution'] = [dict(row._mapping) for row in genre_dist]
        
        # ChromaDB stats
        try:
            client = chromadb.PersistentClient(path=self.vector_db_path)
            collection = client.get_collection(COLLECTION_NAME)
            vector_stats = {
                'total_embeddings': collection.count(),
                'collection_name': COLLECTION_NAME
            }
        except Exception:
            vector_stats = {'total_embeddings': 0, 'collection_name': 'Not created'}
        
        return {
            'database_stats': db_stats,
            'vector_stats': vector_stats,
            'model_info': {
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'vector_db_path': self.vector_db_path
            }
        }