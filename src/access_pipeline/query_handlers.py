# ==========================================
# FILE: access_pipeline/query_handlers.py
# ==========================================
"""Query handling utilities."""

from sqlalchemy import create_engine, text
from typing import Dict, List, Any, Optional
from .models import QueryRequest, EvolutionQuery, SearchRequest


class QueryHandlers:
    """Handles database queries for API."""
    
    def __init__(self, db_config: Dict):
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    
    def query_word_evolution(self, word: str, start_period: str, end_period: str) -> Dict[str, Any]:
        """Track word evolution across periods."""
        query = text("""
            SELECT period, COUNT(*) as frequency, 
                   ARRAY_AGG(DISTINCT original) as variants
            FROM spelling_variants 
            WHERE (original = :word OR normalized = :word)
            AND period BETWEEN :start_period AND :end_period
            GROUP BY period 
            ORDER BY period
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'word': word,
                'start_period': start_period,
                'end_period': end_period
            })
            
            evolution_data = []
            for row in result:
                evolution_data.append({
                    'period': row.period,
                    'frequency': row.frequency,
                    'variants': row.variants
                })
            
            return {
                'word': word,
                'period_range': f"{start_period}-{end_period}",
                'evolution': evolution_data
            }
    
    def linguistic_analysis(self, request: QueryRequest) -> Dict[str, Any]:
        """Perform linguistic pattern analysis."""
        conditions = []
        params = {}
        
        if request.period:
            conditions.append("period = :period")
            params['period'] = request.period
        
        if request.genre:
            conditions.append("genre = :genre")
            params['genre'] = request.genre
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = text(f"""
            SELECT feature_type, feature_name, 
                   SUM(frequency) as total_frequency,
                   AVG(relative_frequency) as avg_relative_frequency
            FROM linguistic_features 
            WHERE {where_clause}
            GROUP BY feature_type, feature_name
            ORDER BY total_frequency DESC
            LIMIT :limit
        """)
        
        params['limit'] = request.limit
        
        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            
            analysis_data = []
            for row in result:
                analysis_data.append({
                    'feature_type': row.feature_type,
                    'feature_name': row.feature_name,
                    'total_frequency': row.total_frequency,
                    'avg_relative_frequency': float(row.avg_relative_frequency)
                })
            
            return {
                'filters': {
                    'period': request.period,
                    'genre': request.genre
                },
                'results': analysis_data
            }
    
    def get_temporal_patterns(self) -> Dict[str, Any]:
        """Get temporal distribution patterns."""
        query = text("""
            SELECT period, genre, COUNT(*) as chunk_count,
                   AVG(token_count) as avg_token_count
            FROM chunks
            GROUP BY period, genre
            ORDER BY period, genre
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query)
            
            patterns = []
            for row in result:
                patterns.append({
                    'period': row.period,
                    'genre': row.genre,
                    'chunk_count': row.chunk_count,
                    'avg_token_count': float(row.avg_token_count)
                })
            
            return {'temporal_patterns': patterns}
    
    def full_text_search(self, query_text: str, period: Optional[str] = None, 
                        limit: int = 50) -> Dict[str, Any]:
        """Full-text search in historical texts."""
        conditions = ["to_tsvector('german', normalized_text) @@ plainto_tsquery('german', :query)"]
        params = {'query': query_text, 'limit': limit}
        
        if period:
            conditions.append("period = :period")
            params['period'] = period
        
        where_clause = " AND ".join(conditions)
        
        search_query = text(f"""
            SELECT chunk_id, period, genre, 
                   ts_headline('german', normalized_text, plainto_tsquery('german', :query)) as highlighted_text,
                   ts_rank(to_tsvector('german', normalized_text), plainto_tsquery('german', :query)) as rank
            FROM chunks
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(search_query, params)
            
            search_results = []
            for row in result:
                search_results.append({
                    'chunk_id': row.chunk_id,
                    'period': row.period,
                    'genre': row.genre,
                    'highlighted_text': row.highlighted_text,
                    'relevance_score': float(row.rank)
                })
            
            return {
                'query': query_text,
                'period_filter': period,
                'results': search_results
            }
