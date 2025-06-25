# ==========================================
# FILE: validation_suite/query_tester.py
# ==========================================
"""Query testing utilities."""

from typing import Dict, Any, List
from .config import SAMPLE_QUERIES


class QueryTester:
    """Tests sample queries for RAG readiness."""
    
    def test_queries(self, documents: List[Dict], tokens_df, linguistic_features: Dict, 
                    validation_results: Dict) -> Dict[str, Any]:
        """Test sample RAG queries."""
        query_results = {}
        
        for query in SAMPLE_QUERIES:
            relevant_data = self._simulate_query_processing(
                query, documents, tokens_df, linguistic_features, validation_results
            )
            query_results[query] = relevant_data
        
        return query_results
    
    def _simulate_query_processing(self, query: str, documents: List[Dict], tokens_df, 
                                 linguistic_features: Dict, validation_results: Dict) -> Dict[str, Any]:
        """Simulate query processing."""
        can_answer = True
        evidence_count = 0
        
        # Spelling evolution queries
        if any(word in query.lower() for word in ['thun', 'tun', 'spelling', 'change']):
            variants = linguistic_features.get('spelling_variants', [])
            evidence_count = len([v for v in variants if 'th' in v.get('original', '').lower()])
            can_answer = evidence_count > 5
        
        # Temporal queries
        elif any(word in query.lower() for word in ['when', 'between', 'evolution']):
            periods = validation_results.get('temporal', {}).get('periods_found', [])
            can_answer = len(periods) >= 2
            evidence_count = len(periods) * 100
        
        # Genre comparison queries
        elif any(word in query.lower() for word in ['religious', 'scientific', 'compare']):
            genres = validation_results.get('completeness', {}).get('genre_distribution', {})
            can_answer = len(genres) >= 2
            evidence_count = sum(genres.values()) if isinstance(genres, dict) and genres else 0
        
        else:
            evidence_count = len(tokens_df) // 10
            can_answer = len(tokens_df) > 1000
        
        return {
            'can_answer': can_answer,
            'evidence_count': evidence_count
        }