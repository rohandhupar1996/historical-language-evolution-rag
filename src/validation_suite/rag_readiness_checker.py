# ==========================================
# FILE: validation_suite/rag_readiness_checker.py
# ==========================================
"""RAG readiness validation."""

from typing import Dict, Any
from .config import THRESHOLDS


class RAGReadinessChecker:
    """Checks if data is ready for RAG pipeline."""
    
    def check_readiness(self, validation_results: Dict, tokens_df, documents: list) -> Dict[str, Any]:
        """Check RAG readiness."""
        rag_issues = []
        readiness_score = 0
        
        # Temporal evolution data (20 points)
        if len(validation_results.get('temporal', {}).get('periods_found', [])) >= 2:
            readiness_score += 20
        else:
            rag_issues.append("Need multiple time periods for evolution tracking")
        
        # Spelling variants (25 points)
        if validation_results.get('spelling_variants', {}).get('total_variants', 0) > THRESHOLDS['min_variants']:
            readiness_score += 25
        else:
            rag_issues.append("Need more spelling variants for language change analysis")
        
        # Data volume (20 points)
        if len(tokens_df) > THRESHOLDS['min_tokens']:
            readiness_score += 20
        else:
            rag_issues.append("Need more tokens for robust analysis")
        
        # Linguistic features (15 points)
        if validation_results.get('linguistic_features', {}).get('pos_tag_count', 0) > 10:
            readiness_score += 15
        else:
            rag_issues.append("Need more linguistic features")
        
        # Data quality (20 points)
        if validation_results.get('text_quality', {}).get('empty_documents', 0) < len(documents) * 0.1:
            readiness_score += 20
        else:
            rag_issues.append("Too many empty/poor quality documents")
        
        is_ready = readiness_score >= THRESHOLDS['rag_readiness_threshold']
        
        return {
            'readiness_score': readiness_score,
            'is_ready': is_ready,
            'issues': rag_issues
        }
