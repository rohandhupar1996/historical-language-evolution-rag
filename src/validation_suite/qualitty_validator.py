# ==========================================
# FILE: validation_suite/quality_validator.py
# ==========================================
"""Text quality validation."""

from typing import List, Dict, Any


class QualityValidator:
    """Validates text extraction quality."""
    
    def validate(self, documents: List[Dict], tokens_df) -> Dict[str, Any]:
        """Validate text quality."""
        short_docs = 0
        empty_docs = 0
        
        for doc in documents:
            token_count = doc.get('token_count', 0)
            if token_count == 0:
                empty_docs += 1
            elif token_count < 50:
                short_docs += 1
        
        avg_token_length = 0
        quality_issues = []
        
        if not tokens_df.empty and 'word_length' in tokens_df.columns:
            avg_token_length = tokens_df['word_length'].mean()
            
            if avg_token_length < 3 or avg_token_length > 15:
                quality_issues.append(f"Unusual average word length: {avg_token_length:.1f}")
        
        return {
            'empty_documents': empty_docs,
            'short_documents': short_docs,
            'average_token_length': avg_token_length,
            'issues': quality_issues,
            'valid': empty_docs < len(documents) * 0.1
        }