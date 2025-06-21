# ==========================================
# FILE: validation_suite/linguistic_validator.py
# ==========================================
"""Linguistic feature validation."""

import pandas as pd
from typing import Dict, Any
from .config import EXPECTED_POS, THRESHOLDS


class LinguisticValidator:
    """Validates POS tags and morphological features."""
    
    def validate(self, tokens_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate linguistic features."""
        pos_tags = tokens_df['pos'].value_counts() if 'pos' in tokens_df.columns else pd.Series()
        
        found_major_pos = 0
        if not pos_tags.empty:
            found_major_pos = sum(1 for pos in EXPECTED_POS if any(pos in tag for tag in pos_tags.index))
        
        morph_features = tokens_df['morphology'].dropna() if 'morphology' in tokens_df.columns else pd.Series()
        
        return {
            'pos_tag_count': len(pos_tags),
            'major_pos_found': found_major_pos,
            'morph_features_present': len(morph_features) > 0,
            'pos_distribution': dict(pos_tags.head(10)) if not pos_tags.empty else {},
            'valid': found_major_pos >= THRESHOLDS['min_major_pos']
        }