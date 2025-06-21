# ==========================================
# FILE: validation_suite/spelling_validator.py
# ==========================================
"""Spelling variant validation."""

import re
from typing import List, Dict, Any
from collections import defaultdict
from .config import ARCHAIC_PATTERNS, KNOWN_CHANGES, THRESHOLDS


class SpellingValidator:
    """Validates spelling variant extraction."""
    
    def validate(self, linguistic_features: Dict, tokens_df) -> Dict[str, Any]:
        """Validate spelling variants."""
        variants = linguistic_features.get('spelling_variants', [])
        
        if not variants:
            return {
                'total_variants': 0,
                'valid_variants': 0,
                'archaic_patterns': 0,
                'variant_rate': 0,
                'known_changes': {},
                'valid': False,
                'critical_error': "NO SPELLING VARIANTS FOUND"
            }
        
        valid_variants = 0
        archaic_patterns = 0
        
        for variant in variants:
            original = variant.get('original', '')
            normalized = variant.get('normalized', '')
            
            if original and normalized and original != normalized:
                valid_variants += 1
                
                if self._has_archaic_patterns(original):
                    archaic_patterns += 1
        
        variant_rate = len(variants) / len(tokens_df) if not tokens_df.empty else 0
        known_changes = self._test_known_changes(variants)
        
        return {
            'total_variants': len(variants),
            'valid_variants': valid_variants,
            'archaic_patterns': archaic_patterns,
            'variant_rate': variant_rate,
            'known_changes': known_changes,
            'valid': len(variants) >= THRESHOLDS['min_variants'] and variant_rate >= THRESHOLDS['min_variant_rate']
        }
    
    def _has_archaic_patterns(self, word: str) -> bool:
        """Check for archaic patterns."""
        return any(re.search(pattern, word.lower()) for pattern in ARCHAIC_PATTERNS)
    
    def _test_known_changes(self, variants: List[Dict]) -> Dict[str, int]:
        """Test for known spelling changes."""
        found_changes = defaultdict(int)
        
        for variant in variants:
            original = variant.get('original', '').lower()
            normalized = variant.get('normalized', '').lower()
            
            for change_type, (old_pattern, new_pattern) in KNOWN_CHANGES.items():
                if re.search(old_pattern, original) and re.search(new_pattern, normalized):
                    found_changes[change_type] += 1
        
        return dict(found_changes)