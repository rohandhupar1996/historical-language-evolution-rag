# ==========================================
# FILE: gate_preprocessor/linguistic_analyzer.py
# ==========================================
"""Linguistic analysis utilities."""

import re
from typing import List
from .config import ARCHAIC_PATTERNS, PUNCTUATION_CHARS


class LinguisticAnalyzer:
    """Analyzes linguistic features of tokens."""
    
    def __init__(self):
        self.archaic_patterns = ARCHAIC_PATTERNS
        self.punctuation_chars = PUNCTUATION_CHARS
    
    def is_archaic_spelling(self, word: str) -> bool:
        """Identify archaic spelling patterns in Early New High German."""
        if not word or len(word) < 2:
            return False
            
        word_lower = word.lower()
        return any(re.search(pattern, word_lower) for pattern in self.archaic_patterns)
    
    def is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation."""
        return all(c in self.punctuation_chars for c in token.strip())
