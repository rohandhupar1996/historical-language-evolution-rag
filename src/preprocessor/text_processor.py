# ==========================================
# FILE: gate_preprocessor/text_processor.py
# ==========================================
"""Text processing utilities."""

import re
from typing import List, Dict, Any
from .config import SENTENCE_ENDERS, PUNCTUATION_CHARS


class TextProcessor:
    """Handles text processing and normalization."""
    
    def __init__(self):
        self.sentence_enders = SENTENCE_ENDERS
        self.punctuation_chars = PUNCTUATION_CHARS
    
    def is_sentence_end(self, token: str) -> bool:
        """Check if token marks sentence boundary."""
        return token.strip() in self.sentence_enders
    
    def is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation."""
        return all(c in self.punctuation_chars for c in token.strip())
    
    def extract_year(self, filename: str) -> int:
        """Extract year from filename."""
        match = re.search(r'_(\d{4})_', filename)
        return int(match.group(1)) if match else None
    
    def extract_region(self, filename: str) -> str:
        """Extract region code from filename."""
        match = re.search(r'_([A-Za-z]+)_\d{4}_', filename)
        return match.group(1) if match else None