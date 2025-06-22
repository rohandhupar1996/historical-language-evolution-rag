# ==========================================
# FILE: prepare_pipeline/feature_extractor.py
# ==========================================
"""Feature extraction utilities."""

import numpy as np
from typing import List, Dict, Any
from collections import Counter
from .config import MIN_WORD_LENGTH


class FeatureExtractor:
    """Extracts linguistic features from chunks."""
    
    def extract_spelling_variants(self, tokens: List) -> List[Dict]:
        """Extract spelling variants from tokens."""
        variants = []
        
        for i, token in enumerate(tokens):
            original = str(token.get('original', ''))
            normalized = str(token.get('normalized', ''))
            
            if original != normalized and len(original) > MIN_WORD_LENGTH:
                variants.append({
                    'original': original,
                    'normalized': normalized,
                    'pos': token.get('pos', ''),
                    'position_in_chunk': i
                })
        
        return variants
    
    def calculate_linguistic_features(self, tokens: List) -> Dict:
        """Calculate chunk-level linguistic features."""
        pos_counts = Counter()
        word_lengths = []
        
        for token in tokens:
            pos = token.get('pos', 'UNKNOWN')
            pos_counts[pos] += 1
            
            original = str(token.get('original', ''))
            if original:
                word_lengths.append(len(original))
        
        return {
            'pos_distribution': dict(pos_counts),
            'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0.0,
            'total_tokens': len(tokens),
            'unique_pos_count': len(pos_counts),
            'spelling_variant_count': sum(1 for token in tokens 
                                        if str(token.get('original', '')) != str(token.get('normalized', '')))
        }
    
    def extract_word_frequencies(self, chunks: List[Dict]) -> List[Dict]:
        """Extract word frequencies for database."""
        word_freq_data = []
        
        for chunk in chunks:
            words = chunk['normalized_text'].lower().split()
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if len(word) > MIN_WORD_LENGTH:
                    word_freq_data.append({
                        'word': word,
                        'period': chunk['period'],
                        'genre': chunk['genre'],
                        'frequency': count,
                        'chunk_id': chunk['chunk_id']
                    })
        
        return word_freq_data
