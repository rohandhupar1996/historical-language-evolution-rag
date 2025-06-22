# ==========================================
# FILE: prepare_pipeline/statistics_calculator.py
# ==========================================
"""Statistics calculation utilities."""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


class StatisticsCalculator:
    """Calculates corpus statistics."""
    
    def calculate_statistics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive corpus statistics."""
        periods = defaultdict(int)
        genres = defaultdict(int)
        token_counts = []
        variant_counts = []
        
        for chunk in chunks:
            periods[chunk['period']] += 1
            genres[chunk['genre']] += 1
            token_counts.append(chunk['token_count'])
            variant_counts.append(len(chunk.get('spelling_variants', [])))
        
        return {
            'total_chunks': len(chunks),
            'period_distribution': dict(periods),
            'genre_distribution': dict(genres),
            'token_statistics': {
                'mean': float(np.mean(token_counts)) if token_counts else 0.0,
                'median': float(np.median(token_counts)) if token_counts else 0.0,
                'total': int(np.sum(token_counts)) if token_counts else 0
            },
            'variant_statistics': {
                'mean_per_chunk': float(np.mean(variant_counts)) if variant_counts else 0.0,
                'total_variants': int(np.sum(variant_counts)) if variant_counts else 0
            }
        }