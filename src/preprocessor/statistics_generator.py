# ==========================================
# FILE: gate_preprocessor/statistics_generator.py
# ==========================================
"""Statistics generation utilities."""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


class StatisticsGenerator:
    """Generates corpus statistics and reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
    
    def generate_and_save_statistics(self, documents: List[Dict], tokens: List[Dict],
                                   linguistic_features: Dict, processing_stats: Dict) -> None:
        """Generate and save comprehensive statistics."""
        stats = self._calculate_statistics(documents, tokens, linguistic_features, processing_stats)
        
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"   - statistics.json")
    
    def _calculate_statistics(self, documents: List[Dict], tokens: List[Dict],
                            linguistic_features: Dict, processing_stats: Dict) -> Dict:
        """Calculate comprehensive corpus statistics."""
        if not documents:
            return {'processing_summary': processing_stats}
        
        # Period and genre distributions
        period_stats = defaultdict(int)
        genre_stats = defaultdict(int)
        year_stats = defaultdict(int)
        
        for doc in documents:
            period_stats[doc['period']] += 1
            genre_stats[doc['genre']] += 1
            if doc.get('year'):
                year_stats[doc['year']] += 1
        
        # Spelling variants by period
        variants_by_period = defaultdict(int)
        for variant in linguistic_features.get('spelling_variants', []):
            variants_by_period[variant['period']] += 1
        
        # Token statistics
        pos_distribution = defaultdict(int)
        archaic_count = 0
        
        for token in tokens:
            if token.get('pos'):
                pos_distribution[token['pos']] += 1
            if token.get('has_archaic_spelling'):
                archaic_count += 1
        
        return {
            'processing_summary': processing_stats,
            'corpus_stats': {
                'total_documents': len(documents),
                'total_tokens': len(tokens),
                'total_sentences': sum(len(doc.get('sentences', [])) for doc in documents),
                'average_tokens_per_doc': len(tokens) / len(documents) if documents else 0,
                'average_sentences_per_doc': sum(len(doc.get('sentences', [])) for doc in documents) / len(documents) if documents else 0
            },
            'temporal_distribution': {
                'period_distribution': dict(period_stats),
                'year_distribution': dict(sorted(year_stats.items()))
            },
            'genre_distribution': dict(genre_stats),
            'linguistic_features': {
                'pos_distribution': dict(pos_distribution),
                'spelling_variants_total': len(linguistic_features.get('spelling_variants', [])),
                'variants_by_period': dict(variants_by_period),
                'archaic_spelling_count': archaic_count,
                'spelling_variant_rate': len(linguistic_features.get('spelling_variants', [])) / len(tokens) if tokens else 0,
                'archaic_spelling_rate': archaic_count / len(tokens) if tokens else 0
            }
        }
