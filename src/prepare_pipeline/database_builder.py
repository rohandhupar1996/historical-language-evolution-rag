# ==========================================
# FILE: prepare_pipeline/database_builder.py
# ==========================================
"""Database table builder for PostgreSQL."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from .config import CHUNK_TABLE_COLUMNS, VARIANT_TABLE_COLUMNS, WORD_FREQ_COLUMNS, FEATURES_TABLE_COLUMNS


class DatabaseBuilder:
    """Builds PostgreSQL-ready data structures."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
    
    def create_chunks_table(self, chunks: List[Dict]) -> pd.DataFrame:
        """Create chunks table for PostgreSQL."""
        chunks_data = []
        
        for chunk in chunks:
            row = {col: chunk.get(col.replace('_', ''), chunk.get(col)) for col in CHUNK_TABLE_COLUMNS}
            chunks_data.append(row)
        
        chunks_df = pd.DataFrame(chunks_data)
        chunks_path = self.output_dir / "chunks_table.csv"
        chunks_df.to_csv(chunks_path, index=False, encoding='utf-8')
        
        return chunks_df
    
    def create_spelling_variants_table(self, chunks: List[Dict]) -> pd.DataFrame:
        """Create spelling variants table."""
        variants_data = []
        
        for chunk in chunks:
            for variant in chunk.get('spelling_variants', []):
                variants_data.append({
                    'chunk_id': chunk['chunk_id'],
                    'period': chunk['period'],
                    'genre': chunk['genre'],
                    'original': variant['original'],
                    'normalized': variant['normalized'],
                    'pos': variant['pos'],
                    'position_in_chunk': variant['position_in_chunk']
                })
        
        if variants_data:
            variants_df = pd.DataFrame(variants_data)
            variants_path = self.output_dir / "spelling_variants_table.csv"
            variants_df.to_csv(variants_path, index=False, encoding='utf-8')
            return variants_df
        
        return pd.DataFrame()
    
    def create_word_frequencies_table(self, word_freq_data: List[Dict]) -> pd.DataFrame:
        """Create word frequencies table."""
        if word_freq_data:
            word_freq_df = pd.DataFrame(word_freq_data)
            word_freq_path = self.output_dir / "word_frequencies_table.csv"
            word_freq_df.to_csv(word_freq_path, index=False, encoding='utf-8')
            return word_freq_df
        
        return pd.DataFrame()
    
    def create_linguistic_features_table(self, chunks: List[Dict]) -> pd.DataFrame:
        """Create linguistic features table."""
        features_data = []
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            period = chunk['period']
            genre = chunk['genre']
            
            # POS features
            for pos, count in chunk.get('linguistic_features', {}).get('pos_distribution', {}).items():
                features_data.append({
                    'chunk_id': chunk_id,
                    'period': period,
                    'genre': genre,
                    'feature_type': 'pos',
                    'feature_name': pos,
                    'frequency': count,
                    'relative_frequency': count / chunk.get('token_count', 1)
                })
            
            # Spelling variants features
            for variant in chunk.get('spelling_variants', []):
                features_data.append({
                    'chunk_id': chunk_id,
                    'period': period,
                    'genre': genre,
                    'feature_type': 'spelling_variant',
                    'feature_name': f"{variant['original']}→{variant['normalized']}",
                    'frequency': 1,
                    'relative_frequency': 1 / chunk.get('token_count', 1)
                })
        
        if features_data:
            features_df = pd.DataFrame(features_data)
            features_path = self.output_dir / "linguistic_features_db.csv"
            features_df.to_csv(features_path, index=False, encoding='utf-8')
            return features_df
        
        return pd.DataFrame()
