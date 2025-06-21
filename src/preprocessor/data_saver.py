# ==========================================
# FILE: gate_preprocessor/data_saver.py
# ==========================================
"""Data saving utilities."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging


class DataSaver:
    """Handles saving processed data in multiple formats."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
    
    def save_all_data(self, documents: List[Dict], tokens: List[Dict], 
                     linguistic_features: Dict) -> None:
        """Save all processed data."""
        self.save_documents(documents)
        self.save_tokens(tokens)
        self.save_linguistic_features(linguistic_features)
        
        self.logger.info(f"Data saved to {self.output_dir}/")
        print(f"📁 Saved to {self.output_dir}/")
        print(f"   - documents.json ({len(documents)} docs)")
        print(f"   - tokens.csv ({len(tokens)} tokens)")
        print(f"   - linguistic_features.json")
    
    def save_documents(self, documents: List[Dict]) -> None:
        """Save documents metadata as JSON."""
        docs_file = self.output_dir / "documents.json"
        
        # Convert sets to lists for JSON serialization
        docs_for_json = []
        for doc in documents:
            doc_copy = doc.copy()
            if 'unique_words' in doc_copy and isinstance(doc_copy['unique_words'], set):
                doc_copy['unique_words'] = list(doc_copy['unique_words'])
            docs_for_json.append(doc_copy)
        
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_for_json, f, indent=2, ensure_ascii=False)
    
    def save_tokens(self, tokens: List[Dict]) -> None:
        """Save tokens as CSV."""
        if tokens:
            tokens_df = pd.DataFrame(tokens)
            tokens_file = self.output_dir / "tokens.csv"
            tokens_df.to_csv(tokens_file, index=False, encoding='utf-8')
    
    def save_linguistic_features(self, linguistic_features: Dict) -> None:
        """Save linguistic features as JSON."""
        features_file = self.output_dir / "linguistic_features.json"
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(dict(linguistic_features), f, indent=2, ensure_ascii=False)