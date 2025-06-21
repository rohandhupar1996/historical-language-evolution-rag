# ==========================================
# FILE: validation_suite/data_loader.py
# ==========================================
"""Data loading utilities for validation."""

import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import logging


class DataLoader:
    """Loads processed data for validation."""
    
    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_all_data(self) -> Tuple[List[Dict], pd.DataFrame, Dict, Dict]:
        """Load all processed data files."""
        try:
            documents = self._load_documents()
            tokens_df = self._load_tokens()
            linguistic_features = self._load_linguistic_features()
            statistics = self._load_statistics()
            
            return documents, tokens_df, linguistic_features, statistics
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            raise
    
    def _load_documents(self) -> List[Dict]:
        """Load documents.json."""
        with open(self.processed_dir / "documents.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_tokens(self) -> pd.DataFrame:
        """Load tokens.csv."""
        return pd.read_csv(self.processed_dir / "tokens.csv")
    
    def _load_linguistic_features(self) -> Dict:
        """Load linguistic_features.json."""
        with open(self.processed_dir / "linguistic_features.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_statistics(self) -> Dict:
        """Load statistics.json."""
        with open(self.processed_dir / "statistics.json", 'r', encoding='utf-8') as f:
            return json.load(f)
