# ==========================================
# FILE: prepare_pipeline/processor.py
# ==========================================
"""Main prepare processor class."""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from src.prepare_pipeline.config_creator import ChunkCreator
from src.prepare_pipeline.feature_extractor import FeatureExtractor
from src.prepare_pipeline.database_builder import DatabaseBuilder
from src.prepare_pipeline.statistics_calculator import StatisticsCalculator
from src.prepare_pipeline.utils import setup_logging, save_json, print_phase_summary
from src.prepare_pipeline.config import DEFAULT_CHUNK_SIZE


class GerManCPrepareProcessor:
    """Main processor for PREPARE phase."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.chunk_creator = ChunkCreator()
        self.feature_extractor = FeatureExtractor()
        self.database_builder = DatabaseBuilder(self.output_dir)
        self.stats_calculator = StatisticsCalculator()
        
        self.logger = setup_logging()
        self._load_data()
    
    def _load_data(self) -> None:
        """Load processed GerManC data."""
        print("📚 Loading data...")
        
        self.tokens_df = pd.read_csv(self.input_dir / "tokens.csv")
        
        with open(self.input_dir / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(self.input_dir / "linguistic_features.json", 'r', encoding='utf-8') as f:
            self.linguistic_features = json.load(f)
        
        print(f"✓ {len(self.tokens_df)} tokens, {len(self.documents)} documents")
    
    def run_prepare_phase(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Tuple[List[Dict], pd.DataFrame, Dict]:
        """Execute complete PREPARE phase."""
        print("🚀 Phase 3: PREPARE")
        print("=" * 40)
        
        # 1. Create temporal chunks
        chunks = self._create_chunks(chunk_size)
        
        # 2. Extract features and variants
        chunks = self._extract_features(chunks)
        
        # 3. Create PostgreSQL tables
        tables = self._create_database_tables(chunks)
        
        # 4. Save outputs
        self._save_outputs(chunks, tables)
        
        # 5. Calculate statistics
        stats = self.stats_calculator.calculate_statistics(chunks)
        
        print_phase_summary(len(chunks), self.output_dir)
        
        return chunks, tables['features'], stats
    
    def _create_chunks(self, chunk_size: int) -> List[Dict]:
        """Create temporal chunks."""
        print(f"\n🔄 Creating temporal chunks (size: {chunk_size})...")
        
        self.chunk_creator.chunk_size = chunk_size
        chunks = self.chunk_creator.create_temporal_chunks(self.documents, self.tokens_df)
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def _extract_features(self, chunks: List[Dict]) -> List[Dict]:
        """Extract features from chunks."""
        print("\n🔍 Extracting features...")
        
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Extract spelling variants
            tokens = chunk.get('tokens', [])
            chunk['spelling_variants'] = self.feature_extractor.extract_spelling_variants(tokens)
            
            # Calculate linguistic features
            chunk['linguistic_features'] = self.feature_extractor.calculate_linguistic_features(tokens)
            
            # Remove tokens to save memory
            chunk.pop('tokens', None)
        
        return chunks
    
    def _create_database_tables(self, chunks: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Create all database tables."""
        print("\n💾 Creating PostgreSQL tables...")
        
        tables = {}
        
        # Chunks table
        tables['chunks'] = self.database_builder.create_chunks_table(chunks)
        
        # Spelling variants table
        tables['variants'] = self.database_builder.create_spelling_variants_table(chunks)
        
        # Word frequencies table
        word_freq_data = self.feature_extractor.extract_word_frequencies(chunks)
        tables['word_freq'] = self.database_builder.create_word_frequencies_table(word_freq_data)
        
        # Linguistic features table
        tables['features'] = self.database_builder.create_linguistic_features_table(chunks)
        
        print(f"✓ PostgreSQL tables created")
        return tables
    
    def _save_outputs(self, chunks: List[Dict], tables: Dict[str, pd.DataFrame]) -> None:
        """Save all outputs."""
        # Save chunks as JSON
        chunks_path = self.output_dir / "temporal_chunks.json"
        save_json(chunks, chunks_path)
        
        # Save statistics
        stats = self.stats_calculator.calculate_statistics(chunks)
        stats_path = self.output_dir / "prepare_statistics.json"
        save_json(stats, stats_path)
