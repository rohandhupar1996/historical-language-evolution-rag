# ==========================================
# FILE: access_pipeline/database_setup.py
# ==========================================
"""Database setup utilities."""

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from typing import Dict
import logging
from .config import SCHEMA_SQL, INDEXES_SQL, DROP_TABLES_SQL


class DatabaseSetup:
    """Handles PostgreSQL database setup."""
    
    def __init__(self, db_config: Dict, prepare_dir: Path):
        self.db_config = db_config
        self.prepare_dir = Path(prepare_dir)
        self.engine = self._create_connection()
        self.logger = logging.getLogger(__name__)
    
    def _create_connection(self):
        """Create SQLAlchemy engine."""
        connection_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string)
    
    def create_schema(self):
        """Create database tables."""
        print("📊 Creating database schema...")
        
        with self.engine.connect() as conn:
            conn.execute(text(SCHEMA_SQL))
            conn.commit()
        
        print("✓ Database schema created")
    
    def create_indexes(self):
        """Create performance indexes."""
        print("🚀 Creating indexes...")
        
        with self.engine.connect() as conn:
            conn.execute(text(INDEXES_SQL))
            conn.commit()
        
        print("✓ Indexes created")
    
    def drop_tables(self):
        """Drop existing tables."""
        with self.engine.connect() as conn:
            conn.execute(text(DROP_TABLES_SQL))
            conn.commit()
    
    def import_data(self):
        """Import CSV data from PREPARE phase."""
        print("📥 Importing data...")
        
        # Drop existing tables first
        self.drop_tables()
        
        # Import chunks
        chunks_path = self.prepare_dir / "chunks_table.csv"
        if chunks_path.exists():
            chunks_df = pd.read_csv(chunks_path)
            chunks_df.to_sql('chunks', self.engine, if_exists='replace', index=False)
            print(f"✓ Imported {len(chunks_df)} chunks")
        
        # Import spelling variants
        variants_path = self.prepare_dir / "spelling_variants_table.csv"
        if variants_path.exists():
            variants_df = pd.read_csv(variants_path)
            variants_df.to_sql('spelling_variants', self.engine, if_exists='replace', index=False)
            print(f"✓ Imported {len(variants_df)} spelling variants")
        
        # Import word frequencies
        word_freq_path = self.prepare_dir / "word_frequencies_table.csv"
        if word_freq_path.exists():
            word_freq_df = pd.read_csv(word_freq_path)
            word_freq_df.to_sql('word_frequencies', self.engine, if_exists='replace', index=False)
            print(f"✓ Imported {len(word_freq_df)} word frequencies")
        
        # Import linguistic features
        features_path = self.prepare_dir / "linguistic_features_db.csv"
        if features_path.exists():
            features_df = pd.read_csv(features_path)
            features_df.to_sql('linguistic_features', self.engine, if_exists='replace', index=False)
            print(f"✓ Imported {len(features_df)} linguistic features")