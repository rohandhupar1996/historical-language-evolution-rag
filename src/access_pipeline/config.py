# ==========================================
# FILE: access_pipeline/config.py
# ==========================================
"""Configuration for ACCESS pipeline."""

# Database configuration
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'germanc_corpus',
    'user': 'rohan',
    'password': '1996'
}

# API configuration
API_CONFIG = {
    'title': "GerManC Historical Linguistics API",
    'version': "1.0.0",
    'host': "127.0.0.1",
    'port': 8000
}

# Database schema
SCHEMA_SQL = """
-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255),
    period VARCHAR(10),
    genre VARCHAR(100),
    year INTEGER,
    filename VARCHAR(255),
    normalized_text TEXT,
    original_text TEXT,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Spelling variants table
CREATE TABLE IF NOT EXISTS spelling_variants (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) REFERENCES chunks(chunk_id),
    period VARCHAR(10),
    genre VARCHAR(100),
    original VARCHAR(255),
    normalized VARCHAR(255),
    pos VARCHAR(50),
    position_in_chunk INTEGER
);

-- Word frequencies table
CREATE TABLE IF NOT EXISTS word_frequencies (
    id SERIAL PRIMARY KEY,
    word VARCHAR(255),
    period VARCHAR(10),
    genre VARCHAR(100),
    frequency INTEGER,
    chunk_id VARCHAR(255) REFERENCES chunks(chunk_id)
);

-- Linguistic features table
CREATE TABLE IF NOT EXISTS linguistic_features (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) REFERENCES chunks(chunk_id),
    period VARCHAR(10),
    genre VARCHAR(100),
    feature_type VARCHAR(50),
    feature_name VARCHAR(255),
    frequency INTEGER,
    relative_frequency FLOAT
);
"""

# Database indexes
INDEXES_SQL = """
-- Temporal indexes
CREATE INDEX IF NOT EXISTS idx_chunks_period ON chunks(period);
CREATE INDEX IF NOT EXISTS idx_chunks_genre ON chunks(period, genre);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_text_gin ON chunks USING gin(to_tsvector('german', normalized_text));

-- Spelling variants indexes
CREATE INDEX IF NOT EXISTS idx_variants_period ON spelling_variants(period);
CREATE INDEX IF NOT EXISTS idx_variants_original ON spelling_variants(original);
CREATE INDEX IF NOT EXISTS idx_variants_normalized ON spelling_variants(normalized);

-- Word frequency indexes
CREATE INDEX IF NOT EXISTS idx_word_freq_word ON word_frequencies(word);
CREATE INDEX IF NOT EXISTS idx_word_freq_period ON word_frequencies(word, period);

-- Features indexes
CREATE INDEX IF NOT EXISTS idx_features_type ON linguistic_features(feature_type);
CREATE INDEX IF NOT EXISTS idx_features_period ON linguistic_features(period, feature_type);
"""

# Drop tables SQL
DROP_TABLES_SQL = """
DROP TABLE IF EXISTS linguistic_features CASCADE;
DROP TABLE IF EXISTS word_frequencies CASCADE;
DROP TABLE IF EXISTS spelling_variants CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
"""
