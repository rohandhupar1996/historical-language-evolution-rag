# ==========================================
# FILE: prepare_pipeline/config.py
# ==========================================
"""Configuration for prepare pipeline."""

# Chunking parameters
DEFAULT_CHUNK_SIZE = 800
MIN_CHUNK_SIZE = 50

# Database table configurations
CHUNK_TABLE_COLUMNS = [
    'chunk_id', 'doc_id', 'period', 'genre', 'year', 
    'filename', 'normalized_text', 'original_text', 'token_count'
]

VARIANT_TABLE_COLUMNS = [
    'chunk_id', 'period', 'genre', 'original', 
    'normalized', 'pos', 'position_in_chunk'
]

WORD_FREQ_COLUMNS = [
    'word', 'period', 'genre', 'frequency', 'chunk_id'
]

FEATURES_TABLE_COLUMNS = [
    'chunk_id', 'period', 'genre', 'feature_type', 
    'feature_name', 'frequency', 'relative_frequency'
]

# Text processing
SENTENCE_ENDINGS = ['.', '!', '?']
MIN_WORD_LENGTH = 2

# Feature types
FEATURE_TYPES = ['pos', 'spelling_variant', 'morphological']