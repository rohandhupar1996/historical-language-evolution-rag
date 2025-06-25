# ==========================================
# FILE: rag_system/config.py
# ==========================================
"""Configuration for RAG system."""

# Embedding model
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Vector database
DEFAULT_VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "german_corpus_chunks"

# Database configuration
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'germanc_corpus',
    'user': 'rohan',
    'password': '1996'
}

# Processing parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEARCH_K = 5
MIN_TEXT_LENGTH = 50

# Time periods for evolution analysis
DEFAULT_PERIODS = ['1050-1350', '1350-1650', '1650-1800', '1800-1900', '1900-2000']

# LLM providers
LLM_PROVIDERS = {
    'simple': 'Simple retrieval without generation',
    'openai': 'OpenAI GPT models',
    'huggingface': 'HuggingFace transformers'
}

# SQL queries
CHUNKS_QUERY = """
SELECT 
    c.chunk_id,
    c.normalized_text as text,
    c.period,
    c.token_count as word_count,
    LENGTH(c.normalized_text) as char_count,
    c.doc_id as document_id,
    ROW_NUMBER() OVER (PARTITION BY c.doc_id ORDER BY c.chunk_id) as chunk_index,
    c.year,
    c.genre,
    c.filename
FROM chunks c
WHERE c.normalized_text IS NOT NULL 
AND LENGTH(c.normalized_text) > {min_length}
ORDER BY c.period, c.doc_id, c.chunk_id
"""