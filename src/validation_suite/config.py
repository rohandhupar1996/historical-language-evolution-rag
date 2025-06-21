# ==========================================
# FILE: validation_suite/config.py
# ==========================================
"""Configuration for validation suite."""

from typing import List

# Archaic spelling patterns for validation
ARCHAIC_PATTERNS: List[str] = [
    r'.*th.*',      # thun -> tun
    r'.*uo.*',      # guot -> gut  
    r'.*ey.*',      # archaic diphthongs
    r'.*ck$',       # archaic endings
    r'^v[aeiou]',   # vmb -> um
]

# Known spelling changes to test
KNOWN_CHANGES = {
    'th_to_t': (r'.*th.*', r'.*t.*'),
    'uo_to_u': (r'.*uo.*', r'.*u.*'),
    'v_to_u': (r'^v.*', r'^u.*'),
    'ck_changes': (r'.*ck.*', r'.*k.*'),
}

# Expected POS categories for German
EXPECTED_POS = ['NN', 'ART', 'VVFIN', 'ADV', 'ADJA', 'APPR', 'PRON']

# Essential fields for RAG pipeline
ESSENTIAL_FIELDS = ['doc_id', 'period', 'genre', 'original', 'normalized']

# Sample queries for testing
SAMPLE_QUERIES = [
    "How did 'thun' become 'tun'?",
    "When did German spelling change?",
    "What words changed between 1600-1800?",
    "Show evolution of verb endings",
    "Compare religious vs scientific language"
]

# Validation thresholds
THRESHOLDS = {
    'min_variant_rate': 0.05,
    'min_major_pos': 4,
    'max_empty_docs_rate': 0.1,
    'min_tokens': 10000,
    'min_variants': 100,
    'rag_readiness_threshold': 80
}