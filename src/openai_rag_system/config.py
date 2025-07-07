# ==========================================
# FILE: src/openai_rag_system/config.py
# ==========================================
"""Configuration for OpenAI RAG system."""

import os
from typing import Dict, List

# OpenAI Configuration
OPENAI_CONFIG = {
    # Models
    'embedding_model': 'text-embedding-3-large',  # Latest and most capable
    'chat_model': 'gpt-4-turbo-preview',          # Best reasoning capabilities
    'fallback_chat_model': 'gpt-3.5-turbo',      # Fallback option
    
    # Embedding settings
    'embedding_dimensions': 3072,  # text-embedding-3-large dimensions
    'embedding_batch_size': 100,   # Batch size for embeddings
    
    # Chat completion settings
    'max_tokens': 4000,           # Max tokens in response
    'temperature': 0.3,           # Low temperature for consistency
    'top_p': 0.9,                # Nucleus sampling
    
    # Rate limiting
    'requests_per_minute': 3000,   # OpenAI rate limits
    'tokens_per_minute': 250000,  # Token rate limits
    
    # Retry settings
    'max_retries': 3,
    'retry_delay': 1,             # Seconds between retries
}

# Database configuration (inherited from main system)
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'germanc_corpus',
    'user': 'rohan',
    'password': '1996'
}

# Vector database configuration
VECTOR_DB_CONFIG = {
    'collection_name': 'german_corpus_openai',
    'distance_metric': 'cosine',
    'index_type': 'hnsw',        # Hierarchical navigable small world
    'ef_construction': 200,       # Index construction parameter
    'ef_search': 100,            # Search parameter
}

# Historical German language periods
HISTORICAL_PERIODS = {
    '1050-1350': 'Middle High German',
    '1350-1650': 'Early New High German', 
    '1650-1800': 'Baroque/Early Modern German',
    '1800-1900': 'Modern German Formation',
    '1900-2000': 'Contemporary German'
}

# Genre classifications for analysis
GENRE_CLASSIFICATIONS = {
    'Religious': ['Sermons', 'Religious'],
    'Legal': ['Legal', 'Administrative'],
    'Scientific': ['Scientific', 'Medical'],
    'Literary': ['Narrative', 'Drama'],
    'Historical': ['Humanities', 'Chronicles'],
    'Popular': ['Newspapers', 'Popular']
}

# Analysis prompts for different capabilities
SYSTEM_PROMPTS = {
    'language_evolution': """You are a specialist in German historical linguistics. Analyze the evolution of German language 
    based on historical texts. Focus on spelling changes, vocabulary development, grammatical evolution, and sociolinguistic factors. 
    Provide scholarly insights with specific examples from the source materials.""",
    
    'semantic_search': """You are analyzing historical German texts for semantic similarity. Consider both literal meaning 
    and historical context. Account for archaic spellings, older grammatical forms, and period-specific vocabulary.""",
    
    'historical_insights': """You are a historical linguist specializing in German language development. Generate scholarly 
    insights about language change, social factors, and cultural influences. Support your analysis with specific evidence 
    from historical sources.""",
    
    'comparative_analysis': """You are conducting comparative linguistic analysis across different periods of German. 
    Identify patterns, changes, and continuities. Consider phonological, morphological, syntactic, and lexical developments.""",
    
    'question_answering': """You are answering questions about German historical linguistics based on primary source materials. 
    Provide accurate, well-sourced answers that demonstrate understanding of historical language development. Cite specific 
    examples from the source texts when possible."""
}

# Question analysis templates
QUESTION_TEMPLATES = {
    'evolution_analysis': """
    Based on the historical German texts provided, analyze how {concept} evolved over time.
    
    Historical sources:
    {sources}
    
    Please provide:
    1. Key changes observed across periods
    2. Specific examples from the texts
    3. Possible social/cultural factors
    4. Linguistic significance
    
    Focus on concrete evidence from the source materials.
    """,
    
    'comparative_study': """
    Compare the usage of {concept} across these historical periods:
    
    {period_data}
    
    Analyze:
    1. Differences in usage patterns
    2. Evolution of meaning or form
    3. Genre-specific variations
    4. Historical context
    
    Support your analysis with specific textual evidence.
    """,
    
    'contextual_analysis': """
    Examine how {concept} appears in these historical German contexts:
    
    {contexts}
    
    Provide analysis of:
    1. Semantic development
    2. Syntactic usage patterns
    3. Social/cultural implications
    4. Comparison with modern usage
    
    Ground your analysis in the provided historical evidence.
    """
}

# Search enhancement parameters
SEARCH_CONFIG = {
    'similarity_threshold': 0.7,    # Minimum similarity for results
    'max_results_per_query': 20,    # Maximum results to consider
    'rerank_top_k': 10,             # Results to rerank with more sophisticated methods
    'context_window': 500,          # Characters of context around matches
    'include_metadata_in_search': True,
    'boost_factors': {
        'period_match': 1.2,        # Boost for period-specific searches
        'genre_match': 1.1,         # Boost for genre-specific searches
        'exact_phrase': 1.3,        # Boost for exact phrase matches
    }
}

# Analysis depth configurations
ANALYSIS_DEPTHS = {
    'quick': {
        'max_sources': 5,
        'max_tokens': 1000,
        'analysis_detail': 'basic',
        'include_examples': True
    },
    'standard': {
        'max_sources': 10,
        'max_tokens': 2500,
        'analysis_detail': 'comprehensive',
        'include_examples': True
    },
    'deep': {
        'max_sources': 20,
        'max_tokens': 4000,
        'analysis_detail': 'scholarly',
        'include_examples': True,
        'include_comparative': True
    }
}

# Linguistic feature extraction patterns
LINGUISTIC_PATTERNS = {
    'archaic_spellings': [
        r'\bth\w+',           # th- words (thun -> tun)
        r'\w*uo\w*',          # uo diphthongs (guot -> gut)
        r'\w*ey\w*',          # ey variants
        r'^v[aeiou]\w*',      # v- beginnings (vmb -> um)
        r'\w*ff\w*',          # double consonants
        r'\w*tz\b',           # tz endings
    ],
    'religious_vocabulary': [
        r'\b[Gg]ott\w*',      # God-related terms
        r'\b[Hh]err\w*',      # Lord-related terms
        r'\b[Cc]hrist\w*',    # Christ-related terms
        r'\b[Kk]irch\w*',     # Church-related terms
        r'\b[Ss]eel\w*',      # Soul-related terms
    ],
    'legal_terminology': [
        r'\b[Rr]echt\w*',     # Law/right terms
        r'\b[Gg]ericht\w*',   # Court terms
        r'\b[Uu]rteil\w*',    # Judgment terms
        r'\b[Oo]rdnung\w*',   # Order/regulation terms
    ]
}

# Error handling and validation
VALIDATION_CONFIG = {
    'min_text_length': 50,          # Minimum text length for processing
    'max_text_length': 8000,        # Maximum text length (OpenAI limit)
    'required_fields': ['text', 'period', 'genre'],
    'valid_periods': list(HISTORICAL_PERIODS.keys()),
    'api_timeout': 30,              # API timeout in seconds
}

# Cost estimation (approximate, for planning)
COST_ESTIMATES = {
    'embedding_cost_per_1k_tokens': 0.00013,  # text-embedding-3-large
    'gpt4_input_cost_per_1k_tokens': 0.01,    # GPT-4 input
    'gpt4_output_cost_per_1k_tokens': 0.03,   # GPT-4 output
    'gpt35_input_cost_per_1k_tokens': 0.0015, # GPT-3.5 input
    'gpt35_output_cost_per_1k_tokens': 0.002, # GPT-3.5 output
}

# Export key configurations
__all__ = [
    'OPENAI_CONFIG',
    'DEFAULT_DB_CONFIG', 
    'VECTOR_DB_CONFIG',
    'HISTORICAL_PERIODS',
    'GENRE_CLASSIFICATIONS',
    'SYSTEM_PROMPTS',
    'QUESTION_TEMPLATES',
    'SEARCH_CONFIG',
    'ANALYSIS_DEPTHS',
    'LINGUISTIC_PATTERNS',
    'VALIDATION_CONFIG',
    'COST_ESTIMATES'
]