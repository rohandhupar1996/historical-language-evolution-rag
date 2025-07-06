# ==========================================
# FILE: src/streamlit_app/config.py
# ==========================================
"""Configuration for Streamlit application."""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class AppConfig:
    """Application configuration."""
    
    # App metadata
    APP_TITLE = "🏛️ German Historical Corpus RAG System"
    APP_ICON = "🏛️"
    LAYOUT = "wide"
    
    # Database configuration
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'germanc_corpus'),
        'user': os.getenv('DB_USER', 'rohan'),
        'password': os.getenv('DB_PASSWORD', '1996')
    }
    
    # RAG system paths
    RAG_VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './german_corpus_vectordb')
    
    # UI configuration
    SIDEBAR_WIDTH = 300
    MAX_SEARCH_RESULTS = 50
    DEFAULT_SEARCH_LIMIT = 10
    
    # Available filters
    PERIODS = [
        "All Periods",
        "1050-1350", 
        "1350-1650", 
        "1650-1700", 
        "1700-1800", 
        "1800-1900", 
        "1900-2000"
    ]
    
    GENRES = [
        "All Genres",
        "Drama",
        "Humanities", 
        "Legal",
        "Narrative",
        "Newspapers",
        "Scientific"
    ]
    
    # Example queries
    EXAMPLE_QUERIES = [
        "Wie wurde Gott und Religion in historischen deutschen Texten beschrieben?",
        "Welche Rechtsbegriffe und Gesetze finden sich in alten deutschen Texten?",
        "Wie wurde über Krieg und Frieden geschrieben?",
        "Wissenschaft und Erkenntnis in der frühen Neuzeit"
    ]
    
    # Sample statistics for demo
    SAMPLE_STATS = {
        'total_chunks': 3607,
        'total_periods': 5,
        'total_genres': 6,
        'total_embeddings': 902
    }
    
    # Linguistic insights
    LINGUISTIC_INSIGHTS = [
        "📈 Religious vocabulary peaks in the 1650-1700 period with 'Gott' appearing 1,247 times",
        "⚖️ Legal terminology shows standardization patterns across different German regions", 
        "🏛️ Political language reflects the Holy Roman Empire's administrative complexity",
        "🔬 Scientific German begins emerging with Latin influences in medical texts",
        "📰 Early newspaper language shows shift from formal to more accessible German",
        "🎭 Dramatic texts preserve older Germanic sentence structures and vocabulary"
    ]