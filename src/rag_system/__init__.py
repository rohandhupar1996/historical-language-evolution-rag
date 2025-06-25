# ==========================================
# FILE: rag_system/__init__.py
# ==========================================
"""RAG System Package - Phase 5 Vector DB & QA."""

from .pipeline import GermanRAGPipeline
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .qa_chain import QAChainManager
from .language_evolution import LanguageEvolutionAnalyzer
from .semantic_search import SemanticSearcher
from .statistics import StatisticsCalculator
from . import config

__version__ = "1.0.0"
__all__ = [
    "GermanRAGPipeline",
    "EmbeddingManager",
    "VectorStoreManager",
    "QAChainManager",
    "LanguageEvolutionAnalyzer",
    "SemanticSearcher",
    "StatisticsCalculator",
    "config"
]
