# ==========================================
# FILE: access_pipeline/utils.py
# ==========================================
"""Utility functions for ACCESS pipeline."""

import logging
from typing import Dict


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)


def print_database_ready(db_config: Dict) -> None:
    """Print database ready message."""
    print("\n✅ Phase 4: ACCESS completed!")
    print("📊 Database ready with tables:")
    print("  - chunks")
    print("  - spelling_variants") 
    print("  - word_frequencies")
    print("  - linguistic_features")
    print("\n🔄 Ready for RAG pipeline!")


def print_api_endpoints() -> None:
    """Print available API endpoints."""
    print("\n📡 API Endpoints:")
    print("  GET  /evolution/{word}/{start_period}/{end_period}")
    print("  POST /linguistic_analysis")
    print("  GET  /temporal_patterns")
    print("  GET  /search/{query}")
