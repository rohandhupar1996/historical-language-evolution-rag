# ==========================================
# FILE: src/streamlit_app/utils/data_processing.py
# ==========================================
"""Data processing utilities for the Streamlit app."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from rag_system import GerManCRAGPipeline
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from ..config import AppConfig

class DataProcessor:
    """Handles data processing for the Streamlit app."""
    
    @staticmethod
    def initialize_rag_system():
        """Initialize RAG system with caching."""
        if not RAG_AVAILABLE:
            return None
        
        try:
            rag = GerManCRAGPipeline(
                AppConfig.DB_CONFIG, 
                AppConfig.RAG_VECTOR_DB_PATH
            )
            rag.setup_qa_system("simple")
            return rag
        except Exception as e:
            return None
    
    @staticmethod
    def get_sample_data():
        """Get sample data for demonstration."""
        return {
            'results': [
                {
                    'text': "Römischen Reichs Müntz-Ordnung / Anno Fünfzehen hundert Fünfzig neun auffgericht / und hernacher auf etlichen Reichs-Tägen / sonderlich aber bei dem in Fünffzehenhundert ein und siebentzichen gehaltenen zu Regenspurg bestätiget und approbiret...",
                    'metadata': {'period': '1650-1700', 'genre': 'Legal', 'filename': 'reichs_muentz_ordnung_1659.txt'},
                    'confidence': 0.89
                },
                {
                    'text': "DAs freimachen soll mit Vorwissen des BergVoigts geschehen / und soll der Freimacher mit ein oder zweyen Zeugen dabey seyn / damit alles ordentlich und recht zugehe...",
                    'metadata': {'period': '1650-1700', 'genre': 'Legal', 'filename': 'berg_ordnung_1681.txt'},
                    'confidence': 0.85
                },
                {
                    'text': "eine Pfeif von dem Stengel hellebori oder Christwurtz gemacht / Kuriere mit ihrem sono die lymphatischen Gefäße und treiben die Melancholey auß...",
                    'metadata': {'period': '1650-1700', 'genre': 'Scientific', 'filename': 'medizin_compendium_1677.txt'},
                    'confidence': 0.82
                }
            ]
        }
    
    @staticmethod
    def perform_search(rag_system, query: str, period: Optional[str] = None, 
                      genre: Optional[str] = None, limit: int = 10):
        """Perform search using RAG system or demo data."""
        if rag_system and RAG_AVAILABLE:
            try:
                # Real RAG search
                period_filter = period if period != "All Periods" else None
                results = rag_system.semantic_search(query, k=limit, period_filter=period_filter)
                
                # Filter by genre if specified
                if genre and genre != "All Genres":
                    results = [r for r in results if r.get('metadata', {}).get('genre') == genre]
                
                return results
            except Exception as e:
                return []
        else:
            # Demo mode - return sample data
            sample_data = DataProcessor.get_sample_data()
            results = sample_data['results'].copy()
            
            # Apply filters
            if period and period != "All Periods":
                results = [r for r in results if r['metadata']['period'] == period]
            if genre and genre != "All Genres":
                results = [r for r in results if r['metadata']['genre'] == genre]
            
            return results[:limit]
    
    @staticmethod
    def analyze_evolution(rag_system, word: str, periods: Optional[List[str]] = None):
        """Analyze word evolution."""
        if rag_system and RAG_AVAILABLE:
            try:
                return rag_system.analyze_language_evolution(word, periods)
            except Exception as e:
                return None
        else:
            # Demo evolution data
            return {
                'word': word,
                'periods': {
                    '1650-1700': {
                        'examples': [{'text': f"Example usage of '{word}' in 1650-1700..."}],
                        'context_count': 5
                    }
                },
                'summary': f"The word '{word}' shows interesting evolution patterns in the available corpus."
            }
