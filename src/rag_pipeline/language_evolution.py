# ==========================================
# FILE: rag_system/language_evolution.py
# ==========================================
"""Language evolution analysis."""

from typing import List, Dict, Any, Optional
from .semantic_search import SemanticSearcher
from .qa_chain import QAChainManager
from .config import DEFAULT_PERIODS


class LanguageEvolutionAnalyzer:
    """Analyzes language evolution across time periods."""
    
    def __init__(self, searcher: SemanticSearcher, qa_manager: QAChainManager):
        self.searcher = searcher
        self.qa_manager = qa_manager
    
    def analyze_word_evolution(self, word: str, periods: List[str] = None) -> Dict[str, Any]:
        """Analyze how a word evolved across periods."""
        if not periods:
            periods = DEFAULT_PERIODS
        
        evolution_analysis = {
            'word': word,
            'periods': {},
            'summary': ''
        }
        
        for period in periods:
            search_query = f"Verwendung des Wortes '{word}'"
            results = self.searcher.search(search_query, k=3, period_filter=period)
            
            evolution_analysis['periods'][period] = {
                'examples': results,
                'context_count': len(results)
            }
        
        # Generate evolution summary
        if self.qa_manager.qa_chain:
            summary_question = f"Wie hat sich das Wort '{word}' in der deutschen Sprache über die Zeit entwickelt?"
            summary = self.qa_manager.ask_question(summary_question)
            evolution_analysis['summary'] = summary['answer']
        else:
            total_contexts = sum(len(data['examples']) for data in evolution_analysis['periods'].values())
            evolution_analysis['summary'] = f"Found {total_contexts} contexts for '{word}' across {len(periods)} time periods."
        
        return evolution_analysis