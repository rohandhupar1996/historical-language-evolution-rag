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
        
        print(f"🔍 Searching for '{word}' in periods: {periods}")
        
        evolution_analysis = {
            'word': word,
            'periods': {},
            'summary': ''
        }
        
        for period in periods:
            # Try multiple search strategies
            search_queries = [
                f"{word}",  # Direct word search
                f"Verwendung des Wortes '{word}'",  # Usage context
                f"{word} sprache",  # Language context
                word.lower(),  # Lowercase version
                word.capitalize()  # Capitalized version
            ]
            
            all_results = []
            for query in search_queries:
                results = self.searcher.search(query, k=2, period_filter=period)
                all_results.extend(results)
            
            # Remove duplicates by chunk_id
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['chunk_id'] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result['chunk_id'])
            
            evolution_analysis['periods'][period] = {
                'examples': unique_results[:5],  # Limit to 5 best results
                'context_count': len(unique_results)
            }
            
            print(f"   {period}: {len(unique_results)} contexts found")
        
        # Generate evolution summary
        total_contexts = sum(len(data['examples']) for data in evolution_analysis['periods'].values())
        if total_contexts > 0:
            if self.qa_manager.qa_chain:
                summary_question = f"Wie wurde das Wort '{word}' in den verfügbaren Texten verwendet?"
                summary = self.qa_manager.ask_question(summary_question)
                evolution_analysis['summary'] = summary['answer']
            else:
                evolution_analysis['summary'] = f"Found {total_contexts} contexts for '{word}' across {len(periods)} time periods."
        else:
            evolution_analysis['summary'] = f"No contexts found for '{word}' in any time period. Try different search terms."
        
        return evolution_analysis