# ==========================================
# FILE: src/openai_rag_system/language_evolution.py
# ==========================================

from typing import List, Dict, Any, Optional
from .semantic_search import OpenAISemanticSearcher
from .qa_chain import OpenAIQAManager

class OpenAILanguageEvolutionAnalyzer:
    def __init__(self, searcher: OpenAISemanticSearcher, qa_manager: OpenAIQAManager):
        self.searcher = searcher
        self.qa_manager = qa_manager
    
    def analyze_evolution(self, word_or_concept: str, periods: List[str] = None,
                         analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze language evolution using OpenAI."""
        if not periods:
            periods = ['1050-1350', '1350-1650', '1650-1800', '1800-1900']
        
        evolution_data = {
            'concept': word_or_concept,
            'period_analysis': {},
            'evolution_insights': '',
            'comparative_summary': ''
        }
        
        # Analyze each period
        for period in periods:
            period_results = self.searcher.search(
                f"{word_or_concept} historical usage development",
                k=3, period_filter=period
            )
            
            evolution_data['period_analysis'][period] = {
                'sources_found': len(period_results),
                'examples': [r['snippet'] for r in period_results[:2]],
                'contexts': period_results
            }
        
        # Generate insights using GPT-4
        if analysis_type == "comprehensive":
            insights_question = f"How did the concept '{word_or_concept}' evolve in German language across these historical periods?"
            insights_result = self.qa_manager.ask_question(insights_question, analysis_depth="deep")
            evolution_data['evolution_insights'] = insights_result['answer']
        
        return evolution_data
    
    def track_changes(self, period_start: str, period_end: str,
                     focus_areas: List[str] = None) -> Dict[str, Any]:
        """Track language changes between periods."""
        if not focus_areas:
            focus_areas = ["spelling", "vocabulary", "syntax"]
        
        change_analysis = {
            'period_comparison': f"{period_start} → {period_end}",
            'focus_areas': focus_areas,
            'changes_detected': {}
        }
        
        for area in focus_areas:
            question = f"What {area} changes occurred in German between {period_start} and {period_end}?"
            result = self.qa_manager.ask_question(question, analysis_depth="standard")
            change_analysis['changes_detected'][area] = result['answer']
        
        return change_analysis