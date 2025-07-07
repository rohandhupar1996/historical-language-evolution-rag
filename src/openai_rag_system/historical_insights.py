# ==========================================
# FILE: src/openai_rag_system/historical_insights.py
# ==========================================

from typing import Dict, Any, List
from .semantic_search import OpenAISemanticSearcher
from .qa_chain import OpenAIQAManager

class HistoricalInsightsGenerator:
    def __init__(self, searcher: OpenAISemanticSearcher, qa_manager: OpenAIQAManager):
        self.searcher = searcher
        self.qa_manager = qa_manager
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
    
    def generate_insights(self, topic: str, insight_type: str = "linguistic_evolution") -> Dict[str, Any]:
        """Generate scholarly insights about historical German language."""
        
        # Search for relevant sources
        sources = self.searcher.search(topic, k=8)
        
        insight_prompts = {
            'linguistic_evolution': f"Analyze the linguistic evolution of {topic} in German historical texts",
            'social_context': f"Examine the social and cultural context of {topic} in German history",
            'comparative_analysis': f"Compare {topic} across different periods and genres of German texts"
        }
        
        prompt = insight_prompts.get(insight_type, insight_prompts['linguistic_evolution'])
        result = self.qa_manager.ask_question(prompt, analysis_depth="deep")
        
        return {
            'topic': topic,
            'insight_type': insight_type,
            'insights': result['answer'],
            'sources_analyzed': len(sources),
            'confidence': result['confidence']
        }
    
    def comparative_analysis(self, periods: List[str], analysis_focus: str = "general") -> Dict[str, Any]:
        """Compare language characteristics across periods."""
        
        comparison_data = {
            'periods_compared': periods,
            'focus': analysis_focus,
            'period_characteristics': {},
            'comparative_insights': ''
        }
        
        for period in periods:
            period_sources = self.searcher.search(
                f"{analysis_focus} language characteristics", 
                k=5, period_filter=period
            )
            comparison_data['period_characteristics'][period] = {
                'source_count': len(period_sources),
                'examples': [s['snippet'] for s in period_sources[:2]]
            }
        
        # Generate comparative analysis
        question = f"Compare {analysis_focus} language features across {', '.join(periods)}"
        result = self.qa_manager.ask_question(question, analysis_depth="standard")
        comparison_data['comparative_insights'] = result['answer']
        
        return comparison_data