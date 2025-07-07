# ==========================================
# FILE: src/openai_rag_system/qa_chain.py
# ==========================================
"""Advanced OpenAI-powered question answering for German historical corpus."""

import openai
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from .config import (
    OPENAI_CONFIG, SYSTEM_PROMPTS, QUESTION_TEMPLATES, 
    ANALYSIS_DEPTHS, HISTORICAL_PERIODS
)
from .vector_store import OpenAIVectorStore
from .embeddings import OpenAIEmbeddingManager
from .utils import estimate_tokens, handle_rate_limits


class OpenAIQAManager:
    """
    Advanced question-answering system using OpenAI GPT models
    with sophisticated prompt engineering for German historical linguistics.
    """
    
    def __init__(self, vector_store: OpenAIVectorStore, 
                 embedding_manager: OpenAIEmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        # OpenAI settings
        self.chat_model = OPENAI_CONFIG['chat_model']
        self.fallback_model = OPENAI_CONFIG['fallback_chat_model']
        self.max_tokens = OPENAI_CONFIG['max_tokens']
        self.temperature = OPENAI_CONFIG['temperature']
        
        # Usage tracking
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        self.questions_answered = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("OpenAI QA Manager initialized")
    
    def setup_gpt4_chain(self):
        """Setup GPT-4 powered QA chain."""
        try:
            # Test OpenAI connection
            test_response = openai.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            
            self.logger.info(f"GPT-4 QA chain ready with model: {self.chat_model}")
            
        except Exception as e:
            self.logger.warning(f"GPT-4 unavailable, falling back to {self.fallback_model}: {e}")
            self.chat_model = self.fallback_model
    
    def ask_question(self, question: str, period_filter: Optional[str] = None,
                    analysis_depth: str = "standard") -> Dict[str, Any]:
        """
        Answer sophisticated questions about German historical language.
        
        Args:
            question: Question about German historical linguistics
            period_filter: Focus on specific historical period
            analysis_depth: "quick", "standard", or "deep"
            
        Returns:
            Comprehensive answer with sources and analysis
        """
        start_time = time.time()
        
        # Get analysis depth configuration
        depth_config = ANALYSIS_DEPTHS.get(analysis_depth, ANALYSIS_DEPTHS['standard'])
        
        # Step 1: Retrieve relevant historical sources
        sources = self._retrieve_relevant_sources(question, period_filter, depth_config)
        
        if not sources:
            return {
                'question': question,
                'answer': "I couldn't find relevant historical sources to answer this question about German language evolution.",
                'sources': [],
                'confidence': 0.0,
                'analysis_depth': analysis_depth
            }
        
        # Step 2: Generate comprehensive answer using GPT-4
        answer_result = self._generate_comprehensive_answer(question, sources, depth_config)
        
        # Step 3: Enhance with linguistic analysis
        enhanced_answer = self._enhance_with_linguistic_analysis(
            question, answer_result, sources, analysis_depth
        )
        
        processing_time = time.time() - start_time
        
        # Update usage tracking
        self.questions_answered += 1
        
        result = {
            'question': question,
            'answer': enhanced_answer['answer'],
            'linguistic_analysis': enhanced_answer.get('linguistic_analysis', {}),
            'sources': sources,
            'confidence': self._calculate_confidence(sources, answer_result),
            'analysis_depth': analysis_depth,
            'processing_time': processing_time,
            'metadata': {
                'model_used': self.chat_model,
                'sources_used': len(sources),
                'tokens_used': answer_result.get('tokens_used', 0),
                'cost_estimate': answer_result.get('cost_estimate', 0.0),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        self.logger.info(f"Question answered in {processing_time:.1f}s using {len(sources)} sources")
        return result
    
    def _retrieve_relevant_sources(self, question: str, period_filter: Optional[str],
                                 depth_config: Dict) -> List[Dict[str, Any]]:
        """Retrieve relevant historical sources for the question."""
        max_sources = depth_config['max_sources']
        
        # Perform semantic search
        search_params = {
            'k': max_sources,
            'period_filter': period_filter,
            'include_context': True,
            'rerank': True,
            'similarity_threshold': 0.6
        }
        
        search_results = self.vector_store.advanced_search(
            question, search_params, self.embedding_manager
        )
        
        # Convert to source format
        sources = []
        for result in search_results.get('results', []):
            source = {
                'text': result['document'],
                'metadata': result['metadata'],
                'similarity_score': result['similarity_score'],
                'snippet': result.get('snippet', ''),
                'period_description': result.get('period_description', ''),
                'text_stats': result.get('text_stats', {})
            }
            sources.append(source)
        
    def _generate_comprehensive_answer(self, question: str, sources: List[Dict],
                                     depth_config: Dict) -> Dict[str, Any]:
        """Generate answer using GPT-4."""
        if not sources:
            return {'answer': 'No relevant sources found.', 'tokens_used': 0, 'cost_estimate': 0.0}
        
        # Prepare context
        context = "\n".join([f"Source {i+1}: {s['text'][:300]}..." for i, s in enumerate(sources[:3])])
        
        try:
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a German historical linguistics expert. Answer based on the provided sources."},
                    {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}\n\nPlease provide a detailed answer."}
                ],
                max_tokens=depth_config.get('max_tokens', 1000),
                temperature=self.temperature
            )
            
            return {
                'answer': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens,
                'cost_estimate': 0.01 * response.usage.total_tokens / 1000
            }
        except Exception as e:
            return {'answer': f'Error generating answer: {str(e)}', 'tokens_used': 0, 'cost_estimate': 0.0}
    
    def _enhance_with_linguistic_analysis(self, question: str, answer_result: Dict,
                                        sources: List[Dict], analysis_depth: str) -> Dict[str, Any]:
        """Add linguistic analysis."""
        return answer_result  # Simplified for now
    
    def _calculate_confidence(self, sources: List[Dict], answer_result: Dict) -> float:
        """Calculate confidence score."""
        if not sources:
            return 0.0
        avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
        return min(avg_similarity, 0.95)
        """Retrieve relevant historical sources for the question."""
        max_sources = depth_config['max_sources']
        
        # Perform semantic search
        search_params = {
            'k': max_sources,
            'period_filter': period_filter,
            'include_context': True,
            'rerank': True,
            'similarity_threshold': 0.6
        }
        
        search_results = self.vector_store.advanced_search(
            question, search_params, self.embedding_manager
        )
        
        # Convert to source format
        sources = []
        for result in search_results.get('results', []):
            source = {
                'text': result['document'],
                'metadata': result['metadata'],
                'similarity_score': result['similarity_score'],
                'snippet': result.get('snippet', ''),
                'period_description': result.get('period_description', ''),
                'text_stats': result.get('text_stats', {})
            }
            sources.append(source)
        
        return sources
    
    def _generate_comprehensive_answer(self, question: str, sources: List[Dict],
                                     depth_config: Dict) -> Dict[str, Any]:
        """Generate comprehensive answer using GPT-4."""
        
        # Prepare context from sources
        context = self._prepare_context_from_sources(sources, depth_config['max_tokens'] // 2)
        
        # Select appropriate system prompt and template
        system_prompt = SYSTEM_PROMPTS['question_answering']
        
        # Determine question type for specialized handling
        question_type = self._classify_question(question)
        
        if question_type in QUESTION_TEMPLATES:
            user_prompt = QUESTION_TEMPLATES[question_type].format(
                concept=question,
                sources=context
            )
        else:
            user_prompt = f"""
            Based on these historical German texts, please answer the following question:
            
            Question: {question}
            
            Historical Sources:
            {context}
            
            Please provide a comprehensive answer that:
            1. Directly addresses the question
            2. Uses specific examples from the source texts
            3. Explains the historical linguistic context
            4. Discusses any relevant language changes or patterns
            5. Maintains scholarly accuracy
            
            Ground your response in the provided historical evidence.
            """
        
        try:
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=depth_config['max_tokens'],
                temperature=self.temperature,
                top_p=OPENAI_CONFIG['top_p']
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Estimate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost_estimate = self._estimate_chat_cost(input_tokens, output_tokens)
            
            # Update tracking
            self.total_tokens_used += tokens_used
            self.total_cost_estimate += cost_estimate
            
            return {
                'answer': answer,
                'tokens_used': tokens_used,
                'cost_estimate': cost_estimate,
                'model_used': self.chat_model
            }
            
        except Exception as e:
            self.logger.error(f"GPT-4 answer generation failed: {e}")
            
            # Fallback to simpler response
            return {
                'answer': self._generate_fallback_answer(question, sources),
                'tokens_used': 0,
                'cost_estimate': 0.0,
                'model_used': 'fallback'
            }
    
    def _enhance_with_linguistic_analysis(self, question: str, answer_result: Dict,
                                        sources: List[Dict], analysis_depth: str) -> Dict[str, Any]:
        """Enhance answer with additional linguistic analysis."""
        
        if analysis_depth == "quick":
            return answer_result
        
        # Extract linguistic patterns from sources
        linguistic_analysis = {
            'period_coverage': self._analyze_period_coverage(sources),
            'genre_distribution': self._analyze_genre_distribution(sources),
            'linguistic_features': self._extract_linguistic_features(sources),
        }
        
        if analysis_depth == "deep":
            # Add deeper analysis for scholarly responses
            linguistic_analysis.update({
                'comparative_analysis': self._generate_comparative_analysis(sources),
                'historical_context': self._analyze_historical_context(sources),
                'language_change_patterns': self._identify_change_patterns(sources)
            })
        
        return {
            'answer': answer_result['answer'],
            'linguistic_analysis': linguistic_analysis,
            'tokens_used': answer_result.get('tokens_used', 0),
            'cost_estimate': answer_result.get('cost_estimate', 0.0)
        }
    
    def _prepare_context_from_sources(self, sources: List[Dict], max_context_length: int) -> str:
        """Prepare context string from sources with length management."""
        context_parts = []
        current_length = 0
        
        for i, source in enumerate(sources):
            # Create source header
            metadata = source['metadata']
            header = f"\n--- Source {i+1} ---"
            header += f"\nPeriod: {metadata.get('period', 'Unknown')}"
            header += f"\nGenre: {metadata.get('genre', 'Unknown')}"
            header += f"\nYear: {metadata.get('year', 'Unknown')}"
            header += f"\nSimilarity: {source['similarity_score']:.2f}"
            header += "\nText:"
            
            # Use snippet if available, otherwise truncate text
            text = source.get('snippet', source['text'])
            if len(text) > 300:
                text = text[:300] + "..."
            
            source_content = f"{header}\n{text}\n"
            
            # Check if adding this source would exceed limit
            if current_length + len(source_content) > max_context_length:
                break
            
            context_parts.append(source_content)
            current_length += len(source_content)
        
        return "\n".join(context_parts)
    
    def _classify_question(self, question: str) -> str:
        """Classify question type for specialized handling."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['evolution', 'change', 'develop', 'transform']):
            return 'evolution_analysis'
        elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'between']):
            return 'comparative_study'
        elif any(word in question_lower for word in ['context', 'usage', 'how', 'when', 'where']):
            return 'contextual_analysis'
        else:
            return 'general_inquiry'
    
    def _calculate_confidence(self, sources: List[Dict], answer_result: Dict) -> float:
        """Calculate confidence score for the answer."""
        if not sources:
            return 0.0
        
        # Base confidence on source quality
        avg_similarity = sum(s['similarity_score'] for s in sources) / len(sources)
        source_count_factor = min(len(sources) / 5, 1.0)  # Max factor at 5+ sources
        
        # Adjust for answer quality indicators
        answer_length = len(answer_result.get('answer', ''))
        length_factor = min(answer_length / 500, 1.0)  # Max factor at 500+ chars
        
        confidence = (avg_similarity * 0.5 + source_count_factor * 0.3 + length_factor * 0.2)
        return min(confidence, 0.95)  # Cap at 95%
    
    def _analyze_period_coverage(self, sources: List[Dict]) -> Dict[str, Any]:
        """Analyze which historical periods are covered by sources."""
        periods = [s['metadata'].get('period', 'Unknown') for s in sources]
        period_counts = {period: periods.count(period) for period in set(periods)}
        
        return {
            'periods_covered': list(period_counts.keys()),
            'period_distribution': period_counts,
            'temporal_span': len(period_counts),
            'dominant_period': max(period_counts.items(), key=lambda x: x[1])[0] if period_counts else None
        }
    
    def _analyze_genre_distribution(self, sources: List[Dict]) -> Dict[str, Any]:
        """Analyze genre distribution in sources."""
        genres = [s['metadata'].get('genre', 'Unknown') for s in sources]
        genre_counts = {genre: genres.count(genre) for genre in set(genres)}
        
        return {
            'genres_represented': list(genre_counts.keys()),
            'genre_distribution': genre_counts,
            'genre_diversity': len(genre_counts)
        }
    
    def _extract_linguistic_features(self, sources: List[Dict]) -> Dict[str, Any]:
        """Extract basic linguistic features from sources."""
        total_words = 0
        total_chars = 0
        periods_analyzed = set()
        
        for source in sources:
            text = source['text']
            words = len(text.split())
            chars = len(text)
            
            total_words += words
            total_chars += chars
            periods_analyzed.add(source['metadata'].get('period', 'Unknown'))
        
        return {
            'total_words_analyzed': total_words,
            'total_characters_analyzed': total_chars,
            'average_words_per_source': total_words / len(sources) if sources else 0,
            'periods_in_analysis': list(periods_analyzed),
            'source_count': len(sources)
        }
    
    def _generate_comparative_analysis(self, sources: List[Dict]) -> Dict[str, Any]:
        """Generate comparative analysis across periods/genres."""
        # Group sources by period
        by_period = {}
        for source in sources:
            period = source['metadata'].get('period', 'Unknown')
            if period not in by_period:
                by_period[period] = []
            by_period[period].append(source)
        
        # Analyze differences
        period_characteristics = {}
        for period, period_sources in by_period.items():
            avg_length = sum(len(s['text']) for s in period_sources) / len(period_sources)
            genres = set(s['metadata'].get('genre', 'Unknown') for s in period_sources)
            
            period_characteristics[period] = {
                'source_count': len(period_sources),
                'average_text_length': avg_length,
                'genres_present': list(genres),
                'representative_snippets': [s.get('snippet', s['text'][:100]) for s in period_sources[:2]]
            }
        
        return {
            'periods_compared': list(by_period.keys()),
            'period_characteristics': period_characteristics,
            'comparison_basis': 'text_length_and_genre_distribution'
        }
    
    def _analyze_historical_context(self, sources: List[Dict]) -> Dict[str, Any]:
        """Analyze historical context of sources."""
        years = []
        for source in sources:
            try:
                year = int(source['metadata'].get('year', 0))
                if year > 1000:  # Valid historical year
                    years.append(year)
            except:
                continue
        
        if not years:
            return {'message': 'No valid historical dates found'}
        
        return {
            'temporal_range': {
                'earliest_year': min(years),
                'latest_year': max(years),
                'span_years': max(years) - min(years)
            },
            'chronological_distribution': {
                'medieval': len([y for y in years if y < 1500]),
                'early_modern': len([y for y in years if 1500 <= y < 1800]),
                'modern': len([y for y in years if y >= 1800])
            },
            'historical_significance': self._assess_historical_significance(years)
        }
    
    def _identify_change_patterns(self, sources: List[Dict]) -> Dict[str, Any]:
        """Identify potential language change patterns."""
        # This is a simplified analysis - in a full implementation,
        # this would involve sophisticated linguistic analysis
        
        patterns = {
            'spelling_variations': [],
            'vocabulary_evolution': [],
            'structural_changes': []
        }
        
        # Look for common archaic patterns
        archaic_patterns = ['th', 'uo', 'ey', 'tz']
        modern_equivalents = ['t', 'u', 'ei', 'z']
        
        for source in sources:
            text = source['text'].lower()
            period = source['metadata'].get('period', '')
            
            for archaic, modern in zip(archaic_patterns, modern_equivalents):
                if archaic in text and '1050-1350' in period:
                    patterns['spelling_variations'].append({
                        'pattern': f"{archaic} -> {modern}",
                        'period': period,
                        'evidence': text[:100] + "..."
                    })
        
        return patterns
    
    def _assess_historical_significance(self, years: List[int]) -> str:
        """Assess historical significance of the time period."""
        if not years:
            return "No temporal data available"
        
        earliest, latest = min(years), max(years)
        
        if earliest < 1350:
            return "Includes Middle High German period - highly significant for language evolution"
        elif earliest < 1650:
            return "Covers Early New High German - important transitional period"
        elif earliest < 1800:
            return "Baroque/Early Modern period - standardization era"
        else:
            return "Modern German period - well-documented language form"
    
    def _estimate_chat_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost of chat completion."""
        if self.chat_model.startswith('gpt-4'):
            input_cost = input_tokens * OPENAI_CONFIG.get('gpt4_input_cost_per_1k_tokens', 0.01) / 1000
            output_cost = output_tokens * OPENAI_CONFIG.get('gpt4_output_cost_per_1k_tokens', 0.03) / 1000
        else:
            input_cost = input_tokens * OPENAI_CONFIG.get('gpt35_input_cost_per_1k_tokens', 0.0015) / 1000
            output_cost = output_tokens * OPENAI_CONFIG.get('gpt35_output_cost_per_1k_tokens', 0.002) / 1000
        
        return input_cost + output_cost
    
    def _generate_fallback_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate fallback answer when GPT fails."""
        if not sources:
            return "I couldn't find relevant historical sources to answer your question about German language evolution."
        
        # Create simple answer from sources
        answer_parts = [
            f"Based on {len(sources)} historical German sources, here's what I found:",
            ""
        ]
        
        for i, source in enumerate(sources[:3]):
            metadata = source['metadata']
            snippet = source.get('snippet', source['text'][:150] + "...")
            
            answer_parts.append(
                f"{i+1}. From {metadata.get('period', 'unknown period')} "
                f"({metadata.get('genre', 'unknown genre')}): {snippet}"
            )
        
        answer_parts.append(
            f"\nThis analysis covers {len(set(s['metadata'].get('period') for s in sources))} "
            f"historical periods and provides insights into German language development."
        )
        
        return "\n".join(answer_parts)
    
    def get_qa_statistics(self) -> Dict[str, Any]:
        """Get comprehensive QA system statistics."""
        return {
            'usage_stats': {
                'questions_answered': self.questions_answered,
                'total_tokens_used': self.total_tokens_used,
                'total_cost_estimate': self.total_cost_estimate,
                'average_tokens_per_question': self.total_tokens_used / max(1, self.questions_answered)
            },
            'model_config': {
                'primary_model': self.chat_model,
                'fallback_model': self.fallback_model,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature
            },
            'capabilities': [
                'Historical linguistics analysis',
                'Period-specific questioning',
                'Comparative language studies',
                'Scholarly source integration',
                'Multi-depth analysis'
            ]
        }