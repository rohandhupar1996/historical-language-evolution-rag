# ==========================================
# FILE: src/openai_rag_system/pipeline.py
# ==========================================
"""
OpenAI-powered RAG pipeline for German historical corpus analysis.
Uses OpenAI embeddings and GPT models for advanced language analysis.
"""

import openai
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
import time

from .embeddings import OpenAIEmbeddingManager
from .vector_store import OpenAIVectorStore
from .semantic_search import OpenAISemanticSearcher
from .qa_chain import OpenAIQAManager
from .language_evolution import OpenAILanguageEvolutionAnalyzer
from .historical_insights import HistoricalInsightsGenerator
from .config import OPENAI_CONFIG, DEFAULT_DB_CONFIG
from .utils import setup_logging, validate_openai_setup


class OpenAIGermanRAGPipeline:
    """
    Advanced OpenAI-powered RAG pipeline for German historical corpus.
    
    Features:
    - OpenAI text-embedding-3-large embeddings
    - GPT-4 powered question answering
    - Historical language evolution analysis
    - Period-specific insights generation
    - Advanced semantic search capabilities
    """
    
    def __init__(self, db_config: Dict[str, Any], vector_db_path: str = "./openai_vector_db"):
        self.db_config = db_config
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Validate OpenAI setup
        validate_openai_setup()
        
        # Database connection
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        # Initialize OpenAI components
        self.embedding_manager = OpenAIEmbeddingManager()
        self.vector_store = OpenAIVectorStore(str(self.vector_db_path))
        self.searcher = OpenAISemanticSearcher(self.vector_store, self.embedding_manager)
        self.qa_manager = OpenAIQAManager(self.vector_store, self.embedding_manager)
        self.evolution_analyzer = OpenAILanguageEvolutionAnalyzer(
            self.searcher, self.qa_manager
        )
        self.insights_generator = HistoricalInsightsGenerator(
            self.searcher, self.qa_manager
        )
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("OpenAI RAG Pipeline initialized")
    
    def load_and_embed_corpus(self, limit: Optional[int] = None, force_rebuild: bool = False):
        """
        Load corpus from database and create OpenAI embeddings.
        
        Args:
            limit: Optional limit for testing
            force_rebuild: Force rebuild of embeddings
        """
        print("🤖 Loading German Historical Corpus for OpenAI Processing...")
        
        # Load chunks from database
        chunks_df = self._load_chunks_from_db(limit)
        
        # Check if embeddings already exist
        if not force_rebuild and self.vector_store.has_embeddings():
            print("✅ OpenAI embeddings already exist")
            return chunks_df
        
        # Create OpenAI embeddings
        print("🧠 Creating OpenAI embeddings...")
        self._create_openai_embeddings(chunks_df)
        
        return chunks_df
    
    def _load_chunks_from_db(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load text chunks from PostgreSQL database."""
        query = """
        SELECT 
            c.chunk_id,
            c.normalized_text as text,
            c.original_text,
            c.period,
            c.genre,
            c.year,
            c.filename,
            c.doc_id,
            c.token_count,
            LENGTH(c.normalized_text) as char_count
        FROM chunks c
        WHERE c.normalized_text IS NOT NULL 
        AND LENGTH(c.normalized_text) > 50
        ORDER BY c.period, c.year, c.chunk_id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print(f"✅ Loaded {len(df)} chunks from database")
        print(f"📊 Periods: {df['period'].unique()}")
        print(f"📊 Genres: {df['genre'].unique()}")
        
        return df
    
    def _create_openai_embeddings(self, chunks_df: pd.DataFrame):
        """Create embeddings using OpenAI API."""
        texts = chunks_df['text'].tolist()
        metadatas = []
        
        # Prepare metadata
        for _, row in chunks_df.iterrows():
            metadata = {
                'chunk_id': str(row['chunk_id']),
                'period': str(row['period']),
                'genre': str(row['genre']),
                'year': str(row['year']),
                'filename': str(row['filename']),
                'doc_id': str(row['doc_id']),
                'token_count': int(row['token_count']),
                'char_count': int(row['char_count'])
            }
            metadatas.append(metadata)
        
        # Create embeddings and store
        self.vector_store.add_texts_with_embeddings(texts, metadatas)
        
        print(f"✅ Created OpenAI embeddings for {len(texts)} chunks")
    
    def setup_advanced_qa_system(self):
        """Setup advanced OpenAI-powered QA system."""
        print("🔗 Setting up OpenAI GPT-4 QA system...")
        
        self.qa_manager.setup_gpt4_chain()
        self.insights_generator.initialize()
        
        print("✅ OpenAI QA system ready")
    
    def semantic_search(self, query: str, k: int = 5, period_filter: Optional[str] = None,
                       include_similarity_scores: bool = True) -> List[Dict]:
        """
        Perform advanced semantic search using OpenAI embeddings.
        
        Args:
            query: Search query
            k: Number of results
            period_filter: Optional period filter
            include_similarity_scores: Include similarity scores
        """
        return self.searcher.search(
            query, k=k, period_filter=period_filter,
            include_similarity_scores=include_similarity_scores
        )
    
    def ask_question(self, question: str, period_filter: Optional[str] = None,
                    analysis_depth: str = "standard") -> Dict[str, Any]:
        """
        Ask sophisticated questions about German language evolution.
        
        Args:
            question: Question about German historical language
            period_filter: Focus on specific period
            analysis_depth: "quick", "standard", or "deep"
        """
        return self.qa_manager.ask_question(
            question, period_filter=period_filter, analysis_depth=analysis_depth
        )
    
    def analyze_language_evolution(self, word_or_concept: str, 
                                 periods: List[str] = None,
                                 analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Comprehensive language evolution analysis using OpenAI.
        
        Args:
            word_or_concept: Word or linguistic concept to analyze
            periods: Time periods to analyze
            analysis_type: "quick", "comprehensive", or "scholarly"
        """
        return self.evolution_analyzer.analyze_evolution(
            word_or_concept, periods=periods, analysis_type=analysis_type
        )
    
    def generate_historical_insights(self, topic: str, 
                                   insight_type: str = "linguistic_evolution") -> Dict[str, Any]:
        """
        Generate scholarly insights about historical German language.
        
        Args:
            topic: Topic for insights (e.g., "religious language", "legal terminology")
            insight_type: Type of analysis ("linguistic_evolution", "social_context", 
                         "comparative_analysis")
        """
        return self.insights_generator.generate_insights(topic, insight_type)
    
    def track_language_changes(self, period_start: str, period_end: str,
                             focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Track specific language changes between periods.
        
        Args:
            period_start: Starting period
            period_end: Ending period  
            focus_areas: Areas to focus on (e.g., ["spelling", "vocabulary", "syntax"])
        """
        return self.evolution_analyzer.track_changes(
            period_start, period_end, focus_areas=focus_areas
        )
    
    def comparative_period_analysis(self, periods: List[str], 
                                  analysis_focus: str = "general") -> Dict[str, Any]:
        """
        Compare language characteristics across multiple periods.
        
        Args:
            periods: List of periods to compare
            analysis_focus: "general", "religious", "legal", "scientific"
        """
        return self.insights_generator.comparative_analysis(periods, analysis_focus)
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # Database stats
        with self.engine.connect() as conn:
            db_stats = {}
            
            # Basic corpus stats
            basic_stats = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT period) as periods,
                    COUNT(DISTINCT genre) as genres,
                    COUNT(DISTINCT doc_id) as documents,
                    MIN(year) as earliest_year,
                    MAX(year) as latest_year,
                    AVG(token_count) as avg_tokens_per_chunk,
                    SUM(token_count) as total_tokens
                FROM chunks
            """)).fetchone()
            
            db_stats['corpus_overview'] = dict(basic_stats._mapping)
            
            # Period distribution
            period_dist = conn.execute(text("""
                SELECT period, COUNT(*) as chunks, 
                       AVG(token_count) as avg_tokens,
                       COUNT(DISTINCT genre) as genres_in_period
                FROM chunks
                GROUP BY period
                ORDER BY period
            """)).fetchall()
            
            db_stats['period_analysis'] = [dict(row._mapping) for row in period_dist]
            
            # Genre analysis
            genre_dist = conn.execute(text("""
                SELECT genre, COUNT(*) as chunks,
                       AVG(token_count) as avg_tokens,
                       COUNT(DISTINCT period) as periods_present
                FROM chunks
                GROUP BY genre
                ORDER BY chunks DESC
            """)).fetchall()
            
            db_stats['genre_analysis'] = [dict(row._mapping) for row in genre_dist]
        
        # Vector store stats
        vector_stats = self.vector_store.get_statistics()
        
        # OpenAI usage stats
        openai_stats = {
            'embedding_model': OPENAI_CONFIG['embedding_model'],
            'chat_model': OPENAI_CONFIG['chat_model'],
            'embedding_dimensions': OPENAI_CONFIG['embedding_dimensions'],
            'max_tokens_per_request': OPENAI_CONFIG['max_tokens']
        }
        
        return {
            'database_stats': db_stats,
            'vector_stats': vector_stats,
            'openai_configuration': openai_stats,
            'pipeline_info': {
                'pipeline_type': 'OpenAI-powered RAG',
                'capabilities': [
                    'Advanced semantic search',
                    'GPT-4 powered question answering',
                    'Language evolution analysis', 
                    'Historical insights generation',
                    'Period-specific analysis',
                    'Comparative studies'
                ]
            }
        }
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests of all OpenAI RAG capabilities."""
        print("\n🧪 Running Comprehensive OpenAI RAG Tests...")
        print("=" * 60)
        
        test_results = {}
        
        # Test 1: Semantic Search
        print("\n1. Testing OpenAI Semantic Search...")
        search_query = "medieval religious language and biblical references"
        search_results = self.semantic_search(search_query, k=3)
        
        test_results['semantic_search'] = {
            'query': search_query,
            'results_found': len(search_results),
            'success': len(search_results) > 0
        }
        
        print(f"   Found {len(search_results)} semantically similar texts")
        
        # Test 2: Advanced Question Answering
        print("\n2. Testing GPT-4 Question Answering...")
        question = "How did German religious vocabulary evolve from medieval to early modern periods?"
        qa_result = self.ask_question(question, analysis_depth="standard")
        
        test_results['question_answering'] = {
            'question': question,
            'answer_length': len(qa_result.get('answer', '')),
            'sources_used': len(qa_result.get('sources', [])),
            'success': len(qa_result.get('answer', '')) > 100
        }
        
        print(f"   Generated {test_results['question_answering']['answer_length']} character answer")
        print(f"   Used {test_results['question_answering']['sources_used']} historical sources")
        
        # Test 3: Language Evolution Analysis
        print("\n3. Testing Language Evolution Analysis...")
        evolution_word = "gott"
        evolution_result = self.analyze_language_evolution(evolution_word, analysis_type="comprehensive")
        
        test_results['language_evolution'] = {
            'analyzed_word': evolution_word,
            'periods_analyzed': len(evolution_result.get('period_analysis', {})),
            'insights_generated': len(evolution_result.get('evolution_insights', '')),
            'success': len(evolution_result.get('evolution_insights', '')) > 50
        }
        
        print(f"   Analyzed '{evolution_word}' across {test_results['language_evolution']['periods_analyzed']} periods")
        
        # Test 4: Historical Insights Generation
        print("\n4. Testing Historical Insights Generation...")
        insights_topic = "legal terminology development"
        insights_result = self.generate_historical_insights(insights_topic, "linguistic_evolution")
        
        test_results['historical_insights'] = {
            'topic': insights_topic,
            'insights_length': len(insights_result.get('insights', '')),
            'success': len(insights_result.get('insights', '')) > 100
        }
        
        print(f"   Generated {test_results['historical_insights']['insights_length']} character insights")
        
        # Summary
        successful_tests = sum(1 for test in test_results.values() if test['success'])
        total_tests = len(test_results)
        
        print(f"\n🎯 Test Results: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 All OpenAI RAG capabilities working perfectly!")
        else:
            print("⚠️ Some capabilities need attention")
        
        return test_results
    
    def save_session_results(self, results: Dict[str, Any], session_name: str = None):
        """Save analysis results for later reference."""
        if not session_name:
            session_name = f"openai_rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results_file = self.vector_db_path / f"{session_name}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Session results saved to: {results_file}")
        return results_file