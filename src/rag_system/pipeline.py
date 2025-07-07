# ==========================================
# FILE: src/rag_system/pipeline.py (FIXED)
# ==========================================
"""Main RAG pipeline class - FIXED to respect LLM provider choice."""

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from typing import Dict, Any, Optional, List
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .semantic_search import SemanticSearcher
from .qa_chain import QAChainManager
from .language_evolution import LanguageEvolutionAnalyzer
from .statistics import StatisticsCalculator
from .utils import setup_logging, prepare_chunk_data, print_test_results
from .config import CHUNKS_QUERY, MIN_TEXT_LENGTH, DEFAULT_BATCH_SIZE


class GermanRAGPipeline:
    """Main RAG pipeline for German historical corpus."""
    
    def __init__(self, db_config: Dict[str, Any], vector_db_path: str = "./chroma_db"):
        self.db_config = db_config
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Database connection
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        # Initialize components
        self.embedding_manager = EmbeddingManager()
        self.vector_manager = VectorStoreManager(str(self.vector_db_path))
        self.searcher = SemanticSearcher(self.vector_manager, self.embedding_manager)
        self.qa_manager = QAChainManager(str(self.vector_db_path), self.vector_manager.collection_name)
        self.evolution_analyzer = LanguageEvolutionAnalyzer(self.searcher, self.qa_manager)
        self.stats_calculator = StatisticsCalculator(db_config, str(self.vector_db_path))
        
        setup_logging()
    
    def load_chunks_from_db(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load text chunks from PostgreSQL."""
        print("📚 Loading chunks from PostgreSQL...")
        
        query = CHUNKS_QUERY.format(min_length=MIN_TEXT_LENGTH)
        if limit:
            query += f" LIMIT {limit}"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print(f"✅ Loaded {len(df)} chunks from database")
        print(f"📊 Periods: {df['period'].unique()}")
        print(f"📊 Genres: {df['genre'].unique()}")
        return df
    
    def create_embeddings(self, chunks_df: pd.DataFrame, batch_size: int = DEFAULT_BATCH_SIZE):
        """Create and store embeddings."""
        print("🧠 Creating embeddings for text chunks...")
        
        collection = self.vector_manager.create_or_get_collection()
        
        # Check for existing embeddings
        try:
            existing_ids = set(collection.get()['ids'])
            new_chunks = chunks_df[~chunks_df['chunk_id'].astype(str).isin(existing_ids)]
            
            if len(new_chunks) == 0:
                print("✅ All chunks already embedded")
                return
            else:
                print(f"🔄 Adding {len(new_chunks)} new chunks")
                chunks_df = new_chunks
        except Exception:
            pass
        
        # Prepare data
        texts, chunk_ids, metadatas = prepare_chunk_data(chunks_df)
        
        try:
            # Create embeddings with error handling
            print(f"📊 Processing {len(texts)} texts in batches of {batch_size}")
            embeddings = self.embedding_manager.encode_texts(texts, batch_size=batch_size)
            
            # Convert to list if it's a numpy array
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = list(embeddings)
            
            # Store in vector database
            self.vector_manager.add_embeddings(
                texts, embeddings_list, metadatas, chunk_ids, batch_size
            )
            
            print(f"✅ Created embeddings for {len(chunks_df)} chunks")
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            print("💡 Trying with smaller batch size...")
            
            # Retry with smaller batch size
            try:
                smaller_batch = max(1, batch_size // 4)
                embeddings = self.embedding_manager.encode_texts(texts, batch_size=smaller_batch)
                
                if hasattr(embeddings, 'tolist'):
                    embeddings_list = embeddings.tolist()
                else:
                    embeddings_list = list(embeddings)
                
                self.vector_manager.add_embeddings(
                    texts, embeddings_list, metadatas, chunk_ids, smaller_batch
                )
                
                print(f"✅ Created embeddings with smaller batch size: {smaller_batch}")
                
            except Exception as e2:
                print(f"❌ Failed even with smaller batch: {e2}")
                print("🔧 Suggested fixes:")
                print("   pip install numpy==1.24.3")
                print("   pip install --upgrade sentence-transformers")
                raise
    
    def setup_qa_system(self, llm_provider: str = "simple"):
        """Setup QA system components - FIXED to respect provider choice."""
        print(f"🔗 Setting up QA system with {llm_provider} provider...")
        self.qa_manager.setup_vectorstore()
        
        # REMOVED: The forced simple mode override
        # Now respects the user's actual choice
        
        if llm_provider == "openai":
            # Check for OpenAI API key
            import os
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ OPENAI_API_KEY not found in environment variables!")
                print("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
                print("🔄 Falling back to simple mode...")
                llm_provider = "simple"
            else:
                print("✅ OpenAI API key found, initializing GPT model...")
        elif llm_provider == "huggingface":
            print("🤗 Initializing HuggingFace model...")
        else:
            print("📝 Using simple retrieval mode (no LLM generation)")
            
        self.qa_manager.setup_qa_chain(llm_provider)
    
    def semantic_search(self, query: str, k: int = 5, period_filter: Optional[str] = None) -> List[Dict]:
        """Perform semantic search."""
        return self.searcher.search(query, k, period_filter)
    
    def ask_question(self, question: str, period_filter: Optional[str] = None) -> Dict[str, Any]:
        """Ask a question using RAG."""
        return self.qa_manager.ask_question(question, period_filter)
    
    def analyze_language_evolution(self, word: str, periods: List[str] = None) -> Dict[str, Any]:
        """Analyze language evolution."""
        return self.evolution_analyzer.analyze_word_evolution(word, periods)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.stats_calculator.get_statistics()
    
    def run_tests(self):
        """Run system tests."""
        print("\n🧪 Testing RAG system...")
        
        # Get actual periods from database
        stats = self.get_statistics()
        actual_periods = [p['period'] for p in stats['database_stats']['period_distribution']]
        print(f"📅 Available periods in database: {actual_periods}")
        
        # Test semantic search
        search_results = self.semantic_search("deutsche Sprache", k=3)
        
        # Test QA
        qa_result = self.ask_question("Wie war die deutsche Sprache in dieser Zeit?")
        
        # Test evolution analysis with actual periods
        if actual_periods:
            evolution = self.analyze_language_evolution("deutsch", periods=actual_periods)
        else:
            evolution = self.analyze_language_evolution("deutsch")
        
        print_test_results(search_results, qa_result, evolution)
        
        # Show statistics
        print("\n📊 System Statistics:")
        print(f"   Total chunks in DB: {stats['database_stats']['chunks']['total_chunks']}")
        print(f"   Total embeddings: {stats['vector_stats']['total_embeddings']}")
        print(f"   Time periods: {stats['database_stats']['chunks']['period_count']}")
        print(f"   Available periods: {actual_periods}")