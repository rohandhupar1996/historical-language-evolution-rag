# ==========================================
# FILE: src/rag_system/openai_embeddings.py
# ==========================================
"""OpenAI embeddings integration for comparison with SentenceTransformers."""

import os
import numpy as np
from typing import List, Dict, Any
import openai
from openai import OpenAI
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OpenAIEmbeddingManager:
    """Manages OpenAI embeddings for semantic search."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        """
        Initialize OpenAI embedding manager.
        
        Args:
            model_name: OpenAI embedding model to use
                - "text-embedding-3-small" (1536 dimensions, cheaper)
                - "text-embedding-3-large" (3072 dimensions, more powerful)
                - "text-embedding-ada-002" (1536 dimensions, legacy)
            api_key: OpenAI API key (will use env var if not provided)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        print(f"🔄 Loading OpenAI embedding model: {self.model_name}")
        print(f"📊 Dimensions: {self.dimensions.get(self.model_name, 'Unknown')}")
        print("✅ OpenAI embedding model ready")
    
    def encode_texts(self, texts: List[str], batch_size: int = 100, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings using OpenAI API.
        
        Args:
            texts: List of texts to encode
            batch_size: Number of texts to process per API call (max 2048 for OpenAI)
            show_progress: Whether to show progress
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                batch_num = i // batch_size + 1
                print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - OpenAI has limits
                if i + batch_size < len(texts):
                    time.sleep(0.1)  # Small delay between batches
                    
            except Exception as e:
                logger.error(f"OpenAI embedding error for batch {batch_num}: {e}")
                raise
        
        return np.array(all_embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimensions.get(self.model_name, 1536)

# ==========================================
# FILE: src/rag_system/embedding_comparator.py
# ==========================================
"""Compare different embedding models."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from .embeddings import EmbeddingManager
from .openai_embeddings import OpenAIEmbeddingManager
from .vector_store import VectorStoreManager
import time

class EmbeddingComparator:
    """Compare SentenceTransformers vs OpenAI embeddings."""
    
    def __init__(self, db_config: Dict, base_vector_path: str):
        self.db_config = db_config
        self.base_vector_path = Path(base_vector_path)
        
        # Create separate vector stores for comparison
        self.sentence_transformer_path = self.base_vector_path / "sentence_transformers"
        self.openai_small_path = self.base_vector_path / "openai_small"
        self.openai_large_path = self.base_vector_path / "openai_large"
        
        # Initialize embedding managers
        self.st_embedder = EmbeddingManager()  # SentenceTransformers
        self.openai_small_embedder = OpenAIEmbeddingManager("text-embedding-3-small")
        self.openai_large_embedder = OpenAIEmbeddingManager("text-embedding-3-large")
        
        # Initialize vector stores
        self.st_store = VectorStoreManager(str(self.sentence_transformer_path), "st_chunks")
        self.openai_small_store = VectorStoreManager(str(self.openai_small_path), "openai_small_chunks")
        self.openai_large_store = VectorStoreManager(str(self.openai_large_path), "openai_large_chunks")
    
    def create_all_embeddings(self, chunks_df: pd.DataFrame, limit: int = 1000):
        """Create embeddings with all three models for comparison."""
        print("🔄 Creating embeddings with multiple models for comparison...")
        
        # Limit chunks for comparison
        if len(chunks_df) > limit:
            chunks_df = chunks_df.head(limit)
            print(f"📊 Using first {limit} chunks for comparison")
        
        texts = chunks_df['text'].tolist()
        chunk_ids = [f"{row['chunk_id']}_{idx}" for idx, row in chunks_df.iterrows()]
        metadatas = []
        
        for idx, row in chunks_df.iterrows():
            metadata = {
                'period': str(row['period']),
                'document_id': str(row['document_id']),
                'chunk_index': int(row['chunk_index']),
                'word_count': int(row['word_count']),
                'genre': str(row.get('genre', '')),
                'year': str(row.get('year', '')),
                'filename': str(row.get('filename', '')),
                'original_chunk_id': str(row['chunk_id'])
            }
            metadatas.append(metadata)
        
        # 1. SentenceTransformers embeddings
        print("\n🤖 Creating SentenceTransformers embeddings...")
        start_time = time.time()
        st_embeddings = self.st_embedder.encode_texts(texts, batch_size=32)
        st_time = time.time() - start_time
        
        st_collection = self.st_store.create_or_get_collection()
        self.st_store.add_embeddings(texts, st_embeddings.tolist(), metadatas, chunk_ids, 32)
        print(f"✅ SentenceTransformers: {len(texts)} embeddings in {st_time:.1f}s")
        
        # 2. OpenAI Small embeddings
        print("\n🔥 Creating OpenAI Small embeddings...")
        start_time = time.time()
        openai_small_embeddings = self.openai_small_embedder.encode_texts(texts, batch_size=50)
        openai_small_time = time.time() - start_time
        
        openai_small_collection = self.openai_small_store.create_or_get_collection()
        self.openai_small_store.add_embeddings(texts, openai_small_embeddings.tolist(), metadatas, chunk_ids, 32)
        print(f"✅ OpenAI Small: {len(texts)} embeddings in {openai_small_time:.1f}s")
        
        # 3. OpenAI Large embeddings (optional - costs more)
        create_large = input("\n❓ Create OpenAI Large embeddings too? (costs more) [y/N]: ").lower().startswith('y')
        
        if create_large:
            print("\n🚀 Creating OpenAI Large embeddings...")
            start_time = time.time()
            openai_large_embeddings = self.openai_large_embedder.encode_texts(texts, batch_size=50)
            openai_large_time = time.time() - start_time
            
            openai_large_collection = self.openai_large_store.create_or_get_collection()
            self.openai_large_store.add_embeddings(texts, openai_large_embeddings.tolist(), metadatas, chunk_ids, 32)
            print(f"✅ OpenAI Large: {len(texts)} embeddings in {openai_large_time:.1f}s")
        
        print("\n🎉 All embedding models ready for comparison!")
        
        return {
            'sentence_transformers': {'time': st_time, 'dimensions': 384},
            'openai_small': {'time': openai_small_time, 'dimensions': 1536},
            'openai_large': {'time': openai_large_time, 'dimensions': 3072} if create_large else None
        }
    
    def compare_search_quality(self, test_queries: List[str], k: int = 5) -> Dict[str, Any]:
        """Compare search quality across different embedding models."""
        print(f"\n🔍 Comparing search quality with {len(test_queries)} test queries...")
        
        results = {
            'queries': test_queries,
            'sentence_transformers': [],
            'openai_small': [],
            'openai_large': []
        }
        
        for i, query in enumerate(test_queries):
            print(f"\n📝 Query {i+1}: '{query}'")
            
            # SentenceTransformers search
            st_results = self.st_store.query(query, k)
            st_distances = st_results['distances'][0] if st_results['distances'] else []
            st_avg_distance = sum(st_distances) / len(st_distances) if st_distances else 1.0
            
            # OpenAI Small search
            openai_small_results = self.openai_small_store.query(query, k)
            openai_small_distances = openai_small_results['distances'][0] if openai_small_results['distances'] else []
            openai_small_avg_distance = sum(openai_small_distances) / len(openai_small_distances) if openai_small_distances else 1.0
            
            print(f"   🤖 SentenceTransformers avg distance: {st_avg_distance:.3f}")
            print(f"   🔥 OpenAI Small avg distance: {openai_small_avg_distance:.3f}")
            
            results['sentence_transformers'].append({
                'query': query,
                'avg_distance': st_avg_distance,
                'results_count': len(st_results['documents'][0]) if st_results['documents'] else 0
            })
            
            results['openai_small'].append({
                'query': query,
                'avg_distance': openai_small_avg_distance,
                'results_count': len(openai_small_results['documents'][0]) if openai_small_results['documents'] else 0
            })
        
        return results

# ==========================================
# FILE: compare_embeddings.py (Main script)
# ==========================================
"""Main script to compare embedding models."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_system.embedding_comparator import EmbeddingComparator
from src.rag_system.pipeline import GermanRAGPipeline
from src.rag_system.config import DEFAULT_DB_CONFIG

def main():
    """Compare embedding models."""
    print("🔬 Embedding Models Comparison for German Historical Corpus")
    print("=" * 60)
    
    # Database config
    db_config = DEFAULT_DB_CONFIG.copy()
    db_config.update({
        'host': 'localhost',
        'port': 5432,
        'database': 'germanc_corpus',
        'user': 'rohan',
        'password': '1996'
    })
    
    # Base path for comparisons
    comparison_path = "./embedding_comparison"
    
    # Initialize comparator
    comparator = EmbeddingComparator(db_config, comparison_path)
    
    # Load chunks from database
    pipeline = GermanRAGPipeline(db_config, "./temp_vectordb")
    chunks_df = pipeline.load_chunks_from_db(limit=500)  # Start with 500 for comparison
    
    # Create embeddings with all models
    embedding_stats = comparator.create_all_embeddings(chunks_df, limit=500)
    
    # Test queries for comparison
    test_queries = [
        "How did German language evolve?",
        "religious language in medieval texts",
        "legal terminology development",
        "archaic spelling patterns",
        "deutsche sprache entwicklung",
        "mittelalterliche rechtsbegriffe"
    ]
    
    # Compare search quality
    comparison_results = comparator.compare_search_quality(test_queries)
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("📊 EMBEDDING COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n⏱️ Processing Times:")
    for model, stats in embedding_stats.items():
        if stats:
            print(f"   {model}: {stats['time']:.1f}s ({stats['dimensions']} dimensions)")
    
    print(f"\n🎯 Search Quality (Average Distance - Lower is Better):")
    st_avg = sum(r['avg_distance'] for r in comparison_results['sentence_transformers']) / len(test_queries)
    openai_avg = sum(r['avg_distance'] for r in comparison_results['openai_small']) / len(test_queries)
    
    print(f"   SentenceTransformers: {st_avg:.3f}")
    print(f"   OpenAI Small: {openai_avg:.3f}")
    
    if openai_avg < st_avg:
        print(f"   🏆 Winner: OpenAI Small (better by {st_avg - openai_avg:.3f})")
    else:
        print(f"   🏆 Winner: SentenceTransformers (better by {openai_avg - st_avg:.3f})")
    
    print(f"\n💡 Recommendations:")
    print(f"   - For cost efficiency: SentenceTransformers (free)")
    print(f"   - For best quality: OpenAI {'Small' if openai_avg < st_avg else 'Large'}")
    print(f"   - For production: Consider hybrid approach")

if __name__ == "__main__":
    main()