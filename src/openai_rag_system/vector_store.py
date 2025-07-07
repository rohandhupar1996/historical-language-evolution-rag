# ==========================================
# FILE: src/openai_rag_system/vector_store.py
# ==========================================
"""Advanced vector store for OpenAI embeddings with German historical corpus optimization."""

import chromadb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import uuid

from .config import VECTOR_DB_CONFIG, HISTORICAL_PERIODS
from .embeddings import OpenAIEmbeddingManager


class OpenAIVectorStore:
    """
    Advanced vector store optimized for OpenAI embeddings and 
    German historical corpus analysis.
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with optimized settings
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = VECTOR_DB_CONFIG['collection_name']
        self.collection = None
        self.embedding_manager = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized OpenAI Vector Store at {db_path}")
    
    def create_or_get_collection(self) -> chromadb.Collection:
        """Create or get the German corpus collection with optimized metadata."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll provide embeddings directly
            )
            
            existing_count = self.collection.count()
            self.logger.info(f"Found existing collection with {existing_count} documents")
            
        except Exception:
            # Create new collection with metadata schema
            metadata_schema = {
                "description": "German Historical Corpus with OpenAI embeddings",
                "embedding_model": "text-embedding-3-large",
                "language": "German (Historical)",
                "created_at": datetime.now().isoformat()
            }
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata_schema,
                embedding_function=None
            )
            
            self.logger.info("Created new OpenAI vector collection")
        
        return self.collection
    
    def add_texts_with_embeddings(self, texts: List[str], metadatas: List[Dict[str, Any]],
                                embedding_manager: OpenAIEmbeddingManager = None):
        """
        Add texts with their OpenAI embeddings to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts for each text
            embedding_manager: OpenAI embedding manager
        """
        if not self.collection:
            self.create_or_get_collection()
        
        if not embedding_manager:
            embedding_manager = OpenAIEmbeddingManager()
        
        print(f"🔄 Adding {len(texts)} texts to OpenAI vector store...")
        
        # Create embeddings using OpenAI
        embedding_result = embedding_manager.embed_texts_batch(texts, show_progress=True)
        embeddings = embedding_result.embeddings
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Enhance metadata with vector store specific info
        enhanced_metadatas = []
        for i, metadata in enumerate(metadatas):
            enhanced_meta = metadata.copy()
            enhanced_meta.update({
                'vector_id': ids[i],
                'text_length': len(texts[i]),
                'embedding_model': 'text-embedding-3-large',
                'added_at': datetime.now().isoformat(),
                'chunk_index': i
            })
            enhanced_metadatas.append(enhanced_meta)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=enhanced_metadatas[i:end_idx]
            )
            
            print(f"   Added batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        print(f"✅ Successfully added {len(texts)} texts to vector store")
        print(f"💰 Embedding cost: ${embedding_result.cost_estimate:.4f}")
        
        return embedding_result
    
    def similarity_search(self, query: str, k: int = 5, 
                         period_filter: Optional[str] = None,
                         genre_filter: Optional[str] = None,
                         similarity_threshold: float = 0.7,
                         embedding_manager: OpenAIEmbeddingManager = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search with OpenAI query embedding.
        
        Args:
            query: Search query
            k: Number of results
            period_filter: Filter by historical period
            genre_filter: Filter by genre
            similarity_threshold: Minimum similarity score
            embedding_manager: OpenAI embedding manager
            
        Returns:
            List of similar documents with metadata
        """
        if not self.collection:
            self.create_or_get_collection()
        
        if not embedding_manager:
            embedding_manager = OpenAIEmbeddingManager()
        
        # Create query embedding
        query_embedding = embedding_manager.embed_query(query)
        
        # Build where clause for filters
        where_conditions = {}
        if period_filter:
            where_conditions['period'] = period_filter
        if genre_filter:
            where_conditions['genre'] = genre_filter
        
        # Search with filters
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,  # Get more results to filter by threshold
            where=where_conditions if where_conditions else None,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Process and filter results
        processed_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # Convert distance to similarity (ChromaDB uses cosine distance)
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                # Filter by similarity threshold
                if similarity >= similarity_threshold:
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': similarity,
                        'distance': distance,
                        'rank': len(processed_results) + 1
                    }
                    processed_results.append(result)
                
                # Stop if we have enough results
                if len(processed_results) >= k:
                    break
        
        self.logger.info(f"Similarity search: {len(processed_results)} results for '{query[:50]}...'")
        return processed_results
    
    def advanced_search(self, query: str, search_params: Dict[str, Any],
                       embedding_manager: OpenAIEmbeddingManager = None) -> Dict[str, Any]:
        """
        Perform advanced search with sophisticated filtering and ranking.
        
        Args:
            query: Search query
            search_params: Advanced search parameters
            embedding_manager: OpenAI embedding manager
            
        Returns:
            Advanced search results with analytics
        """
        # Extract search parameters
        k = search_params.get('k', 10)
        period_filter = search_params.get('period_filter')
        genre_filter = search_params.get('genre_filter')
        year_range = search_params.get('year_range')  # (start_year, end_year)
        include_context = search_params.get('include_context', True)
        rerank_results = search_params.get('rerank', True)
        
        # Initial similarity search
        initial_results = self.similarity_search(
            query, k=k*2, period_filter=period_filter,
            genre_filter=genre_filter, embedding_manager=embedding_manager
        )
        
        # Apply additional filters
        filtered_results = self._apply_advanced_filters(initial_results, search_params)
        
        # Rerank if requested
        if rerank_results and len(filtered_results) > 1:
            filtered_results = self._rerank_results(query, filtered_results, embedding_manager)
        
        # Limit to requested number
        final_results = filtered_results[:k]
        
        # Add context if requested
        if include_context:
            final_results = self._add_context_to_results(final_results)
        
        # Generate search analytics
        analytics = self._generate_search_analytics(query, final_results, search_params)
        
        return {
            'query': query,
            'results': final_results,
            'total_found': len(filtered_results),
            'analytics': analytics,
            'search_params': search_params
        }
    
    def _apply_advanced_filters(self, results: List[Dict], search_params: Dict) -> List[Dict]:
        """Apply advanced filtering to search results."""
        filtered = results.copy()
        
        # Year range filter
        if 'year_range' in search_params:
            start_year, end_year = search_params['year_range']
            filtered = [
                r for r in filtered 
                if start_year <= int(r['metadata'].get('year', 0)) <= end_year
            ]
        
        # Minimum text length filter
        if 'min_text_length' in search_params:
            min_length = search_params['min_text_length']
            filtered = [
                r for r in filtered 
                if len(r['document']) >= min_length
            ]
        
        # Exclude certain terms
        if 'exclude_terms' in search_params:
            exclude_terms = search_params['exclude_terms']
            for term in exclude_terms:
                filtered = [
                    r for r in filtered 
                    if term.lower() not in r['document'].lower()
                ]
        
        return filtered
    
    def _rerank_results(self, query: str, results: List[Dict],
                       embedding_manager: OpenAIEmbeddingManager) -> List[Dict]:
        """Rerank results using additional signals."""
        # For now, use a simple reranking based on multiple factors
        def rerank_score(result):
            base_similarity = result['similarity_score']
            
            # Boost for exact query terms
            query_terms = query.lower().split()
            doc_text = result['document'].lower()
            term_matches = sum(1 for term in query_terms if term in doc_text)
            term_boost = 0.1 * term_matches / len(query_terms)
            
            # Boost for certain periods (more historical = higher)
            period_boost = 0.0
            period = result['metadata'].get('period', '')
            if '1050-1350' in period:
                period_boost = 0.15
            elif '1350-1650' in period:
                period_boost = 0.10
            elif '1650-1800' in period:
                period_boost = 0.05
            
            return base_similarity + term_boost + period_boost
        
        # Sort by rerank score
        results.sort(key=rerank_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1
            result['rerank_score'] = rerank_score(result)
        
        return results
    
    def _add_context_to_results(self, results: List[Dict]) -> List[Dict]:
        """Add contextual information to search results."""
        for result in results:
            metadata = result['metadata']
            
            # Add period description
            period = metadata.get('period', '')
            if period in HISTORICAL_PERIODS:
                result['period_description'] = HISTORICAL_PERIODS[period]
            
            # Add text statistics
            text = result['document']
            result['text_stats'] = {
                'length': len(text),
                'word_count': len(text.split()),
                'sentence_count': text.count('.') + text.count('!') + text.count('?')
            }
            
            # Add snippet (highlighted excerpt)
            result['snippet'] = self._create_snippet(text, 200)
        
        return results
    
    def _create_snippet(self, text: str, max_length: int = 200) -> str:
        """Create a snippet from text."""
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        sentences = text.split('.')
        snippet = ""
        for sentence in sentences:
            if len(snippet + sentence + '.') <= max_length:
                snippet += sentence + '.'
            else:
                break
        
        if not snippet:
            snippet = text[:max_length] + "..."
        
        return snippet.strip()
    
    def _generate_search_analytics(self, query: str, results: List[Dict],
                                 search_params: Dict) -> Dict[str, Any]:
        """Generate analytics for search results."""
        if not results:
            return {'message': 'No results found'}
        
        # Period distribution
        periods = [r['metadata'].get('period', 'Unknown') for r in results]
        period_dist = {period: periods.count(period) for period in set(periods)}
        
        # Genre distribution  
        genres = [r['metadata'].get('genre', 'Unknown') for r in results]
        genre_dist = {genre: genres.count(genre) for genre in set(genres)}
        
        # Similarity statistics
        similarities = [r['similarity_score'] for r in results]
        
        return {
            'query_terms': len(query.split()),
            'results_count': len(results),
            'period_distribution': period_dist,
            'genre_distribution': genre_dist,
            'similarity_stats': {
                'max': max(similarities),
                'min': min(similarities),
                'average': sum(similarities) / len(similarities)
            },
            'year_range': {
                'earliest': min(int(r['metadata'].get('year', 9999)) for r in results),
                'latest': max(int(r['metadata'].get('year', 0)) for r in results)
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive vector store statistics."""
        if not self.collection:
            return {'error': 'No collection exists'}
        
        total_docs = self.collection.count()
        
        if total_docs == 0:
            return {'total_documents': 0, 'message': 'Empty collection'}
        
        # Get sample of documents for analysis
        sample_size = min(100, total_docs)
        sample_results = self.collection.get(limit=sample_size, include=['metadatas'])
        
        # Analyze metadata
        periods = []
        genres = []
        years = []
        
        for metadata in sample_results['metadatas']:
            if metadata.get('period'):
                periods.append(metadata['period'])
            if metadata.get('genre'):
                genres.append(metadata['genre'])
            if metadata.get('year'):
                try:
                    years.append(int(metadata['year']))
                except:
                    pass
        
        return {
            'total_documents': total_docs,
            'collection_name': self.collection_name,
            'sample_analysis': {
                'sample_size': len(sample_results['metadatas']),
                'periods_found': list(set(periods)),
                'genres_found': list(set(genres)),
                'year_range': {
                    'earliest': min(years) if years else None,
                    'latest': max(years) if years else None
                }
            },
            'database_info': {
                'embedding_model': 'text-embedding-3-large',
                'embedding_dimensions': 3072,
                'distance_metric': 'cosine',
                'storage_path': str(self.db_path)
            }
        }
    
    def has_embeddings(self) -> bool:
        """Check if vector store has embeddings."""
        if not self.collection:
            try:
                self.create_or_get_collection()
            except:
                return False
        
        return self.collection.count() > 0
    
    def reset_collection(self):
        """Reset the collection (use with caution)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info("Collection reset")
        except:
            pass
        
        self.collection = None