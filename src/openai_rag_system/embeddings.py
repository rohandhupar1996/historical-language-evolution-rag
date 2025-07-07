# ==========================================
# FILE: src/openai_rag_system/embeddings.py
# ==========================================
"""OpenAI embeddings management for German historical corpus."""

import openai
import numpy as np
import time
import logging
from typing import List, Union, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

from .config import OPENAI_CONFIG, VALIDATION_CONFIG
from .utils import estimate_tokens, handle_rate_limits, validate_text_input


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: List[List[float]]
    total_tokens: int
    cost_estimate: float
    processing_time: float


class OpenAIEmbeddingManager:
    """
    Advanced OpenAI embeddings manager with rate limiting, 
    cost tracking, and batch processing.
    """
    
    def __init__(self):
        self.model = OPENAI_CONFIG['embedding_model']
        self.dimensions = OPENAI_CONFIG['embedding_dimensions']
        self.batch_size = OPENAI_CONFIG['embedding_batch_size']
        self.max_retries = OPENAI_CONFIG['max_retries']
        
        # Initialize tokenizer for cost estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Tracking
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        self.api_calls_made = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized OpenAI Embedding Manager with {self.model}")
    
    def embed_text(self, text: str, **kwargs) -> List[float]:
        """
        Embed a single text using OpenAI API.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector
        """
        # Validate input
        if not validate_text_input(text):
            raise ValueError(f"Invalid text input: {text[:100]}...")
        
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text,
                **kwargs
            )
            
            # Update tracking
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            self.api_calls_made += 1
            
            # Estimate cost
            cost = tokens_used * OPENAI_CONFIG.get('embedding_cost_per_1k_tokens', 0.00013) / 1000
            self.total_cost_estimate += cost
            
            embedding = response.data[0].embedding
            
            self.logger.info(f"Embedded text ({tokens_used} tokens, ${cost:.4f})")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}")
            raise
    
    def embed_texts_batch(self, texts: List[str], show_progress: bool = True) -> EmbeddingResult:
        """
        Embed multiple texts in optimized batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress information
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        all_embeddings = []
        total_tokens = 0
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if show_progress:
                print(f"🧠 Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            # Validate batch texts
            valid_batch = []
            for text in batch:
                if validate_text_input(text):
                    valid_batch.append(text)
                else:
                    # Add zero vector for invalid texts
                    all_embeddings.append([0.0] * self.dimensions)
                    self.logger.warning(f"Skipped invalid text: {text[:50]}...")
            
            if not valid_batch:
                continue
            
            # Create embeddings with retry logic
            batch_embeddings = self._embed_batch_with_retry(valid_batch)
            all_embeddings.extend(batch_embeddings)
            
            # Estimate tokens for this batch
            batch_tokens = sum(estimate_tokens(text, self.tokenizer) for text in valid_batch)
            total_tokens += batch_tokens
            
            # Rate limiting
            handle_rate_limits(self.api_calls_made, self.total_tokens_used)
        
        processing_time = time.time() - start_time
        cost_estimate = total_tokens * OPENAI_CONFIG.get('embedding_cost_per_1k_tokens', 0.00013) / 1000
        
        self.total_cost_estimate += cost_estimate
        
        if show_progress:
            print(f"✅ Embedded {len(texts)} texts in {processing_time:.1f}s")
            print(f"💰 Estimated cost: ${cost_estimate:.4f}")
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            total_tokens=total_tokens,
            cost_estimate=cost_estimate,
            processing_time=processing_time
        )
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                # Update tracking
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used
                self.api_calls_made += 1
                
                return [data.embedding for data in response.data]
                
            except openai.RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
        
        raise Exception(f"Failed to embed batch after {self.max_retries} attempts")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query with query-specific optimizations.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        # Preprocess query for historical German context
        processed_query = self._preprocess_query(query)
        
        return self.embed_text(processed_query)
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for better historical German matching.
        
        Args:
            query: Original query
            
        Returns:
            Processed query
        """
        # Add context for historical German understanding
        if len(query.split()) <= 3:
            # Short queries benefit from historical context
            processed = f"Historical German text about: {query}. Medieval and early modern German language."
        else:
            # Longer queries get minimal processing
            processed = f"In historical German texts: {query}"
        
        return processed
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive embedding statistics."""
        return {
            'model_info': {
                'model': self.model,
                'dimensions': self.dimensions,
                'batch_size': self.batch_size
            },
            'usage_stats': {
                'total_api_calls': self.api_calls_made,
                'total_tokens_used': self.total_tokens_used,
                'estimated_total_cost': self.total_cost_estimate,
                'average_tokens_per_call': self.total_tokens_used / max(1, self.api_calls_made)
            },
            'performance_metrics': {
                'successful_calls': self.api_calls_made,
                'estimated_cost_per_1k_tokens': OPENAI_CONFIG.get('embedding_cost_per_1k_tokens', 0.00013)
            }
        }
    
    def estimate_embedding_cost(self, texts: List[str]) -> Dict[str, float]:
        """
        Estimate the cost of embedding a list of texts.
        
        Args:
            texts: List of texts to estimate
            
        Returns:
            Cost estimation breakdown
        """
        total_tokens = sum(estimate_tokens(text, self.tokenizer) for text in texts)
        cost_per_1k = OPENAI_CONFIG.get('embedding_cost_per_1k_tokens', 0.00013)
        total_cost = (total_tokens / 1000) * cost_per_1k
        
        return {
            'total_texts': len(texts),
            'estimated_tokens': total_tokens,
            'estimated_cost_usd': total_cost,
            'cost_per_1k_tokens': cost_per_1k,
            'average_tokens_per_text': total_tokens / len(texts) if texts else 0
        }
    
    def create_embeddings_for_corpus(self, corpus_texts: List[str], 
                                   metadata: List[Dict] = None,
                                   save_checkpoints: bool = True) -> EmbeddingResult:
        """
        Create embeddings for entire corpus with checkpointing.
        
        Args:
            corpus_texts: All corpus texts
            metadata: Optional metadata for each text
            save_checkpoints: Save progress checkpoints
            
        Returns:
            Complete embedding result
        """
        print(f"🚀 Creating OpenAI embeddings for {len(corpus_texts)} corpus texts")
        print(f"📊 Model: {self.model} ({self.dimensions} dimensions)")
        
        # Estimate cost
        cost_estimate = self.estimate_embedding_cost(corpus_texts)
        print(f"💰 Estimated cost: ${cost_estimate['estimated_cost_usd']:.2f}")
        
        # Create embeddings
        result = self.embed_texts_batch(corpus_texts, show_progress=True)
        
        print(f"✅ Corpus embedding complete!")
        print(f"📈 Total tokens: {result.total_tokens:,}")
        print(f"💸 Actual cost: ${result.cost_estimate:.4f}")
        print(f"⏱️  Processing time: {result.processing_time:.1f} seconds")
        
        return result
    
    def reset_usage_tracking(self):
        """Reset usage tracking counters."""
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        self.api_calls_made = 0
        self.logger.info("Usage tracking reset")


class AdaptiveEmbeddingManager(OpenAIEmbeddingManager):
    """
    Advanced embedding manager with adaptive strategies for 
    historical German texts.
    """
    
    def __init__(self):
        super().__init__()
        self.historical_context_cache = {}
        self.period_specific_strategies = {
            '1050-1350': self._middle_high_german_strategy,
            '1350-1650': self._early_new_high_german_strategy,
            '1650-1800': self._baroque_german_strategy,
            '1800-1900': self._modern_german_strategy,
            '1900-2000': self._contemporary_german_strategy
        }
    
    def embed_historical_text(self, text: str, period: str = None, 
                            genre: str = None) -> List[float]:
        """
        Embed text with period and genre-specific strategies.
        
        Args:
            text: Historical text to embed
            period: Time period (e.g., '1350-1650')
            genre: Text genre (e.g., 'Religious', 'Legal')
            
        Returns:
            Embedding vector optimized for historical context
        """
        # Apply period-specific preprocessing
        if period and period in self.period_specific_strategies:
            processed_text = self.period_specific_strategies[period](text, genre)
        else:
            processed_text = self._default_historical_strategy(text, genre)
        
        return self.embed_text(processed_text)
    
    def _middle_high_german_strategy(self, text: str, genre: str = None) -> str:
        """Strategy for Middle High German texts (1050-1350)."""
        context = "Medieval High German text with archaic spelling and vocabulary. "
        if genre == 'Religious':
            context += "Religious manuscript with Latin influences. "
        elif genre == 'Legal':
            context += "Medieval legal document with formal language. "
        
        return f"{context}{text}"
    
    def _early_new_high_german_strategy(self, text: str, genre: str = None) -> str:
        """Strategy for Early New High German texts (1350-1650)."""
        context = "Early New High German text with transitional spelling. "
        if genre == 'Religious':
            context += "Reformation era religious text. "
        elif genre == 'Legal':
            context += "Early modern legal document. "
        
        return f"{context}{text}"
    
    def _baroque_german_strategy(self, text: str, genre: str = None) -> str:
        """Strategy for Baroque German texts (1650-1800)."""
        context = "Baroque period German text with standardizing orthography. "
        if genre == 'Scientific':
            context += "Early scientific text with emerging terminology. "
        
        return f"{context}{text}"
    
    def _modern_german_strategy(self, text: str, genre: str = None) -> str:
        """Strategy for Modern German texts (1800-1900)."""
        context = "19th century German text with modern standardization. "
        return f"{context}{text}"
    
    def _contemporary_german_strategy(self, text: str, genre: str = None) -> str:
        """Strategy for Contemporary German texts (1900-2000)."""
        context = "20th century German text. "
        return f"{context}{text}"
    
    def _default_historical_strategy(self, text: str, genre: str = None) -> str:
        """Default strategy for unspecified periods."""
        context = "Historical German text. "
        if genre:
            context += f"{genre} genre. "
        
        return f"{context}{text}"