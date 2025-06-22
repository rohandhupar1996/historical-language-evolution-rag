# ==========================================
# FILE: prepare_pipeline/chunk_creator.py
# ==========================================
"""Chunk creation utilities."""

import pandas as pd
from typing import List, Dict, Any
from .config import DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE, SENTENCE_ENDINGS


class ChunkCreator:
    """Creates temporal chunks from tokens."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
    
    def create_temporal_chunks(self, documents: List[Dict], tokens_df: pd.DataFrame) -> List[Dict]:
        """Create temporal chunks from documents and tokens."""
        chunks = []
        
        for doc in documents:
            doc_id = doc['doc_id']
            doc_tokens = tokens_df[tokens_df['doc_id'] == doc_id]
            
            if len(doc_tokens) == 0:
                continue
            
            doc_chunks = self._chunk_document(doc, doc_tokens)
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _chunk_document(self, doc: Dict, tokens: pd.DataFrame) -> List[Dict]:
        """Split document into chunks."""
        chunks = []
        
        if 'position' in tokens.columns:
            tokens = tokens.sort_values('position')
        
        current_chunk = []
        current_size = 0
        chunk_counter = 0
        
        for _, token in tokens.iterrows():
            current_chunk.append(token)
            current_size += 1
            
            if current_size >= self.chunk_size or self._is_sentence_end(token):
                if len(current_chunk) > MIN_CHUNK_SIZE:
                    chunk = self._create_chunk(doc, current_chunk, chunk_counter)
                    chunks.append(chunk)
                    chunk_counter += 1
                
                current_chunk = []
                current_size = 0
        
        # Handle remaining tokens
        if len(current_chunk) > MIN_CHUNK_SIZE:
            chunk = self._create_chunk(doc, current_chunk, chunk_counter)
            chunks.append(chunk)
        
        return chunks
    
    def _is_sentence_end(self, token) -> bool:
        """Check if token ends a sentence."""
        original = str(token.get('original', ''))
        return any(original.endswith(ending) for ending in SENTENCE_ENDINGS)
    
    def _create_chunk(self, doc: Dict, tokens: List, chunk_num: int) -> Dict:
        """Create chunk with metadata."""
        original_text = ' '.join([str(token.get('original', '')) for token in tokens])
        normalized_text = ' '.join([str(token.get('normalized', '')) for token in tokens])
        
        return {
            'chunk_id': f"{doc['doc_id']}_chunk_{chunk_num}",
            'doc_id': doc['doc_id'],
            'original_text': original_text,
            'normalized_text': normalized_text,
            'period': doc.get('period', 'unknown'),
            'genre': doc.get('genre', 'unknown'),
            'year': doc.get('year'),
            'filename': doc.get('filename', ''),
            'token_count': len(tokens),
            'tokens': tokens
        }