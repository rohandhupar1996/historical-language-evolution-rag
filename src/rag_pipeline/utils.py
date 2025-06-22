
# ==========================================
# FILE: rag_system/utils.py
# ==========================================
"""Utility functions for RAG system."""

import logging
import pandas as pd
from typing import List, Dict, Any


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def prepare_chunk_data(chunks_df: pd.DataFrame) -> tuple:
    """Prepare chunk data for embedding."""
    # Remove duplicates
    original_len = len(chunks_df)
    chunks_df = chunks_df.drop_duplicates(subset=['chunk_id'], keep='first')
    deduped_len = len(chunks_df)
    
    if original_len != deduped_len:
        print(f"⚠️ Removed {original_len - deduped_len} duplicate chunk IDs")
    
    texts = chunks_df['text'].tolist()
    
    # Create unique IDs
    chunk_ids = []
    metadatas = []
    
    for idx, row in chunks_df.iterrows():
        base_id = str(row['chunk_id'])
        unique_id = f"{base_id}_{idx}"
        chunk_ids.append(unique_id)
        
        metadata = {
            'period': str(row['period']),
            'document_id': str(row['document_id']),
            'chunk_index': int(row['chunk_index']),
            'word_count': int(row['word_count']),
            'char_count': int(row['char_count']),
            'year': str(row.get('year', '')),
            'genre': str(row.get('genre', '')),
            'filename': str(row.get('filename', '')),
            'original_chunk_id': str(row['chunk_id'])
        }
        metadatas.append(metadata)
    
    return texts, chunk_ids, metadatas


def print_test_results(search_results: List[Dict], qa_result: Dict, evolution: Dict):
    """Print test results."""
    print("\n1. Semantic Search Test:")
    for i, result in enumerate(search_results, 1):
        print(f"   {i}. Period: {result['metadata']['period']}")
        print(f"      Text: {result['text'][:100]}...")
    
    print("\n2. Question Answering Test:")
    print(f"   Question: {qa_result['question']}")
    print(f"   Answer: {qa_result['answer'][:200]}...")
    
    print("\n3. Language Evolution Test:")
    print(f"   Analyzing word: {evolution['word']}")
    for period, data in evolution['periods'].items():
        print(f"   {period}: {data['context_count']} contexts found")