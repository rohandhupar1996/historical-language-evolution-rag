# ==========================================
# FILE: rag_system/vector_store.py
# ==========================================
"""Vector store management using ChromaDB."""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from .config import COLLECTION_NAME, DEFAULT_BATCH_SIZE


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self, db_path: str, collection_name: str = COLLECTION_NAME):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = None
    
    def create_or_get_collection(self):
        """Create or get existing collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"📦 Found existing collection with {self.collection.count()} documents")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "German historical corpus text chunks"}
            )
            print("📦 Created new ChromaDB collection")
        
        return self.collection
    
    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], 
                      metadatas: List[Dict], ids: List[str], batch_size: int = DEFAULT_BATCH_SIZE):
        """Add embeddings to collection in batches."""
        total = len(texts)
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            print(f"Added batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
    
    def query(self, query_text: str, k: int = 5, where_filter: Optional[Dict] = None) -> Dict:
        """Query the vector store."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where_filter
        )
