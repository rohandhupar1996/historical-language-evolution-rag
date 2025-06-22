# ==========================================
# FILE: rag_system/embeddings.py
# ==========================================
"""Embedding management utilities."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .config import EMBEDDING_MODEL


class EmbeddingManager:
    """Manages text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        print(f"🔄 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Embedding model loaded")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.model.encode([text])[0]
