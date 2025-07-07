# ==========================================
# FILE: src/openai_rag_system/utils.py
# ==========================================

import os
import logging
import time
import tiktoken
from typing import List, Dict, Any
import re

def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def validate_openai_setup():
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    try:
        import openai
        openai.models.list()
        print("✅ OpenAI API connection successful")
    except Exception as e:
        raise ConnectionError(f"OpenAI API connection failed: {e}")

def estimate_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    try:
        return len(tokenizer.encode(text))
    except:
        return len(text) // 4

def validate_text_input(text: str) -> bool:
    return isinstance(text, str) and 50 <= len(text) <= 8000 and text.strip()

def handle_rate_limits(api_calls: int, tokens: int):
    if api_calls % 50 == 0:
        time.sleep(0.1)
    if tokens > 200000:  # 80% of 250k limit
        time.sleep(1)