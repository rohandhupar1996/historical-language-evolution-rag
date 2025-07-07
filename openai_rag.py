# ==========================================
# FILE: openai_rag.py (main execution script)
# ==========================================

import argparse
import os
from pathlib import Path
from src.openai_rag_system import OpenAIGermanRAGPipeline
from src.openai_rag_system.config import DEFAULT_DB_CONFIG

def main():
    parser = argparse.ArgumentParser(description="OpenAI RAG System for German Historical Corpus")
    parser.add_argument("--setup", action="store_true", help="Setup embeddings and QA system")
    parser.add_argument("--service", action="store_true", help="Start RAG service")
    parser.add_argument("--test", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--limit", type=int, help="Limit chunks for testing")
    parser.add_argument("--vector-db", default="./openai_vector_db", help="Vector DB path")
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    print("🤖 OpenAI RAG System for German Historical Corpus")
    print("=" * 50)
    
    # Initialize pipeline
    rag = OpenAIGermanRAGPipeline(DEFAULT_DB_CONFIG, args.vector_db)
    
    if args.setup:
        print("📚 Setting up OpenAI embeddings...")
        chunks_df = rag.load_and_embed_corpus(limit=args.limit)
        rag.setup_advanced_qa_system()
        print("✅ Setup complete!")
    
    if args.test:
        print("🧪 Running comprehensive tests...")
        results = rag.run_comprehensive_tests()
        print(f"🎯 Test results: {results}")
    
    if args.service:
        print("🚀 Starting OpenAI RAG service...")
        from src.openai_rag_system.openai_rag_service import run_openai_rag_service
        run_openai_rag_service()
    
    return 0

if __name__ == "__main__":
    exit(main())