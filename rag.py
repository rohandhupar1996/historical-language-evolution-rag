# ==========================================
# FILE: rag.py
# ==========================================
"""Main entry point for RAG pipeline."""

import argparse
from pathlib import Path
from src.rag_system import GermanRAGPipeline
from src.rag_system.config import DEFAULT_DB_CONFIG


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Setup RAG pipeline for German corpus")
    parser.add_argument("--vector-db-path", default="./german_corpus_vectordb", help="Vector database path")
    parser.add_argument("--limit", type=int, help="Limit number of chunks for testing")
    parser.add_argument("--llm-provider", choices=['simple', 'openai', 'huggingface'], 
                       default='simple', help="LLM provider")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    
    args = parser.parse_args()
    
    print("🗃️ Phase 5: RAG Pipeline for German Historical Corpus")
    print("=" * 60)
    
    # Initialize pipeline
    rag = GermanRAGPipeline(DEFAULT_DB_CONFIG, args.vector_db_path)
    
    try:
        # Load and embed chunks
        chunks_df = rag.load_chunks_from_db(limit=args.limit)
        rag.create_embeddings(chunks_df)
        
        # Setup QA system
        rag.setup_qa_system(args.llm_provider)
        
        # Run tests if requested
        if args.test:
            rag.run_tests()
        
        print("\n✅ Phase 5: RAG Pipeline completed successfully!")
        print("\n🎯 Available capabilities:")
        print("   - Semantic search across historical German texts")
        print("   - Question answering about language evolution")  
        print("   - Period-specific analysis")
        print("   - Language change tracking")
        
        return rag
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return None


if __name__ == "__main__":
    rag_system = main()
