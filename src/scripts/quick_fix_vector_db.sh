#!/bin/bash
# ==========================================
# FILE: quick_fix_vector_db.sh
# ==========================================
# Quick fix to link existing vector database

echo "🔧 Quick Fix: Vector Database Path"
echo "=================================="

# Check if vector database exists in your location
if [ -d "/Users/rohan/Downloads/historical-language-evolution-rag/german_corpus_vectordb" ]; then
    echo "✅ Found vector database at: /Users/rohan/Downloads/historical-language-evolution-rag/german_corpus_vectordb"
    
    # Create symbolic link in current directory
    if [ ! -d "./german_corpus_vectordb" ]; then
        echo "🔗 Creating symbolic link..."
        ln -s "/Users/rohan/Downloads/historical-language-evolution-rag/german_corpus_vectordb" "./german_corpus_vectordb"
        echo "✅ Symbolic link created: ./german_corpus_vectordb"
    else
        echo "✅ ./german_corpus_vectordb already exists"
    fi
    
    echo ""
    echo "🎉 Vector database is now accessible!"
    echo "📝 You can now run: ./start_full_system.sh"
    
else
    echo "❌ Vector database not found at expected location"
    echo "📝 Please run: python rag.py --test --limit 1000"
fi