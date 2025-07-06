# ==========================================
# FILE: start_rag_service.sh
# ==========================================
# Startup script for RAG background service

echo "🚀 Starting GerManC RAG Background Service..."
echo "============================================="

# Check if vector database exists
if [ ! -d "./german_corpus_vectordb" ]; then
    echo "❌ Vector database not found!"
    echo "📝 Please run first: python rag.py --test --limit 1000"
    echo "💡 This will create the vector database needed for AI search"
    exit 1
fi

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running!"
    echo "📝 Please start PostgreSQL first:"
    echo "   macOS: brew services start postgresql"
    echo "   Linux: sudo systemctl start postgresql"
    exit 1
fi

# Check if database exists
if ! psql -h localhost -p 5432 -U rohan -d germanc_corpus -c "\q" > /dev/null 2>&1; then
    echo "❌ Database 'germanc_corpus' not found!"
    echo "📝 Please run database setup first: python access.py data/prepared"
    exit 1
fi

echo "✅ Prerequisites check passed!"
echo ""
echo "🔄 Starting RAG service on port 8001..."
echo "💡 Keep this terminal open - the service needs to run continuously"
echo "🌐 Service will be available at: http://127.0.0.1:8001"
echo "📖 API docs will be at: http://127.0.0.1:8001/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo "============================================="

# Start the service
cd "$(dirname "$0")"
python -m src.rag_service.rag_server
