echo "🚀 Starting GerManC RAG Background Service..."
echo "============================================="

# Define possible vector database locations
VECTOR_DB_PATHS=(
    "./german_corpus_vectordb"
    "/Users/rohan/Downloads/historical-language-evolution-rag/german_corpus_vectordb"
    "../german_corpus_vectordb"
    "~/Downloads/historical-language-evolution-rag/german_corpus_vectordb"
)

echo "🔍 Looking for vector database..."
VECTOR_DB_FOUND=false
FOUND_PATH=""

for path in "${VECTOR_DB_PATHS[@]}"; do
    # Expand tilde
    expanded_path="${path/#\~/$HOME}"
    if [ -d "$expanded_path" ]; then
        echo "✅ Vector database found at: $expanded_path"
        VECTOR_DB_FOUND=true
        FOUND_PATH="$expanded_path"
        break
    else
        echo "   ❌ Not found: $expanded_path"
    fi
done

if [ "$VECTOR_DB_FOUND" = false ]; then
    echo ""
    echo "❌ Vector database not found in any expected location!"
    echo "📝 Please either:"
    echo "   1. Run: python rag.py --test --limit 1000"
    echo "   2. Or create a symlink: ln -s /path/to/your/vector/db ./german_corpus_vectordb"
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

echo "✅ All prerequisites check passed!"
echo ""
echo "🔄 Starting RAG service on port 8001..."
echo "💡 Keep this terminal open - the service needs to run continuously"
echo "🌐 Service will be available at: http://127.0.0.1:8001"
echo "📖 API docs will be at: http://127.0.0.1:8001/docs"
echo "📦 Using vector database: $FOUND_PATH"
echo ""
echo "Press Ctrl+C to stop the service"
echo "============================================="

# Set environment variable for the RAG service to use
export VECTOR_DB_PATH="$FOUND_PATH"

# Start the service
cd "$(dirname "$0")"
python -m src.rag_service.rag_server