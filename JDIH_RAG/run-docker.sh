#!/bin/bash

# Script untuk menjalankan JDIH RAG dengan Docker
# Usage: ./run-docker.sh

echo "🚀 Starting JDIH RAG Docker Setup..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  File .env tidak ditemukan!"
    echo "📝 Membuat file .env template..."
    cat > .env << EOF
# OpenAI API Key - GANTI DENGAN API KEY ASLI ANDA
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Uncomment untuk debugging
# TOKENIZERS_PARALLELISM=false
EOF
    echo "✅ File .env dibuat. Silakan edit dan masukkan OpenAI API key Anda."
    echo "📝 Edit file .env dengan: nano .env"
    exit 1
fi

# Check if data folder exists and has PDF files
if [ ! -d "data" ]; then
    echo "❌ Folder data/ tidak ditemukan!"
    echo "📁 Membuat folder data..."
    mkdir -p data
    echo "📋 Silakan copy file PDF Anda ke folder data/"
    exit 1
fi

PDF_COUNT=$(find data -name "*.pdf" | wc -l)
if [ $PDF_COUNT -eq 0 ]; then
    echo "❌ Tidak ada file PDF di folder data/"
    echo "📋 Silakan copy file PDF Anda ke folder data/"
    exit 1
fi

echo "📄 Ditemukan $PDF_COUNT file PDF di folder data/"

# Build and run with docker-compose
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting containers..."
docker-compose up -d

echo "✅ JDIH RAG sudah berjalan!"
echo "🌐 Akses aplikasi di: http://localhost:8501"
echo ""
echo "📋 Useful commands:"
echo "  - Stop containers: docker-compose down"
echo "  - View logs: docker-compose logs -f"
echo "  - Restart: docker-compose restart"
echo ""
echo "🔍 Untuk debug, gunakan: docker-compose logs -f jdih-rag"
