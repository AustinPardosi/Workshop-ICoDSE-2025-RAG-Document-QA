@echo off
REM Script untuk memperbaiki dependency issues
echo 🔧 Fixing OpenAI Dependencies Issues...

REM Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Clean Docker build cache
echo 🧹 Cleaning Docker build cache...
docker system prune -f

REM Rebuild with no cache
echo 🔨 Rebuilding Docker image with updated dependencies...
docker-compose build --no-cache

REM Start containers
echo 🚀 Starting containers...
docker-compose up -d

echo ✅ Dependencies fixed!
echo 🌐 Access app at: http://localhost:8501
echo 🔍 Check logs: docker-compose logs -f jdih-rag
pause
