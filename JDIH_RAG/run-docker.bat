@echo off
REM Script untuk menjalankan JDIH RAG dengan Docker di Windows
REM Usage: run-docker.bat

echo 🚀 Starting JDIH RAG Docker Setup...

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  File .env tidak ditemukan!
    echo 📝 Membuat file .env template...
    (
        echo # OpenAI API Key - GANTI DENGAN API KEY ASLI ANDA
        echo OPENAI_API_KEY=sk-your-openai-api-key-here
        echo.
        echo # Optional: Uncomment untuk debugging
        echo # TOKENIZERS_PARALLELISM=false
    ) > .env
    echo ✅ File .env dibuat. Silakan edit dan masukkan OpenAI API key Anda.
    echo 📝 Edit file .env dengan notepad atau editor lainnya
    pause
    exit /b 1
)

REM Check if data folder exists and has PDF files
if not exist "data" (
    echo ❌ Folder data\ tidak ditemukan!
    echo 📁 Membuat folder data...
    mkdir data
    echo 📋 Silakan copy file PDF Anda ke folder data\
    pause
    exit /b 1
)

REM Count PDF files (Windows way)
set PDF_COUNT=0
for %%f in (data\*.pdf) do (
    set /a PDF_COUNT+=1
)

if %PDF_COUNT%==0 (
    echo ❌ Tidak ada file PDF di folder data\
    echo 📋 Silakan copy file PDF Anda ke folder data\
    pause
    exit /b 1
)

echo 📄 Ditemukan %PDF_COUNT% file PDF di folder data\

REM Build and run with docker-compose
echo 🔨 Building Docker image...
docker-compose build

echo 🚀 Starting containers...
docker-compose up -d

echo ✅ JDIH RAG sudah berjalan!
echo 🌐 Akses aplikasi di: http://localhost:8501
echo.
echo 📋 Useful commands:
echo   - Stop containers: docker-compose down
echo   - View logs: docker-compose logs -f
echo   - Restart: docker-compose restart
echo.
echo 🔍 Untuk debug, gunakan: docker-compose logs -f jdih-rag
pause
