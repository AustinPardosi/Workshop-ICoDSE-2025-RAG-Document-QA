# JDIH RAG - Sistem Tanya Jawab Dokumen Peraturan Akademik ITB

Sistem RAG (Retrieval-Augmented Generation) untuk melakukan tanya jawab terhadap dokumen peraturan akademik ITB menggunakan teknologi AI.

## 🚀 Fitur Utama

### 🔍 **Enhanced Retrieval System**
- **Query Expansion**: Memperluas query dengan sinonim dan konteks
- **Multiple Chunking Strategies**: Recursive, semantic, paragraph-based chunking
- **Advanced Reranking**: Kombinasi similarity score dan keyword matching
- **Metadata-rich Documents**: Tracking sumber dokumen, keywords, dan summary

### 🤖 **AI-Powered Generation**
- **Bahasa Indonesia Support**: Prompt dan response dalam bahasa Indonesia
- **Multi-model Support**: GPT-4o-mini, GPT-4o, GPT-4-turbo
- **Context-aware Responses**: Jawaban berdasarkan konteks dokumen dengan referensi sumber

### 📊 **Quality Metrics & Evaluation**
- **Coverage Score**: Mengukur seberapa baik query terms tercakup dalam hasil
- **Diversity Score**: Mengukur keberagaman dokumen yang diambil
- **Confidence Score**: Skor kepercayaan berdasarkan similarity dan keyword matching
- **Real-time Evaluation**: Metrics ditampilkan untuk setiap query

### 🏭 **Production-Ready Features**
- **Persistent Storage**: Vector database disimpan dan dapat dimuat ulang
- **Scalable Architecture**: Modular design untuk easy maintenance
- **Error Handling**: Comprehensive error handling dan logging
- **Interactive UI**: Streamlit-based web interface

## 📁 Struktur Proyek

```
JDIH_RAG/
├── app/
│   └── streamlit_app.py      # Main Streamlit application
├── rag/
│   ├── __init__.py
│   └── rag_core.py           # Core RAG functionality
├── data/                     # Directory untuk menyimpan PDF (opsional)
├── Memory/                   # Vector database storage (dibuat otomatis)
├── requirements.txt          # Python dependencies
└── README.md                # Dokumentasi ini
```

## 🛠️ Instalasi

1. **Clone atau download project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup OpenAI API Key:**
   Buat file `.env` di root directory dengan:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Siapkan dokumen PDF:**
   - Letakkan file PDF di folder `../Assets/Data/` (atau folder lain sesuai konfigurasi)
   - Pastikan file PDF dapat dibaca dan tidak ter-password

## 🚀 Cara Menjalankan

1. **Jalankan aplikasi Streamlit:**
   ```bash
   cd app
   streamlit run streamlit_app.py
   ```

2. **Akses aplikasi di browser:**
   ```
   http://localhost:8501
   ```

3. **Build index pertama kali:**
   - Atur path folder PDF di sidebar
   - Pilih model dan parameter yang diinginkan
   - Klik "Build/Rebuild Index"
   - Tunggu proses selesai

4. **Mulai bertanya:**
   - Ketik pertanyaan di kolom input
   - Klik "Tanya" atau tekan Enter
   - Lihat jawaban, metrics, dan dokumen referensi

## ⚙️ Konfigurasi

### Model Settings
- **Embedding Models**: text-embedding-3-small (default), text-embedding-3-large
- **Chat Models**: gpt-4o-mini (default), gpt-4o, gpt-4-turbo, gpt-3.5-turbo

### Chunking Options
- **Strategy**: recursive (default), semantic, paragraph, simple
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Overlap**: 50-500 characters (default: 100)

### Retrieval Settings
- **Top-K**: 1-10 dokumen (default: 5)
- **Query Expansion**: Aktif/nonaktif
- **Reranking**: Aktif/nonaktif

## 💡 Contoh Penggunaan

### Pertanyaan yang Bisa Dijawab:
- "Berapa lama cuti hamil untuk mahasiswa ITB?"
- "Apa saja persyaratan untuk mengambil cuti akademik?"
- "Bagaimana prosedur pengajuan skripsi?"
- "Kapan deadline pembayaran SPP?"
- "Siapa yang bisa mengajukan dispensasi ujian?"

### Tips untuk Pertanyaan yang Efektif:
1. **Spesifik**: Gunakan istilah yang spesifik (cuti hamil vs cuti umum)
2. **Konteks**: Tambahkan konteks seperti "mahasiswa ITB" atau "S1"
3. **Bahasa Indonesia**: System dioptimalkan untuk bahasa Indonesia

## 📊 Understanding the Metrics

- **Coverage Score**: % query terms yang ditemukan dalam dokumen hasil
- **Diversity Score**: Keberagaman dokumen (1.0 = sangat beragam, 0.0 = mirip semua)
- **Confidence Score**: Gabungan similarity dan keyword matching
- **Docs Retrieved**: Jumlah dokumen yang diambil untuk menjawab

## 🔧 Advanced Features

### 1. **Document Metadata Extraction**
- Automatic keyword extraction
- Document summarization
- File metadata (size, creation date, page count)

### 2. **Query Enhancement**
- Synonym expansion untuk bahasa Indonesia
- Context addition berdasarkan domain knowledge
- Query rewriting untuk retrieval yang lebih baik

### 3. **Evaluation & Monitoring**
- Real-time quality metrics
- Chat history tracking
- Document source tracking

## 🛠️ Development

### Extending the System:

1. **Menambah Chunking Strategy Baru:**
   Edit `chunk_text_advanced()` di `rag_core.py`

2. **Menambah Model Baru:**
   Update list di `streamlit_app.py`

3. **Menambah Metrics Baru:**
   Edit `calculate_rag_metrics()` di `rag_core.py`

4. **Custom Prompt Engineering:**
   Modifikasi `answer_with_enhanced_rag()` di `rag_core.py`

## 🚨 Troubleshooting

### Common Issues:

1. **"OPENAI_API_KEY not found"**
   - Pastikan file `.env` ada dan berisi API key yang valid
   - Check API key masih aktif dan memiliki credit

2. **"No PDF files found"**
   - Pastikan path folder PDF benar
   - Check file PDF tidak corrupt atau ter-password

3. **Memory Error saat build index**
   - Kurangi chunk_size atau jumlah dokumen
   - Gunakan embedding model yang lebih kecil (text-embedding-3-small)

4. **Jawaban tidak relevan**
   - Aktifkan query expansion dan reranking
   - Coba dengan pertanyaan yang lebih spesifik
   - Check apakah dokumen benar-benar berisi informasi yang dicari

## 📞 Support

Untuk pertanyaan teknis atau bug report, silakan buat issue di repository atau hubungi tim pengembang.

---

**🎯 Dikembangkan untuk Workshop ICoDSE 2025 - Advanced RAG Implementation**