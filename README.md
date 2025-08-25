# Workshop ICoDSE 2025 - RAG Document Q&A

Repository untuk workshop **"Studi Kasus: Document Q&A"** dalam acara **ICoDSE 2025**.

## 📋 Deskripsi

Workshop ini membahas implementasi sistem RAG (Retrieval-Augmented Generation) untuk melakukan tanya jawab terhadap dokumen. Terdapat dua implementasi utama:

1. **CV Screening RAG** - Sistem untuk screening CV/resume
2. **JDIH RAG** - Sistem untuk tanya jawab dokumen peraturan akademik ITB

## 🏗️ Struktur Proyek

```
Workshop-ICoDSE-2025-RAG-Document-QA/
├── Assets/
│   └── Data/                     # Dokumen PDF untuk JDIH RAG
│       ├── Peraturan Kemahasiswaan ITB.pdf
│       ├── doc (12).pdf
│       ├── doc (13).pdf
│       └── doc (8).pdf
├── cv_screening_rag/             # Implementasi CV Screening
│   ├── app/
│   │   └── streamlit_app.py
│   ├── rag/
│   │   └── rag_core.py
│   ├── notebooks/
│   │   └── CV_Screening_RAG.ipynb
│   ├── requirements.txt
│   └── README.md
├── JDIH_RAG/                     # Implementasi JDIH (Enhanced)
│   ├── app/
│   │   └── streamlit_app.py      # Streamlit app dengan fitur industri
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_core.py           # Enhanced RAG core dengan metadata
│   ├── requirements.txt
│   ├── Dockerfile                # Container deployment
│   └── README.md
├── notebooks/
│   └── workshop.ipynb            # Jupyter notebook workshop
├── requirements.txt              # Global requirements
└── README.md                     # Dokumentasi utama
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd Workshop-ICoDSE-2025-RAG-Document-QA

# Install dependencies
pip install -r requirements.txt

# Setup OpenAI API Key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2. Jalankan JDIH RAG (Recommended)

```bash
cd JDIH_RAG
streamlit run app/streamlit_app.py
```

Buka browser di `http://localhost:8501`

### 3. Atau Jalankan CV Screening RAG

```bash
cd cv_screening_rag
streamlit run app/streamlit_app.py
```

### 4. Atau Gunakan Jupyter Notebook

```bash
jupyter notebook notebooks/workshop.ipynb
```

## 🎯 Fitur Utama

### JDIH RAG (Enhanced Implementation)
- ✅ **Multi-strategy Chunking** (recursive, semantic, paragraph)
- ✅ **Query Expansion & Rewriting** dengan bahasa Indonesia
- ✅ **Advanced Reranking** (similarity + keyword matching)
- ✅ **Rich Metadata Extraction** (keywords, summary, file info)
- ✅ **Real-time Quality Metrics** (coverage, diversity, confidence)
- ✅ **Indonesian Language Support** untuk prompt dan response
- ✅ **Persistent Vector Storage** dengan metadata lengkap
- ✅ **Production-ready UI** dengan Streamlit
- ✅ **Docker Support** untuk deployment

### CV Screening RAG (Baseline Implementation)
- ✅ Basic PDF processing dan chunking
- ✅ FAISS vector storage
- ✅ Simple retrieval dan generation
- ✅ Multi-model support (GPT-4o, GPT-3.5, dll)

## 📊 Perbandingan Fitur

| Feature | CV Screening RAG | JDIH RAG | Workshop Notebook |
|---------|------------------|----------|-------------------|
| **Basic RAG Pipeline** | ✅ | ✅ | ✅ |
| **Multiple Models** | ✅ | ✅ | ✅ |
| **Indonesian Language** | ❌ | ✅ | Partial |
| **Advanced Chunking** | ❌ | ✅ | Basic |
| **Query Enhancement** | ❌ | ✅ | ❌ |
| **Metadata Extraction** | ❌ | ✅ | ❌ |
| **Quality Metrics** | ❌ | ✅ | ❌ |
| **Reranking** | ❌ | ✅ | ❌ |
| **Persistent Storage** | ❌ | ✅ | Basic |
| **Production UI** | Basic | ✅ | ❌ |
| **Docker Support** | ✅ | ✅ | ❌ |

## 🛠️ Setup Development

### Prerequisites
- Python 3.10+
- OpenAI API Key dengan credit yang cukup
- Git

### Environment Setup

1. **Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   ```bash
   # .env file
   OPENAI_API_KEY=your-api-key-here
   ```

## 📚 Workshop Materials

### 1. Jupyter Notebook (`notebooks/workshop.ipynb`)
- Implementasi step-by-step RAG pipeline
- Penjelasan konsep indexing dan generation
- Hands-on coding exercises

### 2. Streamlit Applications
- **JDIH RAG**: Advanced implementation dengan fitur industri
- **CV Screening RAG**: Baseline implementation untuk perbandingan

### 3. Documentation
- Comprehensive README untuk setiap modul
- Code comments dalam bahasa Indonesia
- Best practices dan troubleshooting guide

## 🎓 Learning Objectives

Setelah mengikuti workshop ini, peserta akan memahami:

1. **Konsep Dasar RAG**
   - Indexing pipeline (loading, chunking, embedding, storage)
   - Generation pipeline (retrieval, augmentation, generation)

2. **Advanced RAG Techniques**
   - Multiple chunking strategies
   - Query expansion dan rewriting
   - Reranking dan filtering
   - Metadata extraction dan utilization

3. **Production Considerations**
   - Quality metrics dan evaluation
   - Error handling dan logging
   - Scalability dan performance
   - User interface design

4. **Industry Best Practices**
   - Modular architecture
   - Configuration management
   - Testing dan validation
   - Deployment strategies

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```bash
   # Check API key
   echo $OPENAI_API_KEY
   # atau test di Python
   python -c "import openai; print(openai.api_key)"
   ```

2. **Package Installation Issues**
   ```bash
   # Update pip
   pip install --upgrade pip
   # Install with verbose output
   pip install -r requirements.txt -v
   ```

3. **PDF Loading Issues**
   - Pastikan PDF tidak ter-password
   - Check file permissions
   - Coba dengan PDF yang lebih kecil dulu

4. **Memory Issues**
   - Kurangi chunk size atau jumlah dokumen
   - Gunakan embedding model yang lebih kecil
   - Monitoring memory usage

## 🤝 Contributing

Untuk kontribusi atau perbaikan:
1. Fork repository
2. Create feature branch
3. Commit changes dengan deskripsi yang jelas
4. Submit pull request

## 📞 Support

- **Workshop Support**: Hubungi instruktur atau TA
- **Technical Issues**: Create issue di repository
- **Documentation**: Check README di setiap folder modul

## 📄 License

Project ini dibuat untuk keperluan edukasi workshop ICoDSE 2025.

---

**🎯 Selamat belajar dan semoga workshop ini bermanfaat!**

*Dikembangkan dengan ❤️ untuk ICoDSE 2025 Workshop*