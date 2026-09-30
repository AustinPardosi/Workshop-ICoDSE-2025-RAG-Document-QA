# JDIH RAG: Regulation Q&A for ITB

This is a Streamlit app and teaching notebook that answers questions about Institut Teknologi Bandung regulations in Bahasa Indonesia. Every answer is grounded in the source documents and names the ones it used. JDIH stands for *Jaringan Dokumentasi dan Informasi Hukum*, the legal documentation network.

## Pipeline

All of the logic lives in [`rag/rag_core.py`](rag/rag_core.py). The app is a UI on top of it.

| Step | Function | What it does |
|---|---|---|
| Load | `load_documents_with_metadata` | Reads each PDF with `PyPDFLoader` and attaches file metadata, frequency-based keywords and a short extractive summary. |
| Chunk | `chunk_text_advanced` | Offers four strategies: `recursive` (LangChain `RecursiveCharacterTextSplitter`, the default), `semantic` (packs whole sentences up to the chunk size), `paragraph` and `simple` (a sliding window of words). Sizes are in characters. |
| Index | `build_enhanced_vector_store` | Embeds chunks with OpenAI and stores them in a FAISS inner-product index, which gives cosine similarity. The index is saved to `Memory/`. |
| Retrieve | `enhanced_retrieval` | Optionally expands the query (Indonesian synonyms plus question-pattern rewriting), fetches 2·k candidates, reranks them by `0.7·cosine + 0.3·keyword overlap` and keeps the top k. |
| Generate | `answer_with_enhanced_rag` | Sends an Indonesian system prompt with context blocks labelled by source file and relevance, at temperature 0.1. The model is told to say so when the answer is not in the documents. |
| Inspect | `calculate_rag_metrics` | Calculates the coverage, diversity and confidence signals that the app shows next to each answer. |

## Run locally

You need Python 3.10 or 3.11. The pinned `numpy` and `faiss-cpu` versions have no wheels for 3.12+.

```bash
cd JDIH_RAG
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then set OPENAI_API_KEY
streamlit run app/streamlit_app.py
```

Start the app from `JDIH_RAG/` and not from `app/`. The data path (`data`) and the index folder (`Memory/`) are relative to the directory you run it from.

## Run with Docker

```bash
cd JDIH_RAG
cp .env.example .env                                  # then set OPENAI_API_KEY
docker compose up --build                             # http://localhost:8501
```

`run-docker.sh` (macOS/Linux) and `run-docker.bat` (Windows) do the same thing after checking that `.env` and `data/` exist. The index is written to `./Memory` through a volume, so it survives container restarts.

## Using the app

1. In the sidebar, keep the data path as `data` and choose the models and chunking settings.
2. Click **Build/Rebuild Index**.
3. Ask a question, or click one of the example questions.

The saved index is reloaded on restart. Rebuild it whenever you change the embedding model or chunking settings.

| Setting | Options | Default |
|---|---|---|
| Embedding model | `text-embedding-3-small`, `text-embedding-3-large` | small |
| Chat model | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` | `gpt-4o-mini` |
| Chunking strategy | recursive, semantic, paragraph, simple | recursive |
| Chunk size / overlap (characters) | 500–2000 / 50–500 | 1000 / 100 |
| Top-k | 1–10 | 5 |
| Query expansion, reranking | on / off | on |

### Example questions

These are all answerable from the bundled corpus:

- Apa saja tujuan umum dan khusus dari kegiatan MAPAK di ITB?
- Apa saja hal yang dilarang keras dilakukan panitia selama MAPAK?
- Apa sanksi bagi tenan makanan yang melanggar ketentuan K3L saat event di kampus?
- Bagaimana penyesuaian kegiatan pembelajaran saat libur Nyepi dan Idul Fitri 2025?
- Apa saja bentuk organisasi kemahasiswaan yang diakui di ITB?

### What the metrics mean

| Metric | Definition |
|---|---|
| Coverage | The share of question words that appear in the retrieved chunks |
| Diversity | 1 − the mean Jaccard word overlap between retrieved chunks. A higher value means less redundant context. |
| Confidence | The mean rerank score (`0.7·cosine + 0.3·keyword overlap`) |

These are lexical signals for exploring how retrieval behaves. They do not measure answer quality.

## Extending

| To change… | Edit |
|---|---|
| The corpus | Add PDFs to `data/`, then rebuild the index |
| Chunking | `chunk_text_advanced` |
| Synonyms and query rewriting | `expand_query`, `rewrite_query_for_retrieval` |
| Reranking weights | `enhanced_retrieval` |
| The prompt | `answer_with_enhanced_rag` |
| The metrics | `calculate_rag_metrics` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY tidak ditemukan` | Create `.env` in `JDIH_RAG/`, the folder you run `streamlit` from. |
| `pip install` fails while building `numpy` | You are on Python 3.12 or newer. Use 3.10/3.11, or use Docker. |
| `Path data tidak ditemukan` | Run `streamlit` from `JDIH_RAG/`, not from `app/`. |
| Errors after switching the embedding model | The saved index has a different dimension. Click **Build/Rebuild Index**, or delete `Memory/`. |
| An answer says the information was not found | Check that the corpus actually covers the topic, then try a more specific question or a higher top-k. |

## Layout

```
JDIH_RAG/
├── app/streamlit_app.py        # Streamlit UI
├── rag/rag_core.py             # pipeline (see table above)
├── notebooks/workshop_jdih.ipynb
├── data/                       # ITB regulation PDFs
├── Memory/                     # saved FAISS index (generated, git-ignored)
├── Dockerfile, docker-compose.yml, run-docker.sh, run-docker.bat
├── requirements.txt
└── .env.example
```
