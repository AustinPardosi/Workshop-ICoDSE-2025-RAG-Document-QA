# CV Screening RAG

This is a RAG assistant for HR screening. You upload PDF resumes and ask questions in natural language, such as *"Who has strong Python + SQL for data engineering?"*, and get answers grounded in the retrieved CV snippets.

It is the framework-free counterpart to [JDIH RAG](../JDIH_RAG/). The whole pipeline uses only the OpenAI SDK and FAISS, with no LangChain, so every step is visible:

| Step | Where | What it does |
|---|---|---|
| Extract | `app/streamlit_app.py` | Extracts text from the uploaded PDFs with `pypdf` |
| Chunk | `rag_core.chunk_text` | Word window of about 400 tokens with a 60-token overlap (roughly 4 characters per token) |
| Embed and index | `rag_core.embed_texts`, `build_faiss_index` | Batches OpenAI embeddings into a FAISS inner-product index (cosine similarity) |
| Retrieve and answer | `rag_core.answer_with_rag` | Takes the top-k snippets and sends them to the chat model with an HR-assistant prompt |

The index is kept in the Streamlit session and is not saved to disk. It is rebuilt each time you click **(Re)build Index**.

> **Privacy:** CV text and your questions are sent to the OpenAI API. Only use resumes you have permission to process. `uploads/` is git-ignored so that CVs are never committed.

## Run locally

You need Python 3.10+.

```bash
cd cv_screening_rag
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then set OPENAI_API_KEY
streamlit run app/streamlit_app.py                    # http://localhost:8501
```

1. Upload one or more PDF resumes in the sidebar.
2. Click **(Re)build Index**.
3. Ask a question. The answer appears on the left and the retrieved snippets on the right.

If you change the embedding model after building, the app resets the index, because the vector dimensions differ between models.

## Notebook

```bash
jupyter lab notebooks/CV_Screening_RAG.ipynb
```

The notebook reads PDFs from `notebooks/uploads/`. Put a few resumes there before running it.

## Run with Docker

```bash
cd cv_screening_rag
cp .env.example .env                                  # then set OPENAI_API_KEY
docker compose up --build                             # http://localhost:8501
```

To run Jupyter Lab from the same image instead:

```bash
docker build -t cv-screening-rag .
docker run --rm -p 8888:8888 --env-file .env cv-screening-rag \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

## Layout

```
cv_screening_rag/
├── app/streamlit_app.py            # upload, index, ask
├── rag/rag_core.py                 # chunking, embeddings, FAISS, prompting
├── notebooks/CV_Screening_RAG.ipynb
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── .env.example
```

## Next steps

- Add structured filters such as years of experience or location alongside semantic search.
- Improve PDF parsing for multi-column layouts, tables and scanned CVs (OCR).
- Add bias and privacy guardrails before using this on real hiring decisions.
