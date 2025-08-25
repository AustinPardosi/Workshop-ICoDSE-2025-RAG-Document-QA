# CV Screening RAG Chatbot

A hands-on project that builds a Retrieval-Augmented Generation (RAG) chatbot over a collection of resumes (CVs).  
Instead of using external datasets, you **upload your own PDF resumes** to build the searchable index.

**What you get:**
- A step-by-step Jupyter notebook
- A Streamlit chatbot app
- Dockerfile and requirements
- Simple local FAISS vector index
- OpenAI LLM + embeddings for retrieval-augmented answers

---

## 1) Prereqs

- Python 3.10+ (or just use Docker)
- An OpenAI API key and network access
- (Optional) GPUs not required

> ⚠️ **Privacy tip**: The demo sends query text and retrieved chunks to OpenAI. Avoid using real PII during demos unless you have permission.

---

## 2) Quickstart (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then edit OPENAI_API_KEY
```

### Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### Run the notebook
```bash
jupyter lab  # or jupyter notebook
# open notebooks/CV_Screening_RAG_Chatbot.ipynb
```

---

## 3) Quickstart with Docker

Build the image:
```bash
docker build -t cv-screening-rag .
```

### Run Streamlit
```bash
docker run --rm -p 8501:8501 --env-file .env cv-screening-rag
```

### Run Jupyter Lab
```bash
docker run --rm -p 8888:8888 --env-file .env cv-screening-rag jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```
Then run http://localhost:8501 to run the streamlit app, and run the printed URL in terminal to run the jupyter lab.

---

## 4) Dataset

You simply upload your own PDF resumes through the Streamlit interface, and the app:
- Extracts text from PDFs
- Chunks text into passages
- Embeds them with OpenAI embeddings
- Stores them in a FAISS index for semantic search

---

## 5) Project Structure

```
.
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── CV_Screening_RAG_Chatbot.ipynb
├── rag/
│   └── rag_core.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```
---

## 6) Notes

- This repo uses a **local FAISS** index for simplicity.
- The app supports **PDF upload**;
- Feel free to replace FAISS with a managed vector DB (Chroma, PgVector, Pinecone) for production.
