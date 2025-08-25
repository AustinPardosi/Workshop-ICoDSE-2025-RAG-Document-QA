import os
import io
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.rag_core import embed_texts, chunk_text, build_faiss_index, answer_with_rag, DocChunk, get_openai_client

load_dotenv()

st.set_page_config(page_title="CV Screening RAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 CV Screening RAG Chatbot")

# Sidebar config
with st.sidebar:
    uploaded_pdfs = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)
    build_button = st.button("(Re)build Index")
    st.markdown("---")
    st.header("Settings")

    # Dropdown for Chat Models
    chat_model_options = [
        "gpt-4.1-mini",   # default
        "gpt-4.1",
        "gpt-4.1-turbo",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    chat_model = st.selectbox(
        "OpenAI Chat Model",
        options=chat_model_options,
        index=chat_model_options.index("gpt-4.1-mini")
    )

    # Dropdown for Embedding Models
    embed_model_options = [
        "text-embedding-3-small",   # default
        "text-embedding-3-large"
    ]
    embed_model = st.selectbox(
        "OpenAI Embedding Model",
        options=embed_model_options,
        index=embed_model_options.index("text-embedding-3-small")
    )

    top_k = st.slider("Top-K", 1, 10, 5)


# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.meta = None

def build_index_from_pdfs(files: List[io.BytesIO]):
    rows = []
    for f in files:
        try:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            doc_id = getattr(f, "name", "uploaded.pdf")
            for i, ch in enumerate(chunk_text(text)):
                rows.append({"doc_id": doc_id, "chunk_id": i, "text": ch, "category": "Uploaded"})
        except Exception as e:
            st.warning(f"Failed to parse {getattr(f, 'name', 'PDF')}: {e}")
    if not rows:
        return None, None
    chunk_df = pd.DataFrame(rows)
    vecs = embed_texts(chunk_df["text"].tolist(), show_progress=True)
    index = build_faiss_index(vecs)
    meta = {
        "doc_ids": chunk_df["doc_id"].tolist(),
        "chunk_ids": chunk_df["chunk_id"].tolist(),
        "texts": chunk_df["text"].tolist(),
        "categories": chunk_df["category"].tolist(),
    }
    return index, meta

if build_button:
    with st.spinner("Building index from uploaded PDFs..."):
        try:
            if uploaded_pdfs:
                index, meta = build_index_from_pdfs(uploaded_pdfs)
                st.session_state.index = index
                st.session_state.meta = meta
                st.success("Index ready!")
            else:
                st.warning("Please upload at least one PDF to build the index.")
        except Exception as e:
            st.error(f"Error building index: {e}")

# Chat input
query = st.text_input("Ask a question (e.g., *Who has strong Python + SQL for data engineering?*)")
ask = st.button("Ask")

# Display
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Answer")
    if ask:
        if st.session_state.index is None:
            st.warning("Build the index first from the sidebar.")
        else:
            os.environ["OPENAI_EMBEDDING_MODEL"] = embed_model
            os.environ["OPENAI_CHAT_MODEL"] = chat_model

            client = get_openai_client()
            qvec = client.embeddings.create(model=embed_model, input=[query]).data[0].embedding
            qvec = np.array([qvec], dtype="float32")

            D, I = st.session_state.index.search(qvec, int(top_k))

            retrieved = []
            meta = st.session_state.meta
            for idx in I[0]:
                if idx < 0: 
                    continue
                retrieved.append(DocChunk(
                    doc_id=meta["doc_ids"][idx],
                    chunk_id=meta["chunk_ids"][idx],
                    text=meta["texts"][idx],
                    meta={"category": meta["categories"][idx]},
                ))
            answer = answer_with_rag(query, retrieved)
            st.write(answer)

with col2:
    st.subheader("Retrieved snippets")
    if ask and st.session_state.meta:
        meta = st.session_state.meta
        rows = []
        for rank, idx in enumerate(I[0]):
            if idx < 0: 
                continue
            rows.append({
                "rank": rank+1,
                "doc_id": meta["doc_ids"][idx],
                "category": meta["categories"][idx],
                "snippet": meta["texts"][idx][:400] + ("..." if len(meta["texts"][idx])>400 else ""),
            })
        st.dataframe(pd.DataFrame(rows))
    else:
        st.caption("Upload PDFs, build the index, and ask a question to see retrieved context here.")
