import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_core import (
    load_documents_with_metadata, 
    build_enhanced_vector_store,
    enhanced_retrieval,
    answer_with_enhanced_rag,
    calculate_rag_metrics,
    save_vector_store_with_metadata,
    load_vector_store_with_metadata
)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="JDIH RAG - Sistem Tanya Jawab Dokumen Hukum ITB", 
    page_icon="⚖️", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 2rem;
}

.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #FF6B35;
    margin: 0.5rem 0;
}

.doc-source {
    background: #e8f4fd;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    border-left: 3px solid #0066cc;
}

.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #ffc107; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ JDIH RAG</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Sistem Tanya Jawab Dokumen Peraturan Akademik ITB</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    
    # Data source
    st.subheader("📁 Sumber Data")
    data_path = st.text_input("Path folder data PDF:", value="../Assets/Data")
    
    # Model settings
    st.subheader("🤖 Pengaturan Model")
    
    embedding_models = [
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]
    embedding_model = st.selectbox(
        "Model Embedding:",
        options=embedding_models,
        index=0
    )
    
    chat_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    chat_model = st.selectbox(
        "Model Chat:",
        options=chat_models,
        index=0
    )
    
    # Chunking settings
    st.subheader("📝 Pengaturan Chunking")
    
    chunking_strategy = st.selectbox(
        "Strategi Chunking:",
        options=["recursive", "semantic", "paragraph", "simple"],
        index=0
    )
    
    chunk_size = st.slider("Ukuran Chunk:", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Overlap Chunk:", 50, 500, 100, 25)
    
    # Retrieval settings
    st.subheader("🔍 Pengaturan Retrieval")
    top_k = st.slider("Jumlah dokumen yang diambil:", 1, 10, 5)
    
    enable_query_expansion = st.checkbox("Aktifkan Query Expansion", value=True)
    enable_reranking = st.checkbox("Aktifkan Reranking", value=True)
    
    # Build index button
    st.markdown("---")
    build_index = st.button("🔨 Build/Rebuild Index", type="primary")
    
    # Show system info
    st.subheader("📊 Informasi Sistem")
    if "vector_store" in st.session_state:
        st.success("✅ Index sudah siap")
        if "index_metadata" in st.session_state:
            metadata = st.session_state.index_metadata
            st.info(f"📄 {len(metadata)} chunks tersimpan")
    else:
        st.warning("⚠️ Index belum dibuild")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.index_metadata = None
    st.session_state.chat_history = []

# Build Index
if build_index:
    with st.spinner("🔄 Membangun index dari dokumen PDF..."):
        try:
            # Check if data path exists
            full_data_path = os.path.abspath(data_path)
            if not os.path.exists(full_data_path):
                st.error(f"❌ Path data tidak ditemukan: {full_data_path}")
            else:
                # Load documents
                st.info("📖 Memuat dokumen...")
                documents = load_documents_with_metadata(full_data_path)
                
                if not documents:
                    st.error("❌ Tidak ada dokumen PDF ditemukan di folder tersebut")
                else:
                    # Build vector store
                    st.info(f"🏗️ Membangun vector store dari {len(documents)} dokumen...")
                    vector_store, chunks = build_enhanced_vector_store(
                        documents,
                        embedding_model=embedding_model,
                        chunking_strategy=chunking_strategy,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.index_metadata = [
                        {
                            "doc_id": chunk.doc_id,
                            "chunk_id": chunk.chunk_id,
                            "filename": chunk.metadata.filename,
                            "keywords": chunk.metadata.keywords,
                            "summary": chunk.metadata.summary
                        }
                        for chunk in chunks
                    ]
                    
                    # Save to disk
                    st.info("💾 Menyimpan index...")
                    save_path = "./Memory"
                    os.makedirs(save_path, exist_ok=True)
                    save_vector_store_with_metadata(vector_store, chunks, save_path, "JDIH_index")
                    
                    st.success(f"✅ Index berhasil dibangun! ({len(chunks)} chunks dari {len(documents)} dokumen)")
                    
                    # Show document summary
                    with st.expander("📋 Ringkasan Dokumen"):
                        doc_summary = []
                        for text, metadata in documents:
                            doc_summary.append({
                                "Nama File": metadata.filename,
                                "Ukuran": f"{metadata.file_size/1024:.1f} KB",
                                "Halaman": metadata.page_count,
                                "Keywords": ", ".join(metadata.keywords[:5]) if metadata.keywords else "Tidak ada",
                                "Ringkasan": metadata.summary[:100] + "..." if len(metadata.summary) > 100 else metadata.summary
                            })
                        
                        st.dataframe(pd.DataFrame(doc_summary), use_container_width=True)
                        
        except Exception as e:
            st.error(f"❌ Error saat membangun index: {str(e)}")
            st.exception(e)

# Try to load existing index if not in session
if st.session_state.vector_store is None:
    try:
        if os.path.exists("./Memory/JDIH_index.faiss"):
            with st.spinner("📂 Memuat index yang tersimpan..."):
                embeddings = OpenAIEmbeddings(model=embedding_model)
                vector_store, metadata = load_vector_store_with_metadata(
                    "./Memory", "JDIH_index", embeddings
                )
                st.session_state.vector_store = vector_store
                st.session_state.index_metadata = metadata
                st.success("✅ Index berhasil dimuat dari penyimpanan")
    except Exception as e:
        st.warning(f"⚠️ Tidak dapat memuat index tersimpan: {str(e)}")

# Main interface
if st.session_state.vector_store is not None:
    # Query input
    st.subheader("💬 Tanya Dokumen")
    
    # Example queries
    with st.expander("💡 Contoh Pertanyaan"):
        example_queries = [
            "Berapa lama cuti hamil untuk mahasiswa ITB?",
            "Apa saja persyaratan untuk mengambil cuti akademik?",
            "Bagaimana prosedur pengajuan skripsi?",
            "Kapan deadline pembayaran SPP?",
            "Siapa yang bisa mengajukan dispensasi ujian?"
        ]
        
        for i, query in enumerate(example_queries):
            if st.button(f"📝 {query}", key=f"example_{i}"):
                st.session_state.current_query = query
    
    # Query input
    current_query = st.text_input(
        "Masukkan pertanyaan Anda:", 
        value=st.session_state.get("current_query", ""),
        placeholder="Contoh: Berapa lama cuti hamil untuk mahasiswa ITB?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Tanya", type="primary", use_container_width=True)
    with col2:
        clear_history = st.button("🗑️ Hapus Riwayat", use_container_width=True)
    
    if clear_history:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process query
    if ask_button and current_query:
        with st.spinner("🤔 Mencari jawaban..."):
            try:
                # Enhanced retrieval
                retrieved_docs = enhanced_retrieval(
                    st.session_state.vector_store,
                    current_query,
                    k=top_k,
                    query_expansion=enable_query_expansion,
                    rerank=enable_reranking
                )
                
                # Generate answer
                answer = answer_with_enhanced_rag(
                    current_query, 
                    retrieved_docs, 
                    chat_model=chat_model,
                    language="indonesian"
                )
                
                # Calculate metrics
                metrics = calculate_rag_metrics(current_query, retrieved_docs)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now(),
                    "query": current_query,
                    "answer": answer,
                    "retrieved_docs": retrieved_docs,
                    "metrics": metrics
                })
                
                # Display current result
                st.subheader("📝 Jawaban")
                st.write(answer)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    coverage = metrics.get("coverage_score", 0)
                    coverage_class = "confidence-high" if coverage > 0.7 else "confidence-medium" if coverage > 0.4 else "confidence-low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Coverage Score</h4>
                        <p class="{coverage_class}">{coverage:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    diversity = metrics.get("diversity_score", 0)
                    diversity_class = "confidence-high" if diversity > 0.7 else "confidence-medium" if diversity > 0.4 else "confidence-low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🔄 Diversity Score</h4>
                        <p class="{diversity_class}">{diversity:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    confidence = metrics.get("avg_confidence", 0)
                    confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Avg Confidence</h4>
                        <p class="{confidence_class}">{confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📄 Docs Retrieved</h4>
                        <p class="confidence-high">{len(retrieved_docs)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display retrieved documents
                with st.expander(f"📚 Dokumen Referensi ({len(retrieved_docs)} dokumen)"):
                    for i, doc_info in enumerate(retrieved_docs):
                        metadata = doc_info["metadata"]
                        similarity_score = doc_info.get("similarity_score", 0)
                        combined_score = doc_info.get("combined_score", 0)
                        
                        st.markdown(f"""
                        <div class="doc-source">
                            <h5>📄 Dokumen {i+1}: {metadata.get('filename', 'Unknown')}</h5>
                            <p><strong>Similarity Score:</strong> {1-similarity_score:.2%} | 
                               <strong>Combined Score:</strong> {combined_score:.2%}</p>
                            <p><strong>Keywords:</strong> {metadata.get('keywords', 'Tidak ada')}</p>
                            <details>
                                <summary>Lihat isi dokumen</summary>
                                <p>{doc_info['content'][:500]}...</p>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error saat memproses pertanyaan: {str(e)}")
                st.exception(e)
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("📜 Riwayat Percakapan")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 {chat['timestamp'].strftime('%H:%M:%S')} - {chat['query'][:50]}..."):
                st.write("**Pertanyaan:**", chat['query'])
                st.write("**Jawaban:**", chat['answer'])
                
                # Show metrics for this chat
                metrics = chat['metrics']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coverage", f"{metrics.get('coverage_score', 0):.2%}")
                with col2:
                    st.metric("Diversity", f"{metrics.get('diversity_score', 0):.2%}")
                with col3:
                    st.metric("Confidence", f"{metrics.get('avg_confidence', 0):.2%}")

else:
    # No index available
    st.warning("⚠️ **Index belum tersedia**")
    st.info("""
    📋 **Langkah-langkah untuk memulai:**
    
    1. 📁 Pastikan folder data PDF sudah ada dan berisi file PDF
    2. ⚙️ Atur konfigurasi model di sidebar (opsional)
    3. 🔨 Klik tombol "Build/Rebuild Index" di sidebar
    4. ⏳ Tunggu proses build selesai
    5. 💬 Mulai bertanya!
    
    **Catatan:** Proses build pertama kali bisa memakan waktu beberapa menit tergantung jumlah dan ukuran dokumen.
    """)
    
    # Show available files in data directory if exists
    if os.path.exists(data_path):
        pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
        if pdf_files:
            st.subheader("📁 File PDF yang ditemukan:")
            for file in pdf_files:
                file_path = os.path.join(data_path, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    st.write(f"📄 **{file}** ({file_size:.1f} KB)")
        else:
            st.warning(f"❌ Tidak ada file PDF ditemukan di folder: {data_path}")
    else:
        st.error(f"❌ Folder data tidak ditemukan: {data_path}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏛️ <strong>JDIH RAG - Sistem Tanya Jawab Dokumen Hukum ITB</strong></p>
    <p>Dikembangkan untuk Workshop ICoDSE 2025 • Powered by LangChain & OpenAI</p>
</div>
""", unsafe_allow_html=True)