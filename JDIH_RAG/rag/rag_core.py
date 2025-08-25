import os
import json
import math
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime

import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY tidak ditemukan. Pastikan API key sudah di-set di file .env")
    return OpenAI(api_key=api_key)

# ---------- Data Classes ----------

@dataclass
class DocumentMetadata:
    """Metadata untuk dokumen dengan informasi tambahan untuk industri"""
    doc_id: str
    filename: str
    file_size: int
    creation_date: datetime
    page_count: int
    document_type: str = "PDF"
    keywords: List[str] = None
    summary: str = ""
    confidence_score: float = 0.0

@dataclass 
class DocChunk:
    """Enhanced document chunk dengan metadata lengkap"""
    doc_id: str
    chunk_id: int
    text: str
    metadata: DocumentMetadata
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 100

# ---------- Advanced Chunking Strategies ----------

def chunk_text_advanced(text: str, strategy: str = "recursive", max_tokens: int = 1000, overlap_tokens: int = 100) -> List[str]:
    """Advanced chunking dengan berbagai strategi"""
    
    if strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=len,
        )
        return text_splitter.split_text(text)
    
    elif strategy == "semantic":
        # Semantic chunking berdasarkan kalimat
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_tokens:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    elif strategy == "paragraph":
        # Paragraph-based chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk + para) < max_tokens:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    else:
        # Default simple chunking
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Apply overlap
                overlap_words = current_chunk[-overlap_tokens//10:] if len(current_chunk) > overlap_tokens//10 else []
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# ---------- Enhanced Document Processing ----------

def load_documents_with_metadata(data_path: str) -> List[Tuple[str, DocumentMetadata]]:
    """Load dokumen dengan metadata lengkap"""
    documents = []
    
    # Load semua PDF dari directory
    pdf_loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    
    docs = pdf_loader.load()
    
    # Group by source file
    files_content = {}
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        if source not in files_content:
            files_content[source] = []
        files_content[source].append(doc.page_content)
    
    # Create metadata for each file
    for filepath, pages in files_content.items():
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        creation_date = datetime.fromtimestamp(os.path.getctime(filepath)) if os.path.exists(filepath) else datetime.now()
        
        full_text = "\n\n".join(pages)
        
        # Extract keywords sederhana (bisa dikembangkan dengan NLP)
        keywords = extract_keywords_simple(full_text)
        
        # Generate summary singkat
        summary = generate_document_summary(full_text)
        
        metadata = DocumentMetadata(
            doc_id=filename.replace('.pdf', ''),
            filename=filename,
            file_size=file_size,
            creation_date=creation_date,
            page_count=len(pages),
            keywords=keywords,
            summary=summary
        )
        
        documents.append((full_text, metadata))
    
    return documents

def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords sederhana berdasarkan frekuensi"""
    # Bersihkan teks
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Kata-kata yang diabaikan
    stopwords = set(['dan', 'atau', 'yang', 'dari', 'dengan', 'untuk', 'pada', 'di', 'ke', 'dalam',
                    'adalah', 'akan', 'telah', 'dapat', 'harus', 'ini', 'itu', 'tidak', 'juga',
                    'the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'by', 'at', 'on'])
    
    words = [word for word in text.split() if len(word) > 3 and word not in stopwords]
    
    # Hitung frekuensi
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort berdasarkan frekuensi
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:max_keywords]]

def generate_document_summary(text: str, max_length: int = 200) -> str:
    """Generate ringkasan dokumen sederhana"""
    # Ambil kalimat pertama dari beberapa paragraf
    paragraphs = text.split('\n\n')[:3]
    summary_parts = []
    
    for para in paragraphs:
        sentences = para.split('. ')
        if sentences:
            summary_parts.append(sentences[0])
    
    summary = '. '.join(summary_parts)
    
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary

# ---------- Enhanced Vector Store Operations ----------

def build_enhanced_vector_store(documents: List[Tuple[str, DocumentMetadata]], 
                               embedding_model: str = "text-embedding-3-small",
                               chunking_strategy: str = "recursive",
                               chunk_size: int = 1000,
                               chunk_overlap: int = 100) -> FAISS:
    """Build vector store dengan fitur enhanced"""
    
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    # Process semua dokumen
    all_chunks = []
    chunk_metadatas = []
    
    for text, metadata in tqdm(documents, desc="Processing documents"):
        # Chunk dokumen
        chunks = chunk_text_advanced(text, strategy=chunking_strategy, 
                                   max_tokens=chunk_size, overlap_tokens=chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            doc_chunk = DocChunk(
                doc_id=metadata.doc_id,
                chunk_id=i,
                text=chunk,
                metadata=metadata,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            all_chunks.append(doc_chunk)
            
            # Metadata untuk LangChain
            chunk_metadata = {
                "doc_id": metadata.doc_id,
                "chunk_id": i,
                "filename": metadata.filename,
                "file_size": metadata.file_size,
                "page_count": metadata.page_count,
                "keywords": ",".join(metadata.keywords or []),
                "summary": metadata.summary,
                "document_type": metadata.document_type,
                "creation_date": metadata.creation_date.isoformat()
            }
            chunk_metadatas.append(chunk_metadata)
    
    # Create vector store
    index = faiss.IndexFlatIP(1536)  # dimension untuk text-embedding-3-small
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Add documents
    texts = [chunk.text for chunk in all_chunks]
    vector_store.add_texts(texts, metadatas=chunk_metadatas)
    
    return vector_store, all_chunks

# ---------- Query Enhancement ----------

def expand_query(query: str, expansion_type: str = "synonyms") -> str:
    """Expand query untuk hasil retrieval yang lebih baik"""
    
    if expansion_type == "synonyms":
        # Dictionary sinonim sederhana untuk bahasa Indonesia
        synonyms = {
            "cuti": ["izin", "libur", "tidak masuk"],
            "hamil": ["kehamilan", "mengandung"],
            "mahasiswa": ["siswa", "peserta didik", "mahasiswi"],
            "peraturan": ["aturan", "ketentuan", "regulasi", "kebijakan"],
            "ujian": ["tes", "evaluasi", "penilaian"],
            "skripsi": ["tugas akhir", "thesis"],
            "semester": ["semster", "periode kuliah"]
        }
        
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            expanded_terms.append(word)
            if word in synonyms:
                expanded_terms.extend(synonyms[word])
        
        return " ".join(expanded_terms)
    
    elif expansion_type == "context":
        # Tambahkan konteks untuk pertanyaan umum
        context_additions = {
            "cuti": " mahasiswa ITB",
            "ujian": " aturan akademik",
            "skripsi": " persyaratan kelulusan"
        }
        
        for term, addition in context_additions.items():
            if term in query.lower():
                query += addition
        
        return query
    
    return query

def rewrite_query_for_retrieval(query: str) -> str:
    """Rewrite query untuk retrieval yang lebih optimal"""
    
    # Template pertanyaan umum
    question_templates = {
        "berapa lama": "durasi waktu",
        "bagaimana cara": "prosedur langkah",
        "apa saja": "daftar jenis",
        "siapa yang": "pihak yang berwenang",
        "kapan": "waktu periode"
    }
    
    rewritten = query.lower()
    
    for pattern, replacement in question_templates.items():
        if pattern in rewritten:
            rewritten = rewritten.replace(pattern, replacement)
    
    return rewritten

# ---------- Advanced RAG Pipeline ----------

def enhanced_retrieval(vector_store: FAISS, query: str, k: int = 5, 
                      query_expansion: bool = True, 
                      rerank: bool = True) -> List[Dict[str, Any]]:
    """Enhanced retrieval dengan query expansion dan reranking"""
    
    # Query expansion
    if query_expansion:
        expanded_query = expand_query(query)
        rewritten_query = rewrite_query_for_retrieval(expanded_query)
    else:
        rewritten_query = query
    
    # Retrieval
    docs = vector_store.similarity_search_with_score(rewritten_query, k=k*2)  # Ambil lebih banyak untuk reranking
    
    # Basic reranking berdasarkan kecocokan kata kunci
    if rerank:
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc, similarity_score in docs:
            doc_words = set(doc.page_content.lower().split())
            keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words)
            
            combined_score = 0.7 * (1 - similarity_score) + 0.3 * keyword_overlap  # Kombinasi similarity + keyword overlap
            
            scored_docs.append({
                "document": doc,
                "similarity_score": similarity_score,
                "keyword_score": keyword_overlap,
                "combined_score": combined_score,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Sort berdasarkan combined score
        scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_docs[:k]
    
    else:
        return [{"document": doc, "similarity_score": score, "content": doc.page_content, 
                "metadata": doc.metadata} for doc, score in docs[:k]]

def create_enhanced_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Create context yang lebih informatif dengan metadata"""
    context_parts = []
    
    for i, doc_info in enumerate(retrieved_docs):
        doc = doc_info["document"]
        metadata = doc_info["metadata"]
        
        header = f"[Dokumen {i+1}: {metadata.get('filename', 'Unknown')}]"
        content = doc.page_content.strip()
        
        # Tambahkan informasi confidence jika ada
        if "combined_score" in doc_info:
            confidence = doc_info["combined_score"]
            header += f" (Relevansi: {confidence:.2f})"
        
        context_parts.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(context_parts)

def answer_with_enhanced_rag(query: str, retrieved_docs: List[Dict[str, Any]], 
                           chat_model: str = "gpt-4o-mini",
                           language: str = "indonesian") -> str:
    """Generate jawaban dengan enhanced RAG pipeline"""
    
    client = get_openai_client()
    
    # System prompt dalam bahasa Indonesia
    if language == "indonesian":
        system_prompt = (
            "Anda adalah asisten AI yang membantu menjawab pertanyaan tentang peraturan dan dokumen akademik ITB. "
            "Jawab pertanyaan pengguna HANYA berdasarkan konteks dokumen yang diberikan. "
            "Jika informasi tidak tersedia dalam konteks, katakan dengan jelas bahwa informasi tersebut tidak ditemukan. "
            "Berikan jawaban yang akurat, lengkap, dan mudah dipahami dalam bahasa Indonesia. "
            "Jika relevan, sebutkan sumber dokumen dan bagian yang mendukung jawaban Anda."
        )
    else:
        system_prompt = (
            "You are an AI assistant that helps answer questions about ITB academic regulations and documents. "
            "Answer the user's question ONLY based on the provided document context. "
            "If information is not available in the context, clearly state that the information is not found. "
            "Provide accurate, complete, and easy-to-understand answers."
        )
    
    # Create enhanced context
    context = create_enhanced_context(retrieved_docs)
    
    # User prompt
    user_prompt = f"""
Pertanyaan: {query}

Konteks dokumen yang relevan:
{context}

Tolong jawab pertanyaan berdasarkan konteks di atas. Jika memungkinkan, sebutkan dokumen mana yang menjadi rujukan jawaban Anda.
"""
    
    # Generate response
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,  # Low temperature untuk konsistensi
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# ---------- Evaluation Metrics ----------

def calculate_rag_metrics(query: str, retrieved_docs: List[Dict[str, Any]], 
                         ground_truth_keywords: List[str] = None) -> Dict[str, float]:
    """Hitung metrics untuk evaluasi kualitas RAG"""
    
    metrics = {}
    
    # 1. Coverage Score - seberapa banyak query terms yang covered
    query_terms = set(query.lower().split())
    all_doc_terms = set()
    
    for doc_info in retrieved_docs:
        doc_terms = set(doc_info["content"].lower().split())
        all_doc_terms.update(doc_terms)
    
    if query_terms:
        coverage_score = len(query_terms.intersection(all_doc_terms)) / len(query_terms)
    else:
        coverage_score = 0.0
    
    metrics["coverage_score"] = coverage_score
    
    # 2. Diversity Score - diversity antar retrieved documents
    if len(retrieved_docs) > 1:
        doc_similarities = []
        for i in range(len(retrieved_docs)):
            for j in range(i+1, len(retrieved_docs)):
                doc1_terms = set(retrieved_docs[i]["content"].lower().split())
                doc2_terms = set(retrieved_docs[j]["content"].lower().split())
                
                if doc1_terms and doc2_terms:
                    similarity = len(doc1_terms.intersection(doc2_terms)) / len(doc1_terms.union(doc2_terms))
                    doc_similarities.append(similarity)
        
        diversity_score = 1.0 - (sum(doc_similarities) / len(doc_similarities) if doc_similarities else 0)
    else:
        diversity_score = 1.0
    
    metrics["diversity_score"] = diversity_score
    
    # 3. Confidence Score - rata-rata confidence dari retrieved docs
    if retrieved_docs and "combined_score" in retrieved_docs[0]:
        confidence_scores = [doc["combined_score"] for doc in retrieved_docs]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
    else:
        avg_confidence = 0.0
    
    metrics["avg_confidence"] = avg_confidence
    
    # 4. Keyword Relevance (jika ground truth tersedia)
    if ground_truth_keywords:
        relevant_keywords_found = 0
        for keyword in ground_truth_keywords:
            for doc_info in retrieved_docs:
                if keyword.lower() in doc_info["content"].lower():
                    relevant_keywords_found += 1
                    break
        
        keyword_recall = relevant_keywords_found / len(ground_truth_keywords)
        metrics["keyword_recall"] = keyword_recall
    
    return metrics

# ---------- Utility Functions ----------

def save_vector_store_with_metadata(vector_store: FAISS, chunks: List[DocChunk], 
                                   save_path: str, index_name: str):
    """Save vector store beserta metadata lengkap"""
    
    # Save vector store
    vector_store.save_local(folder_path=save_path, index_name=index_name)
    
    # Save metadata chunks
    metadata_path = os.path.join(save_path, f"{index_name}_metadata.json")
    
    chunks_data = []
    for chunk in chunks:
        chunk_data = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text[:500],  # Save first 500 chars for reference
            "embedding_model": chunk.embedding_model,
            "chunk_size": chunk.chunk_size,
            "chunk_overlap": chunk.chunk_overlap,
            "metadata": {
                "filename": chunk.metadata.filename,
                "file_size": chunk.metadata.file_size,
                "creation_date": chunk.metadata.creation_date.isoformat(),
                "page_count": chunk.metadata.page_count,
                "keywords": chunk.metadata.keywords,
                "summary": chunk.metadata.summary,
                "document_type": chunk.metadata.document_type
            }
        }
        chunks_data.append(chunk_data)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

def load_vector_store_with_metadata(save_path: str, index_name: str, 
                                   embeddings: OpenAIEmbeddings) -> Tuple[FAISS, List[Dict]]:
    """Load vector store beserta metadata"""
    
    # Load vector store
    vector_store = FAISS.load_local(
        folder_path=save_path, 
        index_name=index_name, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Load metadata
    metadata_path = os.path.join(save_path, f"{index_name}_metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = []
    
    return vector_store, metadata