"""
Solusi untuk menangani multiple PDF files dalam indexing pipeline RAG
"""

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap


def load_multiple_pdfs(filepath="./Assets/Data"):
    """
    Load multiple PDF files dari direktori yang diberikan

    Args:
        filepath (str): Path ke direktori yang berisi PDF files

    Returns:
        list: List of Document objects dari semua PDF files
    """

    # Method 1: Menggunakan DirectoryLoader (Recommended)
    print("=== Method 1: DirectoryLoader ===")
    loader = DirectoryLoader(
        path=filepath,
        glob="**/*.pdf",  # Load semua PDF files secara recursive
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    pdf_data = loader.load()

    print(f"Total documents loaded: {len(pdf_data)}")
    print(
        f"Files found: {list(set([doc.metadata.get('source', 'Unknown') for doc in pdf_data]))}"
    )

    return pdf_data


def load_multiple_pdfs_manual(filepath="./Assets/Data"):
    """
    Method alternatif: Load PDF files secara manual satu per satu
    """
    print("\n=== Method 2: Manual Loading ===")

    # Get semua PDF files dalam direktori
    pdf_files = [f for f in os.listdir(filepath) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    # Load semua PDF files
    pdf_data = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(filepath, pdf_file)
        print(f"Loading: {pdf_file}")

        try:
            pdfloader = PyPDFLoader(file_path=pdf_path)
            file_data = pdfloader.load()

            # Add metadata untuk mengidentifikasi source file
            for doc in file_data:
                doc.metadata["source"] = pdf_file
                doc.metadata["file_path"] = pdf_path

            pdf_data.extend(file_data)
            print(f"  - Loaded {len(file_data)} pages from {pdf_file}")
        except Exception as e:
            print(f"  - Error loading {pdf_file}: {str(e)}")

    print(f"\nTotal documents loaded: {len(pdf_data)}")
    return pdf_data


def chunk_documents(pdf_data, chunk_size=1000, chunk_overlap=100):
    """
    Chunk semua documents dari multiple PDF files

    Args:
        pdf_data (list): List of Document objects
        chunk_size (int): Ukuran chunk dalam karakter
        chunk_overlap (int): Overlap antar chunks

    Returns:
        list: List of chunked documents
    """

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "."],  # Karakter yang digunakan untuk split
        chunk_size=chunk_size,  # Jumlah karakter dalam setiap chunk
        chunk_overlap=chunk_overlap,  # Jumlah karakter yang overlap antar chunks
    )

    # Process semua documents dari multiple PDF files
    all_texts = [doc.page_content for doc in pdf_data]
    text_chunks = text_splitter.create_documents(all_texts)

    print(f"The number of chunks created: {len(text_chunks)}")

    return text_chunks


def analyze_chunks(text_chunks):
    """
    Analisis distribusi ukuran chunks
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = [len(doc.page_content) for doc in text_chunks]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data)
    plt.title("Box Plot of chunk lengths")
    plt.xlabel("Chunk Lengths")
    plt.ylabel("Values")
    plt.show()

    print(f"The median chunk length is: {round(np.median(data), 2)}")
    print(f"The average chunk length is: {round(np.mean(data), 2)}")
    print(f"The minimum chunk length is: {round(np.min(data), 2)}")
    print(f"The max chunk length is: {round(np.max(data), 2)}")
    print(f"The 75th percentile chunk length is: {round(np.percentile(data, 75), 2)}")
    print(f"The 25th percentile chunk length is: {round(np.percentile(data, 25), 2)}")


# Contoh penggunaan
if __name__ == "__main__":
    # Load multiple PDF files
    pdf_data = load_multiple_pdfs()

    # Tampilkan sample dari document pertama
    if pdf_data:
        print(f"\nSample from first document:")
        print(textwrap.fill(f"{pdf_data[0].page_content[:1000]}", width=150))

    # Chunk documents
    text_chunks = chunk_documents(pdf_data)

    # Analisis chunks
    analyze_chunks(text_chunks)
