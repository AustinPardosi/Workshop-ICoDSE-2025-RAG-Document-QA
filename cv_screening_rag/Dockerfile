FROM python:3.11-slim

WORKDIR /app

# System deps (pdfplumber uses Pillow, which needs basic libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Default environment
ENV PYTHONPATH=/app

# Expose Streamlit (8501) and Jupyter (8888)
EXPOSE 8501 8888

# Default command: Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
