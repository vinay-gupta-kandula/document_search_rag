FROM python:3.11-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure pip timeout and retry settings
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=5

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU and requirements directly using wheels
RUN pip install --no-cache-dir torch>=2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Sentence Transformer and Cross-Encoder models into image cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
