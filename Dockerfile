FROM python:3.10-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU first to avoid heavy CUDA wheels
RUN pip install --no-cache-dir torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
# Filter out torch from requirements.txt to prevent reinstallation
RUN grep -v "torch==" requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt

# Pre-download Sentence Transformer and Cross-Encoder models to bake into image
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy project files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
