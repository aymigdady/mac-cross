FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3.0" \
 && pip install --no-cache-dir -r requirements.txt

ARG CCAPR_DISABLE_BGE_M3_DOWNLOAD=0
ARG CCAPR_DISABLE_BGE_RERANKER_DOWNLOAD=0
RUN if [ "$CCAPR_DISABLE_BGE_M3_DOWNLOAD" != "1" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', cache_folder='/opt/hf-cache')"; \
    fi
RUN if [ "$CCAPR_DISABLE_BGE_RERANKER_DOWNLOAD" != "1" ]; then \
        python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')"; \
    fi

COPY . /app

EXPOSE 8080
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 --timeout 300 -b 0.0.0.0:${PORT} app:app"]
