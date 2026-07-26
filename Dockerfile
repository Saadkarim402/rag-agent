FROM python:3.11-slim

# Set up workspace
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BAAI embeddings and reranker models so the container starts instantly
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')"

# Copy project files
COPY . .

# Grant execute rights to startup script
RUN chmod +x start.sh

# Expose Streamlit port (Hugging Face default)
EXPOSE 7860

# Start both services
CMD ["./start.sh"]
