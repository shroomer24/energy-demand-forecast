FROM python:3.11-slim

# Install system dependencies for XGBoost (OpenMP)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/

# Copy pre-trained model artifacts if they exist
COPY data/*.pkl ./data/ 2>/dev/null || true
COPY data/demand.csv ./data/ 2>/dev/null || true

# Ensure data & logs directories exist
RUN mkdir -p data/charts logs mlruns

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

# Default: start inference API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
