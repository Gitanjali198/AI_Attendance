# Lightweight Python Base
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project code and assets
COPY . .

# Expose default port
EXPOSE 8501

# Run healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Start the Streamlit Web Portal binding to $PORT provided by Render
CMD streamlit run frontend/dashboard.py --server.port ${PORT} --server.address 0.0.0.0
