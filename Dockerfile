# Dockerfile
# Builds the FastAPI CTR prediction service into a container.
#
# How it works:
# Each line is a "layer" - Docker caches each layer separately.
# If you change your code but not requirements.txt,
# Docker reuses the cached packages layer — much faster rebuilds.

# Base image
# Start from official Python 3.10 slim image.
# "slim" means minimal OS — smaller size, faster to download.
# We use 3.10 to match your local venv exactly.
FROM python:3.10-slim

# Working directory
# All commands from here run inside /app inside the container.
# Like doing "cd /app" permanently.
WORKDIR /app

#System dependencies
# Some Python packages need system libraries to compile.
# We install them first so pip install works cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     g++     && rm -rf /var/lib/apt/lists/*

# Python dependencies
# Copy requirements FIRST before copying code.
# Why: Docker caches layers. If requirements.txt hasn't changed,
# Docker skips pip install entirely — saves minutes on rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Application code
# Copy everything else after packages are installed.
# Changes to code don't invalidate the packages cache.
COPY src/ ./src/
COPY mlruns/ ./mlruns/
COPY data/processed/ ./data/processed/
COPY models/ ./models/

#Environment variables
# Set Python to not write .pyc files (cleaner container)
# and to not buffer output (logs appear immediately)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Port
# Tell Docker this container listens on port 8000.
# This is documentation — actual port mapping is in docker-compose.yml
EXPOSE 8000

# Startup command
# Run FastAPI with uvicorn when container starts.
# host 0.0.0.0 means accept connections from outside the container.
# Without this, the API would only be reachable from inside the container.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
