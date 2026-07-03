# Use a lightweight, secure Python base image
FROM python:3.11-slim as runtime-builder

# Set system environment constraints
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

# Install native operating system build tools for schema compilation checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy framework dependencies down first to leverage Docker layer caching architecture
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt fastapi uvicorn pydantic

# Copy our modular package modules and app drivers into the isolated work zone
COPY ./hierarchical_lm /app/hierarchical_lm
COPY ./deployment/app.py /app/app.py

# Expose network ingress port constraints
EXPOSE 8000

# Fire up production uvicorn worker channels to serve traffic
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
