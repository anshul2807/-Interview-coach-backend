FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first (Docker layer cache — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py .

# ── Copy pre-built ChromaDB (must exist before docker build) ──────────────────
# Run `python build_db.py` locally first, then build the image
COPY chroma_db/ ./chroma_db/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Start the API ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]