FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install package and prepare writable runtime data directory
RUN pip install --no-cache-dir . && \
    groupadd --system verarag && \
    useradd --system --gid verarag --home-dir /app --shell /usr/sbin/nologin verarag && \
    mkdir -p /app/data && \
    chown -R verarag:verarag /app

# Default port
EXPOSE 8000

# Data volume
VOLUME /app/data

USER verarag

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=2).read()" || exit 1

CMD ["verarag-web", "--host", "0.0.0.0", "--port", "8000"]
