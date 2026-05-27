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

# Install editable
RUN pip install --no-cache-dir -e ".[all]"

# Default port
EXPOSE 8000

# Data volume
VOLUME /app/data

CMD ["python", "-m", "uvicorn", "web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
