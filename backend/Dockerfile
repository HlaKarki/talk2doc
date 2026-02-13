FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv for fast, reproducible dependency installs from uv.lock.
RUN pip install --no-cache-dir uv

# Install dependencies only first for better Docker layer reuse.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy backend source code after dependencies are ready.
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
