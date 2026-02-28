FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY main.py ./
COPY opencode_proxy/ ./opencode_proxy/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 4141

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4141"]
