# Stage 1: Build Stage
FROM docker.io/library/python:3.13-slim-bookworm AS builder

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv by copying the static binary out of its distroless image —
# no curl|sh, no pip bootstrap, just a COPY.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Stage 2: Runtime Stage
FROM docker.io/library/python:3.13-slim-bookworm AS runtime

# Create a non-root user with a PINNED uid/gid for predictable namespace mapping
RUN groupadd -r -g 1001 agentuser \
    && useradd -r -u 1001 -g agentuser agentuser

WORKDIR /app

# Copy the resolved virtual environment from the builder, not raw packages
COPY --from=builder --chown=agentuser:agentuser /app/.venv /app/.venv
COPY --chown=agentuser:agentuser . .

ENV PATH="/app/.venv/bin:$PATH"

USER agentuser

EXPOSE 8000

# No --reload here. Dev-only flags belong in the compose override, not the image.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]