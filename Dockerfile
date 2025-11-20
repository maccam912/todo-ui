FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH="/root/.local/bin:${PATH}" \
    UV_PROJECT_ENV=.venv \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --all-extras

COPY . /app

EXPOSE 8080

CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8080"]
