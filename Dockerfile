# syntax=docker/dockerfile:1
# NoCodeML API. Build from the repository root:  docker build -t nocodeml-api .
FROM python:3.12-slim

# libgomp1: the OpenMP runtime XGBoost needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY engine/pyproject.toml ./
COPY engine/nocodeml_engine ./nocodeml_engine
# Patient with slow networks, and the cache mount keeps downloaded wheels so a retry doesn't start from zero.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 ".[api]"

RUN useradd --create-home --shell /usr/sbin/nologin app
USER app

# Production mode: no /docs, strict CORS (set NOCODEML_CORS_ORIGINS), HSTS. See docs/deploy.md for every setting.
ENV NOCODEML_ENV=production \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)"

# ONE worker on purpose: rate-limit counters and running trainings live in this process.
CMD ["sh", "-c", "exec uvicorn nocodeml_engine.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
