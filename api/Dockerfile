# Build from the repository root:  docker build -f api/Dockerfile .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
# The fixed income engine, pinned to a release tag of its public repository.
ARG BONDLAB_VERSION=v0.2.0
RUN pip install --no-cache-dir "bondlab @ https://github.com/blessedshoh/bondlab/archive/refs/tags/${BONDLAB_VERSION}.tar.gz"

COPY api ./api

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
