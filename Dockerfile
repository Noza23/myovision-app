FROM python:3.13-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends

FROM base AS builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --no-dev


FROM base AS copier

COPY logging.yaml /app/logging.yaml
COPY backend /app/backend
RUN mkdir -p /static/images



FROM base AS runtime

ENV PATH=/app/.venv/bin:$PATH

COPY --from=builder /app/.venv /app/.venv
COPY --from=copier /app/logging.yaml /app/logging.yaml
COPY --from=copier /app/backend /app/backend

RUN useradd -m -s /bin/bash myovision
USER myovision

WORKDIR /app
EXPOSE 8000

CMD ["fastapi", "run", "backend/main.py", "--host", "0.0.0.0"]
