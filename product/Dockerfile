# uv-in-Docker: copy the official binary, then install from the lockfile.
# https://docs.astral.sh/uv/guides/integration/docker/

FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

RUN groupadd --gid 1000 finscore \
    && useradd --uid 1000 --gid finscore --create-home --shell /bin/bash finscore \
    && chown -R finscore:finscore /app

USER finscore

CMD ["python", "-m", "embat_finscore"]
