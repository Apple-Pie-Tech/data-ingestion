# AWS Lambda Web Adapter: a Lambda extension that bridges the Lambda Runtime API
# to an ordinary HTTP server, so this stays a plain uvicorn app -- no Mangum, no
# handler function, no application code aware of Lambda at all. The same image
# runs under `docker run` and under Lambda.
#
# x86_64 rather than aarch64, recorded because it is a decision not an
# accident: Graviton is ~20% cheaper, which is nothing inside the always-free
# Lambda tier, an arm64 build cannot be verified on an x86_64 host without
# installing qemu binfmt, and story-labeling-api cannot go arm64 at all
# (hdbscan publishes no aarch64 wheel). A uniform x86_64 keeps all three
# services on one verified architecture. Flipping is this one argument.
ARG LWA_ARCH=x86_64
FROM public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0-${LWA_ARCH} AS lambda-adapter


# ---------------------------------------------------------------- build stage
# Separate from the runtime stage so uv, pip's cache and the build tooling never
# reach the shipped image. It is not only size: the runtime image having no
# package manager is one less thing for a compromised request handler to use.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY app ./app

RUN uv sync --frozen --no-dev

# Warm the tokenizer cache at build time. tiktoken downloads its BPE table on
# first use and caches it under $HOME; in Lambda only /tmp is writable, so that
# download would happen inside a request on every cold start. This must match
# SEMANTIC_TOKENIZER_ENCODING -- a different encoding at runtime falls back to
# downloading, which is the behaviour this exists to prevent.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && test -n "$(ls -A "$TIKTOKEN_CACHE_DIR")"


# -------------------------------------------------------------- runtime stage
FROM python:3.12-slim

COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # The adapter listens for Lambda and forwards to this port; its own default
    # is 8080. The app already serves 8000.
    AWS_LWA_PORT=8000 \
    # The adapter's default readiness path is `/`, which this app does not serve.
    # A 404 falls inside its healthy range (100-499), so the default would report
    # ready the moment uvicorn bound a socket rather than when the app could
    # answer -- the first real request would then race startup.
    AWS_LWA_READINESS_CHECK_PATH=/health \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser

# Deliberately NOT chowned to appuser. `chown -R` on the virtualenv rewrites
# every file into the layer, which doubled this image (a 287 MB venv produced a
# 603 MB layer). Nothing here needs to be writable: the app does not write to
# its own install, and in Lambda the filesystem is read-only outside /tmp
# anyway, so root-owned and world-readable is both smaller and tighter.
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY --from=builder /opt/tiktoken /opt/tiktoken

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/app/.venv/bin/python", "-c", "import json, sys, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); payload = json.load(response); sys.exit(0 if payload.get('status') == 'ok' else 1)"]

# The venv's uvicorn directly, not `uv run`: uv is not in this image, and even
# where it is, it would re-resolve the project on every cold start and may try to
# write -- and in Lambda the filesystem is read-only outside /tmp.
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
