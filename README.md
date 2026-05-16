# Apple Pie Data Ingestion Plan

This repo owns the **DB ingestion system** for Apple Pie. It receives captured user inputs, turns them into semantically meaningful text chunks, embeds those chunks, and writes them to Qdrant for later processing and generation.

The repo is currently empty apart from Git metadata, so this plan starts from a minimal, reliable hackathon scaffold instead of adapting existing code.

## MVP goal

Build one Dockerized service that accepts an input-system object and stores searchable story chunks in Qdrant.

Input from the input system:

- `audio`: optional audio binary/file
- `text`: optional text transcript or raw text
- `metadata`: required JSON with at least:
  - `input_id`: unique ID from the input system
  - `user_id`: user identifier
  - `timestamp`: capture timestamp

Output from this component:

- Qdrant points where each point is one semantic text chunk plus vector embedding and metadata payload
- HTTP response confirming the input was accepted or indexed

## Minimal architecture

Use **FastAPI + Python + uv + Docker** for the service. The app is designed to run in a container, not directly on a developer machine.

```text
POST /ingest
  -> validate metadata and audio/text presence
  -> get transcript
      -> if text is supplied, use it
      -> if audio is supplied, call Gradium transcription API
  -> pass transcript to Chonkie SemanticChunker
  -> Chonkie embeds candidate sentences/windows with Azure-hosted text-embedding-3-large
  -> Chonkie returns semantic chunks around meaning shifts
  -> embed final chunks with the same Azure embedding deployment
  -> create or reuse matching Qdrant collection
  -> batch upsert chunk points into Qdrant
  -> return status
```

## Implementation decisions

### 1. API endpoint

Implement a single endpoint:

```text
POST /ingest
Content-Type: multipart/form-data

fields:
  audio: file, optional
  text: string, optional
  metadata: JSON string, required
```

Rules:

- Accept `audio`, `text`, or both.
- If `text` is present, skip transcription and use `text` as the transcript.
- If only `audio` is present, transcribe it with Gradium.
- Reject requests with neither `audio` nor `text`.
- Keep response simple: `{ "input_id": "...", "status": "indexed", "chunks": N }`.

For hackathon speed, this can run synchronously at first. If transcription or embedding latency causes HTTP timeouts, switch to FastAPI `BackgroundTasks` and return `{ "status": "accepted" }` immediately.

### 2. Transcription through Gradium

Use **Gradium** as the audio-to-text provider. Gradium describes managed speech-to-text and text-to-speech APIs with REST and WebSocket transports, so treat it as an external API provider rather than a UI/client library.

For this ingestion service, start with **REST transcription for complete uploaded audio files**. WebSocket transcription is better for live turn-taking, streaming captions, or VAD-driven interactions, which are out of scope for the MVP.

Environment variables:

```text
GRADIUM_API_BASE_URL=<gradium-api-base-url>
GRADIUM_API_KEY=<secret>
GRADIUM_TRANSCRIPTION_MODEL=<model-or-api-choice>
GRADIUM_TRANSCRIPTION_TRANSPORT=rest
```

Implementation shape:

```python
async def transcribe_audio(audio_path: str) -> str:
    # Upload a complete audio file to Gradium's REST transcription API.
    # Keep this behind our own adapter because exact endpoint/payload details
    # should come from Gradium API docs or studio credentials.
    ...
```

Open question resolved: **Is the provider Gradio or Gradium?** It is Gradium. Remove `gradio_client` from the plan and build a `GradiumTranscriber` adapter using `httpx`.

Open question still pending: **What exact Gradium endpoint and payload should we call?** Public docs confirm transcription APIs exist, but the exact request/response contract, file-size limits, auth header, timestamp/diarization support, and retry semantics should be taken from Gradium API docs or the team account before implementation.

### 3. Proper semantic chunking with Chonkie

Use a working semantic chunking dependency instead of hand-writing the chunking algorithm.

Recommended dependency: **Chonkie `SemanticChunker`** with its Azure OpenAI embedding integration.

Why Chonkie is the best fit for this repo:

- It is focused on chunking rather than being a full RAG framework.
- It supports semantic chunking from embedding similarity.
- It has Azure OpenAI embedding support, matching our `text-embedding-3-large` requirement.
- It supports custom embedding providers if the Azure integration needs wrapping.
- It supports batch and async chunking paths, which matters for ingestion throughput.

Alternatives considered:

- **LangChain `SemanticChunker`**: mature and supports percentile, standard-deviation, interquartile, and gradient breakpoint strategies, but it lives in the experimental package and pulls the plan toward LangChain as a framework.
- **LlamaIndex `SemanticSplitterNodeParser`**: good if the app already uses LlamaIndex ingestion, but this repo does not need that larger framework yet.
- **`semantic-chunkers`**: promising and async-friendly, but smaller ecosystem and Azure support comes indirectly through `semantic-router` encoders.
- **`semchunk`**: useful for token-aware splitting, but not embedding-based semantic chunking, so it does not satisfy this requirement.

Chonkie-backed MVP behavior:

1. Normalize transcript whitespace.
2. Pass transcript to `SemanticChunker`.
3. Configure Chonkie to use Azure OpenAI embeddings.
4. Let Chonkie split around embedding-based semantic shifts.
5. Enforce practical min/max chunk constraints where Chonkie supports them, or post-process only for hard safety limits.
6. Embed final chunks before Qdrant upsert.

Initial settings:

```text
SEMANTIC_SIMILARITY_THRESHOLD=<start-with-chonkie-default-or-0.72>
MIN_CHUNK_CHARS=350
MAX_CHUNK_CHARS=1400
CHUNK_OVERLAP_SENTENCES=1
```

These values are intentionally configurable because threshold quality depends on the story style and transcript length. For the hackathon, start with Chonkie's defaults, then tune with 3-5 real examples only if retrieval quality is obviously poor.

Open question resolved: **Do we need to implement semantic chunking ourselves?** No. Use Chonkie as the working dependency and keep `semantic_chunking.py` as a thin adapter around it.

### 4. Embeddings through Azure OpenAI

Use one Azure-hosted embedding deployment for both Chonkie semantic chunking and final chunk vectors.

Required configuration:

```text
AZURE_OPENAI_ENDPOINT=<https://...openai.azure.com/openai/v1/>
AZURE_OPENAI_API_KEY=<secret>
AZURE_OPENAI_API_VERSION=<api-version>
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=<deployment-name-for-text-embedding-3-large>
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
```

Why:

- The team can use the hosted Azure model instead of shipping local model weights.
- The same vectors drive both chunk boundary detection and Qdrant retrieval.
- Docker images stay smaller because no local `sentence-transformers` model is needed.

Azure OpenAI calls should pass the **deployment name** as the model value. Keep the raw model name in config only for observability/logging.

Use batched embedding calls wherever possible:

- one batch through Chonkie for semantic candidate windows
- one batch for final chunks

Open question resolved: **What vector size should Qdrant use?** Default to `3072` for `text-embedding-3-large` unless the Azure deployment/request is configured to return a reduced dimension. The Qdrant collection size must exactly match the actual embedding length returned by Azure.

### 5. Qdrant storage

Use one collection:

```text
QDRANT_COLLECTION=apple_pie_story_chunks
```

Each Qdrant point:

```json
{
  "id": "<input_id>:<chunk_index>",
  "vector": [0.1, 0.2, ...],
  "payload": {
    "input_id": "...",
    "user_id": "...",
    "timestamp": "...",
    "chunk_index": 0,
    "text": "chunk text",
    "source": "audio|text",
    "embedding_model": "text-embedding-3-large",
    "semantic_chunking": {
      "break_threshold": 0.72,
      "overlap_sentences": 1
    }
  }
}
```

Rules:

- Create the collection on startup if it does not exist.
- Collection vector size must exactly match `EMBEDDING_DIM`.
- Batch upsert all chunks for one input.
- Use deterministic point IDs `<input_id>:<chunk_index>` so retries overwrite the same chunks.
- Store original metadata in payload so later systems can label and group points by user/input/time.

Open question resolved: **What is the vector DB object?** One object is one semantic chunk, its Azure embedding vector, and metadata payload.

## Docker and package management

This app should be developed and deployed as a Docker container. Do not rely on running the service directly on the host machine.

Use `uv` for Python package management.

Suggested files:

```text
data-ingestion/
  README.md
  pyproject.toml
  uv.lock
  Dockerfile
  .dockerignore
  .env.example
  app/
    main.py              # FastAPI app and /ingest route
    config.py            # environment settings
    schemas.py           # request metadata models
    transcription.py     # Gradium API adapter
    semantic_chunking.py # thin Chonkie SemanticChunker adapter
    embeddings.py        # Azure OpenAI embedding wrapper
    vector_store.py      # Qdrant collection + upsert
  tests/
    test_semantic_chunking.py
    test_ingest_contract.py
```

Minimal dependencies in `pyproject.toml`:

```text
fastapi
uvicorn[standard]
python-multipart
pydantic-settings
httpx
openai
chonkie[azure-openai]
qdrant-client
numpy
```

Dockerfile shape:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir uv \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser

COPY pyproject.toml uv.lock README.md ./

COPY app ./app

RUN uv sync --frozen --no-dev \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import json, sys, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); payload = json.load(response); sys.exit(0 if payload.get('status') == 'ok' else 1)"]

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

For a hackathon demo, Qdrant can run as a separate service/container or be hosted externally. The ingestion app only needs `QDRANT_URL` and collection settings.

### Local verification and container runtime

Run the default test suite first. The external smoke module is env-gated and skipped unless you explicitly set `RUN_EXTERNAL_SMOKE=1`:

```bash
uv run pytest
```

If you want to verify the smoke module stays skip-safe without real provider credentials, run:

```bash
RUN_EXTERNAL_SMOKE=0 uv run pytest tests/test_external_smoke.py
```

Build the image locally:

```bash
docker build -t apple-pie-data-ingestion:local .
```

Run the container smoke with the safe example env file and verify `/health`:

```bash
docker run -d --name apple-pie-ingest-test -p 8000:8000 --env-file .env.example apple-pie-data-ingestion:local
curl -fsS http://localhost:8000/health
docker rm -f apple-pie-ingest-test
```

For local stack smoke, `docker-compose.yml` starts the app alongside Qdrant and uses fake Azure/Gradium defaults so `/health` stays available without real provider credentials. Override the env values only when you intentionally want to exercise real external calls.

```bash
docker compose up --build app qdrant
```

Text-only ingest smoke command (for use with real provider overrides or a test double stack):

```bash
./scripts/smoke_text_ingest.sh
```

Equivalent curl command used by the smoke script:

```bash
curl -fsS \
  -X POST http://localhost:8000/ingest \
  -F 'text=Alice followed the rabbit hole into a bright hall. She found a tiny golden key on a glass table.' \
  -F 'metadata={"input_id":"sample-input-001","user_id":"user-001","timestamp":"2026-05-16T12:00:00Z"}'
```

For real external smoke with live Azure/Qdrant/Gradium integrations, export these env vars before running `tests/test_external_smoke.py`:

```bash
export RUN_EXTERNAL_SMOKE=1
export QDRANT_URL=http://localhost:6333
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=replace-me
export AZURE_OPENAI_API_VERSION=2024-10-21
export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
export GRADIUM_API_BASE_URL=https://your-gradium-host
export GRADIUM_API_KEY=replace-me
export GRADIUM_TRANSCRIPTION_MODEL=whisper-1
```

Use the same env block above for the smoke commands below. The text-only smoke needs `RUN_EXTERNAL_SMOKE`, `QDRANT_URL`, and `AZURE_OPENAI_*`; the audio smoke also needs the `GRADIUM_*` values and uploads `tests/fixtures/sample.wav`.

```bash
uv run pytest tests/test_external_smoke.py -k text_only
uv run pytest tests/test_external_smoke.py -k audio_external
```

## Missing aspects and answers

| Missing aspect | Hackathon answer |
| --- | --- |
| Exact input contract from input system | Use multipart upload with `audio`, optional `text`, and JSON `metadata`. Align field names later. |
| Audio-to-text provider | Use Gradium, not Gradio. Build a small HTTP adapter around Gradium's transcription API. |
| Exact Gradium API contract | Still needs credentials/API docs confirmation. Keep the adapter isolated so the payload can change without touching ingestion logic. |
| Gradium REST vs WebSocket | Use REST for complete uploaded audio files. Defer WebSocket until live streaming is required. |
| Embedding provider | Azure OpenAI deployment of `text-embedding-3-large`; pass the deployment name as the API model value. |
| Semantic chunking library | Use Chonkie `SemanticChunker` with Azure OpenAI embeddings. Keep a thin adapter so we can swap to LangChain/LlamaIndex later if needed. |
| Package management | Use `uv`, `pyproject.toml`, and committed `uv.lock`. No `requirements.txt`. |
| Deployment | Docker container only. The app should not assume host-local execution. |
| Auth between systems | Use one shared `INGEST_API_KEY` header if exposed beyond private hackathon networking. |
| Async job tracking | Start synchronous; switch to FastAPI `BackgroundTasks` only if transcription or embedding timeout is a problem. |
| Duplicate input handling | Use deterministic point IDs `<input_id>:<chunk_index>` so retries overwrite the same chunks. |
| Qdrant deployment | Run Qdrant separately from the app container or use hosted Qdrant; configure via `QDRANT_URL`. |
| Full transcript storage | Store chunk text in every payload; store full transcript elsewhere later only if needed. |
| Processing-system labels | Not this repo's job. Preserve metadata and chunk IDs so labels can be attached later. |

## Build order

1. Add `pyproject.toml`, `uv.lock`, `Dockerfile`, `.dockerignore`, and `.env.example`.
2. Add FastAPI scaffold and `/health` endpoint.
3. Add `/ingest` request parsing and validation.
4. Add Azure embedding configuration and a small batched embedding test using mocked responses.
5. Add Chonkie semantic chunking adapter and unit tests for topic-boundary behavior.
6. Add Qdrant collection creation and batch upsert.
7. Add Gradium transcription adapter behind a narrow interface.
8. Wire `/ingest` end-to-end.
9. Build the Docker image.
10. Run a text-only smoke test against Qdrant.
11. Run an audio smoke test through Gradium against Qdrant.

## Definition of done

The hackathon MVP is done when:

- Docker image builds successfully with `uv sync --frozen`.
- Container starts and exposes `/health` and `POST /ingest`.
- `POST /ingest` accepts metadata plus text or audio.
- Audio-only requests are transcribed through configured Gradium API.
- Text is semantically chunked through Chonkie using Azure embedding similarity.
- Each final chunk is embedded with Azure-hosted `text-embedding-3-large`.
- Chunks are batch upserted into Qdrant with stable IDs and metadata payload.
- A text-only smoke test and an audio smoke test both produce searchable Qdrant points.

## Risks to watch

- **Gradium API contract unknown:** confirm exact endpoint, auth, file upload format, response schema, limits, and latency before implementation.
- **Gradium feature assumptions:** confirm whether timestamps, diarization, partial results, or language hints are available before planning around them.
- **Azure embedding dimension mismatch:** Qdrant collection size must match the actual embedding length returned by the Azure deployment.
- **Vector storage size:** full `text-embedding-3-large` vectors are 3072 dimensions, roughly 12 KB per vector in float32 before payload/index overhead.
- **Embedding cost/latency:** semantic chunking adds extra embedding calls before final chunk embedding; use batching and avoid re-embedding candidate windows outside Chonkie.
- **Chonkie/Azure integration fit:** confirm the Chonkie Azure embedding wrapper supports the exact Azure endpoint, deployment-name, dimensions, and API-version parameters we need. If not, implement a small custom Chonkie embedding adapter.
- **Threshold tuning:** Chonkie defaults may be good enough, but semantic break thresholds still need a few real transcripts to tune.
- **Docker secrets:** keep Azure, Gradium, and Qdrant credentials in environment variables; never bake them into the image.
- **Large audio files:** set a simple upload size limit if the endpoint is exposed publicly.
- **Over-design:** do not add user management, dashboards, label generation, or generation logic in this repo for the hackathon.
