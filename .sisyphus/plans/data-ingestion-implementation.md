# Apple Pie Data Ingestion Implementation

## TL;DR
> **Summary**: Build a minimal Dockerized FastAPI ingestion service that accepts text/audio + metadata, transcribes audio with Gradium REST, chunks transcripts with Chonkie semantic chunking backed by Azure OpenAI embeddings, and upserts one Qdrant point per semantic chunk.
> **Deliverables**: Docker/uv Python scaffold; `/health`; `POST /ingest`; Gradium adapter; Chonkie adapter; Azure embedding wrapper; Qdrant writer; unit + integration + container smoke tests.
> **Effort**: Medium
> **Parallel**: YES - 3 implementation waves + final verification
> **Critical Path**: Task 1 → Task 2 → Tasks 3/4/5 → Task 6 → Task 7 → Task 8

## Context
### Original Request
- Build the data-ingestion component for Apple Pie.
- Accept input-system objects containing audio, text, and metadata.
- Transcribe audio using **Gradium**, not Gradio.
- Perform proper semantic chunking using embeddings and a working dependency where possible.
- Use Azure-hosted `text-embedding-3-large`.
- Store text + vectors in Qdrant.
- Deploy only as a Docker container.
- Manage packages with `uv`.
- Keep it hackathon-minimal and highly likely to work.

### Interview Summary
- Repo is effectively empty except Git metadata and `README.md`; no existing app patterns to preserve.
- Runtime decision: Python 3.12, FastAPI, `uv`, Docker.
- API contract: one `POST /ingest` multipart endpoint with optional `audio`, optional `text`, required `metadata` JSON string.
- Source precedence: if `text` is supplied and non-empty, use it and skip transcription; if no text and audio exists, call Gradium.
- Semantic chunking dependency decision: use Chonkie `SemanticChunker` with Azure OpenAI embeddings; keep app code as a thin adapter.
- Vector database decision: Qdrant collection `apple_pie_story_chunks`, deterministic point IDs `<input_id>:<chunk_index>`.

### Metis Review (gaps addressed)
- Gradium public docs do not fully lock endpoint/auth/payload. Plan resolves this with a narrow config-driven `GradiumTranscriber` adapter and mocked tests; real smoke test is env-gated.
- Multipart + JSON metadata parsing is a likely failure point. Plan includes strict request contract tests.
- Avoid scope creep: no WebSocket, job queue, user management, dashboard, label generation, generic framework, or local model hosting.
- Verification must be executable by agents with concrete commands and expected outputs.

### Oracle Review (architecture constraints)
- Keep pipeline linear and synchronous for MVP.
- Use only two external boundary adapters: `GradiumTranscriber` and `QdrantVectorStore`; Azure/Chonkie stay internal service wrappers.
- Hard-fail invalid metadata, transcription errors, embedding errors, and Qdrant write errors; do not silently partially succeed.
- Deterministic IDs provide idempotency without adding a relational database.

## Work Objectives
### Core Objective
Create the smallest working ingestion service that can be built in Docker and prove text/audio inputs become semantic Qdrant points.

### Deliverables
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `.dockerignore`, `.env.example`.
- `app/` package with FastAPI app, config, schemas, transcription, chunking, embeddings, vector store, and ingestion orchestration.
- `tests/` suite covering validation, source selection, adapters with mocks, chunking contract, Qdrant point construction, and API behavior.
- `docker-compose.yml` for smoke testing app + Qdrant because repo is empty and Docker-only execution is required.

### Definition of Done (verifiable conditions with commands)
- `uv sync --frozen` succeeds after lockfile exists.
- `uv run pytest` passes.
- `docker build -t apple-pie-data-ingestion:local .` succeeds.
- Container starts with env vars and `/health` returns HTTP 200 JSON `{"status":"ok"}`.
- Text-only `POST /ingest` returns HTTP 200 with `input_id`, `status: indexed`, and `chunks > 0` using mocked external dependencies in tests.
- Qdrant writer tests prove deterministic IDs and payload keys.
- Real external smoke tests are env-gated and skipped unless `RUN_EXTERNAL_SMOKE=1` and provider credentials exist.

### Must Have
- Synchronous MVP pipeline.
- Strict request validation.
- Text-over-audio precedence.
- Gradium REST adapter isolated behind config.
- Chonkie semantic chunking adapter.
- Azure OpenAI embedding wrapper using deployment name.
- Qdrant collection creation/upsert with vector dimension guard.
- Docker-first runtime.

### Must NOT Have
- No Gradio or `gradio_client`.
- No local `sentence-transformers` model.
- No websocket transcription.
- No background job queue unless HTTP timeout becomes a verified blocker.
- No auth system beyond optional single `INGEST_API_KEY` header.
- No processing labels or generative output.
- No dashboard/UI.
- No secrets committed to repo.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-first for each boundary, then implementation.
- Framework: `pytest`, `httpx`/FastAPI test client, mock adapters; Docker CLI for container smoke.
- QA policy: Every task has agent-executed happy and failure scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`.

## Execution Strategy
### Parallel Execution Waves
Wave 1: Task 1 (scaffold/config), Task 2 (schemas/API shell)
Wave 2: Task 3 (Gradium adapter), Task 4 (Azure embeddings + Chonkie), Task 5 (Qdrant store)
Wave 3: Task 6 (orchestrator), Task 7 (Docker runtime), Task 8 (smoke tests/docs alignment)
Final: F1-F4 verification agents in parallel

### Dependency Matrix (full, all tasks)
- Task 1: blocks all implementation tasks.
- Task 2: depends Task 1; blocks Task 6.
- Task 3: depends Task 1; blocks Task 6.
- Task 4: depends Task 1; blocks Task 6.
- Task 5: depends Task 1; blocks Task 6.
- Task 6: depends Tasks 2, 3, 4, 5; blocks Tasks 7, 8.
- Task 7: depends Tasks 1, 6; blocks Task 8.
- Task 8: depends Tasks 6, 7.

### Agent Dispatch Summary
- Wave 1 → 2 tasks → quick / backend-patterns / python-testing
- Wave 2 → 3 tasks → deep / python-patterns / security-review for external providers
- Wave 3 → 3 tasks → unspecified-high / docker-patterns / python-testing
- Final → 4 review tasks → oracle, unspecified-high, unspecified-high, deep

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Scaffold Python/uv/Docker project foundation

  **What to do**: Create `pyproject.toml`, generate `uv.lock`, create `app/__init__.py`, `tests/`, `.env.example`, `.dockerignore`, and base `Dockerfile`. Use Python 3.12. Dependencies: `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic-settings`, `httpx`, `openai`, `chonkie[azure-openai]`, `qdrant-client`, `numpy`, `pytest`, `pytest-asyncio`, `ruff`. Configure `ruff` in `pyproject.toml`. Docker command must run `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`.
  **Must NOT do**: Do not add `requirements.txt`; do not add local ML model dependencies; do not bake secrets into Docker image.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: greenfield scaffold with concrete files.
  - Skills: [`python-patterns`, `docker-patterns`] - Reason: Python packaging and Docker runtime.
  - Omitted: [`frontend-design`] - No UI.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: Tasks 2-8 | Blocked By: none

  **References**:
  - Pattern: `README.md:214-274` - Docker and package management decisions.
  - API/Type: `README.md:243-255` - dependency list.
  - External: `uv` packaging conventions - use `pyproject.toml` + `uv.lock`.

  **Acceptance Criteria**:
  - [ ] `uv sync` exits 0 and creates/updates `uv.lock`.
  - [ ] `uv run python -c "import fastapi, httpx, qdrant_client, openai, chonkie"` exits 0.
  - [ ] `docker build -t apple-pie-data-ingestion:local .` exits 0.
  - [ ] `.env.example` contains every required env var listed in Tasks 3-5.

  **QA Scenarios**:
  ```
  Scenario: Docker image builds from clean checkout
    Tool: Bash
    Steps: docker build -t apple-pie-data-ingestion:local .
    Expected: command exits 0 and image exists locally
    Evidence: .sisyphus/evidence/task-1-scaffold-docker-build.txt

  Scenario: Missing lockfile or dependency mismatch fails fast
    Tool: Bash
    Steps: uv sync --frozen
    Expected: exits 0 after lockfile is committed; if it fails, executor must refresh lockfile and rerun
    Evidence: .sisyphus/evidence/task-1-uv-sync.txt
  ```

  **Commit**: YES | Message: `chore(scaffold): initialize dockerized uv service` | Files: `pyproject.toml`, `uv.lock`, `Dockerfile`, `.dockerignore`, `.env.example`, `app/__init__.py`, `tests/`

- [x] 2. Implement config, schemas, `/health`, and request validation

  **What to do**: Add `app/config.py`, `app/schemas.py`, `app/main.py`. Define `Settings` with env vars: `INGEST_API_KEY` optional, `MAX_AUDIO_BYTES=25000000`, `QDRANT_URL`, `QDRANT_COLLECTION=apple_pie_story_chunks`, `EMBEDDING_DIM=3072`, Azure vars, Gradium vars. Define `IngestMetadata` with `input_id: str`, `user_id: str`, `timestamp: datetime`. Implement `/health` returning `{"status":"ok"}`. Implement `POST /ingest` request parsing with multipart fields `metadata`, optional `text`, optional `audio`; inject an `IngestionService` dependency that tests override with a fake service returning `IngestResult(input_id="sample-input-001", status="indexed", chunks=1)`. Validation rules: metadata must be valid JSON; at least one of non-empty text or audio must exist; text wins over audio; audio size checked.
  **Must NOT do**: Do not call Gradium, Azure, Chonkie, or Qdrant in this task.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: API contract and validation.
  - Skills: [`python-patterns`, `python-testing`] - Reason: Pydantic/FastAPI tests.
  - Omitted: [`security-review`] - Only optional API key; deeper review in final.

  **Parallelization**: Can Parallel: PARTIAL | Wave 1 | Blocks: Task 6 | Blocked By: Task 1

  **References**:
  - Pattern: `README.md:46-68` - endpoint contract and validation rules.
  - API/Type: `README.md:148-157` - required settings.
  - Test: create `tests/test_ingest_contract.py`.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_ingest_contract.py` passes.
  - [ ] `/health` test returns status 200 and JSON `{"status":"ok"}`.
  - [ ] Missing metadata returns HTTP 422 or 400 with stable error field.
  - [ ] Invalid metadata JSON returns HTTP 400.
  - [ ] Missing both text and audio returns HTTP 400.
  - [ ] Text+audio request records source as text and does not require transcription dependency.

  **QA Scenarios**:
  ```
  Scenario: Text-only request validates
    Tool: Bash
    Steps: uv run pytest tests/test_ingest_contract.py -k text_only
    Expected: test passes; response has input_id sample-input-001, status indexed, chunks 1 from mocked service
    Evidence: .sisyphus/evidence/task-2-text-only-contract.txt

  Scenario: Missing content fails
    Tool: Bash
    Steps: uv run pytest tests/test_ingest_contract.py -k missing_text_and_audio
    Expected: test passes; HTTP 400 and message contains "text or audio required"
    Evidence: .sisyphus/evidence/task-2-validation-error.txt
  ```

  **Commit**: YES | Message: `feat(api): add ingest contract validation` | Files: `app/main.py`, `app/config.py`, `app/schemas.py`, `tests/test_ingest_contract.py`

- [x] 3. Implement Gradium REST transcription adapter

  **What to do**: Add `app/transcription.py` with `TranscriptionClient` protocol and `GradiumTranscriber`. Use `httpx.AsyncClient`. Config vars: `GRADIUM_API_BASE_URL`, `GRADIUM_API_KEY`, `GRADIUM_TRANSCRIPTION_PATH=/v1/audio/transcriptions`, `GRADIUM_TRANSCRIPTION_MODEL`, `GRADIUM_TRANSCRIPTION_TRANSPORT=rest`, `GRADIUM_TIMEOUT_SECONDS=60`. Default request: POST `{base_url}{path}` with bearer token, multipart `file`, form `model` if configured. Normalize responses by accepting `{"text":"..."}` first; if response shape differs, raise `TranscriptionError` with provider status/body excerpt. Add tests with `httpx.MockTransport` for success, non-2xx, missing text, timeout.
  **Must NOT do**: Do not implement WebSocket; do not assume timestamps/diarization; do not leak API key in logs/errors.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: external provider adapter and error semantics.
  - Skills: [`python-patterns`, `security-review`] - Reason: async HTTP and secret handling.
  - Omitted: [`fine-tuning-expert`] - No model training.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: Task 6 | Blocked By: Task 1

  **References**:
  - Pattern: `README.md:70-97` - Gradium REST adapter decisions and open contract risk.
  - API/Type: `app/config.py:Settings` from Task 2.
  - Test: create `tests/test_transcription.py`.
  - External: Gradium docs/account for final endpoint; tests must not require live Gradium.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_transcription.py` passes with mocked HTTP.
  - [ ] Success test returns exact transcript `hello apple pie` from mock JSON `{"text":"hello apple pie"}`.
  - [ ] Non-2xx mock raises `TranscriptionError` without API key in string representation.
  - [ ] Missing `text` field raises `TranscriptionError`.

  **QA Scenarios**:
  ```
  Scenario: Mocked Gradium transcript succeeds
    Tool: Bash
    Steps: uv run pytest tests/test_transcription.py -k success
    Expected: test passes; returned transcript equals "hello apple pie"
    Evidence: .sisyphus/evidence/task-3-gradium-success.txt

  Scenario: Provider failure is sanitized
    Tool: Bash
    Steps: uv run pytest tests/test_transcription.py -k provider_failure
    Expected: test passes; exception omits GRADIUM_API_KEY value
    Evidence: .sisyphus/evidence/task-3-gradium-failure.txt
  ```

  **Commit**: YES | Message: `feat(transcription): add gradium rest adapter` | Files: `app/transcription.py`, `tests/test_transcription.py`, `.env.example`

- [x] 4. Implement Azure embeddings and Chonkie semantic chunking adapter

  **What to do**: Add `app/embeddings.py` and `app/semantic_chunking.py`. Implement an `EmbeddingClient` wrapper for Azure OpenAI using `openai.AsyncAzureOpenAI` or compatible Azure client mode. API calls must use `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` as model value. Validate returned embedding length equals `EMBEDDING_DIM`; raise `EmbeddingDimensionError` otherwise. Implement `SemanticChunkerAdapter` around Chonkie `SemanticChunker` using Chonkie's Azure embedding integration if compatible; otherwise implement the smallest custom Chonkie embedding adapter that delegates to `EmbeddingClient`. Adapter returns `Chunk` records with `text`, `chunk_index`, and chunking metadata. Add tests with fake embeddings and deterministic text; do not call Azure in unit tests.
  **Must NOT do**: Do not hand-roll cosine breakpoint logic except as a tiny fallback inside tests; production adapter should use Chonkie. Do not add LangChain/LlamaIndex.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: library integration and dimension guard.
  - Skills: [`python-patterns`, `python-testing`] - Reason: adapter contracts and fake embeddings.
  - Omitted: [`vercel:ai-sdk`] - Python service, not Vercel AI SDK.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: Task 6 | Blocked By: Task 1

  **References**:
  - Pattern: `README.md:99-172` - Chonkie, Azure, dimensions.
  - External: Chonkie `SemanticChunker` and Azure embedding support.
  - Test: create `tests/test_semantic_chunking.py`, `tests/test_embeddings.py`.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_embeddings.py tests/test_semantic_chunking.py` passes.
  - [ ] Fake embedding batch test proves deployment name is passed as model value.
  - [ ] Dimension mismatch test raises `EmbeddingDimensionError`.
  - [ ] Semantic chunking test returns at least 2 chunks for a sample transcript with two obvious topics and preserves all source text content after whitespace normalization.

  **QA Scenarios**:
  ```
  Scenario: Semantic chunks preserve topic boundaries
    Tool: Bash
    Steps: uv run pytest tests/test_semantic_chunking.py -k topic_boundary
    Expected: test passes; chunks length >= 2 and chunk texts contain both "baking apples" and "vector database" in separate chunks or separate topic groups
    Evidence: .sisyphus/evidence/task-4-chonkie-topic-boundary.txt

  Scenario: Azure dimension mismatch fails before Qdrant
    Tool: Bash
    Steps: uv run pytest tests/test_embeddings.py -k dimension_mismatch
    Expected: test passes; error names expected 3072 and actual fake dimension
    Evidence: .sisyphus/evidence/task-4-dimension-mismatch.txt
  ```

  **Commit**: YES | Message: `feat(chunking): add chonkie azure semantic chunking` | Files: `app/embeddings.py`, `app/semantic_chunking.py`, `tests/test_embeddings.py`, `tests/test_semantic_chunking.py`, `.env.example`

- [x] 5. Implement Qdrant vector store adapter

  **What to do**: Add `app/vector_store.py`. Implement `VectorStoreClient` protocol and `QdrantVectorStore`. On startup or first use, ensure collection exists with `VectorParams(size=EMBEDDING_DIM, distance=Cosine)`. If collection exists with different vector size, raise `VectorStoreConfigurationError`. Build points with deterministic IDs `<input_id>:<chunk_index>`, vector, and payload keys: `input_id`, `user_id`, `timestamp`, `chunk_index`, `text`, `source`, `embedding_model`, `semantic_chunking`. Batch upsert points for one input.
  **Must NOT do**: Do not store secrets in payload; do not generate random UUIDs for chunk points; do not create multiple collections.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: external store contract and idempotency.
  - Skills: [`python-patterns`] - Reason: clean adapter and tests.
  - Omitted: [`supabase-postgres-best-practices`] - Qdrant only.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: Task 6 | Blocked By: Task 1

  **References**:
  - Pattern: `README.md:174-212` - collection and payload decisions.
  - API/Type: `app/schemas.py:IngestMetadata` from Task 2.
  - Test: create `tests/test_vector_store.py`.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_vector_store.py` passes.
  - [ ] Point construction test produces ID `sample-input-001:0`.
  - [ ] Payload contains all required keys and no secret-like keys.
  - [ ] Existing collection wrong dimension test raises configuration error.

  **QA Scenarios**:
  ```
  Scenario: Deterministic point construction
    Tool: Bash
    Steps: uv run pytest tests/test_vector_store.py -k deterministic_ids
    Expected: test passes; first point ID equals sample-input-001:0
    Evidence: .sisyphus/evidence/task-5-qdrant-ids.txt

  Scenario: Wrong collection dimension fails fast
    Tool: Bash
    Steps: uv run pytest tests/test_vector_store.py -k wrong_dimension
    Expected: test passes; error prevents upsert
    Evidence: .sisyphus/evidence/task-5-qdrant-dimension-error.txt
  ```

  **Commit**: YES | Message: `feat(vector-store): add qdrant upsert adapter` | Files: `app/vector_store.py`, `tests/test_vector_store.py`, `.env.example`

- [x] 6. Wire linear ingestion orchestration end-to-end

  **What to do**: Add `app/ingestion.py`. Implement `IngestionService.ingest(metadata, text, audio_file)` orchestration: choose transcript source; text wins if non-empty; audio path goes through `GradiumTranscriber`; normalize transcript; reject empty transcript; call `SemanticChunkerAdapter`; embed final chunks; call `QdrantVectorStore.upsert_chunks`; return `IngestResult(input_id, status="indexed", chunks=N)`. Update `app/main.py` to call service dependency. Map known errors to stable HTTP codes: validation 400, transcription 502, embedding 502, vector store 503, unexpected 500.
  **Must NOT do**: Do not swallow partial failures; do not return success if Qdrant upsert fails.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: integration across all adapters.
  - Skills: [`python-patterns`, `python-testing`] - Reason: orchestrator and API tests.
  - Omitted: [`deployment-patterns`] - Docker handled separately.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Tasks 7-8 | Blocked By: Tasks 2-5

  **References**:
  - Pattern: `README.md:29-42` - pipeline flow.
  - Pattern: Oracle review in plan Context - hard-fail semantics.
  - Test: create `tests/test_ingestion_service.py`, extend `tests/test_ingest_contract.py`.

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_ingestion_service.py tests/test_ingest_contract.py` passes.
  - [ ] Text+audio test proves transcriber mock is not called.
  - [ ] Audio-only test proves transcriber mock is called exactly once.
  - [ ] Qdrant failure returns HTTP 503 from API test.
  - [ ] Successful API response has JSON keys `input_id`, `status`, `chunks`.

  **QA Scenarios**:
  ```
  Scenario: Text wins over audio
    Tool: Bash
    Steps: uv run pytest tests/test_ingestion_service.py -k text_wins
    Expected: test passes; transcriber call count is 0 and source payload is text
    Evidence: .sisyphus/evidence/task-6-text-wins.txt

  Scenario: Qdrant write failure returns 503
    Tool: Bash
    Steps: uv run pytest tests/test_ingest_contract.py -k vector_store_failure
    Expected: test passes; HTTP status 503 and response error contains "vector_store_unavailable"
    Evidence: .sisyphus/evidence/task-6-qdrant-failure.txt
  ```

  **Commit**: YES | Message: `feat(ingestion): wire end-to-end ingest pipeline` | Files: `app/ingestion.py`, `app/main.py`, `tests/test_ingestion_service.py`, `tests/test_ingest_contract.py`

- [x] 7. Finalize container runtime and compose smoke stack

  **What to do**: Ensure Dockerfile runs the app as non-root by creating user `appuser`; add a Docker `HEALTHCHECK` that calls `/health` using Python stdlib or `curl` if installed. Add `docker-compose.yml` with services `app` and `qdrant` for local/container smoke only. Compose app env must use fake provider values by default and must not run external calls unless tests override. Add `scripts/smoke_text_ingest.sh` with the README text-only curl command. Ensure container does not rely on host-local Python packages.
  **Must NOT do**: Do not require real Azure/Gradium credentials for basic container startup; do not include secrets in compose.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: Docker runtime verification.
  - Skills: [`docker-patterns`] - Reason: container-first deployment.
  - Omitted: [`kubernetes-architect`] - No Kubernetes.

  **Parallelization**: Can Parallel: PARTIAL | Wave 3 | Blocks: Task 8 | Blocked By: Tasks 1, 6

  **References**:
  - Pattern: `README.md:214-274` - Docker-only requirement.
  - Test: Docker CLI smoke commands.

  **Acceptance Criteria**:
  - [ ] `docker build -t apple-pie-data-ingestion:local .` exits 0.
  - [ ] `docker run -d --name apple-pie-ingest-test -p 8000:8000 --env-file .env.example apple-pie-data-ingestion:local` starts far enough for `/health` when required env examples are accepted for health.
  - [ ] `curl -fsS http://localhost:8000/health` returns JSON containing `ok` while container is running.

  **QA Scenarios**:
  ```
  Scenario: Container health endpoint works
    Tool: Bash
    Steps: docker run -d --name apple-pie-ingest-test -p 8000:8000 --env-file .env.example apple-pie-data-ingestion:local; curl -fsS http://localhost:8000/health; docker rm -f apple-pie-ingest-test
    Expected: curl outputs {"status":"ok"}; cleanup removes container
    Evidence: .sisyphus/evidence/task-7-container-health.txt

  Scenario: Container does not expose secrets
    Tool: Bash
    Steps: docker history apple-pie-data-ingestion:local --no-trunc
    Expected: output does not contain AZURE_OPENAI_API_KEY or GRADIUM_API_KEY values
    Evidence: .sisyphus/evidence/task-7-docker-history.txt
  ```

  **Commit**: YES | Message: `chore(docker): finalize container smoke runtime` | Files: `Dockerfile`, `docker-compose.yml`, `.env.example`, `scripts/smoke_text_ingest.sh`, `README.md`

- [x] 8. Add smoke tests and README alignment for executor handoff

  **What to do**: Add `tests/test_external_smoke.py` with env-gated tests skipped unless `RUN_EXTERNAL_SMOKE=1`. Include text-only external smoke that requires Qdrant URL and Azure credentials but not Gradium. Include audio external smoke requiring `GRADIUM_*`, `AZURE_*`, and a generated tiny valid WAV fixture at `tests/fixtures/sample.wav`. Update README with exact commands: `uv run pytest`, Docker build, health curl, text ingest curl with metadata sample. Ensure README remains consistent with implementation.
  **Must NOT do**: Do not make default test suite depend on paid external APIs; do not commit real audio if licensing unclear; generated simple WAV is okay.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: end-to-end verification and docs sync.
  - Skills: [`python-testing`, `docker-patterns`, `writing`] - Reason: smoke tests and README commands.
  - Omitted: [`playwright-cli`] - No browser/UI.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification | Blocked By: Tasks 6-7

  **References**:
  - Pattern: `README.md:295-320` - build order and done criteria.
  - Test: new `tests/test_external_smoke.py`.

  **Acceptance Criteria**:
  - [ ] `uv run pytest` passes with external smoke skipped by default.
  - [ ] `RUN_EXTERNAL_SMOKE=0 uv run pytest tests/test_external_smoke.py` exits 0 with skip messages.
  - [ ] README includes text-only curl example using `sample-input-001`, `user-001`, and RFC3339 timestamp `2026-05-16T12:00:00Z`.
  - [ ] README includes explicit list of required env vars for real external smoke.

  **QA Scenarios**:
  ```
  Scenario: Full default test suite passes without external credentials
    Tool: Bash
    Steps: uv run pytest
    Expected: exits 0; external smoke tests are skipped, not failed
    Evidence: .sisyphus/evidence/task-8-default-tests.txt

  Scenario: README smoke command is copy-pasteable
    Tool: Bash
    Steps: grep -n "sample-input-001" README.md && grep -n "RUN_EXTERNAL_SMOKE" README.md
    Expected: both commands find documented examples
    Evidence: .sisyphus/evidence/task-8-readme-smoke-docs.txt
  ```

  **Commit**: YES | Message: `test(smoke): add container and external smoke coverage` | Files: `tests/test_external_smoke.py`, `tests/fixtures/sample.wav`, `README.md`, `scripts/smoke_text_ingest.sh`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
  - Verify every README and plan requirement is implemented or explicitly documented as env-gated.
  - Command evidence: `uv run pytest`, `docker build -t apple-pie-data-ingestion:local .`.
- [x] F2. Code Quality Review — unspecified-high
  - Review app boundaries, error handling, type hints, config safety, and no over-engineering.
  - Command evidence: `uv run ruff check .`, `uv run pytest`.
- [x] F3. Real Manual QA — unspecified-high
  - Run container `/health` and mocked/text-only API smoke commands; if real credentials are available, run `RUN_EXTERNAL_SMOKE=1` tests.
  - Evidence files under `.sisyphus/evidence/f3-*`.
- [x] F4. Scope Fidelity Check — deep
  - Confirm no Gradio, no local model weights, no WebSocket, no labels/generative system, no dashboard, no secrets.
  - Command evidence: `grep -R "gradio\|sentence-transformers\|websocket" app tests pyproject.toml README.md` with only allowed negative/docs references.

## Commit Strategy
- Commit after each task if tests for that task pass.
- Use conventional messages listed per task.
- Do not commit `.idea/` or secrets.
- Do not push unless explicitly requested.

## Success Criteria
- The repo contains a working Dockerized FastAPI ingestion MVP.
- The default test suite is deterministic and does not require external credentials.
- External provider integration points are isolated, configurable, and env-gated.
- Qdrant points are deterministic, correctly dimensioned, and metadata-rich.
- The implementation stays hackathon-minimal and avoids all out-of-scope systems.

## Defaults Applied
- Gradium default REST path: `/v1/audio/transcriptions`, configurable via `GRADIUM_TRANSCRIPTION_PATH` because exact account contract is pending.
- Text wins over audio when both are supplied.
- Endpoint remains synchronous for MVP.
- External smoke tests skip unless explicitly enabled.

## Decisions Needed
- None blocking for implementation. If the real Gradium API path/payload differs, update only `GradiumTranscriber` config/defaults and its mocked tests.
