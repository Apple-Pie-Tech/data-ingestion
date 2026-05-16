# Apple Pie Data Ingestion — Agent State

## Repo purpose
- FastAPI service that accepts `text` or `audio` plus `metadata` at `POST /ingest`.
- Audio is transcribed with Gradium, text is chunked semantically, embeddings are generated with Azure OpenAI, and chunks are upserted into Qdrant.

## Current state
- Local and deployed text ingest works.
- Local and deployed WAV ingest works for the tested sample file.
- Koyeb deployment verified healthy after the latest fixes.
- Qdrant upserts verified end-to-end for both text and audio flows.

## Latest fixes
- `4aceb8e` `fix qdrant ids`
  - Qdrant point IDs were changed from `<input_id>:<chunk_index>` strings to deterministic UUIDv5 values because the target Qdrant instance rejected the old string IDs.
- `4f51d52` `fix gradium wav`
  - Gradium transcription adapter now normalizes legacy `/v1/audio/transcriptions` to `/post/speech/asr`.
  - WAV uploads are converted to mono `pcm_16000` before the Gradium REST call.
  - `.env.example` now points `GRADIUM_API_BASE_URL` at `https://api.gradium.ai/api`.

## Verified deployed result
- Deployed app URL: `https://misleading-dotty-trigub-tech-89f74bab.koyeb.app`
- Verified Koyeb deployment: `9afd41d6-221f-49e5-bcc6-9ceb3363c905`
- Verified WAV ingest response:
  - `{"input_id":"koyeb-audio-1778942112","status":"indexed","chunks":3}`
- Verified Qdrant retrieval for that ingest:
  - 3 points retrieved
  - payload `input_id = koyeb-audio-1778942112`
  - payload `source = audio`
  - chunk indices `0`, `1`, `2`

## Important implementation notes
- The README is partially stale.
  - It still describes Qdrant point IDs as `<input_id>:<chunk_index>`.
  - Actual implementation now uses deterministic UUIDv5 IDs derived from that string.
- Gradium REST behavior for this repo is currently:
  - send raw request body, not multipart
  - use `x-api-key`
  - use `/post/speech/asr`
  - for WAV input, convert to `audio/pcm` and set `input_format=pcm_16000`
- The app boundary remains multipart/form-data; only the Gradium adapter uses raw request bodies.

## Known caveats
- Do **not** assume every `.wav` file will work.
  - Standard PCM WAVs are the supported expectation.
  - Compressed WAV variants or float WAVs may still fail.
- Resampling to 16 kHz is now part of the compatibility path.
  - This solved the verified Gradium failure for the tested file.
  - If transcription quality issues appear on other audio shapes, inspect this path first.
- `INGEST_API_KEY` exists in config/env, but route enforcement may still be absent unless added elsewhere later.

## High-signal test commands
- Local unit/integration checks:
  - `uv run pytest tests/test_transcription.py -q`
  - `uv run pytest tests/test_ingestion_service.py tests/test_ingest_contract.py -q`
- Local app run:
  - `uv run uvicorn app.main:app --host 127.0.0.1 --port 8001`
- Local WAV ingest shape:
  - `curl -X POST http://127.0.0.1:8001/ingest -F 'metadata={...}' -F 'audio=@speech_FpoHBz8vLyg.wav;type=audio/wav'`

## If this breaks again
1. Reproduce through `/ingest`, not by calling helpers directly.
2. Verify the Gradium adapter request path, content type, and `input_format`.
3. If Qdrant indexing fails, check deterministic UUID generation first.
4. Confirm the deployed app is actually on the expected Koyeb deployment before debugging runtime behavior.
