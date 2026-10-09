# Deferred: direct-to-S3 audio upload with a presigned PUT

**Status:** deferred by decision, 2026-10-09. The 4 MB cap ships for now.

## Why this exists

`MAX_AUDIO_BYTES` is 4,000,000 in the AWS deployment, against the application's
own default of 25,000,000. That is a transport limit, not a preference:

- A Lambda function URL is a **synchronous** invocation.
- Lambda caps a synchronous request payload at **6 MB** (request and response
  each). Confirmed in the Lambda quotas documentation, under *Invocation
  payload*.
- Binary request bodies are **base64-encoded** into that payload, inflating them
  by about a third, and `POST /ingest` is a multipart form with more overhead on
  top.

So roughly **4.5 MB of audio** is the ceiling before Lambda rejects the request
with a 413 that never reaches this service. 4 MB is set deliberately *below* it
so an oversized upload gets this service's own clear error instead of an opaque
gateway failure.

Raising `MAX_AUDIO_BYTES` past ~4.5 MB does not work. The limit is upstream of
the application.

## What 4 MB buys

Depends entirely on what the client records:

| Encoding | Approx. duration in 4 MB |
| --- | --- |
| 16-bit PCM WAV, 16 kHz mono (what `tests/fixtures/sample.wav` is) | ~2 minutes |
| 16-bit PCM WAV, 44.1 kHz mono | ~45 seconds |
| Opus / WebM at 32 kbps | ~16 minutes |
| AAC / m4a at 128 kbps | ~4 minutes |

`applepie-ui` records through the browser's MediaRecorder on web, which produces
compressed WebM/Opus, so the practical ceiling is minutes rather than seconds.
**This has not been measured against a real recording** -- if users start hitting
the cap, measure before building, because the answer may simply be a codec
setting.

## The design, when it is built

Replace the single multipart `POST /ingest` with three steps:

1. **`POST /ingest/upload-url`** on data-ingestion. Takes the `input_id` and the
   content type, returns a presigned S3 PUT URL plus the object key it will
   write to. The key must come from the server, built by the existing
   `_build_object_key` -- never from the client, or the `input_id` sanitising
   that stops a client escaping its prefix is bypassed.
2. **The client PUTs the audio straight to S3.** No Lambda in the path, so no
   6 MB limit. The presigned URL should be short-lived (minutes) and should pin
   the content type, so a URL minted for audio cannot be used to upload
   something else.
3. **`POST /ingest`** with the object key instead of the bytes. The service
   reads the object from S3 for transcription rather than receiving it.

### Consequences to handle

- **IAM.** data-ingestion's role currently has `s3:PutObject` and
  `s3:DeleteObject` on the audio bucket and deliberately *no* `s3:GetObject`.
  Step 3 needs `GetObject`, and the presigned PUT in step 1 is signed with the
  same role, so the grant stays `PutObject` for that. See
  `infrastructure/aws/iam.tf`, `WriteIngestedAudio`.
- **Orphaned objects.** A client that uploads and then never calls `/ingest`
  leaves an object nothing references. The audio bucket has no lifecycle rule
  for this; add one expiring objects under an `incoming/` prefix, or have the
  ingest step move the object into its final key.
- **The rollback path.** `ingestion.py` deletes the stored audio when indexing
  fails. That still works, but the object now exists *before* ingest runs, so
  "delete what we stored" and "delete what the client uploaded" become the same
  question and need deciding explicitly.
- **CORS on the bucket.** A browser PUT straight to S3 needs a CORS
  configuration on the audio bucket, which it does not have today. That is the
  one piece that makes the bucket's configuration browser-facing, so scope the
  allowed origin to the CloudFront domain.
- **The UI.** `src/features/recording/ingest-client.ts` posts a multipart form.
  It would need the three-step flow, with progress reporting now possible
  because the upload is a plain PUT.
- **`MAX_AUDIO_BYTES` keeps a job.** It stops being a transport ceiling and
  becomes a real product limit, enforced when minting the presigned URL rather
  than when receiving bytes.

## Related

- `infrastructure/aws/README.md`, "Audio uploads are capped by the transport,
  not by preference".
- `infrastructure/aws/variables.tf`, `max_audio_bytes`.
