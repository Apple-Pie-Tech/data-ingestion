#!/usr/bin/env bash

set -euo pipefail

api_url="${API_URL:-http://localhost:8000}"

curl -fsS \
  -X POST "${api_url}/ingest" \
  -F 'text=Alice followed the rabbit hole into a bright hall. She found a tiny golden key on a glass table.' \
  -F 'metadata={"input_id":"sample-input-001","user_id":"user-001","timestamp":"2026-05-16T12:00:00Z"}'
