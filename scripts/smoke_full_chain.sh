#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

ingest_api_url="${INGEST_API_URL:-${API_URL:-http://127.0.0.1:8000}}"
labeling_api_url="${LABELING_API_URL:-http://127.0.0.1:8001}"
provision_api_url="${PROVISION_API_URL:-http://127.0.0.1:8002}"
evidence_file="${EVIDENCE_FILE:-${repo_root}/.sisyphus/evidence/task-5-full-chain-smoke.txt}"
smoke_input_id="${SMOKE_INPUT_ID:-full-chain-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
smoke_user_id="${SMOKE_USER_ID:-smoke-user-001}"
smoke_timestamp="${SMOKE_TIMESTAMP:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
smoke_text="${SMOKE_TEXT:-Alice followed the rabbit hole into a bright hall. She found a tiny golden key on a glass table. Her sister laughed about the tea party the next morning.}"
podcast_label_override="${PODCAST_LABEL:-}"

mkdir -p "$(dirname -- "${evidence_file}")"
exec > >(tee -a "${evidence_file}") 2>&1

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

pass() {
  printf 'PASS %s\n' "$1"
}

fail() {
  printf 'FAIL %s\n' "$1"
  exit 1
}

extract_json_value() {
  local json_file="$1"
  local op="$2"
  local arg="${3:-}"

  # Named operations instead of eval. The previous version eval'd an arbitrary
  # expression with `os` exposed in its globals (ISSUES.md E33); every call site
  # only ever needed one of the handful of accessors below.
  python - "$json_file" "$op" "$arg" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
op = sys.argv[2]
arg = sys.argv[3]

if op == "get":
    value = payload.get(arg, "")
elif op == "get_or_empty":
    value = payload.get(arg, "") or ""
elif op == "get_int":
    value = payload.get(arg, 0)
elif op == "count":
    value = len(payload.get(arg, []))
elif op == "first_label":
    value = next(
        (point.get("label", "") for point in payload.get("points", []) if point.get("label")),
        "",
    )
elif op == "list_contains_id":
    value = any(item.get("id") == arg for item in payload)
else:
    raise SystemExit(f"unknown extract op: {op}")

if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

require_status() {
  local actual="$1"
  local expected="$2"
  local step="$3"

  if [[ "${actual}" != "${expected}" ]]; then
    fail "${step} returned HTTP ${actual}, expected ${expected}"
  fi
}

body_excerpt() {
  local body_file="$1"

  python - "$body_file" <<'PY'
from pathlib import Path
import sys

content = Path(sys.argv[1]).read_text(errors="replace").strip()
normalized = " ".join(content.split())
print(normalized[:240] + ("..." if len(normalized) > 240 else ""))
PY
}

printf '\n=== %s full-chain smoke ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'INGEST_API_URL=%s\n' "${ingest_api_url}"
printf 'LABELING_API_URL=%s\n' "${labeling_api_url}"
printf 'PROVISION_API_URL=%s\n' "${provision_api_url}"
printf 'EVIDENCE_FILE=%s\n' "${evidence_file}"
printf 'SMOKE_INPUT_ID=%s\n' "${smoke_input_id}"

export SMOKE_INPUT_ID="${smoke_input_id}"
export SMOKE_USER_ID="${smoke_user_id}"
export SMOKE_TIMESTAMP="${smoke_timestamp}"

metadata_json="$(python - <<'PY'
import json
import os

print(json.dumps({
    "input_id": os.environ["SMOKE_INPUT_ID"],
    "user_id": os.environ["SMOKE_USER_ID"],
    "timestamp": os.environ["SMOKE_TIMESTAMP"],
}))
PY
)"

ingest_body_file="${tmp_dir}/ingest.json"
# /ingest is authenticated (see ISSUES.md E10). Send the key when one is
# configured; a blank key means the service skips the check, so an empty
# header array is correct rather than a failure.
ingest_auth_header=()
if [[ -n "${INGEST_API_KEY:-}" ]]; then
  ingest_auth_header=(-H "x-api-key: ${INGEST_API_KEY}")
fi

ingest_status="$(curl -sS -o "${ingest_body_file}" -w '%{http_code}' \
  -X POST "${ingest_api_url}/ingest" \
  "${ingest_auth_header[@]}" \
  -F "text=${smoke_text}" \
  -F "metadata=${metadata_json}")"
require_status "${ingest_status}" "200" "POST /ingest"

ingest_input_id="$(extract_json_value "${ingest_body_file}" "get" "input_id")"
ingest_result_status="$(extract_json_value "${ingest_body_file}" "get" "status")"
ingest_chunks="$(extract_json_value "${ingest_body_file}" "get_int" "chunks")"

if [[ "${ingest_input_id}" != "${smoke_input_id}" ]]; then
  fail "POST /ingest returned input_id=${ingest_input_id}, expected ${smoke_input_id}"
fi

if [[ "${ingest_result_status}" != "indexed" ]]; then
  fail "POST /ingest returned status=${ingest_result_status}, expected indexed"
fi

if (( ingest_chunks < 1 )); then
  fail "POST /ingest returned chunks=${ingest_chunks}, expected at least 1"
fi

pass "POST /ingest indexed input_id=${ingest_input_id} with chunks=${ingest_chunks}"

labeling_body_file="${tmp_dir}/cluster-labels.json"
labeling_status="$(curl -sS -o "${labeling_body_file}" -w '%{http_code}' \
  -X POST "${labeling_api_url}/cluster-labels" \
  -H "x-ingest-input-id: ${smoke_input_id}" \
  -H "x-ingest-source: text")"

if [[ ! "${labeling_status}" =~ ^[1-5][0-9][0-9]$ ]]; then
  fail "POST /cluster-labels did not return an HTTP response code"
fi

if [[ "${labeling_status}" == "200" ]]; then
  labeling_result_status="$(extract_json_value "${labeling_body_file}" "get" "status")"
  labeling_points_read="$(extract_json_value "${labeling_body_file}" "get_int" "points_read")"
  labeling_points_updated="$(extract_json_value "${labeling_body_file}" "get_int" "points_updated")"
  labeling_clusters_found="$(extract_json_value "${labeling_body_file}" "get_int" "clusters_found")"

  if [[ -z "${labeling_result_status}" ]]; then
    fail "POST /cluster-labels returned no status field"
  fi

  pass "POST /cluster-labels evidence returned status=${labeling_result_status}, points_read=${labeling_points_read}, clusters_found=${labeling_clusters_found}, points_updated=${labeling_points_updated}"
else
  pass "POST /cluster-labels evidence reached the labeling service with HTTP ${labeling_status} body=$(body_excerpt "${labeling_body_file}")"
fi

universe_body_file="${tmp_dir}/universe.json"
universe_status="$(curl -sS -o "${universe_body_file}" -w '%{http_code}' "${provision_api_url}/universe")"
require_status "${universe_status}" "200" "GET /universe"

universe_point_count="$(extract_json_value "${universe_body_file}" "count" "points")"
universe_edge_count="$(extract_json_value "${universe_body_file}" "count" "edges")"
universe_first_label="$(extract_json_value "${universe_body_file}" "first_label")"

pass "GET /universe returned points=${universe_point_count} edges=${universe_edge_count}"

podcast_label="${podcast_label_override}"
if [[ -z "${podcast_label}" ]]; then
  if [[ -n "${universe_first_label}" ]]; then
    podcast_label="${universe_first_label}"
    printf 'INFO Using first /universe label for podcast smoke: %s\n' "${podcast_label}"
  else
    podcast_label="Synthetic Smoke"
    printf 'INFO /universe returned no labels; falling back to podcast label: %s\n' "${podcast_label}"
  fi
fi

export PODCAST_LABEL="${podcast_label}"

podcast_create_body_file="${tmp_dir}/podcast-create.json"
podcast_create_status="$(curl -sS -o "${podcast_create_body_file}" -w '%{http_code}' \
  -X POST "${provision_api_url}/podcasts" \
  -H 'Content-Type: application/json' \
  --data-binary "$(python - <<'PY'
import json
import os

print(json.dumps({"label": os.environ["PODCAST_LABEL"]}))
PY
)")"
require_status "${podcast_create_status}" "202" "POST /podcasts"

podcast_id="$(extract_json_value "${podcast_create_body_file}" "get" "id")"
podcast_status="$(extract_json_value "${podcast_create_body_file}" "get" "status")"
podcast_response_label="$(extract_json_value "${podcast_create_body_file}" "get" "label")"

if [[ -z "${podcast_id}" ]]; then
  fail "POST /podcasts returned no id"
fi

pass "POST /podcasts created id=${podcast_id} label=${podcast_response_label} status=${podcast_status}"


podcast_list_body_file="${tmp_dir}/podcasts.json"
podcast_list_status="$(curl -sS -o "${podcast_list_body_file}" -w '%{http_code}' "${provision_api_url}/podcasts")"
require_status "${podcast_list_status}" "200" "GET /podcasts"

podcast_in_list="$(extract_json_value "${podcast_list_body_file}" "list_contains_id" "${podcast_id}")"
if [[ "${podcast_in_list}" != "true" ]]; then
  fail "GET /podcasts did not include podcast id=${podcast_id}"
fi

pass "GET /podcasts included id=${podcast_id}"

podcast_detail_body_file="${tmp_dir}/podcast-detail.json"
podcast_detail_status_code="$(curl -sS -o "${podcast_detail_body_file}" -w '%{http_code}' "${provision_api_url}/podcasts/${podcast_id}")"
require_status "${podcast_detail_status_code}" "200" "GET /podcasts/{id}"

podcast_detail_id="$(extract_json_value "${podcast_detail_body_file}" "get" "id")"
podcast_detail_status="$(extract_json_value "${podcast_detail_body_file}" "get" "status")"
podcast_detail_error="$(extract_json_value "${podcast_detail_body_file}" "get_or_empty" "error")"

if [[ "${podcast_detail_id}" != "${podcast_id}" ]]; then
  fail "GET /podcasts/{id} returned id=${podcast_detail_id}, expected ${podcast_id}"
fi

# Poll to a terminal state rather than accepting whatever the first read shows.
podcast_deadline=$((SECONDS + ${PODCAST_WAIT_SECONDS:-300}))
while [[ "${podcast_detail_status}" == "pending" || "${podcast_detail_status}" == "running" ]]; do
  if (( SECONDS >= podcast_deadline )); then
    fail "GET /podcasts/{id} still ${podcast_detail_status} after ${PODCAST_WAIT_SECONDS:-300}s"
  fi
  sleep 5
  podcast_detail_status_code="$(curl -sS -o "${podcast_detail_body_file}" -w '%{http_code}' "${provision_api_url}/podcasts/${podcast_id}")"
  require_status "${podcast_detail_status_code}" "200" "GET /podcasts/{id}"
  podcast_detail_status="$(extract_json_value "${podcast_detail_body_file}" "get" "status")"
  podcast_detail_error="$(extract_json_value "${podcast_detail_body_file}" "get_or_empty" "error")"
done

# A failed podcast is a failed smoke run. Reporting it as a pass (which this
# script used to do) hid the fact that the whole leg was dead -- see ISSUES.md
# X12, E7 and E35.
if [[ "${podcast_detail_status}" != "completed" ]]; then
  fail "GET /podcasts/{id} ended status=${podcast_detail_status} error=${podcast_detail_error}"
fi

if [[ -n "${podcast_detail_error}" ]]; then
  fail "GET /podcasts/{id} completed but carried error=${podcast_detail_error}"
fi

podcast_audio_url="$(extract_json_value "${podcast_detail_body_file}" "get_or_empty" "audio_url")"
if [[ -z "${podcast_audio_url}" ]]; then
  fail "GET /podcasts/{id} completed without an audio_url"
fi

pass "GET /podcasts/{id} completed with audio_url=${podcast_audio_url}"

printf 'SMOKE RESULT full-chain path completed for podcast_id=%s\n' "${podcast_id}"
