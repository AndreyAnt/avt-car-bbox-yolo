#!/bin/zsh

emulate -L zsh
set -euo pipefail

script_dir=${0:A:h}
env_file="$script_dir/.env"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  print "Usage: zsh findCar.sh /path/to/image.jpg [output_path.jpg]" >&2
  exit 1
fi

if [[ ! -f "$env_file" ]]; then
  print "Missing $env_file" >&2
  exit 1
fi

set -a
source "$env_file"
set +a

if [[ -z "${RUNPOD_API_KEY:-}" || "$RUNPOD_API_KEY" == "YOUR_RUNPOD_API_KEY_HERE" ]]; then
  print "Set RUNPOD_API_KEY in $env_file before running this script." >&2
  exit 1
fi

if [[ -z "${RUNPOD_BASE_URL:-}" ]]; then
  print "Set RUNPOD_BASE_URL in $env_file before running this script." >&2
  exit 1
fi

input_image=${1:A}

if [[ ! -f "$input_image" ]]; then
  print "Input image not found: $input_image" >&2
  exit 1
fi

base_url=${RUNPOD_BASE_URL%/}
timeout_seconds=${RUNPOD_WARMUP_TIMEOUT_SECONDS:-180}
poll_interval=${RUNPOD_POLL_INTERVAL_SECONDS:-2}
ping_max_time=${RUNPOD_PING_MAX_TIME_SECONDS:-10}
candidate_limit=${RUNPOD_CANDIDATE_LIMIT:-10}
draw_suggestions=${RUNPOD_DRAW_SUGGESTIONS:-false}
max_det=${RUNPOD_MAX_DET:-300}
roll_verify=${RUNPOD_ROLL_VERIFY:-true}

input_filename=${input_image:t}
input_stem=${input_filename:r}
output_path=${2:-"$PWD/${input_stem}_output.jpg"}
output_path=${output_path:A}
json_output_path=${RUNPOD_JSON_OUTPUT_PATH:-"$PWD/${input_stem}_response.json"}
json_output_path=${json_output_path:A}

tmp_json=$(mktemp)
cleanup() {
  rm -f "$tmp_json"
}
trap cleanup EXIT

print "Waiting for serverless worker readiness at $base_url/ping"

start_time=$SECONDS
attempt=1
while true; do
  print "Ping attempt $attempt: sending /ping request (max ${ping_max_time}s)"
  attempt_start=$SECONDS
  ping_code=$(curl --http1.1 -sS -H "Expect:" \
    --connect-timeout 5 \
    --max-time "$ping_max_time" \
    -o /dev/null \
    -w "%{http_code}" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "$base_url/ping" || true)
  elapsed=$((SECONDS - start_time))
  attempt_elapsed=$((SECONDS - attempt_start))

  if [[ -z "$ping_code" ]]; then
    ping_code="curl_timeout_or_error"
  fi

  print "Ping attempt $attempt: status=$ping_code attempt_elapsed=${attempt_elapsed}s total_elapsed=${elapsed}s"

  if [[ "$ping_code" == "200" ]]; then
    print "Worker is ready after ${elapsed}s."
    break
  fi

  if (( SECONDS - start_time >= timeout_seconds )); then
    print "Timed out waiting for /ping to return 200. Last status: ${ping_code:-none}" >&2
    exit 1
  fi

  print "Waiting ${poll_interval}s before retrying /ping..."
  sleep "$poll_interval"
  ((attempt++))
done

print "Worker is ready. Sending $input_filename for detection."
print "Detection options: candidate_limit=$candidate_limit draw_suggestions=$draw_suggestions max_det=$max_det roll_verify=$roll_verify"

detect_url="$base_url/detect?draw_output=true&candidate_limit=$candidate_limit&draw_suggestions=$draw_suggestions&max_det=$max_det&roll_verify=$roll_verify"

curl_metrics=$(
  curl --http1.1 -sS -H "Expect:" \
    -o "$tmp_json" \
    -w "code=%{http_code} starttransfer=%{time_starttransfer}s total=%{time_total}s size_download=%{size_download}B" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -X POST "$detect_url" \
    -F "file=@$input_image"
)

print "$curl_metrics"

http_code=${${curl_metrics#code=}%% *}
if [[ "$http_code" != "200" ]]; then
  print "Detection request failed. Response body:" >&2
  cat "$tmp_json" >&2 || true
  exit 1
fi

python3 - "$tmp_json" "$output_path" <<'PY'
import base64
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

payload = json.loads(json_path.read_text())

image_b64 = payload.get("output_image_base64")
if not image_b64:
    detail = payload.get("detail") or payload
    raise SystemExit(f"Missing output_image_base64 in response: {detail}")

output_path.write_bytes(base64.b64decode(image_b64))
print(output_path)
PY

print "Saved output image to $output_path"
cp "$tmp_json" "$json_output_path"
print "Saved response JSON to $json_output_path"
