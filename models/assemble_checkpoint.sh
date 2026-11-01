#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTS_DIR="$SCRIPT_DIR/parts"
OUTPUT="$SCRIPT_DIR/aaa_yolo26s_best.pt"
EXPECTED_SHA256="486c8599d8558b7826b9361f19b60bb67d3daf911f1d518436f2726c6bd01643"
EXPECTED_BYTES="23386909"

if ! find "$PARTS_DIR" -maxdepth 1 -type f -name 'aaa_yolo26s_best.pt.part-*' -print -quit | grep -q .; then
  echo "No checkpoint parts found in $PARTS_DIR" >&2
  exit 1
fi

tmp_file="${OUTPUT}.tmp"
trap 'rm -f "$tmp_file"' EXIT

LC_ALL=C find "$PARTS_DIR" -maxdepth 1 -type f \
  -name 'aaa_yolo26s_best.pt.part-*' -print0 \
  | sort -z \
  | xargs -0 cat > "$tmp_file"

actual_bytes="$(wc -c < "$tmp_file" | tr -d ' ')"
if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
  echo "Checkpoint size mismatch: $actual_bytes != $EXPECTED_BYTES" >&2
  exit 1
fi

actual_sha256="$(sha256sum "$tmp_file" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "Checkpoint SHA-256 mismatch: $actual_sha256" >&2
  exit 1
fi

mv "$tmp_file" "$OUTPUT"
trap - EXIT
echo "Verified checkpoint: $OUTPUT"

