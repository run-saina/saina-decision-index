#!/usr/bin/env bash
# Run one Saina model revision through the full Decision Index 0.3 suite and score it.
#
#   MODEL     Hub repo                 (default run-saina/saina-helm-0.8b)
#   REVISION  40-char Hub commit       (default: the pinned release below)
#   DEVICE    torch device             (default cuda)
#   LIMIT     first N requests only    (smoke tests; the score is then not complete)
#   UPLOAD    Hub dataset for results  (e.g. run-saina/saina-decision-index-results)
#   PUBLIC=1  make a newly created UPLOAD dataset public
#
# Output: runs/<model>/<revision>/ — results.jsonl, benchmark-summary.json, index.json,
# scores.json and environment.json (engine provenance, including this repo's commit).
# Interrupted runs resume from results.jsonl; rows that errored are retried.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-run-saina/saina-helm-0.8b}
REVISION=${REVISION:-b85231b8ca2f4f66cfd2e177d59635515474c736}
DEVICE=${DEVICE:-cuda}
OUT="runs/$MODEL/$REVISION"
mkdir -p "$OUT"

# This repo's commit (suffixed -dirty for uncommitted changes) lands in environment.json.
COMMIT=$(git describe --always --dirty --abbrev=40 2>/dev/null || echo unknown)

args=(pipeline --engine saina_decision_index:SainaEngine
      --option "model=$MODEL" --option "revision=\"$REVISION\"" --option "device=$DEVICE"
      --option "repo_commit=\"$COMMIT\""
      --out "$OUT")
[ -n "${LIMIT:-}" ] && args+=(--limit "$LIMIT")
[ -n "${UPLOAD:-}" ] && args+=(--upload "$UPLOAD" --upload-path "runs/$MODEL/$REVISION")
[ "${PUBLIC:-0}" = 1 ] && args+=(--public)
python -m decision_index "${args[@]}"
