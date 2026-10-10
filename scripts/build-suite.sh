#!/usr/bin/env bash
# Build the Decision Index 0.3 suite from its pinned public sources (~7 GB download,
# ~17 GB working space). The suite is not redistributable, so every runner builds it once.
# Prerequisites: `hf auth login`, and the HLE terms accepted at
# https://huggingface.co/datasets/cais/hle
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-work}
export HF_HUB_DISABLE_XET=1

python -m decision_index suite rebuild --work "$WORK"
python -m decision_index suite import \
    --rows "$WORK/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" \
    --added-rows "$WORK/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz" \
    --gsm8k-rows "$WORK/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz"
# Refuses any file whose hash differs from the board's manifest.
python -m decision_index suite verify
