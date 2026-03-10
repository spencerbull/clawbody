#!/usr/bin/env bash
# init_models.sh — download GGUF model files required by the llama.cpp service.
#
# Downloads only the two files needed (model + mmproj) from HuggingFace into
# ~/models so they can be mounted read-only into the container.
#
# Usage:
#   ./init_models.sh
#
# Re-running is safe — huggingface-cli skips files that are already present
# and match their expected checksums.
#
# If the repo is private or gated, set HF_TOKEN before running:
#   HF_TOKEN=hf_... ./init_models.sh

set -euo pipefail

REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
MODEL_FILE="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
MMPROJ_FILE="mmproj-F16.gguf"
LOCAL_DIR="${HOME}/models/${REPO}"

echo "Downloading ${REPO} files to ${LOCAL_DIR} ..."
mkdir -p "${LOCAL_DIR}"

hf download "${REPO}" \
  "${MODEL_FILE}" \
  "${MMPROJ_FILE}" \
  --local-dir "${LOCAL_DIR}"

echo ""
echo "Done. Files written to:"
echo "  ${LOCAL_DIR}/${MODEL_FILE}"
echo "  ${LOCAL_DIR}/${MMPROJ_FILE}"
echo ""
echo "Set LOCAL_MODEL_PATH=${HOME}/models in your .envrc and start the stack:"
echo "  docker compose up model"
