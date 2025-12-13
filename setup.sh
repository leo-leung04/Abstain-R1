#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NLP_PROJECT_ROOT="$ROOT_DIR"
export NLP_PROJECT_DATA="$NLP_PROJECT_ROOT/data"
export NLP_PROJECT_MODELS="${NLP_PROJECT_MODELS:-/workspace/models}"
export NLP_PROJECT_OUTPUT="$NLP_PROJECT_ROOT/results"
export PYTHONPATH="$NLP_PROJECT_ROOT:${PYTHONPATH:-}"

export PATH="$NLP_PROJECT_ROOT/bin:$PATH"
export OPENAI_API_KEY="" # TODO: add your openai api key

mkdir -p "$NLP_PROJECT_DATA" "$NLP_PROJECT_MODELS" "$NLP_PROJECT_OUTPUT"

echo "Project root: $NLP_PROJECT_ROOT"
echo "Data dir:     $NLP_PROJECT_DATA"
echo "Models dir:   $NLP_PROJECT_MODELS"
echo "Output dir:   $NLP_PROJECT_OUTPUT"

