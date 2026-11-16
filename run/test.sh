#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run/test.sh DATA_ROOT CHECKPOINT [evaluation options...]" >&2
    exit 2
fi
DATA_ROOT="$1"
CHECKPOINT="$2"
shift 2
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec "${PYTHON_BIN:-python}" test.py --dataroot "$DATA_ROOT" --ckpt "$CHECKPOINT" "$@"
