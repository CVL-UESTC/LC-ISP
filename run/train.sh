#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run/train.sh DATA_ROOT [training options...]" >&2
    exit 2
fi
DATA_ROOT="$1"
shift
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "$NUM_GPUS" == 1 ]]; then
    exec "$PYTHON_BIN" train.py --dataroot "$DATA_ROOT" "$@"
else
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone \
        --nproc_per_node="$NUM_GPUS" train.py --dataroot "$DATA_ROOT" "$@"
fi
