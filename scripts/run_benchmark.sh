#!/usr/bin/env bash
set -euo pipefail
kv_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
unset PYTHONPATH
export PYTHONNOUSERSITE=1
exec "${KVCOMPRESS_BENCH_PYTHON:-python3}" "$kv_repo/scripts/benchmark/cli.py" "$@"
