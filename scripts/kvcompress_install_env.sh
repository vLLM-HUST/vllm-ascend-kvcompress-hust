#!/usr/bin/env bash
# Persistent plugin environment on the pinned, already-built HUST base.
set -euo pipefail
kv_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
kv_workspace=$(dirname "$kv_repo")
kv_python=${KVCOMPRESS_BASE_PYTHON:-python3.11}
# Select this workspace's durable base when available; other machines may set
# KVCOMPRESS_BASE_PYTHON explicitly. No plugin-version-dependent paths.
if [[ -z ${KVCOMPRESS_BASE_PYTHON:-} && -x "$kv_workspace/../miniconda3/envs/vllm-hust-dev/bin/python" ]]; then
  kv_python="$kv_workspace/../miniconda3/envs/vllm-hust-dev/bin/python"
fi
unset PYTHONPATH
export PYTHONNOUSERSITE=1
exec "$kv_python" "$kv_repo/scripts/kvcompress_install_env.py" "$@"
