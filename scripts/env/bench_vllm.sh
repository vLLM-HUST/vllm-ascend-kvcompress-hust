#!/usr/bin/env bash
# One B1 client executable is used against BOTH servers; no compression client-side.
set -euo pipefail
kv_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
unset VLLM_ASCEND_KVCOMPRESS_CONFIG VLLM_HUST_EXT_CONFIG
export VLLM_ASCEND_KVCOMPRESS_ENABLED=0
export VLLM_PLUGINS=ascend
exec bash "$kv_dir/ascend_env.sh" python -m vllm.entrypoints.cli.main "$@"
