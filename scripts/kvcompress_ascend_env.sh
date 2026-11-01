#!/usr/bin/env bash
# Launch with the persistent, pinned HUST source stack and plugin overlay.
set -euo pipefail
if (( $# == 0 )); then
  echo "usage: $0 COMMAND [ARGS...]" >&2
  exit 2
fi
kv_workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
kv_prefix=${KVCOMPRESS_HOST_ENV:-$kv_workspace/.envs/kvcompress-host}
if [[ ! -x "$kv_prefix/bin/python" || ! -f "$kv_prefix/kvcompress-host.json" ]]; then
  echo "Run scripts/kvcompress_install_env.sh first: $kv_prefix" >&2
  exit 1
fi
mapfile -t kv_paths < <("$kv_prefix/bin/python" -I - "$kv_prefix/kvcompress-host.json" <<'PY'
import json,sys
with open(sys.argv[1]) as stream:
    config=json.load(stream)
for key in ('cann_env','atb_env','triton_source','vendor_env'):
    print(config[key])
PY
)
[[ ${#kv_paths[@]} == 4 ]] || { echo 'Invalid host environment config' >&2; exit 1; }
kv_cann=${KVCOMPRESS_CANN_ENV:-${kv_paths[0]}}
kv_atb=${KVCOMPRESS_ATB_ENV:-${kv_paths[1]}}
kv_triton=${KVCOMPRESS_TRITON_SOURCE:-${kv_paths[2]}}
kv_vendor=${KVCOMPRESS_HOST_VENDOR_ENV:-${kv_paths[3]}}
if [[ ! -f "$kv_cann" || ! -f "$kv_atb" || ! -d "$kv_triton/triton/backends/ascend" || ! -f "$kv_vendor" ]]; then
  echo 'Incomplete HUST sources or CANN/ATB/vendor setup; rerun the installer.' >&2
  exit 1
fi
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD CMAKE_PREFIX_PATH
unset ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH
unset ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="$kv_prefix/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin"
export PYTHONPATH="$kv_triton"
export PYTHONNOUSERSITE=1
export TRITON_HOME="${TRITON_HOME:-$kv_workspace/.cache/kvcompress-triton-home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$kv_workspace/.cache/kvcompress-triton-runtime}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$kv_workspace/.cache/kvcompress-vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$kv_workspace/.cache/kvcompress-torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$kv_workspace/.cache/kvcompress-xdg}"
mkdir -p "$TRITON_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
# Determine the ABI without loading device backends on a CPU-only workspace.
kv_torch_abi=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$kv_prefix/bin/python" -c \
  'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
set +u
source "$kv_cann"
source "$kv_atb" --cxx_abi="$kv_torch_abi"
source "$kv_vendor"
set -u
export LD_LIBRARY_PATH="$kv_prefix/lib:${LD_LIBRARY_PATH:-}"
exec "$@"
