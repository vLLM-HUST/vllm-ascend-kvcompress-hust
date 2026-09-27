#!/usr/bin/env bash
# Launch a command with the isolated, matched CANN/PyTorch host stack.
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 COMMAND [ARGS...]" >&2
  exit 2
fi

kv_workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
kv_prefix=${KVCOMPRESS_HOST_ENV:-$kv_workspace/.envs/kvcompress-0.8-host}
kv_cann=$kv_prefix/Ascend/cann-9.1.0/set_env.sh
kv_atb=$kv_prefix/Ascend/nnal/atb/set_env.sh
kv_vendor_env=${KVCOMPRESS_HOST_VENDOR_ENV:-$kv_workspace/vllm-ascend-hust/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash}

if [[ ! -x "$kv_prefix/bin/python" || ! -f "$kv_cann" || ! -f "$kv_atb" ]]; then
  echo "incomplete isolated Ascend environment: $kv_prefix" >&2
  exit 1
fi

# The interactive development shell still points to CANN 9.0.1 and the old
# Python environment. Do not let either leak into a 9.1/2.13 test process.
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD CMAKE_PREFIX_PATH
unset ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH
unset ATB_HOME_PATH CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="$kv_prefix/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export TRITON_HOME="${TRITON_HOME:-$kv_workspace/.cache/kvcompress-triton-home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$kv_workspace/.cache/kvcompress-triton-runtime}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$kv_workspace/.cache/kvcompress-vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$kv_workspace/.cache/kvcompress-torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$kv_workspace/.cache/kvcompress-xdg}"
mkdir -p "$TRITON_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# Vendor environment scripts read optional unset variables, so disable nounset
# only while sourcing them.
set +u
source "$kv_cann"
# The matched PyTorch 2.13 CPU wheel reports CXX11 ABI=true. Passing it here
# avoids the vendor script importing torch merely to discover the ABI.
source "$kv_atb" --cxx_abi=1
if [[ -f "$kv_vendor_env" ]]; then
  # The host's own build installs standard AscendC kernels here. Its vendor
  # environment must be visible before Python loads the ACL operator resolver.
  source "$kv_vendor_env"
fi
set -u
export LD_LIBRARY_PATH="$kv_prefix/lib:$LD_LIBRARY_PATH"
exec "$@"
