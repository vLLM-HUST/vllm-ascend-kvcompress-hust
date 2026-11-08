#!/usr/bin/env bash
# Install the plugin overlay on an existing persistent HUST Python base.
set -euo pipefail
kv_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
kv_python=${KVCOMPRESS_BASE_PYTHON:-python3}
if ! command -v "$kv_python" >/dev/null 2>&1; then
  echo 'Set KVCOMPRESS_BASE_PYTHON to the existing Python 3.11 HUST base.' >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME BASH_ENV ENV LD_LIBRARY_PATH LD_PRELOAD CMAKE_PREFIX_PATH
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "$kv_python" "$kv_repo/scripts/env/install_b1.py" "$@"
