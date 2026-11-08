#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV ENV
kv_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ge 2 && ( $1 == install || $1 == check ) && ( $2 == b0 || $2 == b1 ) ]]; then
  kv_action=$1
  shift 2
  if [[ $kv_action == check ]]; then
    exec bash "$kv_repo/scripts/env/install_b1.sh" --check "$@"
  fi
  exec bash "$kv_repo/scripts/env/install_b1.sh" "$@"
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
kv_runner=${KVCOMPRESS_ENV_PYTHON:-}
if [[ -z "$kv_runner" ]]; then
  kv_runner=python3
  kv_host=${KVCOMPRESS_HOST_ENV:-$(dirname "$kv_repo")/.envs/kvcompress-host}
  if [[ $# -ge 2 && ( $2 == b0 || $2 == b1 ) && -x "$kv_host/bin/python" ]]; then
    kv_runner="$kv_host/bin/python"
  fi
fi
exec "$kv_runner" "$kv_repo/scripts/env/manage.py" "$@"
