# Current Host Environment Installation

English | [简体中文](environment-installation.zh.md)

This note records the installation issue observed after running dev-hub
`quickstart.sh` option 5 on the 2026-09-11 snapshots. It does not modify the
dev-hub, vLLM-HUST, vLLM-Ascend-HUST, or Triton-Ascend repositories.

## Symptom and cause

Distribution metadata reported `triton-ascend 3.6.0+git8f0a4de8`, but importing
`triton` failed with:

```text
ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'
```

The installed editable build had used the community `setup.py` path and its
CMake cache showed an empty `TRITON_PLUGIN_DIRS`; the resulting
`libtriton.so` included the generic bindings but not the Ascend Python module.
For this source generation, `TRITON_CODEGEN_BACKENDS=ascend pip install -e .`
is not equivalent to the repository's `setup_ascend.py` build entry point.

## Non-mutating recovery

Build from a temporary copy because `setup_ascend.py` applies the repository's
Ascend patch set to its working tree. This keeps the shared checkout unchanged:

```bash
BUILD_DIR="$(mktemp -d /tmp/triton-ascend-build.XXXXXX)"
cp -a /workspace/zjw/triton-ascend-hust/. "$BUILD_DIR/"
rm -rf "$BUILD_DIR/build"

cd "$BUILD_DIR"
env \
  PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  TRITON_BUILD_WITH_CLANG_LLD=0 \
  TRITON_BUILD_PROTON=OFF \
  TRITON_BUILD_TD=OFF \
  TRITON_BUILD_NPUIR=OFF \
  TRITON_WHEEL_NAME=triton-ascend \
  TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF \
  MAX_JOBS=16 \
  /root/miniconda3/envs/vllm-hust-dev/bin/python setup_ascend.py bdist_wheel

/root/miniconda3/envs/vllm-hust-dev/bin/python -m pip install \
  --no-deps --force-reinstall dist/triton_ascend-*.whl
```

`TRITON_BUILD_WITH_CLANG_LLD=0` was required on the observed host because its
Clang rejected `-fuse-ld=lld`; it is not a universal requirement. The build may
download the pinned Ascend LLVM archive into the normal Triton cache.

Validate before running plugin tests:

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python - <<'PY'
import triton
from triton._C.libtriton.ascend import ir
print(triton.__version__)
print(ir)
PY
```

Then reinstall this plugin without attempting dependency resolution:

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python -m pip install \
  --no-deps --no-build-isolation -e /workspace/zjw/vllm-ascend-kvcompress-hust
```

For formal acceptance, record the Triton source commit and built wheel hash.
The locally available 3.2.2 wheel may be useful for diagnosis but is not
evidence for the declared 3.6.0/current-host stack.
