# Benchmark originals

## Leaderboard and mixed historical results

[History manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/manifest.json)
preserves the former result directories. HTML reads pinned benchmark JSON; local checks use:

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
export VLLM_HUST_BENCHMARK_ORIGINALS=/absolute/external/originals
python -m pytest tests/test_leaderboard.py tests/test_benchmark_inputs.py
python scripts/kvcompress_leaderboard.py --check
```

Rebuilding requires new external `--output` and `--frontier-output` files. Missing or changed inputs
fail closed. Kernel rows with only a method-code reference remain method-only, not recovered timings.

This bounded collection is preserved by the benchmark repository. Original versions, failures,
negative results and diagnostic scope remain unchanged; migration is not new runtime acceptance.
Other historical collections are not declared migrated by this change.

- [Original manifest 1](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b36edbc7698e3ca188904d0e20495ed2924a8525/archive/mod-targeted-20261023/compress/manifest.json)

Download and verify each original into an empty directory outside the source checkout:

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
```

The downloader checks the pinned manifest and each file's length and SHA256, rejects unsafe paths
and never executes archived code. CI repeats the readback and rejects the migrated paths returning
to Git. Fixed benchmark originals and unchanged source Git history preserve recoverability.

[compress-release](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5eeccc2d69b2c4a634e273a9f926584c8d252cf7/archive/mod-targeted-20261023/compress-release/manifest.json) — Historical bilingual release guides and LICENSE; paired JSON direct links are checksum-bound. Original failures remain unchanged.

[Historical host records / 历史宿主安装记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4792d99508627b969fbf61449adb2a53d229f071/archive/mod-targeted-20261023/compress-host/manifest.json) — byte-preserved bilingual originals and LICENSE; legacy local wheel consumers remain unresolved.
