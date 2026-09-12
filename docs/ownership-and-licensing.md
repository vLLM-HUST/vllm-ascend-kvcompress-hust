# Ownership and Licensing

English | [简体中文](ownership-and-licensing.zh.md)

## Project ownership

- School: Huazhong University of Science and Technology (HUST)
- Group: CGCL
- Advisor: Prof. Yao Wan (万瑶)
- Project lead: Sichen Liu (刘思辰), [@Seas0](https://github.com/Seas0)
- Maintainers: Jiawan Zhang (张家万),
  [@Jiawan23](https://github.com/Jiawan23); Ruohao Wei (韦若皓),
  [@kotoriqaq0](https://github.com/kotoriqaq0); and Sichen Liu,
  [@Seas0](https://github.com/Seas0)

The maintainers agree to maintain compatibility with vLLM-HUST and
vLLM-Ascend-HUST and to publish this project as an independent Extension
Manager plugin. It must not be described as code built into either host.

## Algorithm and implementation source

The scoring idea is derived from the Apache-2.0-licensed
[TriAttention repository](https://github.com/WeianMao/triattention), pinned in
`NOTICE` to commit `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`, and its associated
paper. This repository reimplements the runtime for Ascend paged KV storage;
it does not copy the upstream CUDA runtime kernels. Local implementation,
packaging, tests, and documentation are licensed under Apache-2.0.

## Redistribution boundary

| Material | Included in source repository | Included in Python wheel | Redistribution statement |
| --- | --- | --- | --- |
| Plugin source, tests, examples, documentation | Yes | Source and manifest only | Apache-2.0; preserve `LICENSE` and `NOTICE` |
| Extension Manager manifest | Yes | Yes | Apache-2.0 |
| Committed `.pt` calibration statistics | Yes | No | Aggregate model-derived data; provenance is incomplete, so repository licensing alone is not a model/dataset redistribution grant |
| Model weights and tokenizer assets | No | No | Obtain under their upstream terms |
| Calibration or benchmark input text | No | No | Operators must verify their right to use and redistribute it |
| Raw service logs, prompts, generated output, profiler traces | No | No | Environment-owned; not granted for redistribution by this repository |
| Public benchmark scripts and aggregate evidence summaries | Yes | No | Apache-2.0 project documentation/code; upstream benchmark text remains under each source dataset's terms |
| Historical measurements in `docs/resuts*.md` | Yes | Documentation only | Apache-2.0 documentation, but not acceptance evidence for version 0.4 |

Before distributing a calibration artifact, record at least its SHA-256,
model identifier and revision, tokenizer revision, generator commit, input-data
provenance, token count, creation date, and responsible maintainer. The two
legacy committed artifacts do not contain all of this information and should
therefore be treated as development samples, not generally redistributable
production assets.
