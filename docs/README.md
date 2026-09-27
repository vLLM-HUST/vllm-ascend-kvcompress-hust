# Documentation map

English | [简体中文](README.zh.md)

## Current 0.8 candidate

- [Frontier benchmark protocol](frontier-benchmarking.md): official SWE prefix reuse and AgentX 256K, 900-second paired B0/B1 measurements, evidence, and current blockers.
- [HTML leaderboard](benchmark-leaderboard.html): separate Frontier measured points and historical engineering results. There are no qualified 0.8 Frontier points yet.
- [Qwen3.5 adaptation](qwen3.5-35b-a3b-adaptation.md): hybrid-cache, TP, APC, MTP2, async, and GDN implementation status.
- [Environment installation](environment-installation.md): synchronized host stack and the isolated PyTorch 2.13/TorchNPU 2.13/CANN 9.1 installation that resolved the earlier mismatch.
- [Method architecture](methods.md), [calibration artifacts](calibration-artifacts.md), and [packaging/release](packaging-and-release.md).

## Historical archive (not 0.8 acceptance gates)

- [V4.6-derived requirements](kv-compress-test-requirements.md) and [A2/A3 procedure](benchmarking.md).
- [Public long-context experiments](public-long-context-benchmarks.md) and [2026-09-20 validation record](validation.md), including negative results.
- [Earlier results](resuts.md) remain under their original filename to preserve links.

Historical files remain at their stable paths and carry archive banners. Moving them would break existing evidence links; the current protocol above supersedes their release-gate language. Do not mix historical percentages or short engineering smokes with the official Frontier 900-second cohorts.
