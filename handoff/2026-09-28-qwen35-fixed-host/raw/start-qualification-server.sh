#!/usr/bin/env bash
set -eo pipefail

run_dir=/root/frontier-kvcompress-runs/qualification/live
mkdir -p "$run_dir"
date -u +%FT%TZ > "$run_dir/server-start-utc.txt"
npu-smi info > "$run_dir/npu-before-server.txt"
cp /root/frontier-kvcompress-runs/server-metadata.json "$run_dir/server-metadata.json"
git -C /root/vllm-hust-pegaflow rev-parse HEAD > "$run_dir/vllm-head.txt"
git -C /root/vllm-hust-pegaflow status --short > "$run_dir/vllm-status.txt"
git -C /root/vllm-ascend-hust rev-parse HEAD > "$run_dir/vllm-ascend-head.txt"
git -C /root/vllm-ascend-kvcompress-hust rev-parse HEAD > "$run_dir/mod-head.txt"
git -C /root/vllm-ascend-kvcompress-hust status --short > "$run_dir/mod-status.txt"
git -C /root/extension-manager rev-parse HEAD > "$run_dir/manager-head.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/vllm-ascend-hust/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=2,3
export PYTHONHASHSEED=0
export PYTHONPATH="/root/vllm-hust-pegaflow:/root/vllm-ascend-hust:/root/vllm-hust-dev-hub/scripts/frontier_pipeline:/root/vllm-hust-dev-hub/scripts/frontier_runtime:${PYTHONPATH:-}"
export VLLM_VERSION=0.25.1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2=1

cd /root/extension-manager
exec /root/frontier-kvmat-env/bin/vllm-hust-ext run --shutdown-grace-seconds 60 -- \
  /root/frontier-kvmat-env/bin/python -m vllm.entrypoints.cli.main serve \
  /models/modelscope_cache/Qwen/Qwen3___5-35B-A3B \
  --host 127.0.0.1 --port 33787 \
  --served-model-name frontier-qwen35-unified \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp --worker-cls pipeline_worker.Worker \
  --dtype bfloat16 --kv-cache-dtype auto --max-model-len 262144 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 --seed 17 --enable-prefix-caching \
  --mamba-cache-mode align --enable-prompt-tokens-details --async-scheduling \
  --shutdown-timeout 60 --additional-config '{"enable_cpu_binding":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[3,6,12,24,48],"max_cudagraph_capture_size":48}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --generation-config vllm \
  --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":-1,"presence_penalty":0.0}' \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --kv-cache-memory-bytes 26038239232
