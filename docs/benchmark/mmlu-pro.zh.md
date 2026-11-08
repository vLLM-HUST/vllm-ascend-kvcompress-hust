# MMLU-Pro 准备与评估

[English](mmlu-pro.md) | 简体中文

完整安装、数据、smoke、B0/B1 配对和性能命令见[统一浦江评测指南](pujiang.zh.md)。
B0/B1 共用锁定的 HUST 基座和宿主环境；B0 关闭压缩插件，B1 使用指定配置启用插件。

客户端固定为 `b6215bf76808b1e9d351295e2ecb401542317372`；数据固定为
`b189ec765aa7ed75c8acfea42df31fdae71f97be`（test12032、validation70）；参考实现固定为
`f418b116db00b065c2aea046518d8fcf74d39872`。
保留 five-shot、关闭 thinking 的 token-ID prompt、32768上下文和2048输出预算，以及上游客户端的本地答案提取规则。
无法提取、越界答案和请求失败记错，不随机猜测。

```bash
scripts/run_benchmark.sh mmlu-pro download-client
scripts/set_env.sh install client --dataset mmlu-pro --python /path/to/python3.11
scripts/run_benchmark.sh mmlu-pro download
scripts/run_benchmark.sh mmlu-pro prepare --model-path /path/to/model
scripts/run_benchmark.sh mmlu-pro run --arm b0 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --output ../.benchmarks/results/mmlu/b0
```

B1 使用同一固定样本、独立新输出目录；`--sample-size` 默认100，`--resample` 创建新版本。
省略 `--smoke` 才运行完整选定评测范围。
客户端不装入服务环境；两侧使用相同的兼容桥和服务参数。
默认性能是独立的200题、256输出预算、`vllm bench serve` chat 请求，不是准确率 token-ID 请求重放。
两侧必须复用同一个压测端版本；实际压缩只能通过服务端提交/回执证明。

模型小文件哈希不证明完整权重一致；smoke 不代表正式成绩，服务身份未知时保持未知。
