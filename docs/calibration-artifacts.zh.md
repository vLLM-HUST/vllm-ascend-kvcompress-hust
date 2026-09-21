# TriAttention 校准产物

[English](calibration-artifacts.md) | 简体中文

TriAttention 需要模型专属的查询统计来为缓存 K 打分。0.5 版本已将结构化 `.pt`
产物生成能力收进插件，不再依赖外部 TriAttention checkout。

## 服务启动时自动生成

`auto_calibrate` 默认是 `true`。当 `stats_path` 不存在时，Ascend worker 按以下顺序
执行：

1. 在 `stats_path` 旁获取排他文件锁，并再次检查目标文件；
2. 通过 Hugging Face Transformers，使用 vLLM 选择的同一模型、tokenizer、
   revision、dtype 和 remote-code 信任策略；
3. 执行一次校准前向，将每层未旋转的 `q_proj` 输出直接归约为逐 head 频率统计；
4. 校验完整 payload，刷写临时文件，再原子重命名到 `stats_path`；
5. 释放临时模型和设备缓存；
6. 调用原始 `NPUModelRunner.load_model()` 加载服务权重。

因此校准模型与服务模型不会同时驻留。直接在线归约也不需要保存所有层的完整长序列
Q 张量。若另一个 worker 在等待期间完成生成，当前 worker 会复用完成的文件；
`stats_path` 不会暴露半写文件。

最小配置如下：

```json
{
  "method_config": {
    "stats_path": "/srv/vllm/calibration/qwen-stats.pt",
    "auto_calibrate": true
  }
}
```

服务账号必须能创建输出目录或在目录中写入，并预留足够空间。manifest 声明了
`filesystem_read`、`filesystem_write`、`device_access` 和 `network_egress`；当模型
文件均在本地且禁止网络访问时，请设置 `calibration_local_files_only=true`。

服务启动不会覆盖已有产物。已有文件使用 `torch.load(..., weights_only=True)` 加载，
并在模型加载前校验。插件新生成的产物还会绑定模型标识、revision 和本地 checkpoint
manifest 指纹；损坏、结构不匹配或绑定到其他模型来源的文件都会使启动失败。设置
`auto_calibrate=false` 可保留旧的“必须预先生成”工作流。

## 校准选项

| 选项 | 默认值 | 作用 |
| --- | --- | --- |
| `stats_path` | 必填 | 产物保存位置及之后的运行时输入 |
| `auto_calibrate` | `true` | 仅当 `stats_path` 缺失时生成 |
| `calibration_input_path` | 内置语料 | 许可清晰的 UTF-8 校准文本 |
| `calibration_max_length` | `4096` | tokenizer 截断长度，最小 128 |
| `calibration_device` | `auto` | 使用 worker 设备，或显式指定 `npu:0` |
| `calibration_attn_implementation` | `eager` | Transformers 后端：`eager`、`sdpa` 或 `flash_attention_2` |
| `calibration_local_files_only` | `false` | 禁止模型/tokenizer 下载 |

内置的中英文系统文本让首次启动不需要额外语料，适用于 bootstrap 和集成验收。
生产环境应提供独立、许可清晰且具有代表性的语料，记录其哈希，并执行目标负载的质量
门槛。校准通常具有一定领域容忍性，但不意味着无需质量验证。

## 在服务启动前生成

wheel 提供同一生成器的命令行入口。需要固定启动耗时，或服务账号没有产物目录写权限
时可使用：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

vllm-ascend-kvcompress-calibrate \
  --model /path/to/Qwen2.5-14B-Instruct \
  --input /path/to/licensed-calibration.txt \
  --output /srv/vllm/calibration/qwen-stats.pt \
  --max-length 32768 \
  --device npu:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --local-files-only
```

Qwen3.5-35B-A3B 的 BF16 权重无法装入单张 910B2，因此需暴露两张设备并增加
`--device-map auto`。服务使用 TP>1 时，启动 hook 会自动选择该分布方式；文件锁
仍保证只有一个 worker 实际生成产物。

省略 `--input` 时使用内置语料。命令默认复用已有有效 payload。`--force` 只在 CLI
中提供，并以原子方式替换文件；只应在受控候选路径使用。正确性和长上下文质量通过前，
建议使用新文件名而不是覆盖上一份已验收产物。

## 生成算法

生成器为每个受支持的全注意力 decoder 层优先挂接投影后的 `q_norm`，否则挂接
`q_proj`，并将输出变形为 `[tokens, query_heads, head_dim]`。对于已经验证的
half-split RoPE 布局，
前后两半分别构成频率分量的实部和虚部，并累计：

- 查询实部均值；
- 查询虚部均值；
- 复数查询幅值均值。

部分 RoPE 模型还会保存每个直通维度的 query 均值。只有当没有 `self_attn` 的混合层
索引与模型配置完全一致时才会跳过这些层；层集合异常会失败关闭。

归约在设备上立即执行，只有很小的和向量移到 CPU。`q_proj` 已经暴露所需的未旋转
查询，因此不再执行“应用 RoPE 后再数值逆变换”。默认 RoPE 优先从模型读取精确
逆频率，否则根据 `rope_theta` 推导。缩放或非默认 RoPE 必须由模型暴露精确
`inv_freq` 和 attention scaling，否则生成失败。

当前已验证的模型形状族包括 Llama、Mistral、Qwen2/Qwen2-MoE、
Qwen3/Qwen3-MoE 和 Qwen3.5/Qwen3.5-MoE，RoPE 布局均为 half-split。没有可用
`q_norm` 或 `q_proj` 的融合/自定义 attention 模块会被拒绝，不进行猜测。

## 结构化 payload

当前生成器写出 schema 3。除 schema 2 字段外，还记录 `rotary_dim`、
`attention_layer_indices`；部分 RoPE 层另含 `q_pass_mean`：

```python
{
    "metadata": {
        "schema_version": 3,
        "generator": "vllm-ascend-kvcompress-hust",
        "generator_version": "0.6.0",
        "model": "/path/or/hf-id",
        "model_revision": "default",
        "model_source_fingerprint": "local-manifest:...",
        "model_config_sha256": "...",
        "input_source": "/path/to/calibration.txt",
        "input_sha256": "...",
        "token_count": 4096,
        "model_type": "qwen2",
        "num_layers": 48,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
        "attention_layer_indices": [0, 1, "...", 47],
        "rope_theta": 1000000.0,
        "rope_style": "half",
        "rope_type": "default",
    },
    "layer_stats": {
        "0": {
            "q_mean_real": Tensor[40, 64],
            "q_mean_imag": Tensor[40, 64],
            "q_abs_mean": Tensor[40, 64],
            "freq_scale_sq": Tensor[1, 64],
            "inv_freq": Tensor[64],
            # 部分 RoPE 还包含 "q_pass_mean": Tensor[40, pass_dim]，
        },
        # 每一层都必须存在。
    },
}
```

loader 继续兼容原始 TriAttention 的扁平 `stats` mapping，以及以整数作为层 key 的
结构化 payload。历史 payload 没有记录新字段，因此无法提供更强的模型来源绑定。

## 仓库中的历史产物

`artifacts/` 下两份文件仅为开发和历史复现保留，不进入 wheel/sdist：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `qwen2.5-coder-14b-stats.pt` | 历史 benchmark 产物 | `d1f43bf5de3ab7d464a0a906060bbc15b828b872af79795bf5d26a7266c8a47a` |
| `qwen2.5-coder-14b-stats-smoke.pt` | 历史启动 smoke | `016965e5d638f467fbb1b2ccc458becab1cc3129e3e30b040b131252d7ea9ab3` |

它们缺少完整模型 revision、校准输入哈希和实际 token 数。不能从本仓库的
Apache-2.0 许可证推导模型或数据集的再分发权。

## 检查与验收

不要用无限制 pickle 加载不可信 `.pt` 文件：

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import torch

path = Path("/srv/vllm/calibration/qwen-stats.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
print("metadata:", payload.get("metadata", {}))
print("layers:", len(payload.get("layer_stats", {})))
PY
```

结构校验只是必要条件。生产使用前，应保存生成器/包版本、模型与 tokenizer revision、
checkpoint 和输入指纹、CANN/PyTorch/torch-npu/Transformers 版本、命令、产物哈希与
大小，以及成对的正确性、长上下文质量、吞吐、时延和 HBM 证据。当模型权重、
tokenizer、查询形状、RoPE 参数、语料或生成算法变化时重新生成。

仅改变 KV budget 或运行时评分/copy 分块大小，不会机械地要求重新生成统计；但新策略
仍必须独立完成质量和性能验收。
