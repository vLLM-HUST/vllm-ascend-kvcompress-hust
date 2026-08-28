# TriAttention 校准产物

[English](calibration-artifacts.md) | 简体中文

TriAttention 在为缓存 key 打分和选择之前，需要针对具体模型生成 query 统计。
`artifacts/` 下的 `.pt` 文件就是这类统计产物。它们离线生成，服务启动时通过
`stats_path` 选择，在整个服务生命周期中按只读数据使用。

`artifacts/` 被有意加入 git ignore。干净 clone 不包含校准文件；运维人员必须为
实际服务的精确模型 revision 生成或安全分发对应产物。

## 当前本地产物

当前 workspace 包含两个 Qwen2.5-Coder-14B-Instruct 产物：

| 文件 | 预期用途 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `artifacts/qwen2.5-coder-14b-stats.pt` | 已验证的服务与 benchmark 产物 | 3,285,943 bytes | `d1f43bf5de3ab7d464a0a906060bbc15b828b872af79795bf5d26a7266c8a47a` |
| `artifacts/qwen2.5-coder-14b-stats-smoke.pt` | 快速启动/集成 smoke 产物；不作为质量参考 | 3,320,539 bytes | `016965e5d638f467fbb1b2ccc458becab1cc3129e3e30b040b131252d7ea9ab3` |

两者都使用参考 TriAttention 校准脚本输出的扁平 payload 格式，包含 48 层 x
40 个 query head，每个 head 在 head dimension 128 下含 64 个频率分量。运行时会
根据模型的 GQA 布局将 40 行 query-head 统计分组为 8 个 KV head。

`-smoke` 后缀只是文件命名约定。loader 不会赋予它不同语义，也不会自动选择它。
`stats_path` 指向哪个文件，哪个就是当前打分产物。

上表完整产物的 SHA-256 与当前 benchmark 证据中的记录一致。现有 payload 未记录
模型 revision、校准输入哈希或实际 token 数，因此无法只从 `.pt` 文件重建完整生成
溯源信息。

## 产物的功能

校准会针对每个 transformer layer 和 query head，在复数频域中汇总去除 RoPE 后的
query 向量：

| 字段 | 含义 | 运行时用途 |
| --- | --- | --- |
| `q_mean_real` | 每个 RoPE 频率上复数 query 均值的实部 | 估计已缓存 post-RoPE key 与未来 query 的对齐程度 |
| `q_mean_imag` | 同一均值的虚部 | 完成对相位敏感的三角打分 |
| `q_abs_mean` | 复数 query 幅值的均值 | 生成幅值回归修正项 |
| `freq_scale_sq` | 可选的频率缩放平方 | scaled RoPE 必需，并作为正数打分乘子 |
| `inv_freq` | 可选的精确 RoPE 逆频率 | scaled 或非默认 RoPE 必须逐层提供 |
| `metadata` | 形状、RoPE 和生成描述 | 字段存在时用于 fail-closed 兼容性检查 |

产物不包含模型权重、KV cache、prompt、生成 token 或请求状态，只包含从校准文本
得到的聚合浮点统计。它可在启动时以较小代价加载到 CPU，然后转换为 FP32 设备张量。

绑定 cache 时，实现会：

1. 使用 `torch.load(..., weights_only=True)` 在 CPU 上加载 payload；
2. 校验 layer 覆盖、head 形状、head dimension、已支持 model type、RoPE style/theta
   以及 scaled-RoPE 必需字段；
3. 根据模型 GQA 比例对 query-head 行分组；
4. 将统计以 FP32 传输到 NPU；
5. 派生频率缩放和幅值修正系数；
6. 预计算 Ascend 打分 kernel 使用的未来 offset 余弦/正弦均值。

每次压缩事务会用这些固定张量为请求当前的 post-RoPE K cache 打分。产物影响
哪些 token 被保留，但不改变 scheduler budget、不自行写入 KV，也不改变语义 RoPE 位置。

## 生成器来源

本插件负责消费校准产物，不重复实现用于生成产物的模型 instrumentation。
请使用匹配的[参考 TriAttention 仓库](https://github.com/WeianMao/triattention)
checkout 中的 `scripts/calibrate.py` 生成扁平格式。

产物溯源中应保存生成器 revision。当前文件符合该脚本输出的 payload schema：脚本加载
Hugging Face 模型，对普通文本执行一次 forward，捕获每个 attention layer 的 query
projection，逆转 RoPE，将成对维度转为复数，沿 token 维约简，最后保存 `metadata`
和 `stats`。

## 校准输入

使用包含自然、连贯内容的 UTF-8 纯文本：

- 提供足够文本，尽量接近选定的 `--max-length`；
- 包含有代表性的语言和结构，但与 benchmark/评估 prompt 隔离；
- 避免损坏文本、空输入以及机械重复的长循环；
- 记录输入 SHA-256 和生成 token 序列所使用的 tokenizer。

校准通常对领域不敏感，但这不代表无需质量评估。新产物在生产使用前必须通过
目标模型的质量护栏。

生成器在一次 forward 中同时持有模型权重和已捕获 query 张量。建议先用短输入执行
smoke，确认兼容后再在内存足够的设备上执行生产长度。

## 生成 Smoke 产物

在本插件仓库根目录执行，将 `TRIATTENTION_CHECKOUT` 指向参考 checkout，并选择空闲 NPU：

```bash
export TRIATTENTION_CHECKOUT=/path/to/triattention
export MODEL=/path/to/Qwen2.5-Coder-14B-Instruct
export CALIBRATION_TEXT=/path/to/calibration.txt
export ASCEND_RT_VISIBLE_DEVICES=5

python "$TRIATTENTION_CHECKOUT/scripts/calibrate.py" \
  --model "$MODEL" \
  --input "$CALIBRATION_TEXT" \
  --output artifacts/qwen2.5-coder-14b-stats-smoke.pt \
  --max-length 2048 \
  --device npu \
  --attn-implementation eager
```

2048-token 上限是建议的 smoke 约定，不是写入输出文件的属性。使用产物前应确认
脚本报告的实际 tokenized length，以及所有预期 layer/head entry 都存在。

## 生成生产产物

使用相同模型、tokenizer、生成器 revision、attention implementation 和设备软件栈，并提供
更长的独立校准文本：

```bash
python "$TRIATTENTION_CHECKOUT/scripts/calibrate.py" \
  --model "$MODEL" \
  --input "$CALIBRATION_TEXT" \
  --output artifacts/qwen2.5-coder-14b-stats.pt \
  --max-length 32768 \
  --device npu \
  --attn-implementation eager
```

脚本只会截断到 `--max-length`，不会对较短输入补齐。在结构校验、服务启动、匹配 A/B
容量/性能测试和模型专用质量检查全部通过前，应将输出视为候选产物。

候选产物通过之前不要覆盖上一个已验证产物。先写入临时产物名、记录哈希、完成验证，
然后再原子更新部署的 `stats_path`。

## 默认 RoPE 与 Scaled RoPE

当前两个 Qwen2.5 产物声明 `rope_style=half`、`rope_type=default`。它们没有
`freq_scale_sq` 和精确 `inv_freq`，因此 loader 使用单位频率缩放，并根据服务模型的
`rope_theta=1000000` 派生标准逆频率。

上述参考扁平生成器不足以生成 YaRN、LongRoPE 或其他 scaled/非默认 RoPE 产物。
这些模型必须使用结构化 `layer_stats` 格式，并提供精确的逐层 `inv_freq` 和正数
`freq_scale_sq`。缺少任一值时 provider 都会主动拒绝 scaled-RoPE 产物；不得从其他
模型复制值或静默回退到默认公式。

## 支持的 Payload Schema

### 扁平参考格式

```python
{
    "metadata": {
        "head_dim": 128,
        "rope_style": "half",
        "rope_type": "default",
        # 建议增加其他溯源和模型形状字段。
    },
    "stats": {
        "layer00_head00": {
            "q_mean_real": Tensor[64],
            "q_mean_imag": Tensor[64],
            "q_abs_mean": Tensor[64],
        },
        # 必须连续提供每一层的每一个 head。
    },
}
```

### 结构化格式

```python
{
    "metadata": {
        "model_type": "qwen2",
        "num_layers": 48,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rope_style": "half",
        "rope_type": "default",
        "rope_theta": 1000000.0,
    },
    "layer_stats": {
        0: {
            "q_mean_real": Tensor[40, 64],
            "q_mean_imag": Tensor[40, 64],
            "q_abs_mean": Tensor[40, 64],
            "freq_scale_sq": Tensor[40, 64],  # 适用时为 scaled RoPE 提供
            "inv_freq": Tensor[64],           # 精确逐层值
        },
        # 必须包含模型的每一层。
    },
}
```

query 统计可以有 `num_attention_heads` 或 `num_kv_heads` 行；后者会扩展到每个 GQA
query group。复数 query 均值也可以通过 `q_mean_complex` 提供：它可以是 complex
tensor，也可以是最后一维长度为 2 的实数 tensor。

## 检查与校验产物

不得使用无限制 pickle 加载检查不可信产物，必须使用 `weights_only=True`：

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import torch

path = Path("artifacts/qwen2.5-coder-14b-stats.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
stats = payload.get("stats", {})
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
print("metadata:", payload.get("metadata", {}))
print("flat entries:", len(stats))
print("first keys:", list(stats)[:3])
PY
```

针对当前 Qwen2.5-Coder-14B 形状，运行插件的完整 loader 与兼容性校验：

```bash
python - <<'PY'
from pathlib import Path
from vllm_ascend_kvcompress.methods.base import ModelShape
from vllm_ascend_kvcompress.methods.triattention.stats import CalibrationStats

path = Path("artifacts/qwen2.5-coder-14b-stats.pt")
model = ModelShape(
    model_type="qwen2",
    num_layers=48,
    num_attention_heads=40,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=1_000_000.0,
    has_rope_scaling=False,
)
reasons = CalibrationStats.load(path).validate(model)
if reasons:
    raise SystemExit("\n".join(reasons))
print("calibration artifact is structurally compatible")
PY
```

结构兼容只是必要条件，不能证明产物确实来自预期权重，也不能证明 token 选择不会
造成质量损失。最终检查是启动真实服务，再执行匹配的质量和容量测试。

## 溯源检查清单

在 `docs/dev/` 中以带日期 manifest 保存开发证据。至少记录：

- 模型路径或仓库 ID、不可变 revision 和 model-config 哈希；
- tokenizer revision 和 `tokenizer.json` 哈希；
- 参考 TriAttention 生成器 revision 和本地 patch 状态；
- 校准输入来源、许可证、SHA-256 和实际 token 数；
- 设备类型、PyTorch、torch-npu、Transformers、dtype、attention implementation 和命令行；
- 产物文件名、字节数、SHA-256、schema、layer/head/频率数；
- 结构校验输出，以及批准该产物的质量/benchmark 证据。

`.pt` 文件继续放在 `artifacts/`；带日期的溯源和实验日志放在 `docs/dev/`。如果将
产物分发到本地 workspace 以外，应同时分发 manifest 和校验和。

## 何时重新生成

以下任一项变化时，都应生成并重新验证产物：

- 模型权重、fine-tune、merge、量化模型实现或 revision；
- 校准使用的 tokenizer 或 prompt 处理策略；
- layer 数、query/KV head 数、head dimension、RoPE style/theta/scaling；
- 校准语料或生成算法；
- 质量回退表明当前统计不再具有代表性。

仅修改 `kv_budget`、`recompute_window`、`protected_recent_window`、
`score_chunk_size` 或 `score_layer_stride` 不会在机制上强制重新生成统计，但任何新
策略仍然必须重新做质量和性能验证。
