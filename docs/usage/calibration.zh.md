# 校准 Query 统计

[English](calibration.md) | 简体中文 | [用户指南](README.zh.md)

TriAttention 使用与模型绑定的 Query 统计给缓存 Key 评分。统计文件按模型、
tokenizer、revision 与本地权重清单生成；不能因为架构名称相同就复用其他
checkpoint 的文件。模型权重、校准文本和生成的 `.pt` 产物均不随插件发布。

## 首次启动自动生成

保持 `auto_calibrate=true`，将 `stats_path` 指向可写目录下尚不存在的文件。
启用的插件会锁定路径、加载临时 Transformers 模型、生成并校验统计、原子写入
文件、释放临时模型，然后加载服务模型。之后启动会检查绑定信息并复用产物。
如果现有文件损坏或模型不匹配，服务启动会失败；自动模式不会悄悄覆盖它。

Qwen3.5-35B-A3B 的 BF16 校准需暴露两张 NPU；自动校准会在两卡分配临时
模型。模型文件已在本地时，设置 `calibration_local_files_only=true`。
内置文本适合初次集成检查；实际部署应设置 `calibration_input_path`，
使用许可明确、有代表性的 UTF-8 语料并记录其哈希。

## 预先生成产物

预生成可缩短服务首次启动时间，也能让服务账号只读访问文件。
Qwen3.5-35B-A3B 的命令示例：

```bash
vllm-ascend-kvcompress-calibrate \
  --model /path/to/Qwen3.5-35B-A3B \
  --input /path/to/licensed-calibration.txt \
  --output /path/to/qwen35-stats.pt \
  --max-length 4096 \
  --device-map auto \
  --dtype bfloat16 \
  --attn-implementation eager \
  --local-files-only
```

将 `stats_path` 指向该文件；若服务绝不能自动生成，设置
`auto_calibrate=false`。`--model` 必须与服务所用模型身份一致。更换
checkpoint 或语料时，应先用新路径生成候选文件。CLI 的 `--force` 会替换现有
文件，只在明确需要替换时使用。分发校准部署前，请阅读
[产物格式与绑定规则](../development/calibration-artifacts.zh.md)；换语料后还需
执行应用质量验证。
