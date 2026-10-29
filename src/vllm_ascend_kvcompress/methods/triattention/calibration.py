# SPDX-License-Identifier: Apache-2.0
"""生成与模型匹配的 TriAttention 校准统计。

服务钩子在 vLLM 加载模型权重前调用本模块，因此临时 Hugging Face
模型会在服务模型分配前释放。进程锁与原子重命名可处理多个 worker
同时发现输出产物缺失的启动竞争。
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ... import __version__
from ...host.model_config import model_shape_from_config
from ..base import ModelShape
from .config import TriAttentionConfig
from .stats import SUPPORTED_MODEL_TYPES, CalibrationStats

_BUILTIN_CALIBRATION_SECTIONS = (
    """A reliable systems service must preserve correctness before optimizing
throughput. Engineers trace each request from tokenization through scheduling,
attention, cache allocation, sampling, and cleanup. They compare semantic token
positions with physical cache slots, record assumptions, and reject ambiguous
states instead of silently producing plausible output.""",
    """Long-context evaluation should contain narrative, source code, tables,
mathematical notation, repeated names, and facts separated by many paragraphs.
The evaluator checks whether an early constraint still controls a late answer,
whether citations refer to the right section, and whether latency measurements
exclude setup work consistently.""",
    """在长上下文推理中，正确性和性能必须分别验收。测试材料应包含中文、英文、数字、标点、代码片段、
跨段落引用和容易混淆的实体。记录需要说明模型版本、输入长度、输出长度、并发数、设备、软件版本以及失败样例，
不能只保留一个平均吞吐数字。""",
    """Consider a cache divided into fixed-size blocks. A logical request may
retain its full history while its physical representation keeps only selected
tokens. Every later position calculation must account for that difference.
Protected recent tokens, recomputation windows, and destination ownership form
independent invariants and are validated at their respective boundaries.""",
    """def verify_record(record):
    required = {"model", "revision", "input_sha256", "token_count"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"missing provenance fields: {missing}")
    return all(record[name] not in (None, "") for name in required)
""",
    """A small observatory logged five signals at regular intervals: temperature
rose from 18.2 to 21.7 degrees, pressure fell by 3.1 units, wind changed direction
twice, humidity remained within one percent, and the backup clock drifted by
forty milliseconds. The final report distinguishes observations from inferred
causes and keeps the original units.""",
)


@dataclass(frozen=True)
# 离线 CLI 与服务启动前自动校准共用的任务描述。
# 它不属于每请求推理状态；output 是持久化统计产物，force 决定能否覆盖既有文件。
class CalibrationRequest:
    model: str
    output: Path
    input_path: Path | None = None
    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    max_length: int = 4096
    device: str = "npu"
    device_map: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    force: bool = False


@dataclass
# 只累计各层 Q 的和、幅度和与 token 数，最终得到均值，不保存整段 Q。
# 捕获点优先 q_norm、否则 q_proj，位于 RoPE 前，和运行时 attention Query 观测不同。
class _LayerAccumulator:
    real_sum: torch.Tensor | None = None
    imag_sum: torch.Tensor | None = None
    abs_sum: torch.Tensor | None = None
    pass_sum: torch.Tensor | None = None
    count: int = 0

    def add(
        self,
        projected: torch.Tensor,
        heads: int,
        head_dim: int,
        rotary_dim: int,
    ) -> None:
        expected = heads * head_dim
        if projected.shape[-1] == expected * 2:
            # Qwen3.5 的 q_proj 将注意力输出门控打包在 Q 之后。
            projected = projected[..., :expected]
        if projected.shape[-1] == expected:
            values = projected.detach().reshape(-1, heads, head_dim).float()
        elif projected.shape[-1] == head_dim and projected.numel() % expected == 0:
            # Q/K 归一化模块通常输出 [..., heads, head_dim] 形状。
            values = projected.detach().reshape(-1, heads, head_dim).float()
        else:
            raise RuntimeError(
                "query projection width does not match model attention shape: "
                f"got {projected.shape[-1]}, expected {expected}"
            )
        # half RoPE 把旋转子空间前后两半视为实/虚部；尾部 pass_values 不参与复旋转。
        # Qwen3.5 的 q_proj 还可能拼接输出 gate，上面先剥离，不能把 gate 纳入 Q 均值。
        frequency_count = rotary_dim // 2
        real = values[..., :frequency_count]
        imag = values[..., frequency_count:rotary_dim]
        pass_values = values[..., rotary_dim:]
        real_sum = real.sum(dim=0).cpu()
        imag_sum = imag.sum(dim=0).cpu()
        abs_sum = torch.sqrt(real.square() + imag.square()).sum(dim=0).cpu()
        self.real_sum = real_sum if self.real_sum is None else self.real_sum + real_sum
        self.imag_sum = imag_sum if self.imag_sum is None else self.imag_sum + imag_sum
        self.abs_sum = abs_sum if self.abs_sum is None else self.abs_sum + abs_sum
        if pass_values.numel():
            pass_sum = pass_values.sum(dim=0).cpu()
            self.pass_sum = (
                pass_sum if self.pass_sum is None else self.pass_sum + pass_sum
            )
        self.count += values.shape[0]


def ensure_calibration_for_runner(runner: Any, config: TriAttentionConfig) -> bool:
    """在加载服务权重前，生成并验证缺失的校准产物。"""
    model_shape = model_shape_from_config(runner.model_config)
    # 服务启动前优先验证已有产物；不因 auto_calibrate=True 就覆盖它。
    # 若缺失且关闭自动生成，后续 method 加载统计时会报告文件缺失。
    if config.stats_path.is_file():
        _validate_artifact(config.stats_path, model_shape, runner.model_config)
        return False
    if not config.auto_calibrate:
        return False

    model_config = runner.model_config
    configured_device = config.calibration_device
    device = (
        str(getattr(runner, "device", "npu"))
        if configured_device == "auto"
        else configured_device
    )
    dtype = _dtype_name(getattr(model_config, "dtype", "bfloat16"))
    runner_config = getattr(runner, "vllm_config", None)
    parallel_config = getattr(runner_config, "parallel_config", None)
    tensor_parallel_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
    request = CalibrationRequest(
        model=str(model_config.model),
        tokenizer=(str(model_config.tokenizer) if model_config.tokenizer else None),
        output=config.stats_path,
        input_path=config.calibration_input_path,
        revision=getattr(model_config, "revision", None),
        tokenizer_revision=getattr(model_config, "tokenizer_revision", None),
        trust_remote_code=bool(getattr(model_config, "trust_remote_code", False)),
        local_files_only=config.calibration_local_files_only,
        max_length=config.calibration_max_length,
        device=device,
        # 单份 Qwen3.5-35B-A3B BF16 权重快照超过一张 910B2 的容量。
        # 若服务拓扑已使用张量并行，
        # 则让 Accelerate 将临时校准模型分散到可见 NPU 上。
        device_map="auto" if tensor_parallel_size > 1 else None,
        dtype=dtype,
        attn_implementation=config.calibration_attn_implementation,
    )
    generated = generate_calibration_artifact(request)
    _validate_artifact(config.stats_path, model_shape, runner.model_config)
    return generated


def generate_calibration_artifact(request: CalibrationRequest) -> bool:
    """生成校准产物，若另一进程已先完成生成，则返回 False。"""
    if request.max_length < 128:
        raise ValueError("calibration max_length must be at least 128")
    output = request.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    # 多 worker 可同时发现文件缺失；文件锁覆盖检查、生成和保存整个临界区。
    # 锁内再次检查，等待者复用胜出者产物；原子替换防止读者看到半写文件。
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if output.is_file() and not request.force:
            _validate_existing_for_request(output, request)
            return False
        payload = _generate_payload(request)
        _atomic_torch_save(payload, output)
    return True


def _generate_payload(request: CalibrationRequest) -> dict[str, Any]:
    if (
        request.device_map is not None
        and importlib.util.find_spec("accelerate") is None
    ):
        raise RuntimeError(
            "distributed calibration requires Accelerate; install the "
            "'vllm-ascend-kvcompress-hust[calibration]' extra in the host "
            "environment"
        )
    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from transformers import __version__ as transformers_version
    except ImportError as error:
        raise RuntimeError(
            "calibration generation requires the Transformers package"
        ) from error

    _prepare_device(request.device)
    input_device = torch.device(
        "cpu" if request.device_map == "auto" else request.device
    )
    dtype = _torch_dtype(request.dtype)
    common = {
        "trust_remote_code": request.trust_remote_code,
        "local_files_only": request.local_files_only,
    }
    config = AutoConfig.from_pretrained(
        request.model,
        revision=request.revision,
        **common,
    )
    shape = _model_shape_from_hf_config(config)
    if shape.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model_type {shape.model_type!r} is not supported for calibration"
        )
    tokenizer_id = request.tokenizer or request.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=request.tokenizer_revision or request.revision,
        **common,
    )
    text, input_source = _load_calibration_text(request)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=request.max_length,
        add_special_tokens=True,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError("tokenizer did not return rank-two input_ids")
    token_count = int(input_ids.shape[1])
    if token_count < 2:
        raise ValueError("calibration input produced fewer than two tokens")

    text_config = getattr(config, "text_config", config)
    is_qwen35_multimodal = text_config is not config and str(
        getattr(config, "model_type", "")
    ).startswith("qwen3_5")
    load_options: dict[str, Any] = {
        "config": text_config if is_qwen35_multimodal else config,
        "revision": request.revision,
        "dtype": dtype,
        "device_map": request.device_map or request.device,
        **common,
    }
    if request.attn_implementation:
        load_options["attn_implementation"] = request.attn_implementation
    if is_qwen35_multimodal:
        # 已发布的 Qwen3.5 检查点将文本权重嵌套在
        # model.language_model 下。只加载因果文本模型，
        # 避免在校准时构造用不到的视觉塔。
        load_options["key_mapping"] = {r"^model\.language_model\.(.+)$": r"model.\1"}

    model: torch.nn.Module | None = None
    handles: list[Any] = []
    try:
        # Qwen3.5 多模态 checkpoint 只加载文本因果模型并做权重键映射。
        # 校准只消费文本 Q，不需要构建未使用的视觉塔。
        model_loader = AutoModelForCausalLM if is_qwen35_multimodal else AutoModel
        model = model_loader.from_pretrained(request.model, **load_options)
        model.eval()
        attention_layers = _find_attention_layers(model)
        actual_layer_indices = tuple(index for index, _ in attention_layers)
        if actual_layer_indices != shape.full_attention_layer_indices:
            raise RuntimeError(
                "located full-attention layers do not match model config: "
                f"got {actual_layer_indices}, expected "
                f"{shape.full_attention_layer_indices}"
            )
        # 每层只存统计累计量与 token 数，不保留全部前向激活。
        # 完整 Q 仅在 hook 回调中短暂使用，均值以实际采样 token 总数为分母。
        accumulators = {
            layer_idx: _LayerAccumulator() for layer_idx, _ in attention_layers
        }
        for layer_idx, attention in attention_layers:
            # 若存在 Q 归一化，必须采集归一化后的 Q，
            # 单采 q_proj 会与服务模型评分不一致。
            # 每层闭包通过默认参数 index 固定层号，
            # 避免循环结束后所有 hook 指向最后一层。
            capture_module = getattr(attention, "q_norm", None)
            if not isinstance(capture_module, torch.nn.Module):
                capture_module = getattr(attention, "q_proj", None)
            if not isinstance(capture_module, torch.nn.Module):
                raise RuntimeError(
                    f"attention layer {layer_idx} exposes neither q_norm nor q_proj"
                )

            def capture(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_idx,
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        f"attention layer {index} q_proj returned a non-tensor"
                    )
                accumulators[index].add(
                    output,
                    shape.num_attention_heads,
                    shape.head_dim,
                    shape.effective_rotary_dim,
                )

            handles.append(capture_module.register_forward_hook(capture))

        model_inputs = {
            name: value.to(input_device)
            for name, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        # 只前向一次、use_cache=False；统计量来自校准输入，不来自实际服务请求。
        # 生成后在 finally 移除 hook 并回收临时模型，随后宿主才加载服务权重。
        with torch.inference_mode():
            model(**model_inputs, use_cache=False)

        rope_type = _rope_type(config)
        rope_values = _extract_rope_values(
            model,
            attention_layers,
            shape,
            rope_type,
        )
        layer_stats: dict[str, dict[str, torch.Tensor]] = {}
        for layer_idx, accumulator in accumulators.items():
            # add 同步初始化三个accumulator累计量；这里只检查外部 forward 是否真正产生了样本。
            if accumulator.count <= 0:
                raise RuntimeError(f"no query values captured for layer {layer_idx}")
            divisor = float(accumulator.count)
            inv_freq, freq_scale_sq = rope_values[layer_idx]
            values = {
                "q_mean_real": (accumulator.real_sum / divisor).contiguous(),
                "q_mean_imag": (accumulator.imag_sum / divisor).contiguous(),
                "q_abs_mean": (accumulator.abs_sum / divisor).contiguous(),
                "inv_freq": inv_freq,
                "freq_scale_sq": freq_scale_sq,
            }
            if accumulator.pass_sum is not None:
                values["q_pass_mean"] = (accumulator.pass_sum / divisor).contiguous()
            layer_stats[str(layer_idx)] = values

        config_payload = (
            config.to_dict() if hasattr(config, "to_dict") else vars(config)
        )
        # 产物同时保存评分所需形状/RoPE 信息和来源信息；二者承担不同验证职责。
        # 统计 loader 检查数值与模型形状，
        # _validate_artifact 对本插件产物额外检查模型来源。
        metadata = {
            "schema_version": 3,
            "generator": "vllm-ascend-kvcompress-hust",
            "generator_version": __version__,
            "transformers_version": transformers_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": request.model,
            "model_revision": request.revision or "default",
            "model_source_fingerprint": _model_source_fingerprint(
                request.model, request.revision
            ),
            "tokenizer": tokenizer_id,
            "tokenizer_revision": request.tokenizer_revision
            or request.revision
            or "default",
            "model_config_sha256": _sha256_json(config_payload),
            "input_source": input_source,
            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count": token_count,
            "requested_max_length": request.max_length,
            "num_traces": 1,
            "model_type": shape.model_type,
            "num_layers": shape.num_layers,
            "num_attention_heads": shape.num_attention_heads,
            "num_kv_heads": shape.num_kv_heads,
            "head_dim": shape.head_dim,
            "rotary_dim": shape.effective_rotary_dim,
            "attention_layer_indices": list(shape.full_attention_layer_indices),
            "rope_theta": shape.rope_theta,
            "rope_style": "half",
            "rope_type": rope_type,
            "dtype": request.dtype,
            "attn_implementation": request.attn_implementation,
        }
        payload = {"metadata": metadata, "layer_stats": layer_stats}
        reasons = _load_payload(payload).validate(shape)
        if reasons:
            raise ValueError(
                "generated calibration is incompatible: " + "; ".join(reasons)
            )
        return payload
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        _empty_device_cache(request.device, all_devices=request.device_map == "auto")


def _load_payload(payload: Mapping[str, Any]) -> CalibrationStats:
    """构造临时载荷，交由安全加载及验证路径处理。"""
    handle, raw_path = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    path = Path(raw_path)
    try:
        torch.save(dict(payload), path)
        return CalibrationStats.load(path)
    finally:
        path.unlink(missing_ok=True)


def _validate_artifact(
    path: Path, shape: ModelShape, model_config: Any | None = None
) -> None:
    artifact = CalibrationStats.load(path)
    reasons = list(artifact.validate(shape))
    if (
        model_config is not None
        and artifact.metadata.get("generator") == "vllm-ascend-kvcompress-hust"
    ):
        expected_model = str(model_config.model)
        expected_revision = getattr(model_config, "revision", None) or "default"
        expected_fingerprint = _model_source_fingerprint(
            expected_model, getattr(model_config, "revision", None)
        )
        checks = {
            "model": expected_model,
            "model_revision": expected_revision,
            "model_source_fingerprint": expected_fingerprint,
        }
        for name, expected in checks.items():
            actual = artifact.metadata.get(name)
            if actual != expected:
                reasons.append(
                    f"calibration {name} {actual!r} does not match {expected!r}"
                )
    if reasons:
        raise ValueError(
            f"calibration artifact {path} is incompatible: " + "; ".join(reasons)
        )


def _validate_existing_for_request(path: Path, request: CalibrationRequest) -> None:
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError(
            "calibration validation requires the Transformers package"
        ) from error
    config = AutoConfig.from_pretrained(
        request.model,
        revision=request.revision,
        trust_remote_code=request.trust_remote_code,
        local_files_only=request.local_files_only,
    )
    shape = _model_shape_from_hf_config(config)
    model_config = type(
        "ModelConfig",
        (),
        {"model": request.model, "revision": request.revision},
    )()
    _validate_artifact(path, shape, model_config)


# 临时文件放在目标同一目录，写完 flush/fsync 后再原子替换。
# 配合外层文件锁，其他 worker 只会读到完整旧文件或完整新文件。
def _atomic_torch_save(payload: Mapping[str, Any], output: Path) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _find_attention_layers(
    model: torch.nn.Module,
) -> list[tuple[int, torch.nn.Module]]:
    backbone = _find_transformer_backbone(model)
    # 查找函数只返回具有 ModuleList 层集合的 backbone。
    layers = backbone.layers
    attention_layers: list[tuple[int, torch.nn.Module]] = []
    for layer_idx, layer in enumerate(list(layers)):
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, torch.nn.Module):
            attention_layers.append((layer_idx, attention))
    return attention_layers


def _find_transformer_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """穿过常见因果语言模型或视觉语言模型包装，定位文本解码层。"""
    candidates = [model]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(getattr(candidate, "layers", None), torch.nn.ModuleList):
            return candidate
        for name in ("model", "language_model", "text_model"):
            child = getattr(candidate, name, None)
            if isinstance(child, torch.nn.Module):
                candidates.append(child)
    raise RuntimeError("cannot locate transformer decoder layers")


# 优先读取模型真实 rotary_emb.inv_freq 与 attention_scaling。
# 只允许 default RoPE 用 theta 补频率；缩放 RoPE 必须暴露真实值，不能猜测。
# 保存 scale² 是为兼容统计格式，运行时 post-RoPE K 评分只补 query-side scale。
def _extract_rope_values(
    model: torch.nn.Module,
    attention_layers: Sequence[tuple[int, torch.nn.Module]],
    shape: ModelShape,
    rope_type: str,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    backbone = _find_transformer_backbone(model)
    shared_rotary = getattr(backbone, "rotary_emb", None)
    values: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    frequency_count = shape.effective_rotary_dim // 2
    for layer_idx, attention in attention_layers:
        rotary = getattr(attention, "rotary_emb", shared_rotary)
        inv_freq = getattr(rotary, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor):
            inv_freq = inv_freq.detach().float().cpu().flatten()
        elif rope_type == "default":
            exponents = torch.arange(
                0, shape.effective_rotary_dim, 2, dtype=torch.float32
            )
            inv_freq = 1.0 / (
                shape.rope_theta ** (exponents / shape.effective_rotary_dim)
            )
        else:
            raise RuntimeError(
                f"scaled RoPE layer {layer_idx} does not expose exact inv_freq"
            )
        if inv_freq.numel() != frequency_count:
            raise RuntimeError(
                f"layer {layer_idx} inv_freq has {inv_freq.numel()} values; "
                f"expected {frequency_count}"
            )
        scaling = getattr(rotary, "attention_scaling", 1.0)
        scale = torch.as_tensor(scaling, dtype=torch.float32).cpu().flatten()
        if scale.numel() == 1:
            scale = scale.expand(frequency_count)
        elif scale.numel() in {shape.head_dim, shape.effective_rotary_dim}:
            scale = scale[:frequency_count]
        elif scale.numel() != frequency_count:
            raise RuntimeError(
                f"layer {layer_idx} attention scaling has {scale.numel()} values"
            )
        values[layer_idx] = (
            inv_freq.contiguous(),
            scale.square().unsqueeze(0).contiguous(),
        )
    return values


def _model_shape_from_hf_config(config: Any) -> ModelShape:
    text_config = getattr(config, "text_config", config)
    wrapper = type("ModelConfig", (), {"hf_text_config": text_config})()
    return model_shape_from_config(wrapper)


def _rope_type(config: Any) -> str:
    config = getattr(config, "text_config", config)
    parameters = getattr(config, "rope_parameters", None)
    scaling = getattr(config, "rope_scaling", None)
    raw = parameters if isinstance(parameters, Mapping) else scaling
    if isinstance(raw, Mapping):
        return str(raw.get("rope_type", raw.get("type", "default")) or "default")
    return "default"


def _load_calibration_text(request: CalibrationRequest) -> tuple[str, str]:
    # 外部文本保持原内容用于来源哈希；没有指定时才生成内置多语种材料。
    # 内置材料按字符数预留长度，最终 token 数仍以 tokenizer 截断结果为准。
    if request.input_path is not None:
        path = request.input_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"calibration input file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"calibration input file is empty: {path}")
        return text, str(path)
    target_characters = request.max_length * 6
    blocks: list[str] = []
    character_count = 0
    round_index = 1
    while character_count < target_characters:
        for section_index, section in enumerate(_BUILTIN_CALIBRATION_SECTIONS, 1):
            heading = f"Calibration round {round_index}, section {section_index}."
            block = f"{heading}\n{section}\n"
            blocks.append(block)
            character_count += len(block)
        round_index += 1
    return "\n".join(blocks), "builtin:multilingual-systems-v1"


def _prepare_device(device: str) -> None:
    if device.startswith("npu"):
        try:
            __import__("torch_npu")
        except ImportError as error:
            raise RuntimeError("NPU calibration requires torch-npu") from error


def _empty_device_cache(device: str, *, all_devices: bool = False) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        current = torch.npu.current_device()
        devices = range(torch.npu.device_count()) if all_devices else (current,)
        for index in devices:
            torch.npu.set_device(index)
            torch.npu.empty_cache()
        torch.npu.set_device(current)
    elif device.startswith("cuda") and torch.cuda.is_available():
        current = torch.cuda.current_device()
        devices = range(torch.cuda.device_count()) if all_devices else (current,)
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.empty_cache()


def _torch_dtype(value: str) -> torch.dtype:
    normalized = _dtype_name(value)
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return choices[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported calibration dtype: {value}") from error


def _dtype_name(value: Any) -> str:
    raw = str(value).replace("torch.", "").lower()
    aliases = {"half": "float16", "float": "float32", "auto": "bfloat16"}
    return aliases.get(raw, raw)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_source_fingerprint(model: str, revision: str | None) -> str:
    """基于本地检查点清单生成指纹，避免重新读取全部权重。"""
    path = Path(model).expanduser()
    if not path.is_dir():
        return (
            "identifier:"
            + hashlib.sha256(f"{model}@{revision or 'default'}".encode()).hexdigest()
        )
    # 此指纹基于文件名、大小、mtime 的本地清单，不读取全部权重内容。
    # 可检测常见 checkpoint 替换，但不是逐字节权重哈希或模型语义等价证明。
    entries: list[tuple[str, int, int]] = []
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.index.json", "config.json")
    candidates = {candidate for pattern in patterns for candidate in path.glob(pattern)}
    for candidate in sorted(candidates):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        entries.append(
            (str(candidate.relative_to(path)), int(stat.st_size), int(stat.st_mtime_ns))
        )
    if not entries:
        raise ValueError(
            f"model directory contains no recognized checkpoint files: {path}"
        )
    return "local-manifest:" + _sha256_json(entries)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate model-matched TriAttention calibration statistics."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--device-map",
        choices=("auto",),
        help="Distribute a large temporary calibration model across visible devices.",
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    request = CalibrationRequest(
        model=args.model,
        output=args.output,
        input_path=args.input_path,
        tokenizer=args.tokenizer,
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        force=args.force,
    )
    generated = generate_calibration_artifact(request)
    print(
        f"{'generated' if generated else 'reused'} calibration artifact: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
