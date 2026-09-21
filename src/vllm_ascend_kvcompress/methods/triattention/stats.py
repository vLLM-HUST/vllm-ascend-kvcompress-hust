# SPDX-License-Identifier: Apache-2.0
"""Safe TriAttention calibration loading and compatibility checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..base import ModelShape

_FLAT_STAT_KEY = re.compile(r"^layer(?P<layer>\d+)_head(?P<head>\d+)$")
SUPPORTED_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
    }
)


@dataclass(frozen=True)
class LayerCalibrationStats:
    q_mean_real: torch.Tensor
    q_mean_imag: torch.Tensor
    q_abs_mean: torch.Tensor
    freq_scale_sq: torch.Tensor | None = None
    inv_freq: torch.Tensor | None = None
    q_pass_mean: torch.Tensor | None = None


@dataclass(frozen=True)
class DeviceLayerCalibrationStats:
    q_mean_real: torch.Tensor
    q_mean_imag: torch.Tensor
    q_abs_mean: torch.Tensor
    freq_scale_sq: torch.Tensor
    omega: torch.Tensor
    rope_style: str
    rotary_dim: int | None = None
    q_pass_mean: torch.Tensor | None = None


@dataclass(frozen=True)
class CalibrationStats:
    metadata: Mapping[str, Any]
    layers: Mapping[int, LayerCalibrationStats]

    @classmethod
    def load(cls, path: Path) -> CalibrationStats:
        if not path.is_file():
            raise ValueError(f"calibration statistics file does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as error:
            raise ValueError(
                "calibration loading requires torch.load(weights_only=True) support"
            ) from error
        if not isinstance(payload, Mapping):
            raise ValueError("calibration statistics payload must be a mapping")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("calibration metadata must be a mapping")

        flat_stats = payload.get("stats")
        if isinstance(flat_stats, Mapping) and flat_stats:
            layers = _load_flat_stats(flat_stats)
        else:
            layer_stats = payload.get("layer_stats")
            if not isinstance(layer_stats, Mapping) or not layer_stats:
                raise ValueError(
                    "calibration payload must contain non-empty 'stats' or "
                    "'layer_stats'"
                )
            layers = _load_layer_stats(layer_stats)
        return cls(metadata=dict(metadata), layers=layers)

    def validate(self, model: ModelShape) -> tuple[str, ...]:
        reasons: list[str] = []
        if model.model_type not in SUPPORTED_MODEL_TYPES:
            reasons.append(
                f"model_type {model.model_type!r} has no verified RoPE layout"
            )
        expected_layers = set(model.full_attention_layer_indices)
        actual_layers = set(self.layers)
        if actual_layers != expected_layers:
            reasons.append(
                "calibration layer set does not match the model: "
                f"expected {sorted(expected_layers)}, got {sorted(actual_layers)}"
            )

        rope_style = str(self.metadata.get("rope_style", "half"))
        if rope_style != "half":
            reasons.append(
                f"calibration rope_style {rope_style!r} is unsupported; expected 'half'"
            )
        metadata_head_dim = self.metadata.get("head_dim")
        if metadata_head_dim is not None and int(metadata_head_dim) != model.head_dim:
            reasons.append(
                f"calibration head_dim {metadata_head_dim} does not match "
                f"model head_dim {model.head_dim}"
            )
        metadata_rotary_dim = self.metadata.get("rotary_dim")
        if (
            metadata_rotary_dim is not None
            and int(metadata_rotary_dim) != model.effective_rotary_dim
        ):
            reasons.append(
                f"calibration rotary_dim {metadata_rotary_dim} does not match "
                f"model rotary_dim {model.effective_rotary_dim}"
            )
        metadata_attention_layers = self.metadata.get("attention_layer_indices")
        if (
            metadata_attention_layers is not None
            and tuple(int(value) for value in metadata_attention_layers)
            != model.full_attention_layer_indices
        ):
            reasons.append("calibration attention_layer_indices do not match the model")
        metadata_model_type = self.metadata.get("model_type")
        if (
            metadata_model_type is not None
            and str(metadata_model_type) != model.model_type
        ):
            reasons.append(
                f"calibration model_type {metadata_model_type!r} does not match "
                f"model_type {model.model_type!r}"
            )
        metadata_num_layers = self.metadata.get("num_layers")
        if (
            metadata_num_layers is not None
            and int(metadata_num_layers) != model.num_layers
        ):
            reasons.append(
                f"calibration num_layers {metadata_num_layers} does not match "
                f"model num_layers {model.num_layers}"
            )
        metadata_query_heads = self.metadata.get("num_attention_heads")
        if (
            metadata_query_heads is not None
            and int(metadata_query_heads) != model.num_attention_heads
        ):
            reasons.append(
                f"calibration num_attention_heads {metadata_query_heads} does "
                f"not match model value {model.num_attention_heads}"
            )
        metadata_kv_heads = self.metadata.get(
            "num_kv_heads", self.metadata.get("num_key_value_heads")
        )
        if (
            metadata_kv_heads is not None
            and int(metadata_kv_heads) != model.num_kv_heads
        ):
            reasons.append(
                f"calibration num_kv_heads {metadata_kv_heads} does not match "
                f"model value {model.num_kv_heads}"
            )
        metadata_theta = self.metadata.get("rope_theta")
        if metadata_theta is not None and not math.isclose(
            float(metadata_theta), model.rope_theta, rel_tol=1e-7, abs_tol=0.0
        ):
            reasons.append(
                f"calibration rope_theta {metadata_theta} does not match "
                f"model rope_theta {model.rope_theta}"
            )

        rope_type = self.metadata.get("rope_type", "default") or "default"

        for layer_idx, layer in self.layers.items():
            frequency_count = model.effective_rotary_dim // 2
            query_shape = (model.num_attention_heads, frequency_count)
            kv_shape = (model.num_kv_heads, frequency_count)
            for name, tensor in (
                ("q_mean_real", layer.q_mean_real),
                ("q_mean_imag", layer.q_mean_imag),
                ("q_abs_mean", layer.q_abs_mean),
            ):
                if tuple(tensor.shape) not in {query_shape, kv_shape}:
                    reasons.append(
                        f"layer {layer_idx} {name} has shape {tuple(tensor.shape)}; "
                        f"expected {query_shape} or {kv_shape}"
                    )
                if not bool(torch.isfinite(tensor).all().item()):
                    reasons.append(
                        f"layer {layer_idx} {name} contains non-finite values"
                    )
            if bool((layer.q_abs_mean < 0).any().item()):
                reasons.append(f"layer {layer_idx} q_abs_mean contains negative values")
            pass_dim = model.head_dim - model.effective_rotary_dim
            if pass_dim:
                pass_query_shape = (model.num_attention_heads, pass_dim)
                pass_kv_shape = (model.num_kv_heads, pass_dim)
                if layer.q_pass_mean is None:
                    reasons.append(
                        f"layer {layer_idx} partial RoPE requires q_pass_mean"
                    )
                elif tuple(layer.q_pass_mean.shape) not in {
                    pass_query_shape,
                    pass_kv_shape,
                }:
                    reasons.append(
                        f"layer {layer_idx} q_pass_mean has shape "
                        f"{tuple(layer.q_pass_mean.shape)}; expected "
                        f"{pass_query_shape} or {pass_kv_shape}"
                    )
                elif not bool(torch.isfinite(layer.q_pass_mean).all().item()):
                    reasons.append(
                        f"layer {layer_idx} q_pass_mean contains non-finite values"
                    )
            if layer.freq_scale_sq is not None and tuple(
                layer.freq_scale_sq.shape
            ) not in {(1, frequency_count), query_shape, kv_shape}:
                reasons.append(
                    f"layer {layer_idx} freq_scale_sq has incompatible shape "
                    f"{tuple(layer.freq_scale_sq.shape)}"
                )
            if layer.freq_scale_sq is not None:
                if not bool(torch.isfinite(layer.freq_scale_sq).all().item()):
                    reasons.append(
                        f"layer {layer_idx} freq_scale_sq contains non-finite values"
                    )
                if bool((layer.freq_scale_sq <= 0).any().item()):
                    reasons.append(f"layer {layer_idx} freq_scale_sq must be positive")
            if model.has_rope_scaling and layer.freq_scale_sq is None:
                reasons.append(
                    f"layer {layer_idx} scaled RoPE requires freq_scale_sq "
                    "calibration values"
                )

            inv_freq = layer.inv_freq
            if inv_freq is None:
                inv_freq = _get_metadata_inv_freq(self.metadata, layer_idx)
            needs_exact_inv_freq = model.has_rope_scaling or rope_type != "default"
            if inv_freq is None and needs_exact_inv_freq:
                reasons.append(
                    f"layer {layer_idx} scaled or non-default RoPE requires "
                    "exact inv_freq calibration values"
                )
            if inv_freq is not None and inv_freq.numel() != frequency_count:
                reasons.append(
                    f"layer {layer_idx} inv_freq has {inv_freq.numel()} "
                    f"values; expected {frequency_count}"
                )
            if inv_freq is not None:
                if not bool(torch.isfinite(inv_freq).all().item()):
                    reasons.append(
                        f"layer {layer_idx} inv_freq contains non-finite values"
                    )
                if bool((inv_freq <= 0).any().item()):
                    reasons.append(f"layer {layer_idx} inv_freq must be positive")
        return tuple(reasons)

    def to_device(
        self,
        model: ModelShape,
        device: torch.device,
        *,
        tensor_parallel_rank: int = 0,
        tensor_parallel_size: int = 1,
    ) -> dict[int, DeviceLayerCalibrationStats]:
        if tensor_parallel_size <= 0 or not (
            0 <= tensor_parallel_rank < tensor_parallel_size
        ):
            raise ValueError("invalid tensor-parallel rank or size")
        if model.num_kv_heads % tensor_parallel_size:
            raise ValueError(
                "TriAttention tensor parallelism requires KV heads divisible "
                "by tensor-parallel size"
            )
        group_size = model.num_attention_heads // model.num_kv_heads
        rotary_dim = model.effective_rotary_dim
        freq_count = rotary_dim // 2
        rope_style = str(self.metadata.get("rope_style", "half"))
        device_layers: dict[int, DeviceLayerCalibrationStats] = {}
        for layer_idx, layer in self.layers.items():
            q_real = layer.q_mean_real.to(device=device, dtype=torch.float32)
            q_imag = layer.q_mean_imag.to(device=device, dtype=torch.float32)
            q_abs = layer.q_abs_mean.to(device=device, dtype=torch.float32)
            q_real = _group_query_stats(q_real, model, group_size, freq_count)
            q_imag = _group_query_stats(q_imag, model, group_size, freq_count)
            q_abs = _group_query_stats(q_abs, model, group_size, freq_count)

            freq_scale = layer.freq_scale_sq
            if freq_scale is None:
                freq_scale = torch.ones(
                    model.num_attention_heads, freq_count, dtype=torch.float32
                )
            if freq_scale.shape[0] == 1:
                freq_scale = freq_scale.expand(model.num_attention_heads, -1)
            freq_scale = _group_query_stats(
                freq_scale.to(device=device, dtype=torch.float32),
                model,
                group_size,
                freq_count,
            )

            q_pass = layer.q_pass_mean
            if q_pass is not None:
                q_pass = _group_query_stats(
                    q_pass.to(device=device, dtype=torch.float32),
                    model,
                    group_size,
                    model.head_dim - rotary_dim,
                )

            local_kv_heads = model.num_kv_heads // tensor_parallel_size
            shard_start = tensor_parallel_rank * local_kv_heads
            shard_stop = shard_start + local_kv_heads
            q_real = q_real[shard_start:shard_stop].contiguous()
            q_imag = q_imag[shard_start:shard_stop].contiguous()
            q_abs = q_abs[shard_start:shard_stop].contiguous()
            freq_scale = freq_scale[shard_start:shard_stop].contiguous()
            if q_pass is not None:
                q_pass = q_pass[shard_start:shard_stop].contiguous()

            inv_freq = layer.inv_freq
            if inv_freq is None:
                inv_freq = _get_metadata_inv_freq(self.metadata, layer_idx)
            if inv_freq is None:
                exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                inv_freq = 1.0 / (model.rope_theta ** (exponents / float(rotary_dim)))
            omega = inv_freq.to(device=device, dtype=torch.float32).contiguous()
            device_layers[layer_idx] = DeviceLayerCalibrationStats(
                q_mean_real=q_real,
                q_mean_imag=q_imag,
                q_abs_mean=q_abs,
                freq_scale_sq=freq_scale,
                omega=omega,
                rope_style=rope_style,
                rotary_dim=rotary_dim,
                q_pass_mean=q_pass,
            )
        return device_layers


def _group_query_stats(
    tensor: torch.Tensor, model: ModelShape, group_size: int, width: int
) -> torch.Tensor:
    """Group query-head rows by KV head, expanding KV-averaged statistics."""
    if tuple(tensor.shape) == (model.num_attention_heads, width):
        return tensor.reshape(model.num_kv_heads, group_size, width).contiguous()
    if tuple(tensor.shape) == (model.num_kv_heads, width):
        return (
            tensor.unsqueeze(1)
            .expand(model.num_kv_heads, group_size, width)
            .contiguous()
        )
    raise ValueError(
        f"calibration tensor shape {tuple(tensor.shape)} cannot be grouped for "
        f"{model.num_attention_heads} query heads and {model.num_kv_heads} KV heads"
    )


def _as_matrix(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"calibration field {name!r} must be a tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"calibration field {name!r} must be rank two")
    return tensor.contiguous()


def _load_flat_stats(
    raw_stats: Mapping[str, Any],
) -> dict[int, LayerCalibrationStats]:
    grouped: dict[int, dict[int, Mapping[str, Any]]] = {}
    for key, entry in raw_stats.items():
        match = _FLAT_STAT_KEY.match(str(key))
        if match is None:
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"calibration entry {key!r} must be a mapping")
        layer_idx = int(match.group("layer"))
        head_idx = int(match.group("head"))
        grouped.setdefault(layer_idx, {})[head_idx] = entry
    if not grouped:
        raise ValueError("flat calibration statistics contain no layer/head entries")

    layers: dict[int, LayerCalibrationStats] = {}
    for layer_idx, heads in grouped.items():
        expected_heads = list(range(max(heads) + 1))
        if sorted(heads) != expected_heads:
            raise ValueError(
                f"layer {layer_idx} calibration heads must be complete and "
                f"contiguous; got {sorted(heads)}"
            )
        real = torch.stack(
            [
                _as_matrix(heads[i]["q_mean_real"], "q_mean_real")[0]
                for i in expected_heads
            ]
        )
        imag = torch.stack(
            [
                _as_matrix(heads[i]["q_mean_imag"], "q_mean_imag")[0]
                for i in expected_heads
            ]
        )
        absolute = torch.stack(
            [
                _as_matrix(heads[i]["q_abs_mean"], "q_abs_mean")[0]
                for i in expected_heads
            ]
        )
        layers[layer_idx] = LayerCalibrationStats(real, imag, absolute)
    return layers


def _load_layer_stats(
    raw_stats: Mapping[Any, Any],
) -> dict[int, LayerCalibrationStats]:
    layers: dict[int, LayerCalibrationStats] = {}
    for raw_layer_idx, entry in raw_stats.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"layer statistics {raw_layer_idx!r} must be a mapping")
        layer_idx = int(raw_layer_idx)
        q_mean = entry.get("q_mean_complex")
        if isinstance(q_mean, torch.Tensor) and q_mean.is_complex():
            real = _as_matrix(q_mean.real, "q_mean_complex.real")
            imag = _as_matrix(q_mean.imag, "q_mean_complex.imag")
        elif isinstance(q_mean, torch.Tensor) and q_mean.ndim == 3:
            if q_mean.shape[-1] != 2:
                raise ValueError("q_mean_complex final dimension must equal two")
            real = _as_matrix(q_mean[..., 0], "q_mean_complex.real")
            imag = _as_matrix(q_mean[..., 1], "q_mean_complex.imag")
        else:
            real = _as_matrix(entry.get("q_mean_real"), "q_mean_real")
            imag = _as_matrix(entry.get("q_mean_imag"), "q_mean_imag")
        absolute = _as_matrix(entry.get("q_abs_mean"), "q_abs_mean")
        freq_scale = entry.get("freq_scale_sq")
        inv_freq = entry.get("inv_freq")
        q_pass_mean = entry.get("q_pass_mean")
        layers[layer_idx] = LayerCalibrationStats(
            q_mean_real=real,
            q_mean_imag=imag,
            q_abs_mean=absolute,
            freq_scale_sq=(
                _as_matrix(freq_scale, "freq_scale_sq")
                if freq_scale is not None
                else None
            ),
            inv_freq=(
                inv_freq.detach().to(device="cpu", dtype=torch.float32).flatten()
                if isinstance(inv_freq, torch.Tensor)
                else None
            ),
            q_pass_mean=(
                _as_matrix(q_pass_mean, "q_pass_mean")
                if q_pass_mean is not None
                else None
            ),
        )
    return layers


def _get_metadata_inv_freq(
    metadata: Mapping[str, Any], layer_idx: int
) -> torch.Tensor | None:
    layer_values = metadata.get("layer_inv_freqs")
    value: Any = None
    if isinstance(layer_values, Mapping):
        value = layer_values.get(layer_idx, layer_values.get(str(layer_idx)))
    elif isinstance(layer_values, (list, tuple)) and layer_idx < len(layer_values):
        value = layer_values[layer_idx]
    if value is None:
        value = metadata.get("inv_freq")
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).flatten()
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32).flatten()
    return None
