# SPDX-License-Identifier: Apache-2.0
"""Generate model-matched TriAttention calibration statistics.

The service hook invokes this module before vLLM loads its model weights.  The
temporary Hugging Face model is therefore released before the serving model is
allocated.  A process lock and atomic rename make startup safe when more than
one worker observes the same missing output path.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
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

from . import __version__
from .methods.base import ModelShape
from .methods.triattention.config import TriAttentionConfig
from .methods.triattention.stats import SUPPORTED_MODEL_TYPES, CalibrationStats
from .model import model_shape_from_config

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
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    force: bool = False


@dataclass
class _LayerAccumulator:
    real_sum: torch.Tensor | None = None
    imag_sum: torch.Tensor | None = None
    abs_sum: torch.Tensor | None = None
    count: int = 0

    def add(self, projected: torch.Tensor, heads: int, head_dim: int) -> None:
        if projected.shape[-1] != heads * head_dim:
            raise RuntimeError(
                "query projection width does not match model attention shape: "
                f"got {projected.shape[-1]}, expected {heads * head_dim}"
            )
        values = projected.detach().reshape(-1, heads, head_dim).float()
        frequency_count = head_dim // 2
        real = values[..., :frequency_count]
        imag = values[..., frequency_count:]
        real_sum = real.sum(dim=0).cpu()
        imag_sum = imag.sum(dim=0).cpu()
        abs_sum = torch.sqrt(real.square() + imag.square()).sum(dim=0).cpu()
        self.real_sum = real_sum if self.real_sum is None else self.real_sum + real_sum
        self.imag_sum = imag_sum if self.imag_sum is None else self.imag_sum + imag_sum
        self.abs_sum = abs_sum if self.abs_sum is None else self.abs_sum + abs_sum
        self.count += values.shape[0]


def ensure_calibration_for_runner(runner: Any, selection: Any) -> bool:
    """Generate and validate a missing artifact before serving weights load."""
    if getattr(selection, "method", None) != "triattention":
        return False
    config = TriAttentionConfig.from_method_config(selection.method_config)
    model_shape = model_shape_from_config(runner.model_config)
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
        dtype=dtype,
        attn_implementation=config.calibration_attn_implementation,
    )
    generated = generate_calibration_artifact(request)
    _validate_artifact(config.stats_path, model_shape, runner.model_config)
    return generated


def generate_calibration_artifact(request: CalibrationRequest) -> bool:
    """Generate one artifact, returning false when another process won the race."""
    if request.max_length < 128:
        raise ValueError("calibration max_length must be at least 128")
    output = request.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if output.is_file() and not request.force:
            _validate_existing_for_request(output, request)
            return False
        payload = _generate_payload(request)
        _atomic_torch_save(payload, output)
    return True


def _generate_payload(request: CalibrationRequest) -> dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers import __version__ as transformers_version
    except ImportError as error:
        raise RuntimeError(
            "calibration generation requires the Transformers package"
        ) from error

    _prepare_device(request.device)
    device = torch.device(request.device)
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

    load_options: dict[str, Any] = {
        "revision": request.revision,
        "dtype": dtype,
        "device_map": request.device,
        **common,
    }
    if request.attn_implementation:
        load_options["attn_implementation"] = request.attn_implementation

    model: torch.nn.Module | None = None
    handles: list[Any] = []
    try:
        model = AutoModel.from_pretrained(request.model, **load_options)
        model.eval()
        attention_layers = _find_attention_layers(model)
        if len(attention_layers) != shape.num_layers:
            raise RuntimeError(
                "located attention layer count does not match model config: "
                f"got {len(attention_layers)}, expected {shape.num_layers}"
            )
        accumulators = [_LayerAccumulator() for _ in attention_layers]
        for layer_idx, attention in enumerate(attention_layers):
            projection = getattr(attention, "q_proj", None)
            if not isinstance(projection, torch.nn.Module):
                raise RuntimeError(
                    f"attention layer {layer_idx} does not expose q_proj"
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
                    output, shape.num_attention_heads, shape.head_dim
                )

            handles.append(projection.register_forward_hook(capture))

        model_inputs = {
            name: value.to(device)
            for name, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
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
        for layer_idx, accumulator in enumerate(accumulators):
            if (
                accumulator.count <= 0
                or accumulator.real_sum is None
                or accumulator.imag_sum is None
                or accumulator.abs_sum is None
            ):
                raise RuntimeError(f"no query values captured for layer {layer_idx}")
            divisor = float(accumulator.count)
            inv_freq, freq_scale_sq = rope_values[layer_idx]
            layer_stats[str(layer_idx)] = {
                "q_mean_real": (accumulator.real_sum / divisor).contiguous(),
                "q_mean_imag": (accumulator.imag_sum / divisor).contiguous(),
                "q_abs_mean": (accumulator.abs_sum / divisor).contiguous(),
                "inv_freq": inv_freq,
                "freq_scale_sq": freq_scale_sq,
            }

        config_payload = (
            config.to_dict() if hasattr(config, "to_dict") else vars(config)
        )
        metadata = {
            "schema_version": 2,
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
        _empty_device_cache(request.device)


def _load_payload(payload: Mapping[str, Any]) -> CalibrationStats:
    """Materialize a short-lived payload for the safe loader/validator path."""
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


def _find_attention_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("cannot locate transformer layers at model.layers")
    attention_layers: list[torch.nn.Module] = []
    for layer_idx, layer in enumerate(list(layers)):
        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, torch.nn.Module):
            raise RuntimeError(f"transformer layer {layer_idx} has no self_attn")
        attention_layers.append(attention)
    return attention_layers


def _extract_rope_values(
    model: torch.nn.Module,
    attention_layers: Sequence[torch.nn.Module],
    shape: ModelShape,
    rope_type: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    backbone = getattr(model, "model", model)
    shared_rotary = getattr(backbone, "rotary_emb", None)
    values: list[tuple[torch.Tensor, torch.Tensor]] = []
    frequency_count = shape.head_dim // 2
    for layer_idx, attention in enumerate(attention_layers):
        rotary = getattr(attention, "rotary_emb", shared_rotary)
        inv_freq = getattr(rotary, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor):
            inv_freq = inv_freq.detach().float().cpu().flatten()
        elif rope_type == "default":
            exponents = torch.arange(0, shape.head_dim, 2, dtype=torch.float32)
            inv_freq = 1.0 / (shape.rope_theta ** (exponents / shape.head_dim))
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
        elif scale.numel() == shape.head_dim:
            scale = scale[:frequency_count]
        elif scale.numel() != frequency_count:
            raise RuntimeError(
                f"layer {layer_idx} attention scaling has {scale.numel()} values"
            )
        values.append((inv_freq.contiguous(), scale.square().unsqueeze(0).contiguous()))
    return values


def _model_shape_from_hf_config(config: Any) -> ModelShape:
    wrapper = type("ModelConfig", (), {"hf_text_config": config})()
    return model_shape_from_config(wrapper)


def _rope_type(config: Any) -> str:
    parameters = getattr(config, "rope_parameters", None)
    scaling = getattr(config, "rope_scaling", None)
    raw = parameters if isinstance(parameters, Mapping) else scaling
    if isinstance(raw, Mapping):
        return str(raw.get("rope_type", raw.get("type", "default")) or "default")
    return "default"


def _load_calibration_text(request: CalibrationRequest) -> tuple[str, str]:
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


def _empty_device_cache(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
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
    """Fingerprint a local checkpoint manifest without rereading all weights."""
    path = Path(model).expanduser()
    if not path.is_dir():
        return (
            "identifier:"
            + hashlib.sha256(f"{model}@{revision or 'default'}".encode()).hexdigest()
        )
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
