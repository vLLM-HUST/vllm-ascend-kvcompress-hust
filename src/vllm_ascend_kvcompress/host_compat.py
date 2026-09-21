# SPDX-License-Identifier: Apache-2.0
"""Explicit compatibility shims for validated host/runtime ABI mismatches."""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Any

from vllm.logger import logger

QWEN_GDN_LIST_COMPAT_ENV = "VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT"
_GDN_OP_NAME = "npu_causal_conv1d_custom"
_GDN_PATCH_MARKER = "_ascend_kvcompress_qwen_gdn_list_compat_v1"
_GDN_METADATA_ARGUMENTS = (
    (5, "query_start_loc_opt"),
    (6, "cache_indices_opt"),
    (7, "initial_state_mode_opt"),
    (8, "num_accepted_tokens_opt"),
)
_MAX_METADATA_CACHE_ENTRIES = 64
_gdn_metadata_cache: OrderedDict[
    int, tuple[weakref.ReferenceType[Any], int | None, list[int]]
] = OrderedDict()


def qwen_gdn_list_compat_enabled() -> bool:
    """Return whether the operator ABI workaround was explicitly requested."""
    return os.getenv(QWEN_GDN_LIST_COMPAT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def clear_qwen_gdn_list_compat_cache() -> None:
    """Start a new model-execution step with fresh device metadata."""
    _gdn_metadata_cache.clear()


def install_qwen_gdn_list_compat(op_namespace: Any | None = None) -> bool:
    """Bridge a legacy ``int[]`` GDN operator ABI without editing the host.

    The current Python host passes device tensors for the four metadata inputs,
    while an older installed ``vllm_ascend_C`` binary declares those inputs as
    integer lists.  This opt-in wrapper is installed only for that exact schema.
    Aligned binaries exposing ``Tensor?`` inputs are left untouched.
    """
    if not qwen_gdn_list_compat_enabled():
        return False

    import torch

    if op_namespace is None:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Ascend custom operators are unavailable")
        op_namespace = torch.ops._C_ascend

    operation = getattr(op_namespace, _GDN_OP_NAME, None)
    if operation is None:
        raise RuntimeError(f"Ascend custom operator {_GDN_OP_NAME!r} is unavailable")
    if getattr(operation, _GDN_PATCH_MARKER, False):
        return True

    overload = getattr(operation, "default", None)
    schema = str(getattr(overload, "_schema", ""))
    if "Tensor? query_start_loc_opt" in schema:
        logger.info("Qwen GDN custom operator already exposes the Tensor metadata ABI")
        return False
    if "int[] query_start_loc_opt" not in schema:
        raise RuntimeError(f"unsupported Qwen GDN custom operator schema: {schema!r}")

    def as_int_list(value: Any, name: str) -> Any:
        if value is None:
            # The legacy schema is non-optional ``int[]``.  Its C++ caller
            # represents absent optional metadata as an empty ArrayRef, so use
            # an empty Python list rather than forwarding ``None``.
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return [int(item) for item in value]
        if not torch.is_tensor(value):
            raise RuntimeError(f"{name} must be a tensor or integer list")
        # Current hybrid-cache metadata carries the complete per-request
        # Mamba block table as a 2-D cache_indices tensor.  The legacy C++ ABI
        # receives the same storage as a flat ArrayRef<int64_t>; preserve that
        # row-major layout when bridging it to Python's int[].  The remaining
        # metadata fields are semantically vectors and must stay one-dimensional.
        if name == "cache_indices_opt":
            value = value.reshape(-1)
        elif value.ndim != 1:
            raise RuntimeError(f"{name} must be one-dimensional")

        identity = id(value)
        try:
            version: int | None = int(value._version)
        except RuntimeError as error:
            if "do not track version counter" not in str(error):
                raise
            version = None
        cached = _gdn_metadata_cache.get(identity)
        if cached is not None and cached[0]() is value and cached[1] == version:
            _gdn_metadata_cache.move_to_end(identity)
            return cached[2]

        converted = [int(item) for item in value.detach().cpu().tolist()]
        _gdn_metadata_cache[identity] = (weakref.ref(value), version, converted)
        _gdn_metadata_cache.move_to_end(identity)
        while len(_gdn_metadata_cache) > _MAX_METADATA_CACHE_ENTRIES:
            _gdn_metadata_cache.popitem(last=False)
        return converted

    def compatible_operation(*args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        for index, name in _GDN_METADATA_ARGUMENTS:
            if index < len(positional):
                positional[index] = as_int_list(positional[index], name)
            elif name in kwargs:
                kwargs[name] = as_int_list(kwargs[name], name)
        return operation(*positional, **kwargs)

    setattr(compatible_operation, _GDN_PATCH_MARKER, True)
    setattr(compatible_operation, f"{_GDN_PATCH_MARKER}_original", operation)
    setattr(op_namespace, _GDN_OP_NAME, compatible_operation)
    logger.warning(
        "Enabled opt-in Qwen GDN list-ABI compatibility for %s; eager execution "
        "is required",
        schema,
    )
    return True
