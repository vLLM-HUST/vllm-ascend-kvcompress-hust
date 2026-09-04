# SPDX-License-Identifier: Apache-2.0
"""Extensible KV-cache compression methods for vLLM-Ascend-HUST."""

from .config import PROVIDER_NAME
from .methods import available_methods, register_method

__all__ = ["PROVIDER_NAME", "available_methods", "register_method"]
__version__ = "0.3.0"
