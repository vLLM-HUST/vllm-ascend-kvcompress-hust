# SPDX-License-Identifier: Apache-2.0
"""Identical ABI-only HUST bridges in both benchmark arms and spawned children."""

import runpy
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder

from vllm_ascend_kvcompress.host import compat
from vllm_ascend_kvcompress.host.plugin import _prepare_current_triton_runtime

_prepare_current_triton_runtime()
compat.install_mamba_prefill_end_compat()


class BridgeLoader(Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        compat.install_removed_routed_experts_types()
        self.wrapped.exec_module(module)
        compat.install_step3p5_fp32_linear_alias()
        compat.install_dcp_length_alias()
        compat.install_removed_output_routing_field()
        compat.install_removed_async_output_routing_field()
        cls = module.NPUModelRunner
        cls.uses_xdrope_dim = getattr(cls, "uses_xdrope_dim", 0)
        if not hasattr(cls, "routed_experts_initialized"):
            cls.routed_experts_initialized = False
        original = cls.load_model

        def load_model(self, *args, **kwargs):
            compat.prepare_model_load(self.vllm_config)
            compat.install_removed_model_routing_flag(self.vllm_config)
            return original(self, *args, **kwargs)

        cls.load_model = load_model


class BridgeFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm_ascend.worker.model_runner_v1":
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = BridgeLoader(spec.loader)
        return spec


sys.meta_path.insert(0, BridgeFinder())
if __name__ == "__main__":
    if sys.argv[1:3] != ["-m", "vllm.entrypoints.cli.main"]:
        raise SystemExit("expected -m vllm.entrypoints.cli.main")
    sys.argv = [sys.argv[2], *sys.argv[3:]]
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
