# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import vllm_ascend_kvcompress


def test_runtime_version_matches_distribution_version() -> None:
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    declared = re.search(r'(?m)^version = "([^"]+)"$', project).group(1)
    assert vllm_ascend_kvcompress.__version__ == declared


def test_wheel_does_not_reresolve_the_hardware_host_stack() -> None:
    project_root = Path(__file__).parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(r"(?m)^dependencies = \[\]$", pyproject)
