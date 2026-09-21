# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import vllm_ascend_kvcompress


def test_runtime_version_matches_distribution_version() -> None:
    assert vllm_ascend_kvcompress.__version__ == "0.6.0"


def test_wheel_does_not_reresolve_the_hardware_host_stack() -> None:
    project_root = Path(__file__).parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(r"(?m)^dependencies = \[\]$", pyproject)


def test_builtin_method_owns_its_implementation_modules() -> None:
    package_root = Path(vllm_ascend_kvcompress.__file__).parent
    method_root = package_root / "methods" / "triattention"

    assert {path.name for path in method_root.glob("*.py")} >= {
        "__init__.py",
        "cache.py",
        "config.py",
        "method.py",
        "scoring.py",
        "selection.py",
        "stats.py",
    }
    for method_specific_module in ("cache.py", "scoring.py", "stats.py"):
        assert not (package_root / method_specific_module).exists()
