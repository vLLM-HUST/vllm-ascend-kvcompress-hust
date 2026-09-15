# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import vllm_ascend_kvcompress


def test_runtime_version_matches_distribution_version() -> None:
    assert vllm_ascend_kvcompress.__version__ == "0.5.0"


def test_builtin_method_owns_its_implementation_modules() -> None:
    package_root = Path(vllm_ascend_kvcompress.__file__).parent
    method_root = package_root / "methods" / "triattention"

    assert {path.name for path in method_root.glob("*.py")} >= {
        "__init__.py",
        "cache.py",
        "config.py",
        "method.py",
        "scoring.py",
        "stats.py",
    }
    for method_specific_module in ("cache.py", "scoring.py", "stats.py"):
        assert not (package_root / method_specific_module).exists()
