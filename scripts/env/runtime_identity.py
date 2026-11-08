"""Read actual server-environment package provenance without loading device code."""

import hashlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path


def main():
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if name and name not in packages:
            packages[name] = {
                "version": distribution.version,
                "direct_url": json.loads(
                    distribution.read_text("direct_url.json") or "null"
                ),
                "root": str(distribution.locate_file("")),
            }

    def file_hash(path):
        path = Path(path)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    config = os.environ.get("VLLM_ASCEND_KVCOMPRESS_CONFIG")
    print(
        json.dumps(
            {
                "base": {
                    "python": sys.version,
                    "prefix": sys.prefix,
                    "packages": packages,
                    "host_config_sha256": file_hash(
                        Path(sys.prefix) / "kvcompress-host.json"
                    ),
                    "bridge_sha256": file_hash(
                        Path(__file__).with_name("benchmark_host.py")
                    ),
                },
                "plugin": {
                    "enabled": os.environ.get("VLLM_ASCEND_KVCOMPRESS_ENABLED") == "1",
                    "plugins": os.environ.get("VLLM_PLUGINS"),
                    "config_sha256": file_hash(config) if config else None,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
