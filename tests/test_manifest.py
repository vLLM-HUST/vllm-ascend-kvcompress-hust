# SPDX-License-Identifier: Apache-2.0

import json
from importlib.resources import files

from vllm_hust_ext.manifest import activation_blocker, parse_manifest


def test_extension_manager_manifest_is_enableable_and_versioned() -> None:
    path = files("vllm_ascend_kvcompress.manifests") / "vllm-hust-extension-v0.2.json"
    raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest = parse_manifest(raw_manifest)

    assert manifest.bundle_id == "org.vllm-hust.ascend-kvcompress"
    assert manifest.bundle_version == "0.4.0"
    assert manifest.host.name == "vllm-ascend"
    assert activation_blocker(manifest) is None
    assert manifest.activation.entry_points[0].name == "ascend_kvcompress"
    assert raw_manifest["implementation"][0]["status"] == "active"
