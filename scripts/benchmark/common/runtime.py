"""Verify that a comparison changes plugin activation on the same server base."""


def compare_runtime(b0, b1):
    if not b0 or not b1:
        return {"status": "unverified", "reason": "server runtime receipt missing"}
    base0, base1 = b0.get("base"), b1.get("base")
    if not base0 or not base1:
        # Compare the available legacy fields before marking incomplete receipts.
        known0, known1 = base0 or b0, base1 or b1
        for key in ("python", "prefix"):
            if key in known0 and key in known1 and known0[key] != known1[key]:
                raise ValueError("B0/B1 server bases differ")
        packages0, packages1 = known0.get("packages", {}), known1.get("packages", {})
        if any(
            packages0[k] != packages1[k] for k in packages0.keys() & packages1.keys()
        ):
            raise ValueError("B0/B1 server bases differ")
        return {"status": "unverified", "reason": "incomplete server runtime receipt"}
    if base0 != base1:
        raise ValueError("B0/B1 server bases differ")
    required = {"python", "prefix", "packages", "host_config_sha256", "bridge_sha256"}
    packages = {"vllm", "vllm-ascend", "torch", "torch-npu", "triton"}
    if (
        not required <= base0.keys()
        or not all(base0.get(k) for k in required)
        or not packages <= base0["packages"].keys()
    ):
        return {"status": "unverified", "reason": "incomplete server base identity"}
    for receipt, enabled in ((b0, False), (b1, True)):
        mode = receipt.get("plugin", {})
        if mode.get("enabled") is not enabled:
            raise ValueError("B0/B1 plugin activation does not match its arm")
        expected = "ascend,ascend_kvcompress" if enabled else "ascend"
        if mode.get("plugins") != expected:
            raise ValueError("unexpected server plugin allowlist")
        if bool(mode.get("config_sha256")) != enabled:
            raise ValueError("unexpected server plugin configuration")
    if b0.get("server") != b1.get("server"):
        raise ValueError("B0/B1 server settings differ")
    if not b0.get("server"):
        return {"status": "unverified", "reason": "server launch settings missing"}
    return {"status": "verified"}
