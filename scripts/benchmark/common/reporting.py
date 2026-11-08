"""Compare only compatible complete arms; preserve upstream result semantics."""

from pathlib import Path

from .files import read, write
from .runtime import compare_runtime


def latest(root):
    paths = list(Path(root).glob("summary-*.json"))
    if not paths:
        paths = list(Path(root).glob("accuracy/summary-*.json"))
    if not paths:
        raise ValueError(f"no evaluation summary under {root}")
    records = [(path, read(path)) for path in paths]
    return max(
        records,
        key=lambda item: item[1].get("finished_unix_ns", item[0].stat().st_mtime_ns),
    )[1]


def numeric(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric(item, f"{prefix}.{key}" if prefix else key))
    elif type(value) in (int, float):
        result[prefix] = value
    return result


def deltas(baseline, candidate):
    b, c = numeric(baseline), numeric(candidate)
    return {
        key: {
            "b0": b[key],
            "b1": c[key],
            "difference": c[key] - b[key],
            "percent": 100 * (c[key] / b[key] - 1) if b[key] else None,
        }
        for key in sorted(b.keys() & c.keys())
    }


def compare(baseline, candidate, output):
    direct = [Path(root) / "performance.json" for root in (baseline, candidate)]
    if any(path.exists() for path in direct):
        if not all(path.exists() for path in direct):
            raise ValueError("both arms must have direct performance results")
        p0, p1 = [read(path) for path in direct]
        result = compare_performance(p0, p1)
        write(Path(output) / "comparison.json", result)
        return result
    b0, b1 = latest(baseline), latest(candidate)
    if b0["arm"] != "b0" or b1["arm"] != "b1":
        raise ValueError("comparison requires b0 baseline and b1 candidate")
    for key in ("dataset", "identity", "selection", "smoke", "client_commit"):
        if b0[key] != b1[key]:
            raise ValueError(f"comparison inputs differ: {key}")
    if b0["status"] != "completed" or b1["status"] != "completed":
        raise ValueError("cannot compare failed/incomplete arms")
    result = {
        "dataset": b0["dataset"],
        "smoke": b0["smoke"],
        "selection_sha256": b0["selection_sha256"],
        "quality": deltas(b0["reports"], b1["reports"]),
        "runtime_b0": b0["runtime"],
        "runtime_b1": b1["runtime"],
        "runtime_verification": compare_runtime(b0.get("runtime"), b1.get("runtime")),
        "note": "Same-base plugin-off/on comparison; smoke is not a formal full score.",
    }
    performance = []
    for root in (baseline, candidate):
        paths = list(Path(root).glob("bench/performance.json"))
        performance.append(read(paths[0]) if paths else None)
    if any(performance):
        if not all(performance):
            raise ValueError("only one arm has performance results")
        result.update(compare_performance(*performance))
    write(Path(output) / "comparison.json", result)
    return result


def compare_performance(p0, p1):
    if p0["arm"] != "b0" or p1["arm"] != "b1":
        raise ValueError("performance requires b0 baseline and b1 candidate")
    for key in (
        "dataset",
        "smoke",
        "selection",
        "bench_client",
        "output_budget",
        "identity",
        "tokenizer_files",
    ):
        if p0[key] != p1[key]:
            raise ValueError(f"performance protocol differs: {key}")
    if p0["status"] != "completed" or p1["status"] != "completed":
        raise ValueError("performance run incomplete")
    if set(p0["groups"]) != set(p1["groups"]):
        raise ValueError("performance coverage differs")
    for name, group in p0["groups"].items():
        other = p1["groups"][name]
        if (
            group.get("input_sha256") != other.get("input_sha256")
            or group["selected"] != other["selected"]
        ):
            raise ValueError(f"performance inputs differ: {name}")
    return {
        "dataset": p0["dataset"],
        "smoke": p0["smoke"],
        "runtime_verification": compare_runtime(p0.get("runtime"), p1.get("runtime")),
        "performance": deltas(p0["groups"], p1["groups"]),
        "compression": {"b0": p0["compression"], "b1": p1["compression"]},
    }
