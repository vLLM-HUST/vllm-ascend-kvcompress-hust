"""Versioned random selections, independent of endpoints and service lifecycles."""

import random
import secrets
from pathlib import Path
from uuid import uuid4

from .files import digest, read, write


def select(client, tasks, directory, size=100, seed=None, resample=False):
    if size < 1:
        raise ValueError("sample-size must be positive")
    ids = [t["case_id"] for t in tasks]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("empty inventory or duplicate case IDs")
    identity = {
        "dataset": client.dataset,
        "revision": client.revision,
        "inventory": digest(
            [{"id": t["case_id"], "track": t.get("track", "test")} for t in tasks]
        ),
        "sample_size": size,
    }
    directory = Path(directory)
    pointer = directory / "current.json"
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two arms choosing different samples concurrently.
    lock = directory / ".selection.lock"
    with lock.open("x"):
        try:
            if pointer.exists() and not resample:
                ref = read(pointer)
                if Path(ref["file"]).name != ref["file"]:
                    raise ValueError("unsafe selection reference")
                manifest = read(directory / ref["file"])
                if digest(manifest) != ref["sha256"]:
                    raise ValueError("selection changed after freezing")
                if manifest["identity"] != identity:
                    raise ValueError("selection identity changed; use --resample")
                if seed is not None and seed != manifest["seed"]:
                    raise ValueError("seed differs; use --resample")
                return manifest
            actual_seed = (
                seed if seed is not None else (secrets.randbits(64) if resample else 17)
            )
            rng = random.Random(actual_seed)
            chosen = set()
            for track in sorted({t.get("track", "test") for t in tasks}):
                members = sorted(
                    t["case_id"] for t in tasks if t.get("track", "test") == track
                )
                chosen.update(rng.sample(members, min(size, len(members))))
            manifest = {
                "schema": "kvcompress-smoke/v1",
                "identity": identity,
                "seed": actual_seed,
                "case_ids": [identifier for identifier in ids if identifier in chosen],
                "counts": {
                    track: sum(
                        t["case_id"] in chosen
                        for t in tasks
                        if t.get("track", "test") == track
                    )
                    for track in sorted({t.get("track", "test") for t in tasks})
                },
            }
            name = f"selection-{uuid4().hex}.json"
            write(directory / name, manifest)
            temp = directory / f"current-{uuid4().hex}.json"
            write(temp, {"file": name, "sha256": digest(manifest)})
            temp.replace(pointer)
            return manifest
        finally:
            lock.unlink()
