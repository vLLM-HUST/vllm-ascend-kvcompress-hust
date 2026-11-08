"""Pinned external clients: acquire sources, check installed code, invoke APIs."""

import importlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .files import WORKSPACE, read, sha


@dataclass(frozen=True)
class Client:
    dataset: str
    repository: str
    commit: str
    module: str
    revision: str
    tracks: tuple[str, ...]

    @property
    def url(self):
        return f"https://github.com/vLLM-HUST/{self.repository}.git"

    def source(self, root=None):
        return Path(root or WORKSPACE / "pujiang-benchmark") / self.repository

    def python(self):
        name = self.dataset.replace("frontier-science", "frontierscience")
        return WORKSPACE / ".envs" / f"kvcompress-{name}" / "bin/python"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def check_source(client, root=None):
    path = client.source(root).resolve()
    if not path.is_dir() or git(path, "rev-parse", "--show-toplevel") != str(path):
        raise ValueError(f"missing client checkout: {path}; run download-client")
    if git(path, "rev-parse", "HEAD") != client.commit:
        raise ValueError(f"client revision mismatch: {path}; refusing checkout/reset")
    origin = git(path, "remote", "get-url", "origin")
    if origin.removesuffix(".git") != client.url.removesuffix(".git"):
        raise ValueError(f"unexpected client origin: {origin}")
    if git(path, "status", "--porcelain"):
        raise ValueError(
            f"client checkout has changes: {path}; preserve or relocate it"
        )
    return path


def download_client(client, root=None):
    path = client.source(root)
    if path.exists():
        return check_source(client, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = WORKSPACE / ".cache" / client.repository
    try:
        check_source(client, WORKSPACE / ".cache")
        local = True
    except (OSError, ValueError, subprocess.CalledProcessError):
        local = False
    if local:
        subprocess.run(
            ["git", "clone", "--no-hardlinks", "--no-checkout", str(cached), str(path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "remote", "set-url", "origin", client.url],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "clone", "--no-checkout", client.url, str(path)], check=True
        )
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--detach", client.commit], check=True
    )
    return check_source(client, root)


def check_installed(client, root=None):
    """Run under the client's interpreter; compare installed Python to pinned source."""
    import importlib.metadata as metadata

    source = check_source(client, root)
    distribution = metadata.distribution(client.repository)
    direct = distribution.read_text("direct_url.json")
    info = {} if direct is None else json.loads(direct)
    vcs = info.get("vcs_info", {})
    parsed = urlparse(info.get("url", ""))
    if vcs.get("commit_id") != client.commit and not (
        parsed.scheme == "file" and Path(unquote(parsed.path)).resolve() == source
    ):
        raise ValueError("installed client provenance differs from the fixed checkout")
    module = importlib.import_module(client.module)
    installed = Path(module.__file__).parent
    expected = source / "src" / client.module
    for file in expected.glob("**/*.py"):
        actual = installed / file.relative_to(expected)
        if not actual.is_file() or sha(actual) != sha(file):
            raise ValueError(f"installed client differs from fixed source: {actual}")
    return source


def clean_client_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return env


def contract_identity(plan):
    c = read(Path(plan) / "contract.json")
    # Endpoint/runtime receipts differ by arm; these semantic identities must not.
    return {
        k: c[k]
        for k in (
            "dataset",
            "dataset_revision",
            "declared_revision",
            "scorer_sha256",
            "prompt_template_sha256",
            "generation",
            "scoring_policy",
            "judge_prompts",
            "solver_prompt",
            "judge_prompt",
            "judge_response_format",
            "model_files_sha256",
        )
        if k in c
    }
