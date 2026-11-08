"""HLE adapter; reuse the author's client prompts, image handling and Judge."""

from scripts.benchmark.common.clients import Client, download_client

CLIENT = Client(
    "hle-verified",
    "hust-eval-client-hle-verified",
    "534c22cd2d826bed2e8cbcf674bbebf4c3004bc1",
    "hust_eval_hle_verified",
    "b705e0fb541c025a1532ce0d60d70ae2f53b00e0",
    ("gold", "revision"),
)


def acquire(root=None):
    return download_client(CLIENT, root)
