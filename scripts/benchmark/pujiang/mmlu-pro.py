"""MMLU-Pro adapter; all prompts and scoring belong to the fixed client."""

from scripts.benchmark.common.clients import Client, download_client

CLIENT = Client(
    "mmlu-pro",
    "hust-eval-client-mmlu-pro",
    "b6215bf76808b1e9d351295e2ecb401542317372",
    "hust_eval_mmlu_pro",
    "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    ("test",),
)


def acquire(root=None):
    return download_client(CLIENT, root)
