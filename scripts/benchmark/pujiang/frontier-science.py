"""FrontierScience adapter; preserve both tracks and upstream grading policy."""

from scripts.benchmark.common.clients import Client, download_client

CLIENT = Client(
    "frontier-science",
    "hust-eval-client-frontierscience",
    "ff7b4315876258b5d0ac9b9413ef79ae27146fa4",
    "hust_eval_frontierscience",
    "25ed67db7da8f4591484e764008ff585544f5a30",
    ("olympiad", "research"),
)


def acquire(root=None):
    return download_client(CLIENT, root)
