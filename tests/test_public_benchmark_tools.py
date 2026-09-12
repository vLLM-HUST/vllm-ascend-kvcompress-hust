from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = _load("kvcompress_benchmark_data")
score = _load("kvcompress_benchmark_score")


def test_public_source_revisions_and_hashes_are_pinned() -> None:
    assert len(data.SOURCES["longbench-v2"]["revision"]) == 40
    assert len(data.SOURCES["longbench-v2"]["sha256"]) == 64
    assert len(data.SOURCES["longbench"]["revision"]) == 40
    assert len(data.SOURCES["longbench"]["sha256"]) == 64


def test_tokenize_accepts_current_transformers_batch_encoding() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    assert data._tokenize(Tokenizer(), "prompt", 8) == [1, 2, 3]

    class MappingOnly(dict):
        pass

    class MappingTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return MappingOnly(input_ids=[4, 5], attention_mask=[1, 1])

    assert data._tokenize(MappingTokenizer(), "prompt", 8) == [4, 5]


def test_longbench_v2_official_answer_extraction() -> None:
    assert score.longbench_v2_answer("The correct answer is (C)") == "C"
    assert score.longbench_v2_answer("The correct answer is D") == "D"
    assert score.longbench_v2_answer("A or B") is None


def test_longbench_retrieval_metric_matches_official_behavior() -> None:
    assert score.retrieval_score("Paragraph 17", "Paragraph 17") == 1.0
    assert score.retrieval_score("17 or 18", "Paragraph 17") == 0.5
    assert score.retrieval_score("unknown", "Paragraph 17") == 0.0


def test_longbench_qa_f1_normalization() -> None:
    assert score.token_f1("The red fox.", "red fox") == 1.0
    assert score.token_f1("red bird", "red fox") == 0.5
    assert score.token_f1("", "red fox") == 0.0
