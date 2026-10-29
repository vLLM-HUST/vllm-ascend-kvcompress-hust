from __future__ import annotations

from collections import UserDict

import pytest

from scripts import kvcompress_benchmark_data as data
from scripts import kvcompress_benchmark_run as run
from scripts import kvcompress_benchmark_score as score
from scripts import kvcompress_long_context_run as long_run


def test_public_source_revisions_and_hashes_are_pinned() -> None:
    assert len(data.SOURCES["longbench-v2"]["revision"]) == 40
    assert len(data.SOURCES["longbench-v2"]["sha256"]) == 64
    assert len(data.SOURCES["longbench"]["revision"]) == 40
    assert len(data.SOURCES["longbench"]["sha256"]) == 64


@pytest.mark.parametrize("runner", [run, long_run], ids=["public", "long-context"])
def test_runners_freeze_the_sampling_protocol(runner) -> None:
    assert runner.FROZEN_SAMPLING_PARAMS == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "n": 1,
        "use_beam_search": False,
        "stop": [],
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": True,
    }


def test_tokenize_accepts_current_transformers_batch_encoding() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            # BatchEncoding 实现 Mapping，但不一定是 dict。
            # 普通 dict 替身无法捕获该回归。
            return UserDict(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])

    assert data._tokenize(Tokenizer(), "prompt", 8) == [1, 2, 3]


def test_tokenize_can_freeze_non_thinking_chat_template() -> None:
    observed = {}

    class Tokenizer:
        def apply_chat_template(self, *_args, **kwargs):
            observed.update(kwargs)
            return [1, 2]

    assert data._tokenize(Tokenizer(), "prompt", 8, disable_thinking=True) == [1, 2]
    assert observed["enable_thinking"] is False


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
