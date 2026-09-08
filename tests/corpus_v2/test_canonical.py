from __future__ import annotations

import pytest

from oilfield_chemical_copilot.corpus_v2.canonical import (
    CorpusV2CanonicalError,
    canonical_jsonl,
    sha256_canonical_jsonl,
)


def test_canonical_jsonl_is_stable_and_newline_terminated() -> None:
    assert canonical_jsonl([{"b": 2, "a": 1}]) == b'{"a":1,"b":2}\n'
    assert sha256_canonical_jsonl([{"a": 1, "b": 2}]) == (
        "e8d38819d39f705646bfb643368eca78f7db476c16471dbc33b941b27326410d"
    )


@pytest.mark.parametrize(
    "record",
    [
        {"score": 1.5},
        {"source_text": "private"},
        {"input_path": "C:/private/document.pdf"},
        {"api_key": "secret"},
        {"body": "private"},
        {"uri": "https://private.example/document"},
        {"password": "secret"},
        {"authorization": "Bearer secret"},
    ],
)
def test_canonical_jsonl_rejects_unsafe_public_aggregate_fields(record: dict[str, object]) -> None:
    with pytest.raises(CorpusV2CanonicalError, match="C2_CANONICAL_INVALID"):
        canonical_jsonl([record])
