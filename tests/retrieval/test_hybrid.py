from __future__ import annotations

import pytest
from pytest import approx

from oilfield_chemical_copilot.retrieval.hybrid import fuse_ranked_hits
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


def _hit(chunk_id: str, retrieval_method: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=0.5,
        retrieval_method=retrieval_method,
        source_file=f"{chunk_id}.md",
        source_path=f"C:/sample/{chunk_id}.md",
        topic="general",
        parser_type="text",
        page_or_sheet="document",
        chunk_index=0,
        metadata={},
    )


def test_fuse_ranked_hits_sums_one_based_ranks_and_keeps_provenance() -> None:
    keyword = [_hit("shared", "keyword"), _hit("keyword-only", "keyword")]
    vector = [_hit("shared", "vector"), _hit("vector-only", "vector")]

    fused = fuse_ranked_hits(keyword, vector, rrf_k=60, limit=5)

    assert [hit.chunk_id for hit in fused] == ["shared", "keyword-only", "vector-only"]
    assert fused[0].score == approx(2 / 61)
    assert fused[0].retrieval_method == "hybrid"
    assert fused[0].metadata == {
        "rrf_methods": ("keyword", "vector"),
        "keyword_rank": 1,
        "vector_rank": 1,
    }


def test_fuse_ranked_hits_keeps_keyword_only_provenance() -> None:
    fused = fuse_ranked_hits([_hit("keyword-only", "keyword")], [], rrf_k=10)

    assert fused[0].score == approx(1 / 11)
    assert fused[0].metadata == {
        "rrf_methods": ("keyword",),
        "keyword_rank": 1,
        "vector_rank": None,
    }


def test_fuse_ranked_hits_keeps_vector_only_provenance() -> None:
    fused = fuse_ranked_hits([], [_hit("vector-only", "vector")], rrf_k=10)

    assert fused[0].score == approx(1 / 11)
    assert fused[0].metadata == {
        "rrf_methods": ("vector",),
        "keyword_rank": None,
        "vector_rank": 1,
    }


def test_fuse_ranked_hits_handles_empty_lists_and_nonpositive_limit() -> None:
    assert fuse_ranked_hits([], []) == []
    assert fuse_ranked_hits([_hit("one", "keyword")], [], limit=0) == []
    assert fuse_ranked_hits([_hit("one", "keyword")], [], limit=-1) == []


def test_fuse_ranked_hits_rejects_nonpositive_rrf_k() -> None:
    with pytest.raises(ValueError, match="^rrf_k must be at least 1$"):
        fuse_ranked_hits([], [], rrf_k=0)


def test_fuse_ranked_hits_ties_sort_by_best_rank_then_chunk_id() -> None:
    fused = fuse_ranked_hits(
        [_hit("z", "keyword"), _hit("a", "keyword")],
        [_hit("a", "vector"), _hit("z", "vector")],
        rrf_k=1,
    )

    assert [hit.chunk_id for hit in fused] == ["a", "z"]
