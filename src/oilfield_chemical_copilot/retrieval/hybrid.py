from __future__ import annotations

from dataclasses import dataclass

from oilfield_chemical_copilot.retrieval.models import RetrievalHit


@dataclass
class _FusedHit:
    hit: RetrievalHit
    score: float = 0.0
    keyword_rank: int | None = None
    vector_rank: int | None = None

    @property
    def best_rank(self) -> int:
        ranks = (rank for rank in (self.keyword_rank, self.vector_rank) if rank is not None)
        return min(ranks)


def fuse_ranked_hits(
    keyword_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    *,
    rrf_k: int = 60,
    limit: int = 5,
) -> list[RetrievalHit]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if limit < 1:
        return []

    fused_hits: dict[str, _FusedHit] = {}
    for rank, hit in enumerate(keyword_hits, start=1):
        fused_hit = fused_hits.setdefault(hit.chunk_id, _FusedHit(hit=hit))
        fused_hit.keyword_rank = rank
        fused_hit.score += 1 / (rrf_k + rank)
    for rank, hit in enumerate(vector_hits, start=1):
        fused_hit = fused_hits.setdefault(hit.chunk_id, _FusedHit(hit=hit))
        fused_hit.vector_rank = rank
        fused_hit.score += 1 / (rrf_k + rank)

    ranked_hits = sorted(
        fused_hits.values(),
        key=lambda fused_hit: (-fused_hit.score, fused_hit.best_rank, fused_hit.hit.chunk_id),
    )
    return [_build_hybrid_hit(fused_hit) for fused_hit in ranked_hits[:limit]]


def _build_hybrid_hit(fused_hit: _FusedHit) -> RetrievalHit:
    hit = fused_hit.hit
    rrf_methods = tuple(
        method
        for method, rank in (("keyword", fused_hit.keyword_rank), ("vector", fused_hit.vector_rank))
        if rank is not None
    )
    return RetrievalHit(
        chunk_id=hit.chunk_id,
        text=hit.text,
        score=fused_hit.score,
        retrieval_method="hybrid",
        source_file=hit.source_file,
        source_path=hit.source_path,
        topic=hit.topic,
        parser_type=hit.parser_type,
        page_or_sheet=hit.page_or_sheet,
        chunk_index=hit.chunk_index,
        metadata={
            "rrf_methods": rrf_methods,
            "keyword_rank": fused_hit.keyword_rank,
            "vector_rank": fused_hit.vector_rank,
        },
    )
