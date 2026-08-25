from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oilfield_chemical_copilot.evaluation.retrieval import (  # noqa: E402
    EvaluationCase,
    EvaluationPrivacyMode,
    EvaluationResult,
    first_expected_rank,
    hit_rate_at_k,
    load_evaluation_cases,
    mean_reciprocal_rank,
    public_sample_chunk_ids,
    validate_public_stored_chunk_ids,
)
from oilfield_chemical_copilot.retrieval.embeddings import EmbeddingSettings, build_embedding_provider  # noqa: E402
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex  # noqa: E402
from oilfield_chemical_copilot.retrieval.models import RetrievalHit  # noqa: E402
from oilfield_chemical_copilot.retrieval.pipeline import (  # noqa: E402
    RetrievalSettings,
    build_retrieval_pipeline,
)
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore  # noqa: E402

RetrievalCallable = Callable[[str, str | None], list[RetrievalHit]]
_ALLOWED_MODES = {"keyword", "vector", "hybrid"}


@dataclass(frozen=True)
class RunProvenance:
    dataset_sha256: str
    corpus_sha256: str
    git_revision: str
    retrieval_mode_settings: dict[str, int | float]
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    k: int
    topic_filter: str


def evaluate_cases(
    cases: list[EvaluationCase], retrieve: RetrievalCallable, *, k: int
) -> list[EvaluationResult]:
    """Evaluate a retriever without retaining private hit content or paths."""
    if k < 1:
        raise ValueError("k must be at least 1")

    results: list[EvaluationResult] = []
    for case in cases:
        started_at = time.perf_counter()
        hits = retrieve(case.question, case.topic)
        latency_ms = (time.perf_counter() - started_at) * 1000
        ranked_chunk_ids = tuple(hit.chunk_id for hit in hits)
        results.append(
            EvaluationResult(
                question_id=case.question_id,
                topic=case.topic,
                ranked_chunk_ids=ranked_chunk_ids,
                expected_rank=first_expected_rank(
                    ranked_chunk_ids, frozenset(case.expected_chunk_ids), k
                ),
                latency_ms=latency_ms,
            )
        )
    return results


def write_report(
    results_by_mode: dict[str, list[EvaluationResult]],
    output_dir: Path,
    *,
    privacy_mode: EvaluationPrivacyMode,
    provenance: RunProvenance,
) -> tuple[Path, Path]:
    """Write sanitized evaluation reports for the requested privacy boundary."""
    summaries = {
        mode: _report_summary(results, include_failures=privacy_mode == "public")
        for mode, results in results_by_mode.items()
    }
    report = {
        "privacy_mode": privacy_mode,
        "provenance": asdict(provenance),
        "modes": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_eval.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "retrieval_eval.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _report_summary(
    results: list[EvaluationResult], *, include_failures: bool = True
) -> dict[str, object]:
    summary: dict[str, object] = {
        "questions": len(results),
        "hit_rate_at_3": hit_rate_at_k(results, 3),
        "hit_rate_at_5": hit_rate_at_k(results, 5),
        "mrr_at_5": mean_reciprocal_rank(results, 5),
        "median_latency_ms": median([result.latency_ms for result in results]) if results else 0.0,
    }
    if include_failures:
        summary["failures"] = [
            {
                "question_id": result.question_id,
                "topic": result.topic,
                "expected_rank": result.expected_rank,
            }
            for result in results
            if result.expected_rank is None
        ]
    return summary


def _markdown_report(report: dict[str, object]) -> str:
    provenance = report["provenance"]
    summaries = report["modes"]
    assert isinstance(provenance, dict)
    assert isinstance(summaries, dict)
    lines = [
        "# Retrieval evaluation",
        "",
        f"Privacy mode: {report['privacy_mode']}",
        "",
        f"Topic filter: {provenance['topic_filter']}",
        "",
        "| mode | questions | Hit Rate@3 | Hit Rate@5 | MRR@5 | median latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, summary in summaries.items():
        assert isinstance(summary, dict)
        lines.append(
            "| {mode} | {questions} | {hit_rate_at_3:.3f} | {hit_rate_at_5:.3f} | "
            "{mrr_at_5:.3f} | {median_latency_ms:.3f} |".format(mode=mode, **summary)
        )
    if report["privacy_mode"] == "public":
        lines.extend(["", "## Failures", ""])
        for mode, summary in summaries.items():
            assert isinstance(summary, dict)
            lines.extend([f"### {mode}", "", "| question ID | topic | expected rank |", "| --- | --- | --- |"])
            failures = summary["failures"]
            assert isinstance(failures, list)
            if failures:
                for failure in failures:
                    lines.append("| {question_id} | {topic} | {expected_rank} |".format(**failure))
            else:
                lines.append("| none | none | none |")
            lines.append("")
    return "\n".join(lines)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrieval evaluation modes.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/public_retrieval_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/evaluation"))
    parser.add_argument("--modes", default="keyword,vector,hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--privacy-mode", choices=("public", "private"), default="public")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    return parser


def _validate_expected_chunk_ids(cases: list[EvaluationCase], stored_hits: list[RetrievalHit]) -> None:
    stored_ids = {hit.chunk_id for hit in stored_hits}
    expected_ids = {chunk_id for case in cases for chunk_id in case.expected_chunk_ids}
    missing_ids = sorted(expected_ids - stored_ids)
    if missing_ids:
        raise ValueError("dataset expected chunk IDs are absent from stored chunks")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _corpus_sha256(stored_hits: list[RetrievalHit]) -> str:
    return _sha256_bytes("\n".join(sorted(hit.chunk_id for hit in stored_hits)).encode("utf-8"))


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", revision) else "unknown"


def _provenance(
    dataset: Path,
    stored_hits: list[RetrievalHit],
    settings: RetrievalSettings,
    embedding_provider: object,
) -> RunProvenance:
    embedding_settings = EmbeddingSettings.from_env()
    return RunProvenance(
        dataset_sha256=_sha256_bytes(dataset.read_bytes()),
        corpus_sha256=_corpus_sha256(stored_hits),
        git_revision=_git_revision(),
        retrieval_mode_settings={
            "max_context_chars": settings.max_context_chars,
            "vector_min_score": settings.min_score,
            "hybrid_candidate_limit": settings.hybrid_candidate_limit,
            "hybrid_rrf_k": settings.hybrid_rrf_k,
            "hybrid_min_rrf_score": settings.hybrid_min_rrf_score,
        },
        embedding_provider=embedding_settings.provider,
        embedding_model="sha256:" + _sha256_bytes(
            str(getattr(embedding_provider, "model_name", "unknown")).encode("utf-8")
        ),
        embedding_dimension=int(getattr(embedding_provider, "dimension", 0)),
        k=5,
        topic_filter="oracle_gold_topic",
    )


def main() -> None:
    args = _build_argument_parser().parse_args()
    if args.k != 5:
        raise ValueError("k must be exactly 5 for fixed evaluation metrics")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes or set(modes) - _ALLOWED_MODES:
        raise ValueError("modes must be a comma-separated subset of keyword, vector, hybrid")

    cases = load_evaluation_cases(args.dataset, privacy_mode=args.privacy_mode)
    public_chunk_ids = public_sample_chunk_ids() if args.privacy_mode == "public" else None
    embedding_settings = EmbeddingSettings.from_env()
    store = PgVectorStore(args.database_url, embedding_dimension=embedding_settings.dimension)
    stored_hits = store.list_chunks()
    if public_chunk_ids is not None:
        validate_public_stored_chunk_ids({hit.chunk_id for hit in stored_hits}, public_chunk_ids)
    _validate_expected_chunk_ids(cases, stored_hits)

    embedding_provider = build_embedding_provider()
    keyword_index = KeywordSearchIndex.from_hits(stored_hits)
    configured_settings = RetrievalSettings.from_env()
    results_by_mode: dict[str, list[EvaluationResult]] = {}
    for mode in modes:
        if mode == "keyword":
            def retrieve(question: str, topic: str | None) -> list[RetrievalHit]:
                return keyword_index.search(question, limit=args.k, topic=topic)
        else:
            settings = replace(configured_settings, retrieval_mode=mode, top_k=args.k)
            pipeline = build_retrieval_pipeline(
                store=store,
                embedding_provider=embedding_provider,
                settings=settings,
                keyword_index=keyword_index if mode == "hybrid" else None,
            )
            retrieve = pipeline.retrieve
        results_by_mode[mode] = evaluate_cases(cases, retrieve, k=args.k)

    provenance = _provenance(args.dataset, stored_hits, configured_settings, embedding_provider)
    json_path, markdown_path = write_report(
        results_by_mode,
        args.output_dir,
        privacy_mode=args.privacy_mode,
        provenance=provenance,
    )
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
