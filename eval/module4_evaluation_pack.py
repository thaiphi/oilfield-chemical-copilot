"""Run the Module 4 public or sealed-local RAG evaluation pack."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oilfield_chemical_copilot.evaluation.answers import load_answer_evaluation_cases  # noqa: E402
from oilfield_chemical_copilot.evaluation.live_rag import (  # noqa: E402
    RecordingAnswerGenerator,
    RecordingRetriever,
    build_live_ollama_generator,
)
from oilfield_chemical_copilot.evaluation.module4_contract import (  # noqa: E402
    Module4Case,
    Module4ContractError,
    consume_one_shot,
    verify_seal,
)
from oilfield_chemical_copilot.evaluation.module4_live import (  # noqa: E402
    ModeRuntime,
    Module4RuntimeError,
    evaluate_module4_modes,
)
from oilfield_chemical_copilot.evaluation.module4_reports import (  # noqa: E402
    ModeSummary,
    build_module4_report,
    write_module4_report,
)
from oilfield_chemical_copilot.evaluation.retrieval import (  # noqa: E402
    public_sample_chunk_ids,
    validate_public_stored_chunk_ids,
)
from oilfield_chemical_copilot.rag.service import BasicRagService  # noqa: E402
from oilfield_chemical_copilot.retrieval.embeddings import (  # noqa: E402
    EmbeddingSettings,
    build_embedding_provider,
)
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex  # noqa: E402
from oilfield_chemical_copilot.retrieval.pipeline import (  # noqa: E402
    RetrievalSettings,
    build_retrieval_pipeline,
)
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore  # noqa: E402


PUBLIC_CASES_PATH = PROJECT_ROOT / "eval" / "public_answer_evaluation.jsonl"
PRIVATE_ROOT = PROJECT_ROOT / ".private" / "evaluation" / "module4_handouts"
LOCAL_REPORT_DIR = PROJECT_ROOT / "docs" / "superpowers" / "reports"
_STATE_FILENAME = re.compile(r"state(?P<suffix>-[a-z0-9]+)?\.json\Z")


class Module4CliError(ValueError):
    """A sanitized Module 4 command-line error."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_private_path(path: Path, category: str) -> Path:
    resolved = path.resolve()
    expected_root = (PRIVATE_ROOT / category).resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError:
        raise Module4CliError("PRIVATE_PATH_REQUIRED") from None
    return resolved


def _require_local_paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    return (
        _require_private_path(arguments.sealed_path, "sealed"),
        _require_private_path(arguments.digest_path, "sealed"),
        _require_private_path(arguments.state_path, "results"),
        _require_private_path(arguments.approval_path, "review"),
    )


def _run_suffix(state_path: Path) -> str:
    match = _STATE_FILENAME.fullmatch(state_path.name)
    if match is None:
        raise Module4CliError("STATE_PATH_INVALID")
    return match.group("suffix") or ""


def _load_approved_digest(approval_path: Path, dataset_sha256: str) -> None:
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise Module4CliError("APPROVAL_REQUIRED") from None
    if approval != {"approved": True, "dataset_sha256": dataset_sha256}:
        raise Module4CliError("APPROVAL_REQUIRED")


def load_local_approval(approval_path: Path, dataset_sha256: str) -> str:
    """Validate controller approval without returning its private file contents."""
    _load_approved_digest(approval_path, dataset_sha256)
    return dataset_sha256


def load_public_cases() -> tuple[Module4Case, ...]:
    return tuple(
        Module4Case(
            case.question_id.lower(),
            case.question,
            "public",
            case.allowed_evidence_ids,
            case.expect_citations,
            case.expect_abstention,
            True,
        )
        for case in load_answer_evaluation_cases(PUBLIC_CASES_PATH)
    )


def public_dataset_sha256() -> str:
    return _sha256(PUBLIC_CASES_PATH)


def build_runtime(database_url: str | None, *, public_scope: bool) -> Callable[[str], ModeRuntime]:
    embedding_settings = EmbeddingSettings.from_env()
    configured_settings = RetrievalSettings.from_env()
    store = PgVectorStore(database_url, embedding_dimension=embedding_settings.dimension)
    stored_hits = store.list_chunks()
    if public_scope:
        validate_public_stored_chunk_ids(
            {hit.chunk_id for hit in stored_hits}, public_sample_chunk_ids()
        )
    embedding_provider = build_embedding_provider()
    keyword_index = KeywordSearchIndex.from_hits(stored_hits)

    def build_service(mode: str) -> ModeRuntime:
        settings = replace(configured_settings, retrieval_mode=mode)
        pipeline = build_retrieval_pipeline(
            store=store,
            embedding_provider=embedding_provider,
            settings=settings,
            keyword_index=keyword_index if mode == "hybrid" else None,
        )
        generator = RecordingAnswerGenerator(build_live_ollama_generator())
        return ModeRuntime(
            service=BasicRagService.from_settings(
                retriever=RecordingRetriever(pipeline),
                generator=generator,
                settings=settings,
                apply_claim_scope_policy=True,
            ),
            generator=generator,
        )

    return build_service


def _markdown_report(report: dict[str, object]) -> str:
    modes = report["modes"]
    assert isinstance(modes, dict)
    lines = [
        "# Module 4 Evaluation",
        "",
        f"Scope: {report['scope']}",
        f"Status: {report['status']}",
        "",
        "| mode | retrieval cases | Hit Rate@5 | MRR@5 | citation pass/fail | abstention pass/fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("vector", "hybrid"):
        summary = modes[mode]
        assert isinstance(summary, dict)
        retrieval = summary["retrieval"]
        deterministic = summary["deterministic"]
        assert isinstance(retrieval, dict)
        assert isinstance(deterministic, dict)
        citations = deterministic["citations"]
        abstention = deterministic["abstention"]
        assert isinstance(citations, dict)
        assert isinstance(abstention, dict)
        lines.append(
            "| {mode} | {count} | {hit:.3f} | {mrr:.3f} | {citation_pass}/{citation_fail} | "
            "{abstention_pass}/{abstention_fail} |".format(
                mode=mode,
                count=summary["retrieval_case_count"],
                hit=retrieval["hit_rate_at_5"],
                mrr=retrieval["mrr_at_5"],
                citation_pass=citations["pass"],
                citation_fail=citations["fail"],
                abstention_pass=abstention["pass"],
                abstention_fail=abstention["fail"],
            )
        )
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, object],
    destination: Path,
    *,
    json_name: str = "module4_evaluation.json",
    markdown_name: str = "module4_evaluation.md",
) -> None:
    write_module4_report(report, destination / json_name)
    markdown_path = destination / markdown_name
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")


def write_local_details(
    cases: tuple[Module4Case, ...], dataset_sha256: str, *, status: str, suffix: str = ""
) -> None:
    destination = _require_private_path(
        PRIVATE_ROOT / "results" / f"details{suffix}.json", "results"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "case_statuses": {case.case_id: status for case in cases},
                "dataset_sha256": dataset_sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _unavailable_modes() -> dict[str, ModeSummary]:
    return {
        mode: ModeSummary(0, 0.0, 0.0, 0, 0, 0, 0)
        for mode in ("vector", "hybrid")
    }


def run_public(arguments: argparse.Namespace) -> dict[str, object]:
    cases = load_public_cases()
    runtime = build_runtime(arguments.database_url, public_scope=True)
    report = build_module4_report(
        scope="public",
        dataset_sha256=public_dataset_sha256(),
        modes=evaluate_module4_modes(cases, build_service=runtime),
    )
    write_reports(report, arguments.output_dir)
    return report


def run_local(arguments: argparse.Namespace) -> dict[str, object]:
    sealed_path, digest_path, state_path, approval_path = _require_local_paths(arguments)
    suffix = _run_suffix(state_path)
    if not sealed_path.is_file() or not digest_path.is_file():
        raise Module4CliError("SEAL_REQUIRED")
    try:
        cases = verify_seal(sealed_path, digest_path)
    except Module4ContractError as error:
        raise Module4CliError("SEAL_REQUIRED") from error
    dataset_sha256 = _sha256(sealed_path)
    load_local_approval(approval_path, dataset_sha256)
    try:
        consume_one_shot(state_path, dataset_sha256)
    except Module4ContractError as error:
        raise Module4CliError("ATTEMPT_UNAVAILABLE") from error
    try:
        runtime = build_runtime(arguments.database_url, public_scope=False)
        modes = evaluate_module4_modes(cases, build_service=runtime)
        status = "success"
        detail_status = "scored"
    except Module4RuntimeError:
        modes = _unavailable_modes()
        status = "unavailable"
        detail_status = "unavailable"
    report = build_module4_report(
        scope="local",
        dataset_sha256=dataset_sha256,
        modes=modes,
        status=status,
    )
    write_local_details(cases, dataset_sha256, status=detail_status, suffix=suffix)
    write_reports(
        report,
        LOCAL_REPORT_DIR,
        json_name=f"module4_evaluation{suffix}.json",
        markdown_name=f"2026-08-15-module-4-local-evaluation{suffix}.md",
    )
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Module 4 RAG evaluation.")
    parser.add_argument("--scope", choices=("public", "local"), required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "evaluation" / "module4-public"
    )
    parser.add_argument("--sealed-path", type=Path, default=PRIVATE_ROOT / "sealed" / "cases.jsonl")
    parser.add_argument("--digest-path", type=Path, default=PRIVATE_ROOT / "sealed" / "cases.sha256")
    parser.add_argument("--state-path", type=Path, default=PRIVATE_ROOT / "results" / "state.json")
    parser.add_argument("--approval-path", type=Path, default=PRIVATE_ROOT / "review" / "approval.json")
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    report = run_public(arguments) if arguments.scope == "public" else run_local(arguments)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
