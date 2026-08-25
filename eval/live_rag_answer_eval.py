"""Compare vector and hybrid live RAG answers using public evaluation cases only."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oilfield_chemical_copilot.evaluation.answers import (  # noqa: E402
    evaluate_answer,
    evaluate_cases,
    load_answer_evaluation_cases,
    write_mode_comparison_report,
)
from oilfield_chemical_copilot.evaluation.judge import AnswerJudge  # noqa: E402
from oilfield_chemical_copilot.evaluation.live_rag import (  # noqa: E402
    RecordingRetriever,
    build_live_ollama_generator,
    capture_live_answer,
)
from oilfield_chemical_copilot.evaluation.live_rag_diagnosis import (  # noqa: E402
    LiveDiagnosticObservation,
    classify_live_failure,
    write_live_failure_diagnosis,
)
from oilfield_chemical_copilot.evaluation.citation_diagnostics import (  # noqa: E402
    local_citation_diagnostic,
    write_local_citation_diagnostics,
)
from oilfield_chemical_copilot.evaluation.abstention_policy import (  # noqa: E402
    classify_claim_scope,
)
from oilfield_chemical_copilot.evaluation.live_rag_policy import (  # noqa: E402
    score_policy_counterfactual,
    write_live_rag_policy_investigation,
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


_BASELINE_LLM_PROVIDER = "ollama"
_BASELINE_JUDGE_PROVIDER = "ollama"
_BASELINE_OLLAMA_MODEL = "granite4.1:8b"
_BASELINE_OLLAMA_BASE_URL = "http://localhost:11434"
_BASELINE_EMBEDDING_PROVIDER = "ollama"
_BASELINE_OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
_BASELINE_DATASET_SHA256 = "0271efed1c11af594a6816ab4478632c84a4f630e64575c54f9856089f5fa4d2"
_BASELINE_EMBEDDING_DIMENSION = 384
_BASELINE_RETRIEVAL_SETTINGS = RetrievalSettings()

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare public live RAG answer retrieval modes.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/public_answer_evaluation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/evaluation/live_rag"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--abstention-policy-shadow", choices=("claim_scope_v1",))
    parser.add_argument("--citation-diagnostics", type=Path)
    return parser



def _require_baseline_configuration() -> None:
    expected = (
        ("LLM_PROVIDER", _BASELINE_LLM_PROVIDER),
        ("ANSWER_EVAL_JUDGE_PROVIDER", _BASELINE_JUDGE_PROVIDER),
        ("OLLAMA_MODEL", _BASELINE_OLLAMA_MODEL),
        ("OLLAMA_BASE_URL", _BASELINE_OLLAMA_BASE_URL),
        ("EMBEDDING_PROVIDER", _BASELINE_EMBEDDING_PROVIDER),
        ("OLLAMA_EMBEDDING_MODEL", _BASELINE_OLLAMA_EMBEDDING_MODEL),
    )
    for name, baseline in expected:
        if os.getenv(name, baseline) != baseline:
            raise ValueError(f"{name} must be '{baseline}' for live evaluation")

def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dataset_sha256(dataset_path: Path) -> str:
    """Hash canonical LF bytes so Git and Windows identify the same JSONL dataset."""
    return _sha256(dataset_path.read_bytes().replace(b"\r\n", b"\n"))


def _require_private_diagnostic_destination(destination: Path, private_root: Path) -> Path:
    resolved_destination = destination.resolve()
    try:
        resolved_destination.relative_to(private_root.resolve())
    except ValueError:
        raise ValueError("citation diagnostics must be stored under .private") from None
    return resolved_destination


def _deterministic_results_from_captures(cases, captures):
    return [
        evaluate_answer(
            case,
            cited_evidence_ids=capture.answer.cited_evidence_ids,
            abstained=capture.answer.abstained,
        )
        for case, capture in zip(cases, captures, strict=True)
    ]


def _require_approved_dataset(dataset_path: Path) -> None:
    if _dataset_sha256(dataset_path) != _BASELINE_DATASET_SHA256:
        raise ValueError("approved canonical dataset hash is required for live evaluation")


def _require_approved_runtime_settings(
    embedding_settings: EmbeddingSettings, retrieval_settings: RetrievalSettings
) -> None:
    if embedding_settings.dimension != _BASELINE_EMBEDDING_DIMENSION:
        raise ValueError("approved embedding dimension is required for live evaluation")
    if retrieval_settings != _BASELINE_RETRIEVAL_SETTINGS:
        raise ValueError("approved retrieval settings are required for live evaluation")


def _comparison_provenance(
    dataset_path: Path,
    public_chunk_ids: frozenset[str],
    retrieval_settings: RetrievalSettings,
) -> dict[str, object]:
    model = _BASELINE_OLLAMA_MODEL
    return {
        "dataset_sha256": _dataset_sha256(dataset_path),
        "corpus_sha256": _sha256("\n".join(sorted(public_chunk_ids)).encode("utf-8")),
        "embedding_provider": _BASELINE_EMBEDDING_PROVIDER,
        "generation_provider": _BASELINE_LLM_PROVIDER,
        "judge_provider": _BASELINE_JUDGE_PROVIDER,
        "generation_model_sha256": _sha256(model.encode("utf-8")),
        "judge_model_sha256": _sha256(model.encode("utf-8")),
        "retrieval_mode_settings": {"vector": "vector", "hybrid": "hybrid"},
        "retrieval_settings": {
            "top_k": retrieval_settings.top_k,
            "min_score": retrieval_settings.min_score,
            "max_context_chars": retrieval_settings.max_context_chars,
            "hybrid_candidate_limit": retrieval_settings.hybrid_candidate_limit,
            "hybrid_rrf_k": retrieval_settings.hybrid_rrf_k,
            "hybrid_min_rrf_score": retrieval_settings.hybrid_min_rrf_score,
        },
        "temperature": 0,
        "topic_filter": "none",
    }
def main() -> None:
    args = _build_argument_parser().parse_args()
    canonical_dataset = PROJECT_ROOT / "eval" / "public_answer_evaluation.jsonl"
    if args.dataset.resolve() != canonical_dataset.resolve():
        raise ValueError("only the approved public answer evaluation dataset is allowed")
    if args.citation_diagnostics is not None and args.abstention_policy_shadow is None:
        raise ValueError("citation diagnostics require the claim-scope policy shadow")
    citation_diagnostics_destination = (
        _require_private_diagnostic_destination(
            args.citation_diagnostics, PROJECT_ROOT / ".private"
        )
        if args.citation_diagnostics is not None
        else None
    )
    _require_baseline_configuration()
    _require_approved_dataset(canonical_dataset)

    embedding_settings = EmbeddingSettings.from_env()
    configured_settings = RetrievalSettings.from_env()
    _require_approved_runtime_settings(embedding_settings, configured_settings)

    cases = load_answer_evaluation_cases(canonical_dataset)
    public_chunk_ids = public_sample_chunk_ids()
    store = PgVectorStore(args.database_url, embedding_dimension=embedding_settings.dimension)
    stored_hits = store.list_chunks()
    validate_public_stored_chunk_ids({hit.chunk_id for hit in stored_hits}, public_chunk_ids)
    verified_preflight = True
    policy_decisions = (
        [classify_claim_scope(case.question) for case in cases]
        if args.abstention_policy_shadow is not None
        else None
    )

    embedding_provider = build_embedding_provider()
    keyword_index = KeywordSearchIndex.from_hits(stored_hits)
    recording_generator = build_live_ollama_generator()
    results_by_mode = {}
    diagnoses_by_mode = {}
    deterministic_results_by_mode = {}
    policy_scores_by_mode = {}
    citation_diagnostics_by_mode = {}
    for mode in ("vector", "hybrid"):
        settings = replace(configured_settings, retrieval_mode=mode)
        pipeline = build_retrieval_pipeline(
            store=store,
            embedding_provider=embedding_provider,
            settings=settings,
            keyword_index=keyword_index if mode == "hybrid" else None,
        )
        service = BasicRagService.from_settings(
            retriever=RecordingRetriever(pipeline),
            generator=recording_generator,
            settings=settings,
            apply_claim_scope_policy=False,
        )
        captures = [capture_live_answer(case, service, recording_generator) for case in cases]
        if citation_diagnostics_destination is None:
            results = evaluate_cases(cases, [capture.answer for capture in captures], AnswerJudge())
            results_by_mode[mode] = results
            deterministic = [result.deterministic for result in results]
        else:
            deterministic = _deterministic_results_from_captures(cases, captures)
        deterministic_results_by_mode[mode] = deterministic
        diagnoses_by_mode[mode] = [
            classify_live_failure(
                LiveDiagnosticObservation(
                    expect_citations=case.expect_citations,
                    evidence_sufficient=case.evidence_sufficient,
                    expect_abstention=case.expect_abstention,
                    allowed_evidence_ids=case.allowed_evidence_ids,
                    retrieved_evidence_ids=capture.retrieved_evidence_ids,
                    cited_evidence_ids=capture.answer.cited_evidence_ids,
                    abstained=capture.answer.abstained,
                    generation_outcome=capture.generation_outcome,
                ),
                deterministic_result,
            )
            for case, capture, deterministic_result in zip(cases, captures, deterministic, strict=True)
        ]
        if policy_decisions is not None:
            policy_scores_by_mode[mode] = [
                score_policy_counterfactual(decision, case, capture, deterministic_result)
                for decision, case, capture, deterministic_result in zip(
                    policy_decisions, cases, captures, deterministic, strict=True
                )
            ]
            if citation_diagnostics_destination is not None:
                citation_diagnostics_by_mode[mode] = [
                    local_citation_diagnostic(case, decision, capture)
                    for case, decision, capture in zip(
                        cases, policy_decisions, captures, strict=True
                    )
                ]

    provenance = _comparison_provenance(canonical_dataset, public_chunk_ids, configured_settings)
    if citation_diagnostics_destination is not None:
        write_local_citation_diagnostics(
            citation_diagnostics_by_mode, citation_diagnostics_destination
        )
        print("local citation diagnostics written")
        return
    json_path, markdown_path = write_mode_comparison_report(
        results_by_mode,
        args.output_dir,
        provenance,
    )
    diagnosis_json_path, diagnosis_markdown_path = write_live_failure_diagnosis(
        diagnoses_by_mode,
        deterministic_results_by_mode,
        args.output_dir,
        provenance,
        verified_preflight=verified_preflight,
    )
    print(json_path)
    print(markdown_path)
    print(diagnosis_json_path)
    print(diagnosis_markdown_path)
    if policy_decisions is not None:
        policy_json_path, policy_markdown_path = write_live_rag_policy_investigation(
            policy_scores_by_mode,
            args.output_dir,
            provenance,
            verified_preflight=verified_preflight,
        )
        print(policy_json_path)
        print(policy_markdown_path)


if __name__ == "__main__":
    main()
