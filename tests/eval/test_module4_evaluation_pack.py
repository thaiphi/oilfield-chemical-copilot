from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import eval.module4_evaluation_pack as module4_cli
from oilfield_chemical_copilot.evaluation.module4_contract import Module4Case
from oilfield_chemical_copilot.evaluation.module4_live import Module4RuntimeError
from oilfield_chemical_copilot.evaluation.module4_reports import ModeSummary
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


def _arguments(tmp_path: Path, scope: str) -> argparse.Namespace:
    private_root = tmp_path / ".private" / "evaluation" / "module4_handouts"
    return argparse.Namespace(
        scope=scope,
        database_url="postgresql://example",
        output_dir=tmp_path / "public-output",
        sealed_path=private_root / "sealed" / "cases.jsonl",
        digest_path=private_root / "sealed" / "cases.sha256",
        state_path=private_root / "results" / "state.json",
        approval_path=private_root / "review" / "approval.json",
    )


@pytest.fixture(autouse=True)
def _local_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        module4_cli, "PRIVATE_ROOT", tmp_path / ".private" / "evaluation" / "module4_handouts"
    )
    monkeypatch.setattr(module4_cli, "LOCAL_REPORT_DIR", tmp_path / "durable-reports")


def _case() -> Module4Case:
    return Module4Case(
        "case-001",
        "How should this be assessed?",
        "scale",
        ("expected",),
        True,
        False,
        True,
    )


def _summaries() -> dict[str, ModeSummary]:
    return {
        mode: ModeSummary(1, 1.0, 1.0, 1, 0, 1, 0)
        for mode in ("vector", "hybrid")
    }


def test_local_mode_rejects_unsealed_fixture_before_runtime_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = _arguments(tmp_path, "local")
    monkeypatch.setattr(module4_cli, "build_runtime", lambda *_args: pytest.fail("runtime built"))

    with pytest.raises(module4_cli.Module4CliError, match="^SEAL_REQUIRED$"):
        module4_cli.run_local(arguments)


def test_local_mode_consumes_state_before_runtime_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = _arguments(tmp_path, "local")
    dataset = json.dumps(
        {
            "case_id": "case-001",
            "question": "How should this be assessed?",
            "topic": "scale",
            "expected_chunk_ids": ["expected"],
            "expect_citations": True,
            "expect_abstention": False,
            "reviewed": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    arguments.sealed_path.parent.mkdir(parents=True)
    arguments.sealed_path.write_bytes(dataset.encode("utf-8"))
    dataset_sha256 = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
    arguments.digest_path.write_text(dataset_sha256 + "\n", encoding="utf-8")
    arguments.approval_path.parent.mkdir(parents=True)
    arguments.approval_path.write_text(
        json.dumps({"approved": True, "dataset_sha256": dataset_sha256}), encoding="utf-8"
    )
    events: list[str] = []
    monkeypatch.setattr(module4_cli, "verify_seal", lambda *_args: (_case(),))
    monkeypatch.setattr(module4_cli, "load_local_approval", lambda *_args: dataset_sha256)
    monkeypatch.setattr(
        module4_cli,
        "consume_one_shot",
        lambda state_path, digest: events.append(f"state:{state_path.name}:{digest}"),
    )
    monkeypatch.setattr(
        module4_cli,
        "build_runtime",
        lambda *_args, **_kwargs: events.append("runtime") or object(),
    )
    monkeypatch.setattr(module4_cli, "evaluate_module4_modes", lambda *_args, **_kwargs: _summaries())

    report = module4_cli.run_local(arguments)

    assert report["scope"] == "local"
    assert events[0].startswith("state:state.json:")
    assert events[1] == "runtime"
    assert json.loads((module4_cli.PRIVATE_ROOT / "results" / "details.json").read_text()) == {
        "case_statuses": {"case-001": "scored"},
        "dataset_sha256": dataset_sha256,
    }
    assert (
        module4_cli.LOCAL_REPORT_DIR / "2026-08-15-module-4-local-evaluation.md"
    ).is_file()


def test_public_mode_uses_committed_fixture_and_writes_aggregate_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = _arguments(tmp_path, "public")
    monkeypatch.setattr(module4_cli, "load_public_cases", lambda: (_case(),))
    monkeypatch.setattr(module4_cli, "public_dataset_sha256", lambda: "a" * 64)
    monkeypatch.setattr(module4_cli, "build_runtime", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module4_cli, "evaluate_module4_modes", lambda *_args, **_kwargs: _summaries())

    report = module4_cli.run_public(arguments)

    assert report["scope"] == "public"
    assert json.loads((arguments.output_dir / "module4_evaluation.json").read_text()) == report
    markdown = (arguments.output_dir / "module4_evaluation.md").read_text(encoding="utf-8")
    assert "How should this be assessed?" not in markdown


def test_public_runtime_rejects_mixed_manifest_before_embedding_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_chunks(self) -> list[RetrievalHit]:
            return [
                RetrievalHit(
                    chunk_id="unexpected",
                    text="irrelevant",
                    score=1.0,
                    retrieval_method="stored",
                    source_file="public.md",
                    source_path="public.md",
                    topic="scale",
                    parser_type="markdown",
                    page_or_sheet="",
                    chunk_index=0,
                )
            ]

    monkeypatch.setattr(module4_cli.EmbeddingSettings, "from_env", lambda: type("S", (), {"dimension": 384})())
    monkeypatch.setattr(module4_cli, "PgVectorStore", FakeStore)
    monkeypatch.setattr(module4_cli, "public_sample_chunk_ids", lambda: frozenset({"expected"}))
    monkeypatch.setattr(
        module4_cli,
        "build_embedding_provider",
        lambda: pytest.fail("embedding provider must not be built"),
    )

    with pytest.raises(ValueError, match="^stored chunk IDs do not match public manifest"):
        module4_cli.build_runtime("postgresql://example", public_scope=True)


def test_local_mode_rejects_second_score_before_runtime_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = _arguments(tmp_path, "local")
    arguments.sealed_path.parent.mkdir(parents=True)
    arguments.sealed_path.write_text("{}\n", encoding="utf-8")
    arguments.digest_path.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(module4_cli, "verify_seal", lambda *_args: (_case(),))
    monkeypatch.setattr(module4_cli, "load_local_approval", lambda *_args: "a" * 64)
    monkeypatch.setattr(
        module4_cli,
        "consume_one_shot",
        lambda *_args: (_ for _ in ()).throw(module4_cli.Module4CliError("ATTEMPT_UNAVAILABLE")),
    )
    monkeypatch.setattr(module4_cli, "build_runtime", lambda *_args: pytest.fail("runtime built"))

    with pytest.raises(module4_cli.Module4CliError, match="^ATTEMPT_UNAVAILABLE$"):
        module4_cli.run_local(arguments)


def test_local_mode_records_aggregate_unavailable_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = _arguments(tmp_path, "local")
    dataset = b'{"sealed":"fixture"}\n'
    arguments.sealed_path.parent.mkdir(parents=True)
    arguments.sealed_path.write_bytes(dataset)
    dataset_sha256 = hashlib.sha256(dataset).hexdigest()
    arguments.digest_path.write_text(dataset_sha256 + "\n", encoding="utf-8")
    arguments.approval_path.parent.mkdir(parents=True)
    arguments.approval_path.write_text(
        json.dumps({"approved": True, "dataset_sha256": dataset_sha256}), encoding="utf-8"
    )
    monkeypatch.setattr(module4_cli, "verify_seal", lambda *_args: (_case(),))
    monkeypatch.setattr(module4_cli, "consume_one_shot", lambda *_args: None)
    monkeypatch.setattr(module4_cli, "build_runtime", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module4_cli,
        "evaluate_module4_modes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Module4RuntimeError("RUNTIME_UNAVAILABLE")),
    )

    report = module4_cli.run_local(arguments)

    assert report["status"] == "unavailable"
    assert report["modes"]["vector"]["retrieval_case_count"] == 0
    assert json.loads((module4_cli.PRIVATE_ROOT / "results" / "details.json").read_text()) == {
        "case_statuses": {"case-001": "unavailable"},
        "dataset_sha256": dataset_sha256,
    }


def test_versioned_local_run_keeps_details_and_aggregate_report_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = _arguments(tmp_path, "local")
    arguments.sealed_path = arguments.sealed_path.with_name("cases-v2.jsonl")
    arguments.digest_path = arguments.digest_path.with_name("cases-v2.sha256")
    arguments.state_path = arguments.state_path.with_name("state-v2.json")
    arguments.approval_path = arguments.approval_path.with_name("approval-v2.json")
    dataset = b'{"sealed":"fixture-v2"}\n'
    arguments.sealed_path.parent.mkdir(parents=True)
    arguments.sealed_path.write_bytes(dataset)
    dataset_sha256 = hashlib.sha256(dataset).hexdigest()
    arguments.digest_path.write_text(dataset_sha256 + "\n", encoding="utf-8")
    arguments.approval_path.parent.mkdir(parents=True)
    arguments.approval_path.write_text(
        json.dumps({"approved": True, "dataset_sha256": dataset_sha256}), encoding="utf-8"
    )
    monkeypatch.setattr(module4_cli, "verify_seal", lambda *_args: (_case(),))
    monkeypatch.setattr(module4_cli, "consume_one_shot", lambda *_args: None)
    monkeypatch.setattr(module4_cli, "build_runtime", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module4_cli, "evaluate_module4_modes", lambda *_args, **_kwargs: _summaries())

    module4_cli.run_local(arguments)

    assert (module4_cli.PRIVATE_ROOT / "results" / "details-v2.json").is_file()
    assert (
        module4_cli.LOCAL_REPORT_DIR / "2026-08-15-module-4-local-evaluation-v2.md"
    ).is_file()
    assert (module4_cli.LOCAL_REPORT_DIR / "module4_evaluation-v2.json").is_file()
