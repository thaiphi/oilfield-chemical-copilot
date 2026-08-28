from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    build_sampling_slots,
    private_sampling_payload_digest,
)
from oilfield_chemical_copilot.evaluation.e1a4_sampling import (
    E1A4SamplingError,
    load_e1a3_prior_allocation,
    mapping_sources_as_sampling_metadata,
    validate_mapping_sources,
)


def _mapped(
    source_id: str = "source-a",
    topic: str = "iron_sulfide",
    source_role: str = "foundational",
    locators: tuple[str, ...] = ("page:1",),
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "topic": topic,
        "source_role": source_role,
        "parser_type": "pdf",
        "locators": list(locators),
    }


def _allocation_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "allocations": [
            {
                **slot.to_mapping(),
                "source_id": f"source-{index}",
                "parser_type": "pdf",
                "locator": f"page:{index}",
            }
            for index, slot in enumerate(build_sampling_slots())
        ],
    }


def _sealed_allocation(tmp_path: Path) -> tuple[Path, Path, Path]:
    private_root = tmp_path / "private"
    payload_path = private_root / "allocation.json"
    manifest_path = private_root / "allocation.sha256"
    payload = _allocation_payload()
    payload_path.parent.mkdir()
    payload_path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    manifest_path.write_bytes((private_sampling_payload_digest(payload) + "\n").encode("ascii"))
    return private_root, payload_path, manifest_path


def test_prior_allocation_requires_manifest_and_exact_96_unique_keys(tmp_path: Path) -> None:
    private_root, payload_path, manifest_path = _sealed_allocation(tmp_path)

    result = load_e1a3_prior_allocation(
        payload_path=payload_path,
        manifest_path=manifest_path,
        private_root=private_root,
    )

    assert result.slot_count == 96
    assert len(result.locator_keys) == 96


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda payload: payload.update({"extra": "field"}), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (lambda payload: payload.update({"schema_version": True}), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (
            lambda payload: payload["allocations"][0].pop("parser_type"),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
        (
            lambda payload: payload["allocations"][0].update({"replicate": True}),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
        (lambda payload: payload["allocations"].pop(), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (
            lambda payload: payload["allocations"].__setitem__(
                1, dict(payload["allocations"][0])
            ),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
        (
            lambda payload: payload["allocations"][1].update(
                {
                    "source_id": payload["allocations"][0]["source_id"],
                    "locator": payload["allocations"][0]["locator"],
                }
            ),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
    ],
)
def test_prior_allocation_rejects_invalid_payload_contract(
    tmp_path: Path, mutation: object, expected_code: str
) -> None:
    private_root, payload_path, manifest_path = _sealed_allocation(tmp_path)
    payload = json.loads(payload_path.read_text())
    mutation(payload)  # type: ignore[operator]
    payload_path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    manifest_path.write_bytes((private_sampling_payload_digest(payload) + "\n").encode("ascii"))

    with pytest.raises(E1A4SamplingError, match=expected_code):
        load_e1a3_prior_allocation(
            payload_path=payload_path,
            manifest_path=manifest_path,
            private_root=private_root,
        )


def test_prior_allocation_rejects_manifest_noncanonical_digest_and_external_paths(tmp_path: Path) -> None:
    private_root, payload_path, manifest_path = _sealed_allocation(tmp_path)
    manifest_path.write_bytes(("0" * 64 + "\n").encode("ascii"))

    with pytest.raises(
        E1A4SamplingError, match="E1A4_PRIOR_ALLOCATION_MANIFEST_INVALID"
    ):
        load_e1a3_prior_allocation(
            payload_path=payload_path,
            manifest_path=manifest_path,
            private_root=private_root,
        )

    manifest_path.write_bytes(
        (private_sampling_payload_digest(_allocation_payload()) + "\n").encode("ascii")
    )
    payload_path.write_text(json.dumps(_allocation_payload(), indent=2) + "\n")
    with pytest.raises(
        E1A4SamplingError, match="E1A4_PRIOR_ALLOCATION_MANIFEST_INVALID"
    ):
        load_e1a3_prior_allocation(
            payload_path=payload_path,
            manifest_path=manifest_path,
            private_root=private_root,
        )

    with pytest.raises(E1A4SamplingError, match="E1A4_PRIVATE_PATH_REQUIRED"):
        load_e1a3_prior_allocation(
            payload_path=tmp_path / "external.json",
            manifest_path=manifest_path,
            private_root=private_root,
        )


def test_mapping_sources_preserve_mixed_roles_for_one_source() -> None:
    sources = validate_mapping_sources(
        (
            _mapped("source-a", "iron_sulfide", "supporting", ("page:2",)),
            _mapped("source-a", "iron_sulfide", "foundational", ("page:1",)),
        )
    )

    assert {item.source_role for item in sources} == {"foundational", "supporting"}
    assert tuple(item.source_role for item in sources) == ("foundational", "supporting")
    metadata = mapping_sources_as_sampling_metadata(sources)
    assert {item.source_role for item in metadata} == {"foundational", "supporting"}
    assert {item.eligibility_status for item in metadata} == {"eligible"}


@pytest.mark.parametrize(
    "record",
    [
        {**_mapped(), "extra": "field"},
        _mapped(topic="unknown"),
        _mapped(source_role="unknown"),
        _mapped(locators=()),
        _mapped(locators=("page:2", "page:1")),
        _mapped(locators=("page:1", "page:1")),
    ],
)
def test_mapping_sources_reject_malformed_records(record: dict[str, object]) -> None:
    with pytest.raises(E1A4SamplingError, match="E1A4_MAPPING_SOURCE_INVALID"):
        validate_mapping_sources((record,))


def test_mapping_sources_reject_duplicate_locator_keys_across_records() -> None:
    with pytest.raises(E1A4SamplingError, match="E1A4_MAPPING_SOURCE_DUPLICATE_LOCATOR"):
        validate_mapping_sources(
            (
                _mapped("source-a", locators=("page:1",)),
                _mapped("source-a", "scale", "supporting", ("page:1",)),
            )
        )


def _frame_mapping_sources() -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for topic in ("iron_sulfide", "scale", "corrosion", "paraffin"):
        for role in ("foundational", "supporting"):
            sources.append(
                _mapped(
                    f"{topic}-{role}",
                    topic,
                    role,
                    tuple(f"fresh:{index:02d}" for index in range(1, 13)),
                )
            )
    sources.extend(
        (
            _mapped("mixed", "iron_sulfide", "foundational", ("page:1",)),
            _mapped("mixed", "iron_sulfide", "supporting", ("page:2",)),
        )
    )
    return sources


def _fake_mapping_seal(tmp_path: Path) -> tuple[object, str]:
    mapping = {
        "schema_version": 1,
        "sources": _frame_mapping_sources(),
    }
    path = tmp_path / "role-mapping.v1.json"
    content = (
        json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path.write_bytes(content)
    binding = "b" * 64
    return (
        SimpleNamespace(
            artifacts=(
                SimpleNamespace(
                    name="role-mapping.v1.json",
                    path=path,
                    sha256=sha256(content).hexdigest(),
                    record_count=10,
                ),
            ),
            binding_sha256=binding,
        ),
        binding,
    )


def _fake_prior() -> SimpleNamespace:
    return SimpleNamespace(
        payload_sha256="c" * 64,
        slot_count=96,
        locator_keys=frozenset(
            f"prior-source-{index}:prior:{index}"
            for index in range(96)
        ),
    )


def _frame_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "reconciliation_root": tmp_path / ".private" / "corpus-reconciliation" / "v1",
        "run_id": "synthetic-run",
        "core_audit_id": "core-audit",
        "supplement_audit_id": "supplement-audit",
        "expected_reconciliation_binding_sha256": "1" * 64,
        "expected_core_binding_sha256": "2" * 64,
        "expected_supplement_binding_sha256": "3" * 64,
        "mapping_root": tmp_path / "mapping",
        "expected_mapping_binding_sha256": "b" * 64,
        "e1a3_allocation_path": tmp_path / "e1a3" / "allocation.json",
        "e1a3_allocation_manifest_path": tmp_path / "e1a3" / "allocation.sha256",
        "e1a3_private_root": tmp_path / "e1a3",
        "database_url": "postgresql://private.invalid/evaluation",
        "index_contract_path": tmp_path / "index-contract.json",
        "output_root": tmp_path / "output",
    }


def _frame_cli_args(tmp_path: Path) -> list[str]:
    kwargs = _frame_kwargs(tmp_path)
    return [
        "--reconciliation-root",
        str(kwargs["reconciliation_root"]),
        "--run-id",
        str(kwargs["run_id"]),
        "--core-audit-id",
        str(kwargs["core_audit_id"]),
        "--supplement-audit-id",
        str(kwargs["supplement_audit_id"]),
        "--expected-reconciliation-binding-sha256",
        str(kwargs["expected_reconciliation_binding_sha256"]),
        "--expected-core-binding-sha256",
        str(kwargs["expected_core_binding_sha256"]),
        "--expected-supplement-binding-sha256",
        str(kwargs["expected_supplement_binding_sha256"]),
        "--mapping-root",
        str(kwargs["mapping_root"]),
        "--expected-mapping-binding-sha256",
        str(kwargs["expected_mapping_binding_sha256"]),
        "--e1a3-allocation-path",
        str(kwargs["e1a3_allocation_path"]),
        "--e1a3-allocation-manifest-path",
        str(kwargs["e1a3_allocation_manifest_path"]),
        "--e1a3-private-root",
        str(kwargs["e1a3_private_root"]),
        "--database-url",
        str(kwargs["database_url"]),
        "--index-contract",
        str(kwargs["index_contract_path"]),
        "--output-root",
        str(kwargs["output_root"]),
    ]


def _patch_frame_trust(
    runner: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    seal, _ = _fake_mapping_seal(tmp_path)
    order: list[str] = []
    monkeypatch.setattr(
        runner,
        "_verify_mapping_trust",
        lambda **_kwargs: order.append("mapping") or seal,
    )
    monkeypatch.setattr(
        runner,
        "verify_e1_index_contract",
        lambda **_kwargs: order.append("index") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner,
        "load_e1a3_prior_allocation",
        lambda **_kwargs: order.append("e1a3") or _fake_prior(),
    )
    kwargs = _frame_kwargs(tmp_path)
    presence = (
        Path(kwargs["reconciliation_root"]) / "reconciliation.sqlite",
        Path(kwargs["mapping_root"])
        / "e1a4-role-mapping"
        / "v1"
        / "sealed"
        / "role-mapping.v1.json",
        Path(kwargs["mapping_root"])
        / "e1a4-role-mapping"
        / "v1"
        / "sealed"
        / "role-mapping.v1.json.sha256",
        Path(kwargs["mapping_root"])
        / "e1a4-role-mapping"
        / "v1"
        / "sealed"
        / "mapping-binding.v1.json",
        Path(kwargs["mapping_root"])
        / "e1a4-role-mapping"
        / "v1"
        / "sealed"
        / "mapping-binding.v1.json.sha256",
        Path(kwargs["e1a3_allocation_path"]),
        Path(kwargs["e1a3_allocation_manifest_path"]),
    )
    for path in presence:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    Path(kwargs["index_contract_path"]).write_text("synthetic-contract\n")
    return order


def test_frame_sealer_authenticates_in_order_and_publishes_exact_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    order = _patch_frame_trust(runner, tmp_path, monkeypatch)
    result = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"

    assert order == ["mapping", "index", "e1a3"]
    assert result.source_record_count == 10
    assert result.slot_count == 96
    assert {path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()} == {
        "sealed/source-register.v1.json",
        "sealed/sampling-allocation.v1.json",
        "manifests/source-register.v1.sha256",
        "manifests/sampling-allocation.v1.sha256",
    }
    source_register = json.loads(
        (final / "sealed" / "source-register.v1.json").read_text()
    )
    allocation = json.loads(
        (final / "sealed" / "sampling-allocation.v1.json").read_text()
    )
    assert {
        item["source_role"]
        for item in source_register["sources"]
        if item["source_id"] == "mixed"
    } == {"foundational", "supporting"}
    assert len(allocation["allocations"]) == 96
    assert len(
        {
            (item["source_id"], item["locator"])
            for item in allocation["allocations"]
        }
    ) == 96


def test_frame_verification_recomputes_exact_bytes_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    before = tuple(
        (path.relative_to(final).as_posix(), path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(item for item in final.rglob("*") if item.is_file())
    )

    verified = runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path))

    assert verified == sealed
    assert tuple(
        (path.relative_to(final).as_posix(), path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(item for item in final.rglob("*") if item.is_file())
    ) == before


@pytest.mark.parametrize(
    "relative_path",
    [
        "sealed/source-register.v1.json",
        "sealed/sampling-allocation.v1.json",
        "manifests/source-register.v1.sha256",
        "manifests/sampling-allocation.v1.sha256",
    ],
)
def test_frame_verification_rejects_each_altered_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    (final / relative_path).write_bytes(b"altered-private-value\n")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_BINDING_MISMATCH",
    ):
        runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path))


def test_frame_rejects_e1a3_reuse_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "load_e1a3_prior_allocation",
        lambda **_kwargs: SimpleNamespace(
            payload_sha256="c" * 64,
            slot_count=96,
            locator_keys=frozenset(("iron_sulfide-foundational:fresh:01",)),
        ),
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_E1A3_REUSE",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert not (tmp_path / "output" / "e1a4" / "sampling-frame" / "v1").exists()


def test_frame_rejects_altered_index_contract_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    calls = 0

    def reject_contract(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise runner.E1IndexPreflightError("private altered digest")

    monkeypatch.setattr(runner, "verify_e1_index_contract", reject_contract)
    monkeypatch.setattr(
        runner,
        "allocate_sampling_slots",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("allocator reached")
        ),
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert calls == 1


def test_frame_publication_failure_leaves_no_final_or_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("private path")),
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    assert not (parent / "v1").exists()
    assert not tuple(parent.glob(".v1.*.tmp"))


def test_sampling_preflight_checks_presence_without_opening_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    required = tuple(tmp_path / name for name in ("one", "two", "three"))
    for path in required:
        path.touch()
    monkeypatch.setattr(
        runner,
        "_presence_paths",
        lambda _args: required,
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("payload opened")
        ),
    )

    assert runner.cli(["--preflight"] + _frame_cli_args(tmp_path)) == 0


def test_sampling_cli_sanitizes_unexpected_failure_without_private_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    private = str(tmp_path / "private-source-id-locator-hash")
    monkeypatch.setattr(
        runner,
        "seal_sampling_frame",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(private)),
    )
    monkeypatch.setattr(runner, "_presence_preflight", lambda _args: None)
    assert runner.cli(["seal"] + _frame_cli_args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_SAMPLING_FRAME_BLOCKED",
        "error_code": "E1A4_SAMPLING_FRAME_OPERATION_FAILED",
    }
    assert private not in captured.err


@pytest.mark.parametrize(
    ("command", "status"),
    [
        ("seal", "E1A4_SAMPLING_FRAME_SEALED"),
        ("verify", "E1A4_SAMPLING_FRAME_VERIFIED"),
    ],
)
def test_sampling_cli_emits_exact_aggregate_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    status: str,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    result = runner.E1A4SamplingFrameSeal(
        source_record_count=10,
        sufficient_strata_count=8,
        slot_count=96,
        source_register_sha256="a" * 64,
        allocation_sha256="b" * 64,
    )
    monkeypatch.setattr(runner, "_presence_preflight", lambda _args: None)
    monkeypatch.setattr(runner, "seal_sampling_frame", lambda **_kwargs: result)
    monkeypatch.setattr(
        runner, "verify_current_sampling_frame", lambda **_kwargs: result
    )

    assert runner.cli([command] + _frame_cli_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": status,
        "source_record_count": 10,
        "sufficient_strata_count": 8,
        "slot_count": 96,
    }
