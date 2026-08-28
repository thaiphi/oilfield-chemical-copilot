from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import stat
from threading import Event, Thread, current_thread
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
from oilfield_chemical_copilot.evaluation.index_preflight import IndexFingerprint


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


def _mapping_fixture_bytes() -> tuple[bytes, bytes]:
    mapping = {
        "schema_version": 1,
        "sources": _frame_mapping_sources(),
    }
    mapping_content = (
        json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    binding = {
        "schema_version": 1,
        "reconciliation_run_id": "synthetic-run",
        "reconciliation_binding_sha256": "1" * 64,
        "core_binding_sha256": "2" * 64,
        "supplement_binding_sha256": "3" * 64,
        "e1a3_allocation_sha256": "c" * 64,
        "mapping_payload_sha256": sha256(mapping_content).hexdigest(),
        "source_record_count": 10,
        "unique_locator_count": 98,
        "stratum_locator_counts": {
            "iron_sulfide:foundational": 13,
            "iron_sulfide:supporting": 13,
            "scale:foundational": 12,
            "scale:supporting": 12,
            "corrosion:foundational": 12,
            "corrosion:supporting": 12,
            "paraffin:foundational": 12,
            "paraffin:supporting": 12,
        },
        "allocator_available": True,
        "allocator_slot_count": 96,
        "e1a3_excluded_before_allocation": True,
    }
    binding_content = (
        json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return mapping_content, binding_content


def _fake_mapping_seal(tmp_path: Path) -> tuple[object, str]:
    mapping_content, binding_content = _mapping_fixture_bytes()
    mapping_path = tmp_path / "role-mapping.v1.json"
    binding_path = tmp_path / "mapping-binding.v1.json"
    mapping_path.write_bytes(mapping_content)
    binding_path.write_bytes(binding_content)
    binding = sha256(binding_content).hexdigest()
    return (
        SimpleNamespace(
            artifacts=(
                SimpleNamespace(
                    name="role-mapping.v1.json",
                    path=mapping_path,
                    sha256=sha256(mapping_content).hexdigest(),
                    record_count=10,
                ),
                SimpleNamespace(
                    name="mapping-binding.v1.json",
                    path=binding_path,
                    sha256=binding,
                    record_count=1,
                ),
            ),
            binding_sha256=binding,
        ),
        binding,
    )


def _index_fingerprint(inventory_sha256: str = "d" * 64) -> IndexFingerprint:
    return IndexFingerprint(
        chunk_count=98,
        distinct_source_count=10,
        embedding_models=("synthetic-model",),
        embedding_dimensions=(3,),
        inventory_sha256=inventory_sha256,
    )


def _index_contract_bytes(fingerprint: IndexFingerprint) -> bytes:
    return (
        json.dumps(fingerprint.to_mapping(), sort_keys=True) + "\n"
    ).encode()


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
    _, binding_content = _mapping_fixture_bytes()
    return {
        "reconciliation_root": tmp_path / ".private" / "corpus-reconciliation" / "v1",
        "run_id": "synthetic-run",
        "core_audit_id": "core-audit",
        "supplement_audit_id": "supplement-audit",
        "expected_reconciliation_binding_sha256": "1" * 64,
        "expected_core_binding_sha256": "2" * 64,
        "expected_supplement_binding_sha256": "3" * 64,
        "mapping_root": tmp_path / "mapping",
        "expected_mapping_binding_sha256": sha256(binding_content).hexdigest(),
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
        lambda **_kwargs: order.append("index") or _index_fingerprint(),
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
    Path(kwargs["index_contract_path"]).write_bytes(
        _index_contract_bytes(_index_fingerprint())
    )
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


def test_public_frame_rejects_mapping_swapped_after_trust_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    seal, _ = _fake_mapping_seal(tmp_path)
    mapping_artifact = next(
        artifact
        for artifact in seal.artifacts
        if artifact.name == "role-mapping.v1.json"
    )
    payload = json.loads(mapping_artifact.path.read_text())
    payload["sources"][0]["parser_type"] = "swapped-parser"
    swapped = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    def verify_then_swap(**_kwargs: object) -> object:
        mapping_artifact.path.write_bytes(swapped)
        mapping_artifact.sha256 = sha256(swapped).hexdigest()
        return seal

    monkeypatch.setattr(runner, "_verify_mapping_trust", verify_then_swap)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_MAPPING_INVALID",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert not (tmp_path / "output" / "e1a4" / "sampling-frame" / "v1").exists()


def test_public_frame_allocator_is_called_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    real_allocate = runner.allocate_sampling_slots
    calls = 0

    def allocate_once(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_allocate(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner, "allocate_sampling_slots", allocate_once)

    result = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    assert result.slot_count == 96
    assert calls == 1


@pytest.mark.parametrize("mutation", ["short", "duplicate", "coverage"])
def test_public_frame_rejects_malformed_allocator_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    valid = runner.allocate_sampling_slots(
        slots=build_sampling_slots(),
        sources=runner.mapping_sources_as_sampling_metadata(
            runner.validate_mapping_sources(_frame_mapping_sources())
        ),
    )
    calls = 0

    def malformed(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if mutation == "short":
            return valid[:-1]
        if mutation == "duplicate":
            return (
                valid[0],
                replace(
                    valid[1],
                    source_id=valid[0].source_id,
                    locator=valid[0].locator,
                ),
                *valid[2:],
            )
        return tuple(
            replace(
                item,
                topic="iron_sulfide",
                source_role="foundational",
            )
            for item in valid
        )

    monkeypatch.setattr(runner, "allocate_sampling_slots", malformed)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_ALLOCATION_INVALID",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert calls == 1
    assert not (tmp_path / "output" / "e1a4" / "sampling-frame" / "v1").exists()


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


def test_frame_rejects_index_contract_a_to_b_to_a_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    contract_path = Path(_frame_kwargs(tmp_path)["index_contract_path"])
    original = contract_path.read_bytes()
    verified_fingerprint = _index_fingerprint("e" * 64)

    def verify_different_fingerprint(**_kwargs: object) -> IndexFingerprint:
        contract_path.write_bytes(
            _index_contract_bytes(verified_fingerprint)
        )
        contract_path.write_bytes(original)
        return verified_fingerprint

    monkeypatch.setattr(
        runner, "verify_e1_index_contract", verify_different_fingerprint
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert not (tmp_path / "output" / "e1a4" / "sampling-frame" / "v1").exists()


def test_frame_publication_failure_leaves_no_final_or_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    rename_no_replace = runner._rename_no_replace
    monkeypatch.setattr(
        runner,
        "_rename_no_replace",
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

    monkeypatch.setattr(runner, "_rename_no_replace", rename_no_replace)
    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert runner.verify_current_sampling_frame(
        **_frame_kwargs(tmp_path)
    ) == sealed


def test_frame_retry_removes_abandoned_staging_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    abandoned = parent / ".v1.abandoned.tmp"
    abandoned.mkdir(parents=True)
    (abandoned / "sentinel").write_text("synthetic private material")

    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    assert not abandoned.exists()
    assert not tuple(parent.glob(".v1.*.tmp"))
    assert runner.verify_current_sampling_frame(
        **_frame_kwargs(tmp_path)
    ) == sealed


def test_frame_publisher_lock_fails_closed_then_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    parent.mkdir(parents=True)

    with runner._publisher_lock(parent):
        with pytest.raises(
            runner.E1A4SamplingFrameError,
            match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
        ):
            runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
        assert not (parent / "v1").exists()

    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert runner.verify_current_sampling_frame(
        **_frame_kwargs(tmp_path)
    ) == sealed


def test_frame_rejects_symlink_publisher_lock_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    parent.mkdir(parents=True)
    external = tmp_path / "external-lock-target"
    external.write_bytes(b"synthetic sentinel")
    lock_path = parent / ".v1.publish.lock"
    try:
        lock_path.symlink_to(external)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"file symlinks unavailable: {type(error).__name__}")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    assert external.read_bytes() == b"synthetic sentinel"
    assert not (parent / "v1").exists()


def test_frame_rejects_non_regular_publisher_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    lock_path = parent / ".v1.publish.lock"
    lock_path.mkdir(parents=True)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    assert lock_path.is_dir()
    assert not (parent / "v1").exists()


def test_frame_rejects_injected_windows_reparse_lock_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    parent.mkdir(parents=True)
    lock_path = parent / ".v1.publish.lock"
    lock_path.write_bytes(b"0")
    real_lstat = runner.os.lstat

    class ReparseStat:
        def __init__(self, observed: object) -> None:
            self._observed = observed
            self.st_file_attributes = (
                getattr(observed, "st_file_attributes", 0)
                | stat.FILE_ATTRIBUTE_REPARSE_POINT
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._observed, name)

    def lstat_with_reparse(path: object, *args: object, **kwargs: object) -> object:
        observed = real_lstat(path, *args, **kwargs)
        if Path(path) == lock_path:
            return ReparseStat(observed)
        return observed

    monkeypatch.setattr(runner.os, "lstat", lstat_with_reparse)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    assert lock_path.read_bytes() == b"0"
    assert not (parent / "v1").exists()


def test_failed_publisher_cleans_owned_final_before_releasing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    real_verify = runner.verify_sampling_frame
    real_remove = runner._remove_owned_publication
    cleanup_started = Event()
    publisher_two_done = Event()
    results: dict[str, object] = {}

    def fail_first_verification(**kwargs: object) -> object:
        if current_thread().name == "publisher-1":
            raise runner.E1A4SamplingFrameError(
                "E1A4_SAMPLING_FRAME_VERIFY_FAILED"
            )
        return real_verify(**kwargs)

    def coordinated_remove(final: Path, identity: object) -> None:
        if current_thread().name == "publisher-1":
            cleanup_started.set()
            assert publisher_two_done.wait(10)
        real_remove(final, identity)

    monkeypatch.setattr(runner, "verify_sampling_frame", fail_first_verification)
    monkeypatch.setattr(runner, "_remove_owned_publication", coordinated_remove)

    def publish(name: str) -> None:
        try:
            results[name] = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
        except runner.E1A4SamplingFrameError as error:
            results[name] = error
        finally:
            if name == "publisher-2":
                publisher_two_done.set()

    first = Thread(target=publish, args=("publisher-1",), name="publisher-1")
    second = Thread(target=publish, args=("publisher-2",), name="publisher-2")
    first.start()
    assert cleanup_started.wait(10)
    second.start()
    first.join(10)
    second.join(10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert str(results["publisher-1"]) == "E1A4_SAMPLING_FRAME_VERIFY_FAILED"
    assert str(results["publisher-2"]) == "E1A4_SAMPLING_FRAME_WRITE_FAILED"
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    assert not (parent / "v1").exists()
    assert not tuple(parent.glob(".v1.*.tmp"))


def test_frame_publication_race_preserves_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    final = parent / "v1"

    def concurrent_publish(_staged: Path, destination: Path) -> None:
        assert destination == final
        destination.mkdir()
        (destination / "concurrent-owner").write_text("preserve me")
        raise FileExistsError("private concurrent detail")

    monkeypatch.setattr(
        runner,
        "_rename_no_replace",
        concurrent_publish,
        raising=False,
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert (final / "concurrent-owner").read_text() == "preserve me"
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


@pytest.mark.parametrize("verification_fails", [False, True])
def test_frame_mapping_trust_attempts_every_close_and_blocks_on_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verification_fails: bool,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    seal, _ = _fake_mapping_seal(tmp_path)
    closed: list[str] = []

    class FailingClose:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)
            raise RuntimeError(f"private close detail {self.name}")

    store = FailingClose("store")
    core = FailingClose("core")
    supplement = FailingClose("supplement")
    monkeypatch.setattr(
        runner.ReconciliationStore,
        "open",
        classmethod(lambda _cls, **_kwargs: store),
    )
    monkeypatch.setattr(
        runner.FoundationalAuditStore,
        "open",
        classmethod(lambda _cls, **_kwargs: core),
    )
    monkeypatch.setattr(
        runner.IronSulfideSupplementAuditStore,
        "open",
        classmethod(lambda _cls, **_kwargs: supplement),
    )
    def verify(**_kwargs: object) -> object:
        if verification_fails:
            raise runner.E1A4MappingApplicationError(
                "E1A4_MAPPING_AUTHENTICATION_FAILED"
            )
        return seal

    monkeypatch.setattr(runner, "verify_e1a4_role_mapping", verify)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_CLOSE_FAILED",
    ):
        runner._verify_mapping_trust(**_frame_kwargs(tmp_path))
    assert closed == ["supplement", "core", "store"]


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


def test_sampling_cli_sanitizes_store_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    monkeypatch.setattr(runner, "_presence_preflight", lambda _args: None)
    monkeypatch.setattr(
        runner,
        "seal_sampling_frame",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.E1A4SamplingFrameError(
                "E1A4_SAMPLING_FRAME_CLOSE_FAILED"
            )
        ),
    )

    assert runner.cli(["seal"] + _frame_cli_args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_SAMPLING_FRAME_BLOCKED",
        "error_code": "E1A4_SAMPLING_FRAME_CLOSE_FAILED",
    }


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
