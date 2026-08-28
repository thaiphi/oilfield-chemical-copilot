from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from threading import Thread, current_thread
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
        "source_register_sha256": "0123456789abcdef" * 4,
        "slot_count": 96,
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
    assert result.payload_sha256 == sha256(payload_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda payload: payload.update({"extra": "field"}), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (lambda payload: payload.update({"schema_version": True}), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (lambda payload: payload.update({"schema_version": 2}), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (
            lambda payload: payload.update({"source_register_sha256": "A" * 64}),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
        (
            lambda payload: payload.update({"source_register_sha256": "g" * 64}),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
        (
            lambda payload: payload.update({"source_register_sha256": "a" * 63}),
            "E1A4_PRIOR_ALLOCATION_INVALID",
        ),
        (lambda payload: payload.update({"slot_count": True}), "E1A4_PRIOR_ALLOCATION_INVALID"),
        (lambda payload: payload.update({"slot_count": 95}), "E1A4_PRIOR_ALLOCATION_INVALID"),
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


@pytest.mark.parametrize(
    "field",
    ("schema_version", "source_register_sha256", "slot_count", "allocations"),
)
def test_prior_allocation_rejects_missing_top_level_field(
    tmp_path: Path, field: str
) -> None:
    private_root, payload_path, manifest_path = _sealed_allocation(tmp_path)
    payload = json.loads(payload_path.read_text())
    payload.pop(field)
    payload_path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    manifest_path.write_bytes((private_sampling_payload_digest(payload) + "\n").encode("ascii"))

    with pytest.raises(E1A4SamplingError, match="E1A4_PRIOR_ALLOCATION_INVALID"):
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


def test_frame_publication_failure_preserves_owned_staging_for_manual_review(
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
    assert len(tuple(parent.glob(".v1.*.tmp"))) == 1

    monkeypatch.setattr(runner, "_rename_no_replace", rename_no_replace)
    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))


def test_frame_blocks_and_preserves_abandoned_staging_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    abandoned = parent / ".v1.abandoned.tmp"
    abandoned.mkdir(parents=True)
    (abandoned / "sentinel").write_text("synthetic private material")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    assert (abandoned / "sentinel").read_text() == "synthetic private material"
    assert not (parent / "v1").exists()


def test_frame_verifier_rejects_symlinked_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    target = tmp_path / "external-frame"
    final.rename(target)
    try:
        final.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_VERIFY_FAILED",
    ):
        runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path))


def test_frame_verifier_rejects_symlinked_sealed_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    member = final / "sealed" / runner.SOURCE_REGISTER_NAME
    replacement = tmp_path / "external-member"
    replacement.write_bytes(member.read_bytes())
    member.unlink()
    try:
        member.symlink_to(replacement)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_VERIFY_FAILED",
    ):
        runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX FIFO")
def test_frame_verifier_rejects_fifo_member_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing

    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    member = final / "sealed" / runner.SOURCE_REGISTER_NAME
    member.unlink()
    os.mkfifo(member)

    def verify_in_child() -> None:
        try:
            runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path))
        except runner.E1A4SamplingFrameError:
            return
        raise AssertionError("FIFO member was accepted")

    process = multiprocessing.Process(target=verify_in_child)
    process.start()
    process.join(1)
    blocked = process.is_alive()
    if blocked:
        process.terminate()
        process.join(5)
    assert not blocked
    assert process.exitcode == 0


def test_abandoned_staging_detection_does_not_resolve_or_delete_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "parent"
    parent.mkdir()
    candidate = parent / ".v1.abandoned.tmp"
    candidate.mkdir()
    (candidate / "owned").write_text("owned")
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("staging detection must not resolve")
        ),
    )

    with pytest.raises(OSError):
        runner._remove_abandoned_staging(parent)

    assert (candidate / "owned").read_text() == "owned"


class _FakeFrameWindowsReader:
    def __init__(
        self,
        members: dict[str, bytes],
        *,
        unsafe_member: str | None = None,
        replaced_member: str | None = None,
        unsafe_root: bool = False,
        replaced_root: bool = False,
        fail_close: object | None = None,
    ) -> None:
        self.members = members
        self.unsafe_member = unsafe_member
        self.replaced_member = replaced_member
        self.unsafe_root = unsafe_root
        self.replaced_root = replaced_root
        self.fail_close = fail_close
        self.root_validations = 0
        self.opens: dict[str, int] = {}
        self.closed: list[object] = []

    def open_directory(self, _path: Path) -> str:
        if self.unsafe_root:
            raise OSError("frame directory is reparse")
        return "root"

    def open_child(self, _parent: object, name: str) -> str:
        return name

    def _validate_handle(self, handle: object, *, directory: bool) -> tuple[object, ...]:
        if directory:
            if handle == "root":
                self.root_validations += 1
                return (
                    handle,
                    "replacement"
                    if self.replaced_root and self.root_validations > 1
                    else "original",
                )
            return (handle, "directory")
        raise AssertionError("member validation uses member_snapshot")

    def directory_entries(self, handle: object) -> set[str]:
        if handle == "root":
            return {"sealed", "manifests"}
        return {
            key.removeprefix(f"{handle}/")
            for key in self.members
            if key.startswith(f"{handle}/")
        }

    def open_member(self, directory: object, name: str) -> tuple[str, int]:
        key = f"{directory}/{name}"
        self.opens[key] = self.opens.get(key, 0) + 1
        return key, self.opens[key]

    def member_snapshot(self, handle: tuple[str, int]) -> tuple[object, ...]:
        key, generation = handle
        if key == self.unsafe_member:
            raise OSError("member is reparse")
        identity = (
            "replacement"
            if key == self.replaced_member and generation > 1
            else "original"
        )
        return key, identity, len(self.members[key])

    def read_member(self, handle: tuple[str, int]) -> bytes:
        return self.members[handle[0]]

    def close_handle(self, handle: object) -> None:
        self.closed.append(handle)
        if handle == self.fail_close:
            raise OSError("synthetic close failure")


def test_frame_windows_reader_rejects_reparse_member_via_native_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    members = {
        path.relative_to(final).as_posix(): path.read_bytes()
        for path in final.rglob("*")
        if path.is_file()
    }
    api = _FakeFrameWindowsReader(
        members, unsafe_member=f"sealed/{runner.SOURCE_REGISTER_NAME}"
    )
    monkeypatch.setattr(
        runner._mapping_application, "_windows_seal_reader_api", lambda: api
    )
    monkeypatch.setattr(
        runner,
        "_open_windows_child_directory",
        lambda _api, parent, name: api.open_child(parent, name),
    )

    with pytest.raises(OSError, match="member is reparse"):
        runner._read_windows_frame_members(final)

    assert f"sealed/{runner.SOURCE_REGISTER_NAME}" in api.opens
    assert "root" in api.closed


def test_frame_windows_reader_rejects_reparse_final_directory_via_native_seam(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    api = _FakeFrameWindowsReader({}, unsafe_root=True)
    monkeypatch.setattr(
        runner._mapping_application, "_windows_seal_reader_api", lambda: api
    )

    with pytest.raises(OSError, match="directory is reparse"):
        runner._read_windows_frame_members(Path("frame"))

    assert not api.closed


def test_frame_windows_reader_rejects_member_replacement_via_native_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    final = tmp_path / "output" / "e1a4" / "sampling-frame" / "v1"
    members = {
        path.relative_to(final).as_posix(): path.read_bytes()
        for path in final.rglob("*")
        if path.is_file()
    }
    replaced = f"sealed/{runner.SOURCE_REGISTER_NAME}"
    api = _FakeFrameWindowsReader(members, replaced_member=replaced)
    monkeypatch.setattr(
        runner._mapping_application, "_windows_seal_reader_api", lambda: api
    )
    monkeypatch.setattr(
        runner,
        "_open_windows_child_directory",
        lambda _api, parent, name: api.open_child(parent, name),
    )

    with pytest.raises(OSError, match="frame member changed"):
        runner._read_windows_frame_members(final)

    assert api.opens[replaced] == 2


def test_frame_windows_reader_rejects_final_directory_replacement_via_native_seam(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    members = {
        f"sealed/{runner.SOURCE_REGISTER_NAME}": b"source",
        f"sealed/{runner.ALLOCATION_NAME}": b"allocation",
        f"manifests/{runner._manifest_name(runner.SOURCE_REGISTER_NAME)}": b"digest\n",
        f"manifests/{runner._manifest_name(runner.ALLOCATION_NAME)}": b"digest\n",
    }
    api = _FakeFrameWindowsReader(members, replaced_root=True)
    monkeypatch.setattr(
        runner._mapping_application, "_windows_seal_reader_api", lambda: api
    )
    monkeypatch.setattr(
        runner,
        "_open_windows_child_directory",
        lambda _api, parent, name: api.open_child(parent, name),
    )

    with pytest.raises(OSError, match="frame directory changed"):
        runner._read_windows_frame_members(Path("frame"))


def test_frame_windows_reader_attempts_all_closes_after_close_failure(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    members = {
        f"sealed/{runner.SOURCE_REGISTER_NAME}": b"source",
        f"sealed/{runner.ALLOCATION_NAME}": b"allocation",
        f"manifests/{runner._manifest_name(runner.SOURCE_REGISTER_NAME)}": b"digest\n",
        f"manifests/{runner._manifest_name(runner.ALLOCATION_NAME)}": b"digest\n",
    }
    api = _FakeFrameWindowsReader(members, fail_close="manifests")
    monkeypatch.setattr(
        runner._mapping_application, "_windows_seal_reader_api", lambda: api
    )
    monkeypatch.setattr(
        runner,
        "_open_windows_child_directory",
        lambda _api, parent, name: api.open_child(parent, name),
    )

    with pytest.raises(OSError, match="synthetic close failure"):
        runner._read_windows_frame_members(Path("frame"))

    assert api.closed[-1] == "root"


class _FakeFramePosixOS:
    O_RDONLY = 0x01
    O_CLOEXEC = 0x02
    O_DIRECTORY = 0x04
    O_NOFOLLOW = 0x08
    O_NONBLOCK = 0x10

    def __init__(self, members: dict[str, bytes], *, fifo: str) -> None:
        self.members = members
        self.fifo = fifo
        self.events: list[tuple[object, ...]] = []
        self.next_fd = 10
        self.handles: dict[int, str] = {}
        self.read_done: set[int] = set()

    def open(self, path: object, flags: int, *, dir_fd: int | None = None) -> int:
        name = str(path)
        self.events.append(("open", name, flags, dir_fd))
        if dir_fd is not None and "/" in self.handles[dir_fd]:
            if name == self.fifo:
                raise AssertionError("blocking FIFO open attempted")
            key = f"{self.handles[dir_fd]}/{name}"
        elif dir_fd is not None and name in {"sealed", "manifests"}:
            key = name
        elif dir_fd is None:
            key = "root"
        else:
            key = f"{self.handles[dir_fd]}/{name}"
        fd = self.next_fd
        self.next_fd += 1
        self.handles[fd] = key
        return fd

    def fstat(self, fd: int) -> object:
        key = self.handles[fd]
        if key in {"root", "sealed", "manifests"}:
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=fd)
        content = self.members[key]
        return SimpleNamespace(
            st_mode=(stat.S_IFIFO if key.endswith(self.fifo) else stat.S_IFREG) | 0o600,
            st_nlink=1,
            st_dev=1,
            st_ino=100 + sorted(self.members).index(key),
            st_size=len(content),
            st_mtime_ns=1,
            st_ctime_ns=1,
        )

    def stat(self, path: object, *, dir_fd: int, follow_symlinks: bool) -> object:
        key = f"{self.handles[dir_fd]}/{path}"
        self.events.append(("stat", key, follow_symlinks))
        content = self.members[key]
        return SimpleNamespace(
            st_mode=(stat.S_IFIFO if key.endswith(self.fifo) else stat.S_IFREG) | 0o600,
            st_nlink=1,
            st_dev=1,
            st_ino=100 + sorted(self.members).index(key),
            st_size=len(content),
            st_mtime_ns=1,
            st_ctime_ns=1,
        )

    def listdir(self, fd: int) -> list[str]:
        key = self.handles[fd]
        if key == "root":
            return ["sealed", "manifests"]
        return [
            item.removeprefix(f"{key}/")
            for item in self.members
            if item.startswith(f"{key}/")
        ]

    def read(self, fd: int, _count: int) -> bytes:
        key = self.handles[fd]
        if fd in self.read_done:
            return b""
        self.read_done.add(fd)
        return self.members[key]

    def close(self, _fd: int) -> None:
        return None


def test_frame_posix_reader_rejects_fifo_before_opening_it() -> None:
    import eval.seal_e1a4_sampling_frame as runner

    fifo = runner.SOURCE_REGISTER_NAME
    members = {
        f"sealed/{runner.SOURCE_REGISTER_NAME}": b"source",
        f"sealed/{runner.ALLOCATION_NAME}": b"allocation",
        f"manifests/{runner._manifest_name(runner.SOURCE_REGISTER_NAME)}": b"digest\n",
        f"manifests/{runner._manifest_name(runner.ALLOCATION_NAME)}": b"digest\n",
    }
    fake_os = _FakeFramePosixOS(members, fifo=fifo)

    with pytest.raises(OSError, match="unsafe sealed member"):
        runner._read_posix_frame_members(Path("frame"), os_api=fake_os)

    assert ("stat", f"sealed/{fifo}", False) in fake_os.events
    assert not any(
        event[:2] == ("open", fifo) for event in fake_os.events
    )


def test_frame_publisher_lock_fails_closed_then_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    parent.mkdir(parents=True)
    result: list[object] = []

    def contend() -> None:
        try:
            result.append(runner.seal_sampling_frame(**_frame_kwargs(tmp_path)))
        except runner.E1A4SamplingFrameError as error:
            result.append(error)

    with runner._publisher_lock(parent):
        contender = Thread(target=contend)
        contender.start()
        contender.join(10)
        assert not contender.is_alive()
        assert [str(item) for item in result] == [
            "E1A4_SAMPLING_FRAME_WRITE_FAILED"
        ]
        assert not (parent / "v1").exists()

    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert runner.verify_current_sampling_frame(
        **_frame_kwargs(tmp_path)
    ) == sealed


def _fake_windows_mutex(
    wait_result: int,
    on_create: object | None = None,
    create_error: Exception | None = None,
) -> tuple[SimpleNamespace, list[tuple[object, ...]]]:
    events: list[tuple[object, ...]] = []

    def owner_sid() -> str:
        events.append(("owner-sid",))
        return "S-1-5-21-111-222-333-1001"

    def build_security_attributes(policy: str) -> SimpleNamespace:
        events.append(("build-security", policy))
        return SimpleNamespace(attributes="secure-attributes", descriptor=77)

    def create_mutex(name: str, attributes: object) -> int:
        events.append(("create", name, attributes))
        if callable(on_create):
            on_create()
        if create_error is not None:
            raise create_error
        return 91

    def free_security_descriptor(security: object) -> None:
        events.append(("free-security", security))

    def wait(handle: int, timeout_ms: int) -> int:
        events.append(("wait", handle, timeout_ms))
        return wait_result

    def release_mutex(handle: int) -> None:
        events.append(("release", handle))

    def close_handle(handle: int) -> None:
        events.append(("close", handle))

    return (
        SimpleNamespace(
            owner_sid=owner_sid,
            build_security_attributes=build_security_attributes,
            create_mutex=create_mutex,
            free_security_descriptor=free_security_descriptor,
            wait=wait,
            release_mutex=release_mutex,
            close_handle=close_handle,
        ),
        events,
    )


@pytest.mark.parametrize("wait_result", (0, 0x00000080))
def test_windows_mutex_acquired_or_abandoned_never_opens_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_result: int,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    parent.mkdir()
    trap = parent / ".v1.publish.lock"

    def install_final_component_trap() -> None:
        trap.mkdir()
        (trap / "sentinel").write_text("preserve")

    api, events = _fake_windows_mutex(wait_result, install_final_component_trap)
    monkeypatch.setattr(runner, "_windows_mutex_api", lambda: api, raising=False)
    real_open = runner.os.open

    def forbid_filesystem_lock(path: object, *args: object, **kwargs: object) -> int:
        if Path(path).name == ".v1.publish.lock":
            raise AssertionError("filesystem lock opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", forbid_filesystem_lock)

    with runner._publisher_lock(parent):
        pass

    policy = str(events[1][1])
    assert policy == (
        "O:S-1-5-21-111-222-333-1001"
        "D:P"
        "(A;;0x001F0001;;;S-1-5-21-111-222-333-1001)"
        "(A;;0x001F0001;;;SY)"
    )
    assert "WD" not in policy
    mutex_name = str(events[2][1])
    assert mutex_name.startswith("Global\\E1A4SamplingFrame-")
    assert len(mutex_name.removeprefix("Global\\E1A4SamplingFrame-")) == 64
    assert str(parent) not in mutex_name
    assert events[2][2] == "secure-attributes"
    assert events[3:] == [
        ("free-security", SimpleNamespace(attributes="secure-attributes", descriptor=77)),
        ("wait", 91, 0),
        ("release", 91),
        ("close", 91),
    ]
    assert (trap / "sentinel").read_text() == "preserve"


def test_windows_mutex_contention_closes_without_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    parent.mkdir()
    api, events = _fake_windows_mutex(0x00000102)
    monkeypatch.setattr(runner, "_windows_mutex_api", lambda: api, raising=False)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        with runner._publisher_lock(parent):
            pass

    assert [event[0] for event in events] == [
        "owner-sid",
        "build-security",
        "create",
        "free-security",
        "wait",
        "close",
    ]
    assert not (parent / ".v1.publish.lock").exists()


def test_windows_mutex_access_denial_frees_security_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    parent.mkdir()
    api, events = _fake_windows_mutex(
        0,
        create_error=PermissionError("private access detail"),
    )
    monkeypatch.setattr(runner, "_windows_mutex_api", lambda: api)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        with runner._publisher_lock(parent):
            pass

    assert [event[0] for event in events] == [
        "owner-sid",
        "build-security",
        "create",
        "free-security",
    ]
    assert not (parent / ".v1.publish.lock").exists()


def test_windows_mutex_parent_swap_never_mutates_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    displaced = tmp_path / "displaced-frame"
    parent.mkdir()

    def swap_parent() -> None:
        parent.rename(displaced)
        parent.mkdir()
        trap = parent / ".v1.publish.lock"
        trap.mkdir()
        (trap / "sentinel").write_text("replacement")

    api, events = _fake_windows_mutex(0, swap_parent)
    monkeypatch.setattr(runner, "_windows_mutex_api", lambda: api, raising=False)

    with runner._publisher_lock(parent):
        pass

    assert [event[0] for event in events] == [
        "owner-sid",
        "build-security",
        "create",
        "free-security",
        "wait",
        "release",
        "close",
    ]
    assert not (displaced / ".v1.publish.lock").exists()
    assert (parent / ".v1.publish.lock" / "sentinel").read_text() == "replacement"


class _FakePosixOS:
    O_RDONLY = 0x01
    O_RDWR = 0x02
    O_CREAT = 0x04
    O_EXCL = 0x08
    O_CLOEXEC = 0x10
    O_DIRECTORY = 0x20
    O_NOFOLLOW = 0x40
    O_NONBLOCK = 0x80

    def __init__(
        self,
        *,
        parent_stat: object,
        lock_stat: object,
        parent_open_stat: object | None = None,
        initial_lock_stat: object | None = None,
        lock_open_stat: object | None = None,
    ) -> None:
        self.parent_stat = parent_stat
        self.parent_open_stat = parent_open_stat or parent_stat
        self.lock_stat = lock_stat
        self.initial_lock_stat = initial_lock_stat
        self.lock_open_stat = lock_open_stat or lock_stat
        self.lock_created = initial_lock_stat is not None
        self.calls: list[tuple[object, ...]] = []

    def lstat(self, path: object) -> object:
        self.calls.append(("lstat-parent", path))
        return self.parent_stat

    def open(
        self,
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        self.calls.append(("open", path, flags, mode, dir_fd))
        if dir_fd is None:
            return 41
        self.lock_created = True
        return 42

    def fstat(self, descriptor: int) -> object:
        self.calls.append(("fstat", descriptor))
        if descriptor == 41:
            return self.parent_open_stat
        return self.lock_open_stat

    def stat(
        self, path: object, *, dir_fd: int, follow_symlinks: bool
    ) -> object:
        self.calls.append(("stat-lock", path, dir_fd, follow_symlinks))
        if not self.lock_created:
            raise FileNotFoundError
        return self.initial_lock_stat or self.lock_stat

    def close(self, descriptor: int) -> None:
        self.calls.append(("close", descriptor))


class _FakeFlock:
    LOCK_EX = 0x01
    LOCK_NB = 0x02
    LOCK_UN = 0x04

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def flock(self, descriptor: int, operation: int) -> None:
        self.calls.append((descriptor, operation))


def test_posix_lock_creates_only_by_relative_validated_parent_fd(
    tmp_path: Path,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    parent.mkdir()
    reference = tmp_path / "reference-lock"
    reference.touch()
    fake_os = _FakePosixOS(
        parent_stat=parent.stat(),
        lock_stat=reference.stat(),
    )
    fake_flock = _FakeFlock()
    posix_lock = getattr(runner, "_posix_publisher_lock", None)
    assert posix_lock is not None

    with posix_lock(parent, os_api=fake_os, flock_api=fake_flock):
        pass

    parent_open = next(call for call in fake_os.calls if call[:2] == ("open", parent.resolve()))
    relative_open = next(
        call for call in fake_os.calls if call[0:2] == ("open", ".v1.publish.lock")
    )
    assert parent_open[2] & fake_os.O_DIRECTORY
    assert parent_open[2] & fake_os.O_NOFOLLOW
    assert relative_open[4] == 41
    assert relative_open[2] & fake_os.O_CREAT
    assert relative_open[2] & fake_os.O_EXCL
    assert relative_open[2] & fake_os.O_NOFOLLOW
    assert relative_open[2] & fake_os.O_NONBLOCK
    assert fake_flock.calls == [(42, 0x03), (42, 0x04)]
    assert fake_os.calls[-2:] == [("close", 42), ("close", 41)]


def test_posix_parent_swap_fails_before_relative_lock_open(
    tmp_path: Path,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    replacement = tmp_path / "replacement-frame"
    reference = tmp_path / "reference-lock"
    parent.mkdir()
    replacement.mkdir()
    reference.touch()
    fake_os = _FakePosixOS(
        parent_stat=parent.stat(),
        parent_open_stat=replacement.stat(),
        lock_stat=reference.stat(),
    )
    fake_flock = _FakeFlock()
    posix_lock = getattr(runner, "_posix_publisher_lock", None)
    assert posix_lock is not None

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        with posix_lock(parent, os_api=fake_os, flock_api=fake_flock):
            pass

    assert not any(call[-1] == 41 for call in fake_os.calls if call[0] == "open")
    assert not (replacement / ".v1.publish.lock").exists()
    assert not fake_flock.calls


def test_posix_final_component_swap_fails_before_flock(
    tmp_path: Path,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    parent = tmp_path / "sampling-frame"
    initial = tmp_path / "initial-lock"
    replacement = tmp_path / "replacement-lock"
    parent.mkdir()
    initial.touch()
    replacement.write_text("preserve")
    fake_os = _FakePosixOS(
        parent_stat=parent.stat(),
        lock_stat=initial.stat(),
        initial_lock_stat=initial.stat(),
        lock_open_stat=replacement.stat(),
    )
    fake_flock = _FakeFlock()
    posix_lock = getattr(runner, "_posix_publisher_lock", None)
    assert posix_lock is not None

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        with posix_lock(parent, os_api=fake_os, flock_api=fake_flock):
            pass

    assert replacement.read_text() == "preserve"
    assert not fake_flock.calls


def test_failed_publisher_preserves_owned_final_for_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    real_verify = runner.verify_sampling_frame

    def fail_first_verification(**kwargs: object) -> object:
        if current_thread().name == "publisher-1":
            raise runner.E1A4SamplingFrameError(
                "E1A4_SAMPLING_FRAME_VERIFY_FAILED"
            )
        return real_verify(**kwargs)

    monkeypatch.setattr(runner, "verify_sampling_frame", fail_first_verification)
    failures: list[BaseException] = []

    def publish() -> None:
        try:
            runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
        except BaseException as error:
            failures.append(error)

    first = Thread(target=publish, name="publisher-1")
    first.start()
    first.join(10)

    assert not first.is_alive()
    assert [str(error) for error in failures] == [
        "E1A4_SAMPLING_FRAME_VERIFY_FAILED"
    ]
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    assert (parent / "v1").exists()
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
    assert len(tuple(parent.glob(".v1.*.tmp"))) == 1


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
