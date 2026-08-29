from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
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
        "approved_private_root": tmp_path,
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
        "--approved-private-root",
        str(kwargs["approved_private_root"]),
        "--database-url",
        str(kwargs["database_url"]),
        "--index-contract",
        str(kwargs["index_contract_path"]),
        "--output-root",
        str(kwargs["output_root"]),
    ]


def _replace_frame_cli_path(
    arguments: list[str], option: str, value: Path
) -> list[str]:
    replaced = list(arguments)
    replaced[replaced.index(option) + 1] = str(value)
    return replaced


@pytest.mark.parametrize(
    ("option", "escape"),
    (
        ("--mapping-root", "public"),
        ("--output-root", "public"),
        ("--output-root", "sibling-prefix"),
    ),
)
def test_sampling_cli_rejects_private_path_escape_before_presence_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    escape: str,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    kwargs = _frame_kwargs(tmp_path)
    approved = Path(kwargs["approved_private_root"])
    escaped = (
        tmp_path.parent / "public-output"
        if escape == "public"
        else approved.with_name(approved.name + "-sibling")
    )
    arguments = _replace_frame_cli_path(
        _frame_cli_args(tmp_path), option, escaped
    )
    monkeypatch.setattr(
        runner,
        "_presence_preflight",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("presence checked before private boundary validation")
        ),
    )

    assert runner.cli(["seal"] + arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_SAMPLING_FRAME_BLOCKED",
        "error_code": "E1A4_SAMPLING_FRAME_PRIVATE_ROOT_INVALID",
    }
    assert str(escaped) not in captured.err


def test_sampling_cli_rejects_symlinked_private_output_ancestor_before_presence_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    approved = Path(_frame_kwargs(tmp_path)["approved_private_root"])
    target = approved / "real-output"
    target.mkdir()
    linked = approved / "linked-output"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error.__class__.__name__}")
    arguments = _replace_frame_cli_path(
        _frame_cli_args(tmp_path), "--output-root", linked
    )
    monkeypatch.setattr(
        runner,
        "_presence_preflight",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("presence checked before private boundary validation")
        ),
    )

    assert runner.cli(["seal"] + arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_SAMPLING_FRAME_BLOCKED",
        "error_code": "E1A4_SAMPLING_FRAME_PRIVATE_ROOT_INVALID",
    }


def test_sampling_cli_rejects_public_worktree_as_approved_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    arguments = _replace_frame_cli_path(
        _frame_cli_args(tmp_path),
        "--approved-private-root",
        runner.PROJECT_ROOT,
    )
    arguments = _replace_frame_cli_path(
        arguments, "--mapping-root", runner.PROJECT_ROOT / "public-mapping"
    )
    arguments = _replace_frame_cli_path(
        arguments, "--output-root", runner.PROJECT_ROOT / "public-output"
    )
    monkeypatch.setattr(
        runner,
        "_presence_preflight",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("presence checked before private boundary validation")
        ),
    )

    assert runner.cli(["seal"] + arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_SAMPLING_FRAME_BLOCKED",
        "error_code": "E1A4_SAMPLING_FRAME_PRIVATE_ROOT_INVALID",
    }


def test_sampling_private_boundary_rejects_windows_reparse_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    approved = tmp_path / "approved"
    reparse = approved / "junction"
    reparse.mkdir(parents=True)
    real_lstat = Path.lstat

    def mark_reparse(path: Path) -> object:
        observed = real_lstat(path)
        if path == reparse:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_file_attributes=0x400,
            )
        return observed

    monkeypatch.setattr(Path, "lstat", mark_reparse)

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_PRIVATE_ROOT_INVALID$",
    ):
        runner._validate_private_paths(approved, (reparse / "output",))


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


def _synthetic_frame_members(
    source_register: dict[str, object],
    allocation: dict[str, object],
) -> dict[str, bytes]:
    payloads = {
        "source-register.v1.json": (
            json.dumps(source_register, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode(),
        "sampling-allocation.v1.json": (
            json.dumps(allocation, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode(),
    }
    return {
        **{f"sealed/{name}": content for name, content in payloads.items()},
        **{
            f"manifests/{name.removesuffix('.json')}.sha256": (
                sha256(content).hexdigest() + "\n"
            ).encode("ascii")
            for name, content in payloads.items()
        },
    }


def _write_synthetic_frame(
    final: Path, members: dict[str, bytes]
) -> None:
    for relative_name, content in members.items():
        path = final / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _frame_payloads(marker: str) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "schema_version": 1,
            "source_record_count": 1,
            "marker": marker,
        },
        {"schema_version": 1, "slot_count": 96, "marker": marker},
    )


def test_locked_frame_verification_never_reopens_output_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    source_register, allocation = _frame_payloads("authenticated")
    output = tmp_path / "output"
    approved = tmp_path
    runner._publish_sampling_frame(
        source_register=source_register,
        allocation=allocation,
        approved_private_root=approved,
        output_root=output,
    )
    displaced = tmp_path / "authenticated-output"
    replacement = tmp_path / "replacement-output"
    replacement_members = _synthetic_frame_members(
        *_frame_payloads("replacement")
    )
    real_acquire = runner.authenticated_publication_directory
    retarget_succeeded = False

    @contextmanager
    def retarget_after_acquisition(**kwargs: object) -> object:
        nonlocal retarget_succeeded
        with real_acquire(**kwargs) as publication:
            try:
                output.rename(displaced)
                _write_synthetic_frame(
                    output / "e1a4" / "sampling-frame" / "v1",
                    replacement_members,
                )
                retarget_succeeded = True
            except OSError:
                pass
            try:
                yield publication
            finally:
                if retarget_succeeded:
                    output.rename(replacement)
                    displaced.rename(output)

    monkeypatch.setattr(
        runner,
        "authenticated_publication_directory",
        retarget_after_acquisition,
    )
    monkeypatch.setattr(
        runner,
        "_read_frame_members",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("locked verification reopened output by pathname")
        ),
        raising=False,
    )

    verified = runner._publish_sampling_frame(
        source_register=source_register,
        allocation=allocation,
        approved_private_root=approved,
        output_root=output,
    )

    assert verified.source_register_sha256 == sha256(
        _synthetic_frame_members(source_register, allocation)[
            "sealed/source-register.v1.json"
        ]
    ).hexdigest()


def test_standalone_frame_verification_reads_through_authenticated_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    real_acquire = runner.authenticated_publication_directory
    events: list[str] = []

    @contextmanager
    def observe_capability(**kwargs: object) -> object:
        events.append("acquire")
        with real_acquire(**kwargs) as publication:
            original_read = publication.read_exact_tree

            def read_exact_tree(*args: object, **read_kwargs: object) -> object:
                events.append("read")
                return original_read(*args, **read_kwargs)

            monkeypatch.setattr(publication, "read_exact_tree", read_exact_tree)
            yield publication

    monkeypatch.setattr(
        runner, "authenticated_publication_directory", observe_capability
    )
    monkeypatch.setattr(
        runner,
        "_read_frame_members",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("standalone verification reopened output by pathname")
        ),
        raising=False,
    )

    assert runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path)) == sealed
    assert events == ["acquire", "read"]


def test_frame_existing_final_is_verified_from_locked_tree_not_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    original_source, original_allocation = _frame_payloads("authenticated")
    replacement_source, replacement_allocation = _frame_payloads("replacement")
    output = tmp_path / "output"
    runner._publish_sampling_frame(
        source_register=original_source,
        allocation=original_allocation,
        approved_private_root=tmp_path,
        output_root=output,
    )
    displaced = tmp_path / "authenticated-output"
    replacement = tmp_path / "replacement-output"
    real_acquire = runner.authenticated_publication_directory
    retarget_succeeded = False
    retarget_denial: OSError | None = None

    @contextmanager
    def retarget_after_acquisition(**kwargs: object) -> object:
        nonlocal retarget_succeeded, retarget_denial
        with real_acquire(**kwargs) as publication:
            try:
                output.rename(displaced)
                _write_synthetic_frame(
                    output / "e1a4" / "sampling-frame" / "v1",
                    _synthetic_frame_members(
                        replacement_source, replacement_allocation
                    ),
                )
                retarget_succeeded = True
            except OSError as error:
                retarget_denial = error
            try:
                yield publication
            finally:
                if retarget_succeeded:
                    output.rename(replacement)
                    displaced.rename(output)

    monkeypatch.setattr(
        runner,
        "authenticated_publication_directory",
        retarget_after_acquisition,
    )

    observed_error: runner.E1A4SamplingFrameError | None = None
    try:
        runner._publish_sampling_frame(
            source_register=replacement_source,
            allocation=replacement_allocation,
            approved_private_root=tmp_path,
            output_root=output,
        )
    except runner.E1A4SamplingFrameError as error:
        observed_error = error
    if retarget_denial is not None:
        pytest.skip(
            "authenticated capability denied ancestor retarget: "
            f"{retarget_denial.__class__.__name__}"
        )
    assert retarget_succeeded
    assert str(observed_error) == "E1A4_SAMPLING_FRAME_BINDING_MISMATCH"


def test_frame_publisher_preserves_directory_sync_order_with_shared_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    source_register, allocation = _frame_payloads("ordered")
    members = _synthetic_frame_members(source_register, allocation)
    events: list[tuple[object, ...]] = []

    class Staging:
        def mkdir(self, name: str) -> None:
            events.append(("mkdir", name))

        def write_exclusive(self, name: str, content: bytes) -> None:
            events.append(("write", name, content))

        def sync_directory(self, name: str) -> None:
            events.append(("sync", name))

        def sync_root(self) -> None:
            events.append(("sync-root",))

    class Publication:
        staging = Staging()
        published = False

        def ensure_no_staging(self, prefix: str, suffix: str) -> None:
            events.append(("residue", prefix, suffix))

        def final_exists(self, name: str) -> bool:
            events.append(("exists", name))
            return self.published

        def create_staging(self, prefix: str, suffix: str) -> Staging:
            events.append(("create", prefix, suffix))
            return self.staging

        def publish_no_replace(self, staging: Staging, name: str) -> None:
            assert staging is self.staging
            events.append(("publish", name))
            self.published = True

        def sync_parent(self) -> None:
            events.append(("sync-parent",))

        def read_exact_tree(
            self, name: str, layout: object
        ) -> dict[str, bytes]:
            events.append(("read", name, layout))
            return members

    @contextmanager
    def capability(**kwargs: object) -> object:
        events.append(("acquire", kwargs))
        yield Publication()

    monkeypatch.setattr(
        runner, "authenticated_publication_directory", capability, raising=False
    )

    sealed = runner._publish_sampling_frame(
        source_register=source_register,
        allocation=allocation,
        approved_private_root=tmp_path,
        output_root=tmp_path / "output",
    )

    assert sealed.slot_count == 96
    labels = [event[:2] for event in events]
    assert labels == [
        ("acquire", {
            "approved_private_root": tmp_path,
            "publication_parent": tmp_path
            / "output"
            / "e1a4"
            / "sampling-frame",
            "lock_name": ".v1.publish.lock",
        }),
        ("residue", ".v1."),
        ("exists", "v1"),
        ("create", ".v1."),
        ("mkdir", "sealed"),
        ("mkdir", "manifests"),
        ("write", "sealed/source-register.v1.json"),
        ("write", "manifests/source-register.v1.sha256"),
        ("write", "sealed/sampling-allocation.v1.json"),
        ("write", "manifests/sampling-allocation.v1.sha256"),
        ("sync", "sealed"),
        ("sync", "manifests"),
        ("sync-root",),
        ("publish", "v1"),
        ("sync-parent",),
        ("read", "v1"),
    ]


def test_frame_runner_has_no_duplicate_platform_publication_implementation() -> None:
    import eval.seal_e1a4_sampling_frame as runner

    duplicate_symbols = (
        "_PosixPublicationDirectory",
        "_WindowsPublicationDirectory",
        "_publisher_lock",
        "_authenticated_posix_publisher_lock",
        "_authenticated_windows_publisher_lock",
        "_acquire_posix_publication_parent",
        "_acquire_windows_publication_parent",
        "_rename_no_replace_at",
    )
    assert not [name for name in duplicate_symbols if hasattr(runner, name)]


def test_frame_publication_failure_preserves_owned_staging_for_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    real_acquire = runner.authenticated_publication_directory

    @contextmanager
    def fail_publish(**kwargs: object) -> object:
        with real_acquire(**kwargs) as publication:
            class PublicationProxy:
                def __getattr__(self, name: str) -> object:
                    return getattr(publication, name)

                def publish_no_replace(
                    self, _staging: object, _name: str
                ) -> None:
                    raise OSError("private path")

            yield PublicationProxy()

    monkeypatch.setattr(
        runner, "authenticated_publication_directory", fail_publish
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    assert not (parent / "v1").exists()
    assert len(tuple(parent.glob(".v1.*.tmp"))) == 1

    monkeypatch.setattr(
        runner, "authenticated_publication_directory", real_acquire
    )
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


def test_standalone_frame_verify_blocks_and_preserves_staging_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    sealed = runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    residue = parent / ".v1.abandoned.tmp"
    residue.mkdir()
    (residue / "sentinel").write_text("preserve")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path))

    assert (residue / "sentinel").read_text() == "preserve"
    assert sealed.slot_count == 96


def test_standalone_frame_verify_fails_closed_on_publisher_lock_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    result: list[object] = []

    def verify() -> None:
        try:
            result.append(runner.verify_current_sampling_frame(**_frame_kwargs(tmp_path)))
        except runner.E1A4SamplingFrameError as error:
            result.append(error)

    with runner.authenticated_publication_directory(
        approved_private_root=tmp_path,
        publication_parent=parent,
        lock_name=".v1.publish.lock",
    ):
        contender = Thread(target=verify)
        contender.start()
        contender.join(10)
        assert not contender.is_alive()

    assert [str(item) for item in result] == [
        "E1A4_SAMPLING_FRAME_WRITE_FAILED"
    ]


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


def test_failed_publisher_preserves_owned_final_for_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    real_verify = runner._verify_sampling_frame_members

    def fail_first_verification(**kwargs: object) -> object:
        if current_thread().name == "publisher-1":
            raise runner.E1A4SamplingFrameError(
                "E1A4_SAMPLING_FRAME_VERIFY_FAILED"
            )
        return real_verify(**kwargs)

    monkeypatch.setattr(
        runner, "_verify_sampling_frame_members", fail_first_verification
    )
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
    real_acquire = runner.authenticated_publication_directory

    @contextmanager
    def race_at_publish(**kwargs: object) -> object:
        with real_acquire(**kwargs) as publication:
            class PublicationProxy:
                def __getattr__(self, name: str) -> object:
                    return getattr(publication, name)

                def publish_no_replace(
                    self, staging: object, name: str
                ) -> None:
                    final.mkdir()
                    (final / "concurrent-owner").write_text("preserve me")
                    publication.publish_no_replace(staging, name)

            yield PublicationProxy()

    monkeypatch.setattr(
        runner, "authenticated_publication_directory", race_at_publish
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="E1A4_SAMPLING_FRAME_WRITE_FAILED",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))
    assert (final / "concurrent-owner").read_text() == "preserve me"
    assert len(tuple(parent.glob(".v1.*.tmp"))) == 1


def test_frame_publisher_rejects_symlinked_output_ancestor_without_mutation(
    tmp_path: Path,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    approved = tmp_path / "approved"
    actual = approved / "actual"
    actual.mkdir(parents=True)
    linked = approved / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error.__class__.__name__}")
    sentinel = actual / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        runner._publish_sampling_frame(
            source_register={"schema_version": 1},
            allocation={"schema_version": 1},
            approved_private_root=approved,
            output_root=linked,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (actual / "e1a4").exists()


@pytest.mark.parametrize("failed_sync", ("sealed", "manifests", "staging", "parent"))
def test_frame_publisher_fails_closed_when_directory_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_sync: str,
) -> None:
    import eval.seal_e1a4_sampling_frame as runner

    _patch_frame_trust(runner, tmp_path, monkeypatch)
    calls: list[str] = []
    real_acquire = runner.authenticated_publication_directory

    def record_sync(label: str) -> None:
        calls.append(label)
        if label == failed_sync:
            raise OSError("synthetic directory sync failure")

    @contextmanager
    def fail_sync(**kwargs: object) -> object:
        with real_acquire(**kwargs) as publication:
            class StagingProxy:
                def __init__(self, staging: object) -> None:
                    self.staging = staging

                def __getattr__(self, name: str) -> object:
                    return getattr(self.staging, name)

                def sync_directory(self, name: str) -> None:
                    record_sync(name)
                    self.staging.sync_directory(name)

                def sync_root(self) -> None:
                    record_sync("staging")
                    self.staging.sync_root()

            class PublicationProxy:
                staging: StagingProxy | None = None

                def __getattr__(self, name: str) -> object:
                    return getattr(publication, name)

                def create_staging(
                    self, prefix: str, suffix: str
                ) -> StagingProxy:
                    self.staging = StagingProxy(
                        publication.create_staging(prefix, suffix)
                    )
                    return self.staging

                def publish_no_replace(
                    self, staging: StagingProxy, name: str
                ) -> None:
                    publication.publish_no_replace(staging.staging, name)

                def sync_parent(self) -> None:
                    record_sync("parent")
                    publication.sync_parent()

            yield PublicationProxy()

    monkeypatch.setattr(
        runner, "authenticated_publication_directory", fail_sync
    )

    with pytest.raises(
        runner.E1A4SamplingFrameError,
        match="^E1A4_SAMPLING_FRAME_WRITE_FAILED$",
    ):
        runner.seal_sampling_frame(**_frame_kwargs(tmp_path))

    parent = tmp_path / "output" / "e1a4" / "sampling-frame"
    assert failed_sync in calls
    if failed_sync == "parent":
        assert (parent / "v1").is_dir()
        assert not tuple(parent.glob(".v1.*.tmp"))
    else:
        assert not (parent / "v1").exists()
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
