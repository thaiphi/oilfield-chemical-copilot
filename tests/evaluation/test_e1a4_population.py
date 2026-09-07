from __future__ import annotations

from dataclasses import replace
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import pytest

TOPICS = ("iron_sulfide", "scale", "corrosion", "paraffin")
ROLES = ("foundational", "supporting")
FORMS = ("definition_mechanism", "diagnostic_interpretive", "operational_procedural")
DEPTHS = ("single_claim", "multi_claim")


def _cases():
    from oilfield_chemical_copilot.evaluation.e1a4_population import E1A4PopulationCase

    return tuple(
        E1A4PopulationCase(
            question_id=f"e1a4-{index:03d}",
            question=f"Private question {index}?",
            topic=topic,
            expected_source=f"private-{topic}-{role}.pdf",
            expected_locator=f"page:{index}",
            expected_source_role=role,
            question_form=form,
            evidence_depth=depth,
        )
        for index, (topic, role, form, depth, _) in enumerate(
            product(TOPICS, ROLES, FORMS, DEPTHS, (1, 2)), start=1
        )
    )


def _claims(cases):
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4CanonicalClaim,
        E1A4ClaimFixture,
    )

    return tuple(
        E1A4ClaimFixture(
            question_id=case.question_id,
            claims=tuple(
                E1A4CanonicalClaim(
                    claim_id=f"{case.question_id}:claim:{claim_index}",
                    claim=f"Private claim {claim_index}.",
                )
                for claim_index in range(1, (1 if case.evidence_depth == "single_claim" else 2) + 1)
            ),
        )
        for case in cases
    )


def _allocations(cases):
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        E1A3SlotAllocation,
        build_sampling_slots,
    )

    return tuple(
        E1A3SlotAllocation(
            slot_id=slot.slot_id,
            topic=slot.topic,
            source_role=slot.source_role,
            question_form=slot.question_form,
            evidence_depth=slot.evidence_depth,
            replicate=slot.replicate,
            source_id=case.expected_source,
            parser_type="pdf",
            locator=case.expected_locator,
        )
        for slot, case in zip(build_sampling_slots(), cases, strict=True)
    )


def _source_register_payload(allocations) -> dict[str, object]:
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    for item in allocations:
        key = (item.source_id, item.source_role, item.topic, item.parser_type)
        grouped.setdefault(key, []).append(item.locator)
    return {
        "schema_version": 1,
        "e1a3_source_role_config_sha256": "e" * 64,
        "sources": [
            {
                "source_id": source_id,
                "source_role": source_role,
                "topic": topic,
                "parser_type": parser_type,
                "locators": locators,
                "eligibility_status": "eligible",
            }
            for (source_id, source_role, topic, parser_type), locators in sorted(grouped.items())
        ],
    }


def test_population_contract_accepts_exact_fresh_grid_and_atomic_claim_limit() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import validate_population_and_claims

    cases = _cases()
    validate_population_and_claims(
        cases=cases,
        claims=_claims(cases),
        earlier_questions=("Earlier private question?",),
        e1a3_locators=("other.pdf:page:1",),
    )


def test_population_contract_rejects_normalized_prior_or_within_population_wording() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        validate_population_and_claims,
    )

    cases = list(_cases())
    cases[1] = replace(cases[1], question="  private   question 1? ")
    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_QUESTION_DUPLICATE"):
        validate_population_and_claims(
            cases=tuple(cases),
            claims=_claims(cases),
            earlier_questions=(),
            e1a3_locators=(),
        )


def test_population_contract_rejects_question_ids_that_collide_after_trimming() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        validate_population_and_claims,
    )

    cases = list(_cases())
    cases[1] = replace(cases[1], question_id=f" {cases[0].question_id} ")

    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_CASE_INVALID"):
        validate_population_and_claims(
            cases=tuple(cases),
            claims=_claims(cases),
            earlier_questions=(),
            e1a3_locators=(),
        )

    cases = _cases()
    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_NOT_UNSEEN"):
        validate_population_and_claims(
            cases=cases,
            claims=_claims(cases),
            earlier_questions=(" PRIVATE question 1? ",),
            e1a3_locators=(),
        )


def test_population_contract_rejects_any_e1a3_locator_reuse() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        validate_population_and_claims,
    )

    cases = _cases()
    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_LOCATOR_REUSED"):
        validate_population_and_claims(
            cases=cases,
            claims=_claims(cases),
            earlier_questions=(),
            e1a3_locators=(f"{cases[0].expected_source}:{cases[0].expected_locator}",),
        )


def test_population_contract_normalizes_locator_keys_before_reuse_check() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        validate_population_and_claims,
    )

    cases = list(_cases())
    cases[0] = replace(
        cases[0],
        expected_source=f"  {cases[0].expected_source}  ",
        expected_locator=f" {cases[0].expected_locator} ",
    )
    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_LOCATOR_REUSED"):
        validate_population_and_claims(
            cases=tuple(cases),
            claims=_claims(cases),
            earlier_questions=(),
            e1a3_locators=("private-iron_sulfide-foundational.pdf:page:1",),
        )


def test_population_contract_rejects_more_than_three_canonical_claims() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4CanonicalClaim,
        E1A4ClaimFixture,
        E1A4PopulationError,
        validate_population_and_claims,
    )

    cases = _cases()
    claims = list(_claims(cases))
    target = next(item for item in claims if len(item.claims) == 2)
    claims[claims.index(target)] = E1A4ClaimFixture(
        question_id=target.question_id,
        claims=target.claims
        + (
            E1A4CanonicalClaim(
                claim_id=f"{target.question_id}:claim:3",
                claim="Private claim 3.",
            ),
            E1A4CanonicalClaim(
                claim_id=f"{target.question_id}:claim:4",
                claim="Private claim 4.",
            ),
        ),
    )

    with pytest.raises(E1A4PopulationError, match="E1A4_CANONICAL_CLAIMS_DEPTH_INVALID"):
        validate_population_and_claims(
            cases=cases,
            claims=tuple(claims),
            earlier_questions=(),
            e1a3_locators=(),
        )


def test_population_contract_rejects_missing_joint_grid_slot() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        validate_population_and_claims,
    )

    cases = _cases()[:-1]
    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_GRID_INVALID"):
        validate_population_and_claims(
            cases=cases,
            claims=_claims(cases),
            earlier_questions=(),
            e1a3_locators=(),
        )


def test_population_payload_binds_exact_sampling_allocation_and_claims() -> None:
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        build_population_and_claim_payloads,
    )

    cases = _cases()
    population, population_digest, claims, claims_digest = build_population_and_claim_payloads(
        cases=cases,
        claims=_claims(cases),
        allocations=_allocations(cases),
        allocation_sha256="a" * 64,
        earlier_questions=("Earlier private question?",),
        e1a3_locators=("other.pdf:page:1",),
    )

    assert population["allocation_sha256"] == "a" * 64
    assert len(population["cases"]) == 96
    assert claims["population_sha256"] == population_digest
    assert len(claims["fixtures"]) == 96
    assert private_sampling_payload_digest(population) == population_digest
    assert private_sampling_payload_digest(claims) == claims_digest


def test_population_payload_rejects_case_not_in_sealed_sampling_allocation() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        build_population_and_claim_payloads,
    )

    allocated_cases = _cases()
    authored_cases = list(allocated_cases)
    authored_cases[0] = replace(authored_cases[0], expected_locator="page:not-allocated")

    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_LINEAGE_INVALID"):
        build_population_and_claim_payloads(
            cases=tuple(authored_cases),
            claims=_claims(authored_cases),
            allocations=_allocations(allocated_cases),
            allocation_sha256="a" * 64,
            earlier_questions=(),
            e1a3_locators=(),
        )


def test_population_payload_rejects_allocation_without_parser_provenance() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        build_population_and_claim_payloads,
    )

    cases = _cases()
    allocations = list(_allocations(cases))
    allocations[0] = replace(allocations[0], parser_type="")

    with pytest.raises(E1A4PopulationError, match="E1A4_ALLOCATION_GRID_INVALID"):
        build_population_and_claim_payloads(
            cases=cases,
            claims=_claims(cases),
            allocations=tuple(allocations),
            allocation_sha256="a" * 64,
            earlier_questions=(),
            e1a3_locators=(),
        )


def test_population_payload_rejects_boolean_allocation_replicate() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        build_population_and_claim_payloads,
    )

    cases = _cases()
    allocations = list(_allocations(cases))
    allocations[0] = replace(allocations[0], replicate=True)

    with pytest.raises(E1A4PopulationError, match="E1A4_ALLOCATION_GRID_INVALID"):
        build_population_and_claim_payloads(
            cases=cases,
            claims=_claims(cases),
            allocations=tuple(allocations),
            allocation_sha256="a" * 64,
            earlier_questions=(),
            e1a3_locators=(),
        )


def test_population_draft_parser_rejects_extra_fields() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationCase,
        E1A4PopulationError,
    )

    record = _cases()[0].to_mapping()
    record["candidate_output"] = "must never enter the population contract"

    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_CASE_INVALID"):
        E1A4PopulationCase.from_mapping(record)


def test_canonical_claim_draft_parser_rejects_extra_fields() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4ClaimFixture,
        E1A4PopulationError,
    )

    record = _claims(_cases()[:1])[0].to_mapping()
    record["support_label"] = "must remain independently blinded"

    with pytest.raises(E1A4PopulationError, match="E1A4_CANONICAL_CLAIMS_INVALID"):
        E1A4ClaimFixture.from_mapping(record)


def test_population_sealer_atomically_writes_population_claims_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oilfield_chemical_copilot.evaluation import e1a3_sampling
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        seal_population_and_claims,
    )

    monkeypatch.setattr(e1a3_sampling, "PRIVATE_RETRIEVAL_ROOT", tmp_path)
    root = tmp_path / "e1a4"
    population_path = root / "sealed" / "population.v1.json"
    population_manifest_path = root / "manifests" / "population.v1.sha256"
    claims_path = root / "sealed" / "canonical-claims.v1.json"
    claims_manifest_path = root / "manifests" / "canonical-claims.v1.sha256"
    cases = _cases()

    population_digest, claims_digest = seal_population_and_claims(
        cases=cases,
        claims=_claims(cases),
        allocations=_allocations(cases),
        allocation_sha256="a" * 64,
        earlier_questions=(),
        e1a3_locators=(),
        population_path=population_path,
        population_digest_path=population_manifest_path,
        claims_path=claims_path,
        claims_digest_path=claims_manifest_path,
        private_root=root,
    )

    assert population_manifest_path.read_text(encoding="ascii").strip() == population_digest
    assert claims_manifest_path.read_text(encoding="ascii").strip() == claims_digest
    assert json.loads(claims_path.read_text(encoding="utf-8"))["population_sha256"] == population_digest
    assert len(json.loads(population_path.read_text(encoding="utf-8"))["cases"]) == 96
    assert not tuple(path for path in root.rglob("*") if path.is_file() and path.name.startswith("."))


def test_population_sealer_leaves_no_partial_set_when_any_destination_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oilfield_chemical_copilot.evaluation import e1a3_sampling
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        seal_population_and_claims,
    )

    monkeypatch.setattr(e1a3_sampling, "PRIVATE_RETRIEVAL_ROOT", tmp_path)
    root = tmp_path / "e1a4"
    population_path = root / "sealed" / "population.v1.json"
    population_manifest_path = root / "manifests" / "population.v1.sha256"
    claims_path = root / "sealed" / "canonical-claims.v1.json"
    claims_manifest_path = root / "manifests" / "canonical-claims.v1.sha256"
    claims_path.parent.mkdir(parents=True)
    claims_path.write_text("existing seal\n", encoding="utf-8")
    cases = _cases()

    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_SEAL_FAILED"):
        seal_population_and_claims(
            cases=cases,
            claims=_claims(cases),
            allocations=_allocations(cases),
            allocation_sha256="a" * 64,
            earlier_questions=(),
            e1a3_locators=(),
            population_path=population_path,
            population_digest_path=population_manifest_path,
            claims_path=claims_path,
            claims_digest_path=claims_manifest_path,
            private_root=root,
        )

    assert claims_path.read_text(encoding="utf-8") == "existing seal\n"
    assert not population_path.exists()
    assert not population_manifest_path.exists()
    assert not claims_manifest_path.exists()


def test_population_sealer_rolls_back_published_files_after_mid_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oilfield_chemical_copilot.evaluation import e1a3_sampling
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
        seal_population_and_claims,
    )

    monkeypatch.setattr(e1a3_sampling, "PRIVATE_RETRIEVAL_ROOT", tmp_path)
    real_replace = e1a3_sampling.replace
    replace_count = 0

    def fail_on_second_publish(source, destination) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("synthetic publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(e1a3_sampling, "replace", fail_on_second_publish)
    root = tmp_path / "e1a4"
    outputs = (
        root / "sealed" / "population.v1.json",
        root / "manifests" / "population.v1.sha256",
        root / "sealed" / "canonical-claims.v1.json",
        root / "manifests" / "canonical-claims.v1.sha256",
    )
    cases = _cases()

    with pytest.raises(E1A4PopulationError, match="E1A4_POPULATION_SEAL_FAILED"):
        seal_population_and_claims(
            cases=cases,
            claims=_claims(cases),
            allocations=_allocations(cases),
            allocation_sha256="a" * 64,
            earlier_questions=(),
            e1a3_locators=(),
            population_path=outputs[0],
            population_digest_path=outputs[1],
            claims_path=outputs[2],
            claims_digest_path=outputs[3],
            private_root=root,
        )

    assert replace_count == 2
    assert not any(path.exists() for path in outputs)
    assert not tuple(path for path in root.rglob("*") if path.is_file() and path.name.startswith("."))


def _seal_runner_module() -> object:
    path = Path(__file__).resolve().parents[2] / "eval" / "seal_e1a4_population.py"
    spec = importlib.util.spec_from_file_location("test_e1a4_population_sealer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_payload_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def test_existing_population_seal_is_verified_without_rewrite(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )

    runner = _seal_runner_module()
    population = {"schema_version": 1, "allocation_sha256": "a" * 64, "cases": []}
    population_digest = private_sampling_payload_digest(population)
    claims = {"schema_version": 1, "population_sha256": population_digest, "fixtures": []}
    claims_digest = private_sampling_payload_digest(claims)
    population_path = tmp_path / "population.json"
    population_manifest_path = tmp_path / "population.sha256"
    claims_path = tmp_path / "claims.json"
    claims_manifest_path = tmp_path / "claims.sha256"
    population_path.write_bytes(_canonical_payload_bytes(population))
    population_manifest_path.write_text(population_digest, encoding="ascii")
    claims_path.write_bytes(_canonical_payload_bytes(claims))
    claims_manifest_path.write_text(claims_digest, encoding="ascii")
    paths = (population_path, population_manifest_path, claims_path, claims_manifest_path)
    before = tuple(path.read_bytes() for path in paths)

    assert runner._verify_existing_seal(
        population=population,
        population_digest=population_digest,
        claims=claims,
        claims_digest=claims_digest,
        population_path=population_path,
        population_manifest_path=population_manifest_path,
        claims_path=claims_path,
        claims_manifest_path=claims_manifest_path,
    )
    assert tuple(path.read_bytes() for path in paths) == before


def test_existing_population_seal_rejects_semantically_equal_noncanonical_bytes(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    population = {"schema_version": 1, "allocation_sha256": "a" * 64, "cases": []}
    population_digest = private_sampling_payload_digest(population)
    claims = {"schema_version": 1, "population_sha256": population_digest, "fixtures": []}
    claims_digest = private_sampling_payload_digest(claims)
    population_path = tmp_path / "population.json"
    population_manifest_path = tmp_path / "population.sha256"
    claims_path = tmp_path / "claims.json"
    claims_manifest_path = tmp_path / "claims.sha256"
    population_path.write_text(json.dumps(population, indent=2), encoding="utf-8")
    population_manifest_path.write_text(population_digest, encoding="ascii")
    claims_path.write_bytes(_canonical_payload_bytes(claims))
    claims_manifest_path.write_text(claims_digest, encoding="ascii")

    with pytest.raises(E1A4PopulationError, match="E1A4_ARTIFACT_MISMATCH"):
        runner._verify_existing_seal(
            population=population,
            population_digest=population_digest,
            claims=claims,
            claims_digest=claims_digest,
            population_path=population_path,
            population_manifest_path=population_manifest_path,
            claims_path=claims_path,
            claims_manifest_path=claims_manifest_path,
        )


def test_existing_population_seal_rejects_incomplete_artifact_set(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    population = {"schema_version": 1, "allocation_sha256": "a" * 64, "cases": []}
    population_digest = private_sampling_payload_digest(population)
    claims = {"schema_version": 1, "population_sha256": population_digest, "fixtures": []}
    population_path = tmp_path / "population.json"
    population_path.write_text(json.dumps(population), encoding="utf-8")

    with pytest.raises(E1A4PopulationError, match="E1A4_ARTIFACT_INCOMPLETE"):
        runner._verify_existing_seal(
            population=population,
            population_digest=population_digest,
            claims=claims,
            claims_digest=private_sampling_payload_digest(claims),
            population_path=population_path,
            population_manifest_path=tmp_path / "population.sha256",
            claims_path=tmp_path / "claims.json",
            claims_manifest_path=tmp_path / "claims.sha256",
        )


def _write_payload_with_manifest(payload: dict[str, object], payload_path: Path, manifest_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(_canonical_payload_bytes(payload))
    manifest_path.write_text(private_sampling_payload_digest(payload), encoding="ascii")


def test_population_sealer_cli_validates_and_seals_synthetic_private_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from oilfield_chemical_copilot.evaluation import e1a3_sampling
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        E1A3SlotAllocation,
        build_sampling_slots,
    )

    runner = _seal_runner_module()
    evaluation_root = tmp_path / "retrieval-evaluation" / "v1"
    root = evaluation_root / "e1a4"
    monkeypatch.setattr(e1a3_sampling, "PRIVATE_RETRIEVAL_ROOT", evaluation_root)
    monkeypatch.setattr(runner, "PRIVATE_RETRIEVAL_ROOT", evaluation_root)
    cases = _cases()
    allocations = _allocations(cases)
    source_register = _source_register_payload(allocations)
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )

    source_register_digest = private_sampling_payload_digest(source_register)
    _write_payload_with_manifest(
        source_register,
        root / "sealed" / "source-register.v1.json",
        root / "manifests" / "source-register.v1.sha256",
    )
    allocation_payload = {
        "schema_version": 1,
        "source_register_sha256": source_register_digest,
        "prior_e1a3_allocation_sha256": "c" * 64,
        "slot_count": 96,
        "allocations": [item.to_mapping() for item in allocations],
    }
    _write_payload_with_manifest(
        allocation_payload,
        root / "sealed" / "sampling-allocation.v1.json",
        root / "manifests" / "sampling-allocation.v1.sha256",
    )
    (root / "draft").mkdir(parents=True)
    (root / "draft" / "population.v1.json").write_text(
        json.dumps([case.to_mapping() for case in cases]), encoding="utf-8"
    )
    (root / "draft" / "canonical-claims.v1.json").write_text(
        json.dumps([fixture.to_mapping() for fixture in _claims(cases)]), encoding="utf-8"
    )

    development_path = evaluation_root / "sealed" / "development.jsonl"
    development_path.parent.mkdir(parents=True)
    development_path.write_text('{"question":"Earlier development question?"}\n', encoding="utf-8")
    e1a2_path = evaluation_root / "e1a2-fresh" / "sealed" / "candidate-pool.jsonl"
    e1a2_path.parent.mkdir(parents=True)
    e1a2_path.write_text('{"question":"Earlier E1a-2 question?"}\n', encoding="utf-8")

    e1a3_root = evaluation_root / "e1a3"
    e1a3_population = {
        "schema_version": 1,
        "allocation_sha256": "f" * 64,
        "cases": [{"question": f"Earlier E1a-3 question {index}?"} for index in range(1, 97)],
    }
    _write_payload_with_manifest(
        e1a3_population,
        e1a3_root / "sealed" / "population.v1.json",
        e1a3_root / "manifests" / "population.v1.sha256",
    )
    prior_allocations = tuple(
        E1A3SlotAllocation(
            slot_id=slot.slot_id,
            topic=slot.topic,
            source_role=slot.source_role,
            question_form=slot.question_form,
            evidence_depth=slot.evidence_depth,
            replicate=slot.replicate,
            source_id="prior-private.pdf",
            parser_type="pdf",
            locator=f"page:prior-{index}",
        )
        for index, slot in enumerate(build_sampling_slots(), start=1)
    )
    e1a3_allocation = {
        "schema_version": 1,
        "source_register_sha256": "d" * 64,
        "slot_count": 96,
        "allocations": [item.to_mapping() for item in prior_allocations],
    }
    _write_payload_with_manifest(
        e1a3_allocation,
        e1a3_root / "sealed" / "sampling-allocation.v4.json",
        e1a3_root / "manifests" / "sampling-allocation.v4.sha256",
    )
    allocation_payload["prior_e1a3_allocation_sha256"] = private_sampling_payload_digest(e1a3_allocation)
    _write_payload_with_manifest(
        allocation_payload,
        root / "sealed" / "sampling-allocation.v1.json",
        root / "manifests" / "sampling-allocation.v1.sha256",
    )

    monkeypatch.setattr(sys, "argv", ["seal_e1a4_population.py", "--private-root", str(root)])
    runner.main()

    assert json.loads(capsys.readouterr().out) == {
        "status": "E1A4_POPULATION_SEALED",
        "case_count": 96,
        "claim_fixture_count": 96,
    }
    assert (root / "sealed" / "population.v1.json").is_file()
    assert (root / "sealed" / "canonical-claims.v1.json").is_file()
    assert (root / "manifests" / "population.v1.sha256").is_file()
    assert (root / "manifests" / "canonical-claims.v1.sha256").is_file()
    assert not (root / "sealed" / "selected-case-ids.v1.json").exists()


def test_population_sealer_rejects_tampered_sampling_allocation_manifest(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    root = tmp_path / "e1a4"
    cases = _cases()
    allocations = _allocations(cases)
    source_register = _source_register_payload(allocations)
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )

    source_register_digest = private_sampling_payload_digest(source_register)
    _write_payload_with_manifest(
        source_register,
        root / "sealed" / "source-register.v1.json",
        root / "manifests" / "source-register.v1.sha256",
    )
    allocation_payload = {
        "schema_version": 1,
        "source_register_sha256": source_register_digest,
        "prior_e1a3_allocation_sha256": "c" * 64,
        "slot_count": 96,
        "allocations": [item.to_mapping() for item in allocations],
    }
    payload_path = root / "sealed" / "sampling-allocation.v1.json"
    manifest_path = root / "manifests" / "sampling-allocation.v1.sha256"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(allocation_payload), encoding="utf-8")
    manifest_path.write_text("0" * 64, encoding="ascii")

    with pytest.raises(E1A4PopulationError, match="E1A4_ALLOCATION_MANIFEST_INVALID"):
        runner._load_e1a4_allocation(root)


def test_population_sealer_rejects_noncanonical_sampling_allocation_bytes(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
        private_sampling_payload_digest,
    )
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    root = tmp_path / "e1a4"
    allocations = _allocations(_cases())
    source_register = _source_register_payload(allocations)
    source_register_digest = private_sampling_payload_digest(source_register)
    (root / "sealed").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    (root / "sealed" / "source-register.v1.json").write_bytes(_canonical_payload_bytes(source_register))
    (root / "manifests" / "source-register.v1.sha256").write_text(
        source_register_digest, encoding="ascii"
    )
    allocation_payload = {
        "schema_version": 1,
        "source_register_sha256": source_register_digest,
        "prior_e1a3_allocation_sha256": "c" * 64,
        "slot_count": 96,
        "allocations": [item.to_mapping() for item in allocations],
    }
    (root / "sealed" / "sampling-allocation.v1.json").write_text(
        json.dumps(allocation_payload, indent=2), encoding="utf-8"
    )
    (root / "manifests" / "sampling-allocation.v1.sha256").write_text(
        private_sampling_payload_digest(allocation_payload), encoding="ascii"
    )

    with pytest.raises(E1A4PopulationError, match="E1A4_ALLOCATION_MANIFEST_INVALID"):
        runner._load_e1a4_allocation(root)


def test_population_sealer_rejects_allocation_without_its_source_register(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    root = tmp_path / "e1a4"
    allocations = _allocations(_cases())
    allocation_payload = {
        "schema_version": 1,
        "source_register_sha256": "b" * 64,
        "prior_e1a3_allocation_sha256": "c" * 64,
        "slot_count": 96,
        "allocations": [item.to_mapping() for item in allocations],
    }
    _write_payload_with_manifest(
        allocation_payload,
        root / "sealed" / "sampling-allocation.v1.json",
        root / "manifests" / "sampling-allocation.v1.sha256",
    )

    with pytest.raises(E1A4PopulationError, match="E1A4_SOURCE_REGISTER_INVALID"):
        runner._load_e1a4_allocation(root)


def test_population_sealer_rejects_boolean_allocation_schema_version(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    root = tmp_path / "e1a4"
    allocation_payload = {
        "schema_version": True,
        "source_register_sha256": "b" * 64,
        "prior_e1a3_allocation_sha256": "c" * 64,
        "slot_count": 96,
        "allocations": [item.to_mapping() for item in _allocations(_cases())],
    }
    _write_payload_with_manifest(
        allocation_payload,
        root / "sealed" / "sampling-allocation.v1.json",
        root / "manifests" / "sampling-allocation.v1.sha256",
    )

    with pytest.raises(E1A4PopulationError, match="E1A4_ALLOCATION_MANIFEST_INVALID"):
        runner._load_e1a4_allocation(root)


def test_population_sealer_rejects_boolean_allocation_replicate() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    records = [item.to_mapping() for item in _allocations(_cases())]
    records[0]["replicate"] = True

    with pytest.raises(E1A4PopulationError, match="E1A4_ALLOCATION_MANIFEST_INVALID"):
        runner._parse_allocation_records(records)


def test_population_sealer_rejects_truncated_e1a3_question_inventory(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    evaluation_root = tmp_path / "retrieval-evaluation" / "v1"
    root = evaluation_root / "e1a4"
    development_path = evaluation_root / "sealed" / "development.jsonl"
    development_path.parent.mkdir(parents=True)
    development_path.write_text('{"question":"Earlier development question?"}\n', encoding="utf-8")
    e1a2_path = evaluation_root / "e1a2-fresh" / "sealed" / "candidate-pool.jsonl"
    e1a2_path.parent.mkdir(parents=True)
    e1a2_path.write_text('{"question":"Earlier E1a-2 question?"}\n', encoding="utf-8")
    e1a3_population = {
        "schema_version": 1,
        "allocation_sha256": "f" * 64,
        "cases": [{"question": "Only one E1a-3 question?"}],
    }
    _write_payload_with_manifest(
        e1a3_population,
        evaluation_root / "e1a3" / "sealed" / "population.v1.json",
        evaluation_root / "e1a3" / "manifests" / "population.v1.sha256",
    )

    with pytest.raises(E1A4PopulationError, match="E1A4_PRIOR_QUESTION_INVALID"):
        runner._load_prior_questions(root)


def test_population_sealer_rejects_boolean_e1a3_inventory_schema(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    evaluation_root = tmp_path / "retrieval-evaluation" / "v1"
    root = evaluation_root / "e1a4"
    development_path = evaluation_root / "sealed" / "development.jsonl"
    development_path.parent.mkdir(parents=True)
    development_path.write_text('{"question":"Earlier development question?"}\n', encoding="utf-8")
    e1a2_path = evaluation_root / "e1a2-fresh" / "sealed" / "candidate-pool.jsonl"
    e1a2_path.parent.mkdir(parents=True)
    e1a2_path.write_text('{"question":"Earlier E1a-2 question?"}\n', encoding="utf-8")
    e1a3_population = {
        "schema_version": True,
        "allocation_sha256": "f" * 64,
        "cases": [{"question": f"Earlier E1a-3 question {index}?"} for index in range(1, 97)],
    }
    _write_payload_with_manifest(
        e1a3_population,
        evaluation_root / "e1a3" / "sealed" / "population.v1.json",
        evaluation_root / "e1a3" / "manifests" / "population.v1.sha256",
    )

    with pytest.raises(E1A4PopulationError, match="E1A4_PRIOR_QUESTION_INVALID"):
        runner._load_prior_questions(root)


def test_population_sealer_cli_rejects_root_outside_private_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    allowed_root = tmp_path / "allowed-private"
    outside_root = tmp_path / "outside" / "e1a4"
    monkeypatch.setattr(runner, "PRIVATE_RETRIEVAL_ROOT", allowed_root)
    monkeypatch.setattr(sys, "argv", ["seal_e1a4_population.py", "--private-root", str(outside_root)])

    with pytest.raises(E1A4PopulationError, match="E1A4_PRIVATE_ROOT_INVALID"):
        runner.main()


def test_population_sealer_cli_rejects_a_different_private_cohort_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_population import (
        E1A4PopulationError,
    )

    runner = _seal_runner_module()
    allowed_root = tmp_path / "allowed-private"
    wrong_cohort = allowed_root / "e1a3"
    monkeypatch.setattr(runner, "PRIVATE_RETRIEVAL_ROOT", allowed_root)
    monkeypatch.setattr(sys, "argv", ["seal_e1a4_population.py", "--private-root", str(wrong_cohort)])

    with pytest.raises(E1A4PopulationError, match="E1A4_PRIVATE_ROOT_INVALID"):
        runner.main()


def test_population_sealer_preflight_reports_readiness_without_private_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _seal_runner_module()
    evaluation_root = tmp_path / "retrieval-evaluation" / "v1"
    root = evaluation_root / "e1a4"
    monkeypatch.setattr(runner, "PRIVATE_RETRIEVAL_ROOT", evaluation_root)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["seal_e1a4_population.py", "--private-root", str(root), "--preflight"],
    )

    exit_code = runner.cli()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "E1A4_TASK2_NOT_READY",
        "database_url_configured": False,
        "index_contract_present": False,
        "e1a3_prerequisites_complete": False,
        "historical_inventories_present": False,
        "sampling_frame_complete": False,
        "population_drafts_complete": False,
        "population_seal_complete": False,
        "ready_to_build_sampling_frame": False,
        "ready_to_seal_population": False,
        "task2_complete": False,
    }
    assert str(root) not in captured.out


def test_population_sealer_cli_sanitizes_missing_prerequisite_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _seal_runner_module()
    evaluation_root = tmp_path / "retrieval-evaluation" / "v1"
    root = evaluation_root / "e1a4"
    monkeypatch.setattr(runner, "PRIVATE_RETRIEVAL_ROOT", evaluation_root)
    monkeypatch.setattr(sys, "argv", ["seal_e1a4_population.py", "--private-root", str(root)])

    exit_code = runner.cli()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_POPULATION_BLOCKED",
        "error_code": "E1A4_POPULATION_PREREQUISITES_MISSING",
    }
    assert "Traceback" not in captured.err
    assert str(root) not in captured.err
