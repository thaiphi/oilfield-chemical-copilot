"""Seal the fresh private E1a-4 population and canonical claims."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path[:0] = [str(PROJECT_ROOT), str(SRC_DIR)]

from oilfield_chemical_copilot.evaluation.e1a3_sampling import (  # noqa: E402
    E1A3SamplingError,
    E1A3SourceMetadata,
    E1A3SlotAllocation,
    build_sampling_slots,
)
from oilfield_chemical_copilot.evaluation.e1a4_population import (  # noqa: E402
    E1A4ClaimFixture,
    E1A4PopulationCase,
    E1A4PopulationError,
    build_population_and_claim_payloads,
    normalize_question,
    seal_population_and_claims,
)
from oilfield_chemical_copilot.evaluation.private_retrieval import (  # noqa: E402
    PRIVATE_RETRIEVAL_ROOT,
)

_ALLOCATION_FIELDS = {
    "slot_id",
    "topic",
    "source_role",
    "question_form",
    "evidence_depth",
    "replicate",
    "source_id",
    "parser_type",
    "locator",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _read_json(path: Path, *, code: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E1A4PopulationError(code) from error


def _read_manifest(path: Path, *, code: str) -> str:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise E1A4PopulationError(code) from error
    if not value:
        raise E1A4PopulationError(code)
    return value


def _read_json_artifact(path: Path, *, code: str) -> tuple[object, str]:
    try:
        content = path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E1A4PopulationError(code) from error
    return payload, sha256(content).hexdigest()


def _verify_existing_seal(
    *,
    population: dict[str, object],
    population_digest: str,
    claims: dict[str, object],
    claims_digest: str,
    population_path: Path,
    population_manifest_path: Path,
    claims_path: Path,
    claims_manifest_path: Path,
) -> bool:
    """Accept an existing seal only when all four exact artifacts verify."""
    paths = (population_path, population_manifest_path, claims_path, claims_manifest_path)
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths):
        raise E1A4PopulationError("E1A4_ARTIFACT_INCOMPLETE")
    existing_population, existing_population_digest = _read_json_artifact(
        population_path, code="E1A4_ARTIFACT_MISMATCH"
    )
    existing_claims, existing_claims_digest = _read_json_artifact(
        claims_path, code="E1A4_ARTIFACT_MISMATCH"
    )
    population_manifest = _read_manifest(population_manifest_path, code="E1A4_ARTIFACT_MISMATCH")
    claims_manifest = _read_manifest(claims_manifest_path, code="E1A4_ARTIFACT_MISMATCH")
    if (
        existing_population != population
        or existing_claims != claims
        or not isinstance(existing_population, dict)
        or not isinstance(existing_claims, dict)
        or existing_population_digest != population_manifest
        or existing_claims_digest != claims_manifest
        or existing_population_digest != population_digest
        or existing_claims_digest != claims_digest
    ):
        raise E1A4PopulationError("E1A4_ARTIFACT_MISMATCH")
    return True


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _parse_allocation_records(records: object) -> tuple[E1A3SlotAllocation, ...]:
    if not isinstance(records, list) or any(
        not isinstance(item, dict)
        or set(item) != _ALLOCATION_FIELDS
        or type(item["replicate"]) is not int
        for item in records
    ):
        raise E1A4PopulationError("E1A4_ALLOCATION_MANIFEST_INVALID")
    try:
        allocations = tuple(E1A3SlotAllocation(**item) for item in records)
    except (TypeError, ValueError) as error:
        raise E1A4PopulationError("E1A4_ALLOCATION_MANIFEST_INVALID") from error
    expected = {
        (slot.slot_id, slot.topic, slot.source_role, slot.question_form, slot.evidence_depth, slot.replicate)
        for slot in build_sampling_slots()
    }
    observed = {
        (
            item.slot_id,
            item.topic,
            item.source_role,
            item.question_form,
            item.evidence_depth,
            item.replicate,
        )
        for item in allocations
    }
    if (
        len(allocations) != 96
        or len(observed) != len(allocations)
        or observed != expected
        or any(
            not isinstance(value, str) or not value.strip()
            for item in allocations
            for value in (item.source_id, item.parser_type, item.locator)
        )
    ):
        raise E1A4PopulationError("E1A4_ALLOCATION_MANIFEST_INVALID")
    return allocations


def _load_source_register(root: Path) -> tuple[tuple[E1A3SourceMetadata, ...], str]:
    payload, payload_digest = _read_json_artifact(
        root / "sealed" / "source-register.v1.json",
        code="E1A4_SOURCE_REGISTER_INVALID",
    )
    manifest = _read_manifest(
        root / "manifests" / "source-register.v1.sha256",
        code="E1A4_SOURCE_REGISTER_INVALID",
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "e1a3_source_role_config_sha256", "sources"}
        or not _exact_int(payload["schema_version"], 1)
        or not _valid_sha256(payload["e1a3_source_role_config_sha256"])
        or not isinstance(payload["sources"], list)
        or not payload["sources"]
        or payload_digest != manifest
    ):
        raise E1A4PopulationError("E1A4_SOURCE_REGISTER_INVALID")
    source_fields = {
        "source_id",
        "source_role",
        "topic",
        "parser_type",
        "locators",
        "eligibility_status",
    }
    sources: list[E1A3SourceMetadata] = []
    try:
        for item in payload["sources"]:
            if not isinstance(item, dict) or set(item) != source_fields or not isinstance(item["locators"], list):
                raise E1A4PopulationError("E1A4_SOURCE_REGISTER_INVALID")
            sources.append(
                E1A3SourceMetadata(
                    source_id=item["source_id"],
                    source_role=item["source_role"],
                    topic=item["topic"],
                    parser_type=item["parser_type"],
                    locators=tuple(item["locators"]),
                    eligibility_status=item["eligibility_status"],
                )
            )
    except (E1A3SamplingError, TypeError, ValueError) as error:
        raise E1A4PopulationError("E1A4_SOURCE_REGISTER_INVALID") from error
    return tuple(sources), manifest


def _load_e1a4_allocation(root: Path) -> tuple[tuple[E1A3SlotAllocation, ...], str, str]:
    payload, payload_digest = _read_json_artifact(
        root / "sealed" / "sampling-allocation.v1.json",
        code="E1A4_ALLOCATION_MANIFEST_INVALID",
    )
    manifest = _read_manifest(
        root / "manifests" / "sampling-allocation.v1.sha256",
        code="E1A4_ALLOCATION_MANIFEST_INVALID",
    )
    expected_fields = {
        "schema_version",
        "source_register_sha256",
        "prior_e1a3_allocation_sha256",
        "slot_count",
        "allocations",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or not _exact_int(payload["schema_version"], 1)
        or not _exact_int(payload["slot_count"], 96)
        or not _valid_sha256(payload["source_register_sha256"])
        or not _valid_sha256(payload["prior_e1a3_allocation_sha256"])
        or payload_digest != manifest
    ):
        raise E1A4PopulationError("E1A4_ALLOCATION_MANIFEST_INVALID")
    allocations = _parse_allocation_records(payload["allocations"])
    sources, source_register_digest = _load_source_register(root)
    available = {
        (source.source_id, source.source_role, source.topic, source.parser_type, locator)
        for source in sources
        for locator in source.locators
    }
    if payload["source_register_sha256"] != source_register_digest or any(
        (item.source_id, item.source_role, item.topic, item.parser_type, item.locator) not in available
        for item in allocations
    ):
        raise E1A4PopulationError("E1A4_SOURCE_REGISTER_INVALID")
    return allocations, manifest, payload["prior_e1a3_allocation_sha256"]


def _read_questions_jsonl(path: Path) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise E1A4PopulationError("E1A4_PRIOR_QUESTION_INVALID") from error
    questions: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise E1A4PopulationError("E1A4_PRIOR_QUESTION_INVALID") from error
        if not isinstance(record, dict) or not isinstance(record.get("question"), str) or not record["question"].strip():
            raise E1A4PopulationError("E1A4_PRIOR_QUESTION_INVALID")
        questions.append(record["question"])
    return tuple(questions)


def _load_prior_questions(root: Path) -> tuple[str, ...]:
    evaluation_root = root.parent
    development_path = evaluation_root / "sealed" / "development.jsonl"
    e1a2_paths = tuple(sorted(evaluation_root.glob("e1a2*/sealed/candidate-pool.jsonl")))
    e1a3_root = evaluation_root / "e1a3"
    e1a3_population, e1a3_population_digest = _read_json_artifact(
        e1a3_root / "sealed" / "population.v1.json",
        code="E1A4_PRIOR_QUESTION_INVALID",
    )
    e1a3_manifest = _read_manifest(
        e1a3_root / "manifests" / "population.v1.sha256",
        code="E1A4_PRIOR_QUESTION_INVALID",
    )
    if (
        not development_path.is_file()
        or not e1a2_paths
        or not isinstance(e1a3_population, dict)
        or set(e1a3_population) != {"schema_version", "allocation_sha256", "cases"}
        or not _exact_int(e1a3_population["schema_version"], 1)
        or not _valid_sha256(e1a3_population["allocation_sha256"])
        or e1a3_population_digest != e1a3_manifest
        or not isinstance(e1a3_population.get("cases"), list)
    ):
        raise E1A4PopulationError("E1A4_PRIOR_QUESTION_INVALID")
    e1a3_questions: list[str] = []
    for item in e1a3_population["cases"]:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not item["question"].strip():
            raise E1A4PopulationError("E1A4_PRIOR_QUESTION_INVALID")
        e1a3_questions.append(item["question"])
    if len(e1a3_questions) != 96 or len({normalize_question(question) for question in e1a3_questions}) != 96:
        raise E1A4PopulationError("E1A4_PRIOR_QUESTION_INVALID")
    return tuple(
        question
        for path in (development_path, *e1a2_paths)
        for question in _read_questions_jsonl(path)
    ) + tuple(e1a3_questions)


def _load_prior_e1a3_locators(root: Path) -> tuple[tuple[str, ...], str]:
    payload, payload_digest = _read_json_artifact(
        root / "sealed" / "sampling-allocation.v4.json",
        code="E1A4_PRIOR_ALLOCATION_INVALID",
    )
    manifest = _read_manifest(
        root / "manifests" / "sampling-allocation.v4.sha256",
        code="E1A4_PRIOR_ALLOCATION_INVALID",
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "source_register_sha256", "slot_count", "allocations"}
        or not _exact_int(payload["schema_version"], 1)
        or not _exact_int(payload["slot_count"], 96)
        or not _valid_sha256(payload["source_register_sha256"])
        or payload_digest != manifest
    ):
        raise E1A4PopulationError("E1A4_PRIOR_ALLOCATION_INVALID")
    try:
        allocations = _parse_allocation_records(payload["allocations"])
    except E1A4PopulationError as error:
        raise E1A4PopulationError("E1A4_PRIOR_ALLOCATION_INVALID") from error
    locator_keys = tuple(f"{item.source_id.strip()}:{item.locator.strip()}" for item in allocations)
    if len(locator_keys) != len(set(locator_keys)):
        raise E1A4PopulationError("E1A4_PRIOR_ALLOCATION_INVALID")
    return locator_keys, manifest


def task2_readiness(root: Path) -> dict[str, object]:
    """Report prerequisite presence without opening or revealing private artifacts."""
    evaluation_root = root.parent
    e1a3_root = evaluation_root / "e1a3"
    database_url_configured = bool(os.environ.get("DATABASE_URL", "").strip())
    index_contract_present = (evaluation_root / "contracts" / "index-contract.json").is_file()
    e1a3_prerequisites_complete = all(
        path.is_file()
        for path in (
            e1a3_root / "draft" / "source-role-config.json",
            e1a3_root / "sealed" / "sampling-allocation.v4.json",
            e1a3_root / "manifests" / "sampling-allocation.v4.sha256",
        )
    )
    historical_inventories_present = (
        (evaluation_root / "sealed" / "development.jsonl").is_file()
        and bool(tuple(evaluation_root.glob("e1a2*/sealed/candidate-pool.jsonl")))
        and all(
            path.is_file()
            for path in (
                e1a3_root / "sealed" / "population.v1.json",
                e1a3_root / "manifests" / "population.v1.sha256",
                e1a3_root / "sealed" / "sampling-allocation.v4.json",
                e1a3_root / "manifests" / "sampling-allocation.v4.sha256",
            )
        )
    )
    sampling_frame_complete = all(
        path.is_file()
        for path in (
            root / "sealed" / "source-register.v1.json",
            root / "manifests" / "source-register.v1.sha256",
            root / "sealed" / "sampling-allocation.v1.json",
            root / "manifests" / "sampling-allocation.v1.sha256",
        )
    )
    population_drafts_complete = all(
        path.is_file()
        for path in (
            root / "draft" / "population.v1.json",
            root / "draft" / "canonical-claims.v1.json",
        )
    )
    population_seal_complete = all(
        path.is_file()
        for path in (
            root / "sealed" / "population.v1.json",
            root / "manifests" / "population.v1.sha256",
            root / "sealed" / "canonical-claims.v1.json",
            root / "manifests" / "canonical-claims.v1.sha256",
        )
    )
    ready_to_build_sampling_frame = (
        database_url_configured and index_contract_present and e1a3_prerequisites_complete
    )
    ready_to_seal_population = (
        historical_inventories_present and sampling_frame_complete and population_drafts_complete
    )
    task2_complete = population_seal_complete
    if task2_complete:
        status = "E1A4_TASK2_COMPLETE"
    elif ready_to_seal_population:
        status = "E1A4_TASK2_READY_TO_SEAL"
    elif ready_to_build_sampling_frame:
        status = "E1A4_TASK2_READY_TO_BUILD_SAMPLING_FRAME"
    else:
        status = "E1A4_TASK2_NOT_READY"
    return {
        "status": status,
        "database_url_configured": database_url_configured,
        "index_contract_present": index_contract_present,
        "e1a3_prerequisites_complete": e1a3_prerequisites_complete,
        "historical_inventories_present": historical_inventories_present,
        "sampling_frame_complete": sampling_frame_complete,
        "population_drafts_complete": population_drafts_complete,
        "population_seal_complete": population_seal_complete,
        "ready_to_build_sampling_frame": ready_to_build_sampling_frame,
        "ready_to_seal_population": ready_to_seal_population,
        "task2_complete": task2_complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal E1a-4 questions and canonical claims.")
    parser.add_argument("--private-root", type=Path, default=PRIVATE_RETRIEVAL_ROOT / "e1a4")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.private_root
    if root.resolve() != (PRIVATE_RETRIEVAL_ROOT / "e1a4").resolve():
        raise E1A4PopulationError("E1A4_PRIVATE_ROOT_INVALID")
    readiness = task2_readiness(root)
    if args.preflight:
        print(json.dumps(readiness))
        return 0 if readiness["task2_complete"] else 1
    if not readiness["ready_to_seal_population"]:
        raise E1A4PopulationError("E1A4_POPULATION_PREREQUISITES_MISSING")
    allocations, allocation_digest, expected_prior_digest = _load_e1a4_allocation(root)
    e1a3_locators, prior_digest = _load_prior_e1a3_locators(root.parent / "e1a3")
    if expected_prior_digest != prior_digest:
        raise E1A4PopulationError("E1A4_ALLOCATION_MANIFEST_INVALID")
    population_draft = _read_json(root / "draft" / "population.v1.json", code="E1A4_POPULATION_DRAFT_INVALID")
    claims_draft = _read_json(root / "draft" / "canonical-claims.v1.json", code="E1A4_POPULATION_DRAFT_INVALID")
    if not isinstance(population_draft, list) or not isinstance(claims_draft, list):
        raise E1A4PopulationError("E1A4_POPULATION_DRAFT_INVALID")
    try:
        cases = tuple(E1A4PopulationCase.from_mapping(item) for item in population_draft)
        claims = tuple(E1A4ClaimFixture.from_mapping(item) for item in claims_draft)
    except E1A4PopulationError as error:
        raise E1A4PopulationError("E1A4_POPULATION_DRAFT_INVALID") from error
    earlier_questions = _load_prior_questions(root)
    population, population_digest, canonical_claims, claims_digest = build_population_and_claim_payloads(
        cases=cases,
        claims=claims,
        allocations=allocations,
        allocation_sha256=allocation_digest,
        earlier_questions=earlier_questions,
        e1a3_locators=e1a3_locators,
    )
    population_path = root / "sealed" / "population.v1.json"
    population_manifest_path = root / "manifests" / "population.v1.sha256"
    claims_path = root / "sealed" / "canonical-claims.v1.json"
    claims_manifest_path = root / "manifests" / "canonical-claims.v1.sha256"
    if _verify_existing_seal(
        population=population,
        population_digest=population_digest,
        claims=canonical_claims,
        claims_digest=claims_digest,
        population_path=population_path,
        population_manifest_path=population_manifest_path,
        claims_path=claims_path,
        claims_manifest_path=claims_manifest_path,
    ):
        print(json.dumps({"status": "E1A4_POPULATION_ALREADY_SEALED_VERIFIED", "case_count": len(cases)}))
        return 0
    seal_population_and_claims(
        cases=cases,
        claims=claims,
        allocations=allocations,
        allocation_sha256=allocation_digest,
        earlier_questions=earlier_questions,
        e1a3_locators=e1a3_locators,
        population_path=population_path,
        population_digest_path=population_manifest_path,
        claims_path=claims_path,
        claims_digest_path=claims_manifest_path,
        private_root=root,
    )
    print(
        json.dumps(
            {
                "status": "E1A4_POPULATION_SEALED",
                "case_count": len(cases),
                "claim_fixture_count": len(claims),
            }
        )
    )
    return 0


def cli() -> int:
    """Convert safe contract failures into aggregate-only process output."""
    try:
        return main()
    except E1A4PopulationError as error:
        error_code = str(error)
        if re.fullmatch(r"E1A4_[A-Z0-9_]+", error_code) is None:
            error_code = "E1A4_POPULATION_FAILED"
        print(
            json.dumps({"status": "E1A4_POPULATION_BLOCKED", "error_code": error_code}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
