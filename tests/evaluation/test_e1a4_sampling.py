from __future__ import annotations

import json
from pathlib import Path

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
