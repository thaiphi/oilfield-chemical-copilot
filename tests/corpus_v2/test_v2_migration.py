from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.corpus_v2_apply_migrations import (
    CorpusV2MigrationError,
    V2_MIGRATIONS_DIR,
    assert_v2_migrations_isolated,
    validate_migration_target,
    validate_v2_schema,
)
from oilfield_chemical_copilot.corpus_v2.models import ReleaseConfig


SHA = "a" * 64


@pytest.fixture
def config(tmp_path: Path) -> ReleaseConfig:
    return ReleaseConfig(
        release_id="corpus-v2-test",
        release_root=tmp_path,
        candidate_database_name="corpus_v2_candidate",
        configured_database_name="configured",
        legacy_database_names=("legacy",),
        expected_source_count=1,
        source_register_sha256=SHA,
        critical_source_register_sha256=SHA,
    )


def test_migration_target_rejects_equal_candidate_and_legacy_urls(config) -> None:
    with pytest.raises(CorpusV2MigrationError, match="C2_DATABASE_TARGET_INVALID"):
        validate_migration_target(
            "postgresql://user@host/corpus_v2_candidate",
            "postgresql://user@host/corpus_v2_candidate",
            config,
        )


def test_migration_target_rejects_legacy_name_before_any_write_logic(config) -> None:
    with pytest.raises(CorpusV2MigrationError, match="C2_DATABASE_TARGET_INVALID"):
        validate_migration_target(
            "postgresql://user@host/legacy", "postgresql://user@host/legacy2", config
        )


def test_v2_migrations_are_separate_from_legacy_runner() -> None:
    legacy_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    assert_v2_migrations_isolated(legacy_dir)
    assert V2_MIGRATIONS_DIR.resolve() != legacy_dir.resolve()
    assert all(path.parent.resolve() == legacy_dir.resolve() for path in legacy_dir.glob("*.sql"))


def test_v2_schema_requires_immutable_release_and_provenance_bound_chunks() -> None:
    validate_v2_schema(V2_MIGRATIONS_DIR / "0001_corpus_v2_release.sql")
