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


def test_v2_schema_rejects_required_tokens_hidden_in_block_comment(tmp_path: Path) -> None:
    schema = tmp_path / "comment-only.sql"
    schema.write_text(
        "/* create table corpus_release (release_id text not null unique, "
        "index_manifest_sha256 char(64) not null, unique (release_id, index_manifest_sha256)); "
        "create table chunks (chunk_id text not null, release_id text not null, "
        "source_id text not null, source_sha256 char(64) not null, text_sha256 char(64) not null, "
        "manifest_sha256 char(64) not null, embedding vector(384) not null, "
        "embedding_sha256 char(64) not null, foreign key (release_id, manifest_sha256) "
        "references corpus_release (release_id, index_manifest_sha256)); */"
    )
    with pytest.raises(CorpusV2MigrationError, match="C2_MIGRATION_SCHEMA_INVALID"):
        validate_v2_schema(schema)
