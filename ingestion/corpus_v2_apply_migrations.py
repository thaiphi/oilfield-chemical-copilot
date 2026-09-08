"""Validate a Corpus V2 migration target without connecting or applying SQL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oilfield_chemical_copilot.corpus_v2.models import ReleaseConfig
from oilfield_chemical_copilot.corpus_v2.store import CorpusV2StoreError, database_name_from_url


V2_MIGRATIONS_DIR = PROJECT_ROOT / "db" / "corpus_v2_migrations"
LEGACY_MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"


class CorpusV2MigrationError(ValueError):
    """Raised before any candidate migration action is possible."""


def validate_migration_target(candidate_url: str, legacy_url: str, config: ReleaseConfig) -> None:
    """Reject an unsafe target using URL parsing only; never connect or write."""
    try:
        candidate_name = database_name_from_url(candidate_url)
        legacy_name = database_name_from_url(legacy_url)
    except CorpusV2StoreError as error:
        raise CorpusV2MigrationError("C2_DATABASE_TARGET_INVALID") from error
    forbidden = {config.configured_database_name, *config.legacy_database_names, legacy_name}
    if candidate_name != config.candidate_database_name or candidate_name in forbidden:
        raise CorpusV2MigrationError("C2_DATABASE_TARGET_INVALID")


def assert_v2_migrations_isolated(legacy_migrations_dir: Path = LEGACY_MIGRATIONS_DIR) -> None:
    if V2_MIGRATIONS_DIR.resolve() == legacy_migrations_dir.resolve():
        raise CorpusV2MigrationError("C2_MIGRATION_ISOLATION_INVALID")
    if not V2_MIGRATIONS_DIR.is_dir() or not any(V2_MIGRATIONS_DIR.glob("*.sql")):
        raise CorpusV2MigrationError("C2_MIGRATION_ISOLATION_INVALID")


def validate_v2_schema(schema_path: Path) -> None:
    """Check the checked-in SQL carries the non-negotiable V2 isolation contract."""
    try:
        schema = schema_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CorpusV2MigrationError("C2_MIGRATION_SCHEMA_INVALID") from error
    uncommented = re.sub(r"--[^\r\n]*", "", schema).lower()
    release_table = re.search(r"create table corpus_release\s*\((.*?)\n\);", uncommented, re.DOTALL)
    chunks_table = re.search(r"create table chunks\s*\((.*?)\n\);", uncommented, re.DOTALL)
    required_release_fragments = (
        "release_id text not null unique",
        "index_manifest_sha256 char(64) not null",
        "unique (release_id, index_manifest_sha256)",
    )
    required_chunk_fragments = (
        "chunk_id text not null",
        "release_id text not null",
        "source_id text not null",
        "source_sha256 char(64) not null",
        "text_sha256 char(64) not null",
        "manifest_sha256 char(64) not null",
        "embedding vector(384) not null",
        "embedding_sha256 char(64) not null",
        "foreign key (release_id, manifest_sha256)",
        "references corpus_release (release_id, index_manifest_sha256)",
    )
    if (
        release_table is None
        or chunks_table is None
        or not all(fragment in release_table.group(1) for fragment in required_release_fragments)
        or not all(fragment in chunks_table.group(1) for fragment in required_chunk_fragments)
        or re.search(
            r"create trigger corpus_release_immutable\s+before update or delete on corpus_release",
            uncommented,
        )
        is None
        or "create index v2_chunks_embedding_hnsw_idx on chunks using hnsw (embedding vector_cosine_ops);"
        not in uncommented
    ):
        raise CorpusV2MigrationError("C2_MIGRATION_SCHEMA_INVALID")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Corpus V2 migration target; does not apply SQL.")
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--legacy-url", required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = ReleaseConfig.from_mapping(
            json.loads(args.release_config.read_text(encoding="utf-8"))
        )
        validate_migration_target(args.candidate_url, args.legacy_url, config)
        assert_v2_migrations_isolated()
        validate_v2_schema(V2_MIGRATIONS_DIR / "0001_corpus_v2_release.sql")
    except (CorpusV2MigrationError, OSError, json.JSONDecodeError, ValueError):
        return 2
    # This command validates only. An executor is intentionally absent pending separate authorization.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
