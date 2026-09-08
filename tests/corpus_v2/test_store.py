from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from oilfield_chemical_copilot.corpus_v2.models import CorpusV2Chunk, ReleaseBinding, ReleaseConfig
from oilfield_chemical_copilot.corpus_v2.store import (
    CorpusV2Store,
    CorpusV2StoreError,
    V2ChunkManifestEntry,
    V2Embedding,
    validate_v2_index,
    vector_sha256,
)


SHA = "a" * 64
OTHER_SHA = "b" * 64
CONTENT_SHA = hashlib.sha256(b"evidence").hexdigest()
VECTOR = (0.1,) * 384
OTHER_VECTOR = (0.2,) * 384


@pytest.fixture
def config(tmp_path):
    return ReleaseConfig(
        release_id="corpus-v2-test",
        release_root=tmp_path,
        candidate_database_name="corpus_v2_candidate",
        configured_database_name="configured",
        legacy_database_names=("legacy",),
        expected_source_count=2,
        source_register_sha256=SHA,
        critical_source_register_sha256=SHA,
    )


@pytest.fixture
def binding() -> ReleaseBinding:
    return ReleaseBinding(
        release_id="corpus-v2-test",
        source_register_sha256=SHA,
        index_manifest_sha256=SHA,
        chunk_count=2,
    )


@pytest.fixture
def manifest() -> tuple[V2ChunkManifestEntry, ...]:
    return (
        V2ChunkManifestEntry(
            CorpusV2Chunk("doc-1/chunk-0", "doc-1", 0, CONTENT_SHA, 8),
            SHA,
            SHA,
            "page:1",
            "local-model",
            384,
            "evidence",
        ),
        V2ChunkManifestEntry(
            CorpusV2Chunk("doc-2/chunk-0", "doc-2", 0, CONTENT_SHA, 8),
            OTHER_SHA,
            SHA,
            "page:2",
            "local-model",
            384,
            "evidence",
        ),
    )


def embeddings() -> tuple[V2Embedding, ...]:
    return (
        V2Embedding("doc-1/chunk-0", "local-model", vector_sha256(VECTOR), VECTOR),
        V2Embedding("doc-2/chunk-0", "local-model", vector_sha256(OTHER_VECTOR), OTHER_VECTOR),
    )


def test_v2_store_refuses_legacy_database_name(config) -> None:
    with pytest.raises(CorpusV2StoreError, match="C2_DATABASE_TARGET_INVALID"):
        CorpusV2Store("postgresql://user@host/legacy", "postgresql://user@host/legacy", config)


def test_v2_store_refuses_candidate_url_with_configured_wrong_database(config) -> None:
    with pytest.raises(CorpusV2StoreError, match="C2_DATABASE_TARGET_INVALID"):
        CorpusV2Store(
            "postgresql://user@host/not-the-candidate",
            "postgresql://user@host/legacy",
            config,
        )


def test_store_builds_only_provenance_bound_rows(config, manifest) -> None:
    store = CorpusV2Store(
        "postgresql://user@host/corpus_v2_candidate", "postgresql://user@host/legacy", config
    )
    rows = store.prepare_rows(manifest, embeddings())
    assert [row.chunk_id for row in rows] == ["doc-1/chunk-0", "doc-2/chunk-0"]
    assert rows[0].release_id == "corpus-v2-test"


def test_store_rejects_embedding_without_exact_manifest_entry(config, manifest) -> None:
    store = CorpusV2Store(
        "postgresql://user@host/corpus_v2_candidate", "postgresql://user@host/legacy", config
    )
    extra = (*embeddings(), V2Embedding("doc-3/chunk-0", "local-model", vector_sha256(VECTOR), VECTOR))
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_EXACT_SET_MISMATCH"):
        store.prepare_rows(manifest, extra)


def test_store_rejects_non_finite_or_wrong_dimension_vector() -> None:
    with pytest.raises(CorpusV2StoreError, match="C2_EMBEDDING_INVALID"):
        V2Embedding("doc-1/chunk-0", "local-model", SHA, (0.1,))
    with pytest.raises(CorpusV2StoreError, match="C2_EMBEDDING_INVALID"):
        V2Embedding("doc-1/chunk-0", "local-model", SHA, (float("nan"),) * 384)


def test_validate_release_rejects_extra_or_orphan_database_chunk(config, binding, manifest) -> None:
    store = CorpusV2Store(
        "postgresql://user@host/corpus_v2_candidate", "postgresql://user@host/legacy", config
    )
    store.replace_fake_rows(store.prepare_rows(manifest, embeddings()))
    store.insert_fake_row(
        replace(store.read_rows()[0], chunk_id="doc-3/chunk-0", source_id="doc-3")
    )
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_EXACT_SET_MISMATCH"):
        validate_v2_index(store, binding, manifest)


def test_validate_release_rejects_wrong_release_or_manifest_metadata(config, binding, manifest) -> None:
    store = CorpusV2Store(
        "postgresql://user@host/corpus_v2_candidate", "postgresql://user@host/legacy", config
    )
    rows = list(store.prepare_rows(manifest, embeddings()))
    rows[0] = replace(rows[0], release_id="corpus-v2-other")
    store.replace_fake_rows(rows)
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_EXACT_SET_MISMATCH"):
        validate_v2_index(store, binding, manifest)


def test_validate_release_rejects_altered_candidate_content(config, binding, manifest) -> None:
    store = CorpusV2Store(
        "postgresql://user@host/corpus_v2_candidate", "postgresql://user@host/legacy", config
    )
    rows = list(store.prepare_rows(manifest, embeddings()))
    rows[0] = replace(rows[0], content="altered evidence")
    store.replace_fake_rows(rows)
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_EXACT_SET_MISMATCH"):
        validate_v2_index(store, binding, manifest)


def test_validate_release_rejects_vector_with_swapped_sealed_identity(config, binding, manifest) -> None:
    store = CorpusV2Store(
        "postgresql://user@host/corpus_v2_candidate", "postgresql://user@host/legacy", config
    )
    rows = list(store.prepare_rows(manifest, embeddings()))
    rows[0] = replace(rows[0], vector=OTHER_VECTOR)
    store.replace_fake_rows(rows)
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_EXACT_SET_MISMATCH"):
        validate_v2_index(store, binding, manifest)


def test_store_rejects_manifest_with_non_384_vector_dimension(config) -> None:
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_METADATA_INVALID"):
        V2ChunkManifestEntry(
            CorpusV2Chunk("doc-1/chunk-0", "doc-1", 0, CONTENT_SHA, 8),
            SHA,
            SHA,
            "page:1",
            "local-model",
            3,
            "evidence",
        )
