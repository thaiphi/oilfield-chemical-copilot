"""Synthetic processing identities must survive every typed storage boundary."""

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from oilfield_chemical_copilot.corpus_v2.canonical import (
    CorpusV2CanonicalError, canonical_jsonl, sha256_canonical_jsonl,
)
from oilfield_chemical_copilot.corpus_v2.models import (
    CorpusV2Chunk, CorpusV2ContractError, EmbeddingRecord, ReleaseBinding, ReleaseConfig,
)
from oilfield_chemical_copilot.corpus_v2.processing import build_v2_chunk
from oilfield_chemical_copilot.corpus_v2.store import (
    CorpusV2Store, CorpusV2StoreError, V2ChunkManifestEntry, V2Embedding,
    validate_v2_index, vector_sha256,
)


def test_processing_sha256_survives_typed_manifest_embedding_and_store(tmp_path):
    release_id = "corpus-v2-2026-09-09"
    source_sha = hashlib.sha256(b"synthetic source").hexdigest()
    loaded = build_v2_chunk(
        release_id=release_id, source_id="doc-1", source_byte_sha256=source_sha,
        parser_policy_version="parser-v1", chunk_policy_version="chunk-v1",
        location="sheet:Private synthetic label", ordinal=0, text="synthetic evidence",
        sealed_source_ids=("doc-1",),
    )
    typed = CorpusV2Chunk.from_loaded_chunk(loaded, release_id=release_id)
    assert typed.chunk_id == loaded.metadata.chunk_id
    encoded = canonical_jsonl((typed,))
    public = json.loads(encoded)
    assert public["chunk_id"] == loaded.metadata.chunk_id
    assert "provenance" not in public
    assert b"Private synthetic label" not in encoded
    with pytest.raises(CorpusV2CanonicalError):
        canonical_jsonl((public,))  # Raw hash projections cannot prove derivation.
    with pytest.raises(CorpusV2CanonicalError):
        canonical_jsonl((asdict(typed),))  # Private provenance is never public JSONL.

    digest = sha256_canonical_jsonl((typed,))
    vector = (0.1,) * 384
    embedding_record = EmbeddingRecord(typed.chunk_id, "local-model", vector_sha256(vector), 384)
    assert json.loads(canonical_jsonl((asdict(embedding_record),)))["chunk_id"] == typed.chunk_id
    manifest = V2ChunkManifestEntry(typed, source_sha, digest, loaded.metadata.page_or_sheet,
                                    "local-model", 384, loaded.text, embedding_record.embedding_sha256)
    config = ReleaseConfig(release_id, tmp_path, "candidate", "configured", ("legacy",),
                           1, "a" * 64, "b" * 64)
    store = CorpusV2Store("postgresql://user@host/candidate", "postgresql://user@host/legacy", config)
    embedding = V2Embedding(typed.chunk_id, "local-model", embedding_record.embedding_sha256, vector)
    rows = store.prepare_rows((manifest,), (embedding,))
    assert rows[0].chunk_id == loaded.metadata.chunk_id
    store.replace_fake_rows(rows)
    validate_v2_index(store, ReleaseBinding(release_id, "a" * 64, digest, 1), (manifest,))

    for changes in ({"source_id": "doc-2"}, {"ordinal": 1}, {"text_sha256": "c" * 64},
                    {"chunk_id": "d" * 64}):
        with pytest.raises(CorpusV2ContractError, match="C2_CHUNK_PROVENANCE_INVALID"):
            replace(typed, **changes)
    for changes in ({"release_id": "corpus-v2-other"}, {"source_byte_sha256": "c" * 64},
                    {"parser_policy_version": "parser-v2"}, {"chunk_policy_version": "chunk-v2"},
                    {"location": "page:2"}):
        with pytest.raises(CorpusV2ContractError, match="C2_CHUNK_PROVENANCE_INVALID"):
            replace(typed, provenance=replace(typed.provenance, **changes))
    for changes in ({"source_sha256": "c" * 64}, {"location": "page:2"}):
        with pytest.raises(CorpusV2StoreError, match="C2_INDEX_METADATA_INVALID"):
            replace(manifest, **changes)
    with pytest.raises(CorpusV2ContractError, match="C2_CHUNK_PROVENANCE_INVALID"):
        CorpusV2Chunk.from_loaded_chunk(loaded, release_id="corpus-v2-other")
    wrong_release = CorpusV2Store("postgresql://user@host/candidate", "postgresql://user@host/legacy",
                                  replace(config, release_id="corpus-v2-other"))
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_METADATA_INVALID"):
        wrong_release.prepare_rows((manifest,), (embedding,))
