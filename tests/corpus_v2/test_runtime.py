import json
from hashlib import sha256

import pytest

from oilfield_chemical_copilot.corpus_v2 import runtime


def fixture_config(**changes):
    facts = dict(mode="v2", release_id="corpus-v2-r1",
                 schema_version=1, database_name="candidate", index_manifest_sha256="c" * 64,
                 source_register_sha256="a" * 64, source_count=1, chunk_count=1,
                 database_identity=sha256(b"postgresql://localhost/candidate").hexdigest(),
                 embedding_provider="deterministic", embedding_model="deterministic-token-hash-384",
                 embedding_dimension=384)
    facts.update(changes)
    content = (json.dumps(facts, sort_keys=True, separators=(",", ":")) + "\n").encode()
    binding = dict(schema_version=1, release_id=facts["release_id"],
                   candidate_database_name="candidate", critical_source_register_sha256="a" * 64,
                   digests={key: "a" * 64 for key in runtime.REQUIRED_DIGESTS})
    binding["digests"]["index_contract"] = sha256(content).hexdigest()
    sealed = (json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode()
    env = dict(CORPUS_MODE=facts["mode"], CORPUS_RELEASE_ID=facts["release_id"],
               CORPUS_DATABASE_URL="postgresql://localhost/candidate",
               CORPUS_EMBEDDING_PROVIDER=facts["embedding_provider"],
               CORPUS_EMBEDDING_MODEL=facts["embedding_model"],
               CORPUS_EMBEDDING_DIMENSION=str(facts["embedding_dimension"]),
               CORPUS_RELEASE_MANIFEST_SHA256=sha256(sealed).hexdigest(),
               CORPUS_RELEASE_BINDING_PATH="synthetic-binding")
    return env, runtime.AuthenticatedRuntimeBinding(sealed, content)


def test_runtime_requires_every_explicit_value_without_fallback():
    env, content = fixture_config()
    for key in env:
        incomplete = dict(env)
        del incomplete[key]
        with pytest.raises(runtime.CorpusV2RuntimeError, match="C2_RUNTIME_RELEASE_INVALID"):
            runtime.load_runtime_release(incomplete, binding_reader=lambda _: content)


@pytest.mark.parametrize("field,value", [
    ("CORPUS_DATABASE_URL", "postgresql://localhost/legacy"),
    ("CORPUS_EMBEDDING_MODEL", "wrong"),
    ("CORPUS_EMBEDDING_DIMENSION", "768"),
    ("CORPUS_RELEASE_MANIFEST_SHA256", "b" * 64),
    ("CORPUS_MODE", "legacy"),
])
def test_runtime_rejects_binding_mismatch(field, value):
    env, content = fixture_config()
    env[field] = value
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.load_runtime_release(env, binding_reader=lambda _: content)


def test_canonical_binding_and_explicit_legacy_are_verified():
    env, content = fixture_config(mode="legacy", release_id="legacy-r1",
                                  source_count=198, chunk_count=4797)
    # A candidate Task6 binding can never authenticate legacy mode.
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.load_runtime_release(env, binding_reader=lambda _: content)
    sealed = (json.dumps(dict(schema_version=1, kind="legacy-runtime",
        index_contract_sha256=sha256(content.index_contract_bytes).hexdigest()),
        sort_keys=True, separators=(",", ":")) + "\n").encode()
    env["CORPUS_RELEASE_MANIFEST_SHA256"] = sha256(sealed).hexdigest()
    content = runtime.AuthenticatedLegacyRuntimeBinding(sealed, content.index_contract_bytes)
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    assert release.mode == "legacy"
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.load_runtime_release(env, binding_reader=lambda _: runtime.AuthenticatedRuntimeBinding(
            content.binding_bytes + b" ", content.index_contract_bytes))


def test_unconfigured_reader_fails_closed_and_hides_paths():
    env, _ = fixture_config()
    with pytest.raises(runtime.CorpusV2RuntimeError) as error:
        runtime.load_runtime_release(env)
    assert str(error.value) == "C2_RUNTIME_RELEASE_INVALID"


def test_database_verifier_requires_read_only_exact_contract():
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    session = FakeSession()
    store = runtime.open_verified_runtime_store(release, reader=session.reader)
    assert session.active
    assert store.list_chunks() == ["synthetic"]
    store.close()
    assert not session.active
    with pytest.raises(runtime.CorpusV2RuntimeError):
        store.list_chunks()


def test_database_unavailable_hides_private_error():
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    def unavailable(*args, **kwargs):
        raise RuntimeError("private database secret")
    with pytest.raises(runtime.CorpusV2RuntimeError) as error:
        runtime.open_verified_runtime_store(release, reader=unavailable)
    assert str(error.value) == "C2_RUNTIME_DATABASE_UNAVAILABLE"


def test_database_result_for_another_release_is_rejected():
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    session = FakeSession(metadata=[("other", "c" * 64, "a" * 64)])
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.open_verified_runtime_store(release, reader=session.reader)
    assert not session.active


def test_index_contract_cannot_be_replaced_under_sealed_binding():
    env, content = fixture_config()
    replaced = runtime.AuthenticatedRuntimeBinding(content.binding_bytes, b"{}\n")
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.load_runtime_release(env, binding_reader=lambda _: replaced)


class FakeSession:
    """Synthetic SQL transport; verifier must request and inspect independent rows."""
    def __init__(self, *, readonly="on", database="candidate", metadata=None,
                 rows=None, index_type="vector(384)", isolation="repeatable read",
                 vector_index=True):
        self.active = False
        self.readonly, self.database, self.index_type = readonly, database, index_type
        self.isolation = isolation
        self.vector_index = vector_index
        self.metadata = metadata if metadata is not None else [
            ("corpus-v2-r1", "c" * 64, "a" * 64)]
        self.rows = rows if rows is not None else [(1, 1, "deterministic-token-hash-384", 384,
                                                  "corpus-v2-r1", "c" * 64)]
        self.store = self

    def reader(self, release, *, default_transaction_read_only):
        assert default_transaction_read_only is True
        return self

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False

    def execute(self, query):
        assert self.active
        if query == "SHOW transaction_read_only":
            self.result = [(self.readonly,)]
        elif query == "SHOW transaction_isolation":
            self.result = [(self.isolation,)]
        elif query == "SELECT current_database()":
            self.result = [(self.database,)]
        elif "FROM public.corpus_release" in query:
            self.result = self.metadata
        elif "FROM pg_attribute" in query:
            self.result = [(self.index_type,)]
        elif "FROM pg_index" in query:
            self.result = [(self.vector_index,)]
        elif "FROM public.chunks" in query:
            self.result = self.rows
        else:
            raise AssertionError(query)
        return self

    def fetchall(self):
        return self.result

    def list_chunks(self):
        assert self.active
        return ["synthetic"]


@pytest.mark.parametrize("damage", [dict(readonly="off"), dict(database="other"),
    dict(isolation="read committed"),
    dict(vector_index=False),
    dict(metadata=[]), dict(rows=[]), dict(index_type="vector(768)"),
    dict(rows=[(1, 1, "wrong", 384, "corpus-v2-r1", "c" * 64)])])
def test_database_independent_queries_reject_wrong_target(damage):
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    session = FakeSession(**damage)
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.open_verified_runtime_store(release, reader=session.reader)
    assert not session.active


def legacy_fixture():
    env, candidate = fixture_config(mode="legacy", release_id="legacy-r1",
                                   source_count=198, chunk_count=4797)
    sealed = (json.dumps(dict(schema_version=1, kind="legacy-runtime",
        index_contract_sha256=sha256(candidate.index_contract_bytes).hexdigest()),
        sort_keys=True, separators=(",", ":")) + "\n").encode()
    env["CORPUS_RELEASE_MANIFEST_SHA256"] = sha256(sealed).hexdigest()
    binding = runtime.AuthenticatedLegacyRuntimeBinding(sealed, candidate.index_contract_bytes)
    return runtime.load_runtime_release(env, binding_reader=lambda _: binding)


@pytest.mark.parametrize("source_count,chunk_count", [(198, 4797), (197, 4797), (198, 4796)])
def test_separate_legacy_reader_validates_fingerprint_without_v2_metadata(source_count, chunk_count):
    release = legacy_fixture()
    session = FakeSession(metadata=[], rows=[
        (chunk_count, source_count, "deterministic-token-hash-384", 384)])
    if (source_count, chunk_count) != (198, 4797):
        with pytest.raises(runtime.CorpusV2RuntimeError):
            runtime.open_verified_runtime_store(release, legacy_reader=session.reader)
        assert not session.active
    else:
        store = runtime.open_verified_runtime_store(release, legacy_reader=session.reader)
        assert store.list_chunks() == ["synthetic"]
        store.close()


def test_legacy_never_opens_v2_reader():
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.open_verified_runtime_store(legacy_fixture(), reader=lambda *a, **k: pytest.fail())


def test_store_detects_lost_readonly_and_closes_owned_session():
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    session = FakeSession()
    store = runtime.open_verified_runtime_store(release, reader=session.reader)
    session.readonly = "off"
    with pytest.raises(runtime.CorpusV2RuntimeError):
        store.list_chunks()
    assert not session.active


def test_sealed_source_register_must_match_index_contract():
    env, content = fixture_config(source_register_sha256="f" * 64)
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.load_runtime_release(env, binding_reader=lambda _: content)
