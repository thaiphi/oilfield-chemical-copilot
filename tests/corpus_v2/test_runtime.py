import json
from hashlib import sha256

import pytest

from oilfield_chemical_copilot.corpus_v2 import runtime


def fixture_config(**changes):
    facts = dict(mode="v2", release_id="corpus-v2-r1",
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
    env, content = fixture_config(mode="legacy", release_id="legacy-r1")
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
    from contextlib import contextmanager
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    store = object()

    @contextmanager
    def reader(selected, *, default_transaction_read_only):
        assert selected == release
        assert default_transaction_read_only is True
        yield runtime.VerifiedRuntimeStore(release, True, store)

    assert runtime.open_verified_runtime_store(release, reader=reader) is store

    @contextmanager
    def writable(*args, **kwargs):
        yield runtime.VerifiedRuntimeStore(release, False, store)

    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.open_verified_runtime_store(release, reader=writable)


def test_database_unavailable_hides_private_error():
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    def unavailable(*args, **kwargs):
        raise RuntimeError("private database secret")
    with pytest.raises(runtime.CorpusV2RuntimeError) as error:
        runtime.open_verified_runtime_store(release, reader=unavailable)
    assert str(error.value) == "C2_RUNTIME_DATABASE_UNAVAILABLE"


def test_database_result_for_another_release_is_rejected():
    from contextlib import contextmanager
    from dataclasses import replace
    env, content = fixture_config()
    release = runtime.load_runtime_release(env, binding_reader=lambda _: content)
    @contextmanager
    def wrong_release(*args, **kwargs):
        yield runtime.VerifiedRuntimeStore(replace(release, release_id="other"), True, object())
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.open_verified_runtime_store(release, reader=wrong_release)


def test_index_contract_cannot_be_replaced_under_sealed_binding():
    env, content = fixture_config()
    replaced = runtime.AuthenticatedRuntimeBinding(content.binding_bytes, b"{}\n")
    with pytest.raises(runtime.CorpusV2RuntimeError):
        runtime.load_runtime_release(env, binding_reader=lambda _: replaced)
