"""Explicit runtime contracts; operational readers must be supplied by the operator.

The binding reader must authenticate the private publication and return its verified
canonical runtime contract. No filesystem or database adapter is enabled implicitly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from collections.abc import Callable, Mapping
from contextlib import ExitStack
import weakref
from typing import Protocol

from .store import database_name_from_url
from .release import REQUIRED_DIGESTS
from .index_contract import IndexContract, parse_index_contract


class CorpusV2RuntimeError(ValueError):
    pass


class RuntimeQueryResult(Protocol):
    def fetchall(self) -> list[tuple]: ...


class RuntimeDatabaseSession(Protocol):
    """SQL transport and retrieval facade share one retained transaction."""
    store: object

    def execute(self, query: str) -> RuntimeQueryResult: ...

    def verify_connection_target(self, database_url: str) -> bool:
        """Authenticate the actual retained transport against the complete URL.

        An adapter must compare its connected server/port, database, authenticated
        principal and required TLS identity with the URL, resolving host aliases
        using its authenticated transport. Echoing the supplied URL or querying
        current_database alone does not satisfy this boundary. No default adapter
        is supplied; absence or anything other than True fails closed.
        """
        ...


@dataclass(frozen=True)
class AuthenticatedRuntimeBinding:
    """Bytes from the authenticated, exact private publication tree.

    The reader verifies the entire Task 6 release before supplying these members.
    The sealed index_contract member carries the runtime identity contract.
    """
    binding_bytes: bytes = field(repr=False)
    index_contract_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class AuthenticatedLegacyRuntimeBinding:
    """A separately authenticated legacy publication, never a V2 candidate binding."""
    binding_bytes: bytes = field(repr=False)
    index_contract_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class RuntimeRelease:
    mode: str
    release_id: str
    database_identity: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    manifest_sha256: str
    database_url: str = field(repr=False)
    index_contract: IndexContract | None = field(default=None, repr=False)


def load_runtime_release(
    environ: Mapping[str, str] | None = None,
    *, binding_reader: Callable[[str], AuthenticatedRuntimeBinding] | None = None,
) -> RuntimeRelease:
    """Check the authenticated private binding before any resource cache lookup."""
    env = os.environ if environ is None else environ
    try:
        names = ("MODE", "RELEASE_ID", "DATABASE_URL", "EMBEDDING_PROVIDER",
                 "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "RELEASE_MANIFEST_SHA256",
                 "RELEASE_BINDING_PATH")
        values = {name: env[f"CORPUS_{name}"] for name in names}
        if any(not value or value != value.strip() for value in values.values()):
            raise ValueError
        database_name = database_name_from_url(values["DATABASE_URL"])
        dimension = int(values["EMBEDDING_DIMENSION"])
        if dimension <= 0 or values["MODE"] not in {"v2", "legacy"}:
            raise ValueError
        if values["EMBEDDING_PROVIDER"] not in {"ollama", "sentence-transformers", "deterministic"}:
            raise ValueError
        facts = dict(mode=values["MODE"], release_id=values["RELEASE_ID"],
                     database_identity=sha256(values["DATABASE_URL"].encode()).hexdigest(),
                     embedding_provider=values["EMBEDDING_PROVIDER"],
                     embedding_model=values["EMBEDDING_MODEL"], embedding_dimension=dimension)
        if binding_reader is None:
            raise ValueError
        verified = binding_reader(values["RELEASE_BINDING_PATH"])
        expected_type = (AuthenticatedRuntimeBinding if values["MODE"] == "v2"
                         else AuthenticatedLegacyRuntimeBinding)
        if type(verified) is not expected_type:
            raise ValueError
        content = verified.index_contract_bytes
        contract = parse_index_contract(content)
        if (any(getattr(contract, name) != value for name, value in facts.items())
                or contract.database_name != database_name):
            raise ValueError
        binding = json.loads(verified.binding_bytes)
        if values["MODE"] == "v2":
            valid = (set(binding) == {"schema_version", "release_id", "candidate_database_name",
                             "critical_source_register_sha256", "digests"}
                and type(binding["schema_version"]) is int and binding["schema_version"] == 1
                and binding["release_id"] == values["RELEASE_ID"]
                and binding["candidate_database_name"] == database_name
                and set(binding["digests"]) == REQUIRED_DIGESTS
                and binding["digests"]["source_register"] == contract.source_register_sha256
                and binding["digests"]["index_contract"] == sha256(content).hexdigest())
        else:
            valid = binding == dict(schema_version=1, kind="legacy-runtime",
                                    index_contract_sha256=sha256(content).hexdigest())
        if (not valid or verified.binding_bytes != (json.dumps(binding, sort_keys=True,
                                               separators=(",", ":")) + "\n").encode()
                or sha256(verified.binding_bytes).hexdigest() != values["RELEASE_MANIFEST_SHA256"]):
            raise ValueError
        return RuntimeRelease(**facts, manifest_sha256=values["RELEASE_MANIFEST_SHA256"],
                              database_url=values["DATABASE_URL"], index_contract=contract)
    except Exception:
        raise CorpusV2RuntimeError("C2_RUNTIME_RELEASE_INVALID") from None


class VerifiedRuntimeStore:
    """Own the verified read-only session for every delegated retrieval operation."""
    def __init__(self, session, stack):
        self._session = session
        self._finalizer = weakref.finalize(self, stack.close)

    def close(self):
        self._finalizer()

    def _call(self, name, *args, **kwargs):
        if not self._finalizer.alive:
            raise CorpusV2RuntimeError("C2_RUNTIME_DATABASE_UNAVAILABLE")
        try:
            if self._session.execute("SHOW transaction_read_only").fetchall() != [("on",)]:
                raise ValueError
            return getattr(self._session.store, name)(*args, **kwargs)
        except Exception:
            self.close()
            raise CorpusV2RuntimeError("C2_RUNTIME_DATABASE_UNAVAILABLE") from None

    def list_chunks(self):
        return self._call("list_chunks")

    def search(self, *args, **kwargs):
        return self._call("search", *args, **kwargs)


def open_verified_runtime_store(release: RuntimeRelease, *, reader=None, legacy_reader=None):
    """Query independent database facts through an injected SQL session.

    Reader opens a repeatable-read, read-only transaction and yields a session with
    execute(sql).fetchall() tuple rows and a retrieval store using that SAME session.
    No connector is enabled by default. The returned owner keeps the context alive.
    Legacy uses only its separate reader and fingerprint, never V2 release metadata.
    """
    stack = ExitStack()
    try:
        contract = release.index_contract
        selected_reader = reader if release.mode == "v2" else legacy_reader
        if selected_reader is None or contract is None or contract.mode != release.mode:
            raise ValueError
        session: RuntimeDatabaseSession = stack.enter_context(
            selected_reader(release, default_transaction_read_only=True))
        def rows(query):
            return session.execute(query).fetchall()
        if (session.verify_connection_target(release.database_url) is not True
                or rows("SHOW transaction_read_only") != [("on",)]
                or rows("SHOW transaction_isolation") != [("repeatable read",)]
                or rows("SELECT current_database()") != [(contract.database_name,)]
                or rows("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                        "WHERE attrelid = 'public.chunks'::regclass AND attname = 'embedding' "
                        "AND NOT attisdropped") != [(f"vector({contract.embedding_dimension})",)]
                or rows("SELECT EXISTS (SELECT 1 FROM pg_index i "
                        "JOIN pg_class c ON c.oid = i.indexrelid "
                        "JOIN pg_am am ON am.oid = c.relam "
                        "JOIN pg_opclass op ON op.oid = i.indclass[0] "
                        "JOIN pg_attribute a ON a.attrelid = i.indrelid "
                        "AND a.attnum = i.indkey[0] "
                        "WHERE i.indrelid = 'public.chunks'::regclass "
                        "AND i.indisvalid AND i.indisready AND i.indpred IS NULL "
                        "AND i.indnkeyatts = 1 AND a.attname = 'embedding' "
                        "AND am.amname = 'hnsw' AND op.opcname = 'vector_cosine_ops')")
                    != [(True,)]):
            raise ValueError
        if release.mode == "legacy":
            if rows("SELECT count(*), count(DISTINCT source_path), embedding_model, "
                    "vector_dims(embedding) FROM public.chunks "
                    "GROUP BY embedding_model, vector_dims(embedding)") != [
                        (4797, 198, contract.embedding_model, contract.embedding_dimension)]:
                raise ValueError
        elif (rows("SELECT release_id, index_manifest_sha256, source_register_sha256 "
                   "FROM public.corpus_release") != [(contract.release_id,
                       contract.index_manifest_sha256, contract.source_register_sha256)]
                or rows("SELECT count(*), count(DISTINCT source_id), embedding_model, "
                        "vector_dims(embedding), release_id, manifest_sha256 FROM public.chunks "
                        "GROUP BY embedding_model, vector_dims(embedding), release_id, manifest_sha256")
                    != [(contract.chunk_count, contract.source_count, contract.embedding_model,
                         contract.embedding_dimension, contract.release_id,
                         contract.index_manifest_sha256)]):
            raise ValueError
        return VerifiedRuntimeStore(session, stack)
    except Exception:
        stack.close()
        raise CorpusV2RuntimeError("C2_RUNTIME_DATABASE_UNAVAILABLE") from None
