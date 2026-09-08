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

from .store import database_name_from_url
from .release import REQUIRED_DIGESTS


class CorpusV2RuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class AuthenticatedRuntimeBinding:
    """Bytes from the authenticated, exact private publication tree.

    The reader verifies the entire Task 6 release before supplying these members.
    The sealed index_contract member carries the runtime identity contract.
    """
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
        if not isinstance(verified, AuthenticatedRuntimeBinding):
            raise ValueError
        content = verified.index_contract_bytes
        canonical = (json.dumps(facts, sort_keys=True, separators=(",", ":")) + "\n").encode()
        binding = json.loads(verified.binding_bytes)
        if (set(binding) != {"schema_version", "release_id", "candidate_database_name",
                             "critical_source_register_sha256", "digests"}
                or binding["schema_version"] != 1
                or binding["release_id"] != values["RELEASE_ID"]
                or binding["candidate_database_name"] != database_name
                or set(binding["digests"]) != REQUIRED_DIGESTS
                or binding["digests"]["index_contract"] != sha256(content).hexdigest()
                or verified.binding_bytes != (json.dumps(binding, sort_keys=True,
                                               separators=(",", ":")) + "\n").encode()
                or content != canonical
                or sha256(verified.binding_bytes).hexdigest() != values["RELEASE_MANIFEST_SHA256"]):
            raise ValueError
        return RuntimeRelease(**facts, manifest_sha256=values["RELEASE_MANIFEST_SHA256"],
                              database_url=values["DATABASE_URL"])
    except Exception:
        raise CorpusV2RuntimeError("C2_RUNTIME_RELEASE_INVALID") from None


@dataclass(frozen=True)
class VerifiedRuntimeStore:
    release: RuntimeRelease
    read_only: bool
    store: object = field(repr=False)


def open_verified_runtime_store(release: RuntimeRelease, *, reader=None):
    """Reserved operational boundary: verify exact DB contract read-only, then return store.

    Deployment must supply an adapter that checks the V2 release metadata and exact
    index contract (or the separate legacy contract) in a read-only transaction.
    Until that adapter is supplied this boundary refuses every database connection.
    """
    try:
        if reader is None:
            raise ValueError
        with reader(release, default_transaction_read_only=True) as verified:
            if (not isinstance(verified, VerifiedRuntimeStore)
                    or verified.release != release or verified.read_only is not True
                    or verified.store is None):
                raise ValueError
            return verified.store
    except Exception:
        raise CorpusV2RuntimeError("C2_RUNTIME_DATABASE_UNAVAILABLE") from None
