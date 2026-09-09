"""Canonical index identity shared by sealing and runtime verification."""
from dataclasses import asdict, dataclass
import json
import re


@dataclass(frozen=True)
class IndexContract:
    mode: str
    release_id: str
    database_name: str
    database_identity: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    index_manifest_sha256: str
    source_register_sha256: str
    source_count: int
    chunk_count: int
    schema_version: int = 1

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version != 1
                or self.mode not in {"v2", "legacy"}
                or self.embedding_provider not in {"deterministic", "ollama", "sentence-transformers"}
                or type(self.embedding_dimension) is not int or self.embedding_dimension != 384
                or any(not isinstance(v, str) or not v.strip() or v != v.strip()
                       for v in (self.release_id, self.database_name, self.embedding_model))
                or any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
                       for v in (self.database_identity, self.index_manifest_sha256,
                                 self.source_register_sha256))
                or any(type(v) is not int or v <= 0 for v in (self.source_count, self.chunk_count))
                or self.source_count > self.chunk_count
                or (self.mode == "legacy" and (self.source_count, self.chunk_count) != (198, 4797))
                or (self.embedding_provider == "deterministic"
                    and self.embedding_model != "deterministic-token-hash-384")):
            raise ValueError("C2_INDEX_CONTRACT_INVALID")


def index_contract_bytes(contract: IndexContract) -> bytes:
    return (json.dumps(asdict(contract), sort_keys=True, separators=(",", ":")) + "\n").encode()


def parse_index_contract(content: bytes) -> IndexContract:
    contract = IndexContract(**json.loads(content))
    if index_contract_bytes(contract) != content:
        raise ValueError("C2_INDEX_CONTRACT_INVALID")
    return contract
