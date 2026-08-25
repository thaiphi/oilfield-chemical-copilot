from __future__ import annotations

from dataclasses import dataclass, field

FALLBACK_MESSAGE = "I do not have enough retrieved evidence to answer confidently."


class RagError(Exception):
    """Base error for safe RAG failures."""


class RagConfigurationError(RagError):
    """Raised when required runtime configuration is missing or invalid."""


class RagGenerationError(RagError):
    """Raised when answer generation fails or returns malformed output."""


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    chunk_id: str
    source_file: str
    page_or_sheet: str
    topic: str
    excerpt: str
    score: float
    retrieval_method: str = "vector"
    retrieval_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagPrompt:
    system_prompt: str
    user_prompt: str
    sources: list[SourceEvidence]


@dataclass(frozen=True)
class RagDraft:
    answer: str
    why_this_matters: str
    cited_source_ids: list[str]
    recommended_next_checks: list[str]
    limitations: str

    def validate(self, allowed_source_ids: set[str]) -> None:
        missing_fields = [
            name
            for name, value in (
                ("answer", self.answer),
                ("why_this_matters", self.why_this_matters),
                ("limitations", self.limitations),
            )
            if not value.strip()
        ]
        if missing_fields:
            raise RagGenerationError(f"Generated answer is missing: {', '.join(missing_fields)}")
        if len(self.recommended_next_checks) != 3:
            raise RagGenerationError("Generated answer must include exactly three next checks")
        if not self.cited_source_ids:
            raise RagGenerationError("Generated answer must include at least one cited source")
        unknown_sources = set(self.cited_source_ids) - allowed_source_ids
        if unknown_sources:
            raise RagGenerationError("Generated answer cited unknown sources")


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[SourceEvidence] = field(default_factory=list)
    weak_evidence: bool = False