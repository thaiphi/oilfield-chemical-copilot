from __future__ import annotations

import re
from dataclasses import replace
from pathlib import PurePosixPath, PureWindowsPath

from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagAnswer, RagDraft, SourceEvidence


_FIELD_TREATMENT_DIRECTIVE = re.compile(
    r"\b(?:implement(?:s|ed|ing)?|inject(?:s|ed|ing)?|appl(?:y|ies|ied|ying)|select(?:s|ed|ing)?|dos(?:e|es|ed|ing)?)\b.*\b(?:treatments?|chemicals?|inhibitors?|dispersants?|dissolvers?|biocides?|scale\s+inhibition)\b",
    re.IGNORECASE,
)
_NUMERIC_VALUE = re.compile(r"\b\d+(?:\.\d+)?\b")
_MEASUREMENT = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg\s*/\s*l|ppm|ppb|psi|bbl\s*/\s*d|gal\s*/\s*d|percent|%|(?:degrees?\s*)?[fc])(?=$|\W)",
    re.IGNORECASE,
)
_COMPARATOR_PATTERNS = (
    ("range", re.compile(r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)),
    ("above", re.compile(r"\b(?:above|greater than|more than)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)),
    ("at_least", re.compile(r"\b(?:at least|no less than)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)),
    ("below", re.compile(r"\b(?:below|less than|under)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)),
    ("at_most", re.compile(r"\b(?:not exceed|no more than|at most)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)),
)
_STRONG_CONCLUSION = re.compile(
    r"\b(?:prove(?:s|d)?|confirm(?:s|ed)?|demonstrate(?:s|d)?|cause(?:s|d)?|will)\b",
    re.IGNORECASE,
)
_LIMITING_EVIDENCE = re.compile(
    r"\b(?:may|might|could|possible|associated|not definitive|does not establish|insufficient|no established|no validated)\b",
    re.IGNORECASE,
)
_CONFLICT_EVIDENCE = re.compile(
    r"\b(?:no direct|does not identify|does not establish|insufficient|cannot identify|unable to identify)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_LANGUAGE = re.compile(
    r"\b(?:may|might|could|possible|associated|conflict(?:ing)?|differ(?:s|ent|ing)?|mixed|uncertain|inconclusive|insufficient|cannot|does not establish|limited)\b",
    re.IGNORECASE,
)
_ENGINEERING_REVIEW_CHECK = "Obtain qualified engineering review before any field treatment change."


def format_answer(draft: RagDraft, sources: list[SourceEvidence], *, question: str) -> RagAnswer:
    selected_sources = select_supported_sources(question=question, draft=draft, sources=sources)
    selected_draft = replace(draft, cited_source_ids=[source.source_id for source in selected_sources])
    source_by_id = {source.source_id: source for source in sources}
    selected_draft.validate(set(source_by_id))
    cited_sources = [source_by_id[source_id] for source_id in selected_draft.cited_source_ids]
    if _has_unsupported_numeric_claim(selected_draft.answer, cited_sources):
        return weak_evidence_answer(
            limitations="Retrieved evidence did not support a numeric claim in the generated answer."
        )
    if _has_unsupported_semantic_claim(selected_draft.answer, cited_sources):
        return weak_evidence_answer(
            limitations="Retrieved evidence did not preserve the technical claim in the generated answer."
        )
    evidence_lines = [_evidence_line(source) for source in cited_sources]
    if not evidence_lines:
        evidence_lines = ["- No retrieved source was cited."]
    checks = [
        f"{index}. {_safe_next_check(check)}"
        for index, check in enumerate(draft.recommended_next_checks, start=1)
    ]
    text = (
        f"Answer:\n{selected_draft.answer.strip()}\n\n"
        f"Why this matters:\n{selected_draft.why_this_matters.strip()}\n\n"
        "Evidence from retrieved sources:\n"
        + "\n".join(evidence_lines)
        + "\n\nRecommended next checks:\n"
        + "\n".join(checks)
        + f"\n\nLimitations:\n{selected_draft.limitations.strip()}"
    )
    return RagAnswer(text=text, sources=cited_sources, weak_evidence=False)


def select_supported_sources(
    *, question: str, draft: RagDraft, sources: list[SourceEvidence]
) -> list[SourceEvidence]:
    del question
    sources_by_id = {source.source_id: source for source in sources}
    selected: list[SourceEvidence] = []
    seen_ids: set[str] = set()
    for source_id in draft.cited_source_ids:
        if source_id in seen_ids:
            continue
        source = sources_by_id.get(source_id)
        if source is not None:
            selected.append(source)
            seen_ids.add(source_id)
    return selected


def _safe_next_check(check: str) -> str:
    if _FIELD_TREATMENT_DIRECTIVE.search(check):
        return _ENGINEERING_REVIEW_CHECK
    return check


def _has_unsupported_numeric_claim(answer: str, sources: list[SourceEvidence]) -> bool:
    cited_text = " ".join(source.excerpt for source in sources)
    return any(value not in cited_text for value in _NUMERIC_VALUE.findall(answer))


def _has_unsupported_semantic_claim(answer: str, sources: list[SourceEvidence]) -> bool:
    cited_text = " ".join(source.excerpt for source in sources)
    if not _measurements_supported(answer, cited_text):
        return True
    if not _comparators_supported(answer, cited_text):
        return True
    if _STRONG_CONCLUSION.search(answer) and not _STRONG_CONCLUSION.search(cited_text):
        return True
    if _LIMITING_EVIDENCE.search(cited_text) and _asserts_without_uncertainty(answer):
        return True
    return len(sources) > 1 and bool(_CONFLICT_EVIDENCE.search(cited_text)) and not bool(
        _UNCERTAINTY_LANGUAGE.search(answer)
    )


def _measurements_supported(answer: str, cited_text: str) -> bool:
    cited_measurements = {
        (match.group("value"), _normalize_unit(match.group("unit")))
        for match in _MEASUREMENT.finditer(cited_text)
    }
    return all(
        (match.group("value"), _normalize_unit(match.group("unit"))) in cited_measurements
        for match in _MEASUREMENT.finditer(answer)
    )


def _normalize_unit(unit: str) -> str:
    return re.sub(r"\s+", "", unit.casefold())


def _comparators_supported(answer: str, cited_text: str) -> bool:
    cited_comparators = _comparators(cited_text)
    return _comparators(answer).issubset(cited_comparators)


def _comparators(text: str) -> set[tuple[str, ...]]:
    return {
        (kind, *match.groups())
        for kind, pattern in _COMPARATOR_PATTERNS
        for match in pattern.finditer(text)
    }


def _asserts_without_uncertainty(answer: str) -> bool:
    return bool(re.search(r"\b(?:is|are|was|were|proves?|confirms?|causes?|will)\b", answer, re.IGNORECASE)) and not bool(
        _UNCERTAINTY_LANGUAGE.search(answer)
    )


def weak_evidence_answer(*, limitations: str) -> RagAnswer:
    draft = RagDraft(
        answer=FALLBACK_MESSAGE,
        why_this_matters="A grounded answer needs retrieved evidence that is relevant enough to cite.",
        cited_source_ids=[],
        recommended_next_checks=[
            "Rephrase the question with the chemical, symptom, and operating context.",
            "Confirm the sample corpus has been parsed, embedded, and indexed.",
            "Lower the evidence threshold only after reviewing retrieval quality.",
        ],
        limitations=limitations,
    )
    text = (
        f"Answer:\n{draft.answer}\n\n"
        f"Why this matters:\n{draft.why_this_matters}\n\n"
        "Evidence from retrieved sources:\n- No qualifying retrieved source was available.\n\n"
        "Recommended next checks:\n"
        + "\n".join(
            f"{index}. {check}" for index, check in enumerate(draft.recommended_next_checks, start=1)
        )
        + f"\n\nLimitations:\n{draft.limitations}"
    )
    return RagAnswer(text=text, sources=[], weak_evidence=True)


def scope_limited_answer(*, category: str) -> RagAnswer:
    limitation_by_category = {
        "site_specific_determination": "The question asks for a site-specific determination that requires site data and engineering review.",
        "field_ready_prescription": "The question asks for a field-ready prescription that requires site data and engineering review.",
        "complete_input_substitution": "The question asks the public corpus to replace a complete analysis or input set.",
    }
    limitations = limitation_by_category.get(
        category,
        "The question requires a determination outside this assistant's grounded general-review scope.",
    )
    text = (
        "Answer:\nThis assistant cannot provide a field-ready prescription or site-specific determination from a general corpus.\n\n"
        "Why this matters:\nOperational treatment decisions require complete, current field inputs and qualified engineering review.\n\n"
        "Evidence from retrieved sources:\n- Retrieval was not run because the question exceeds the supported claim scope.\n\n"
        "Recommended next checks:\n"
        "1. Gather the applicable site, fluid, operating, and laboratory data.\n"
        "2. Define the decision that needs qualified engineering review.\n"
        "3. Use this assistant for a general evidence review after that scope is separated.\n\n"
        f"Limitations:\n{limitations}"
    )
    return RagAnswer(text=text, sources=[], weak_evidence=True)


def _evidence_line(source: SourceEvidence) -> str:
    return (
        f"- {source.source_id}: {_safe_source_file(source.source_file)}, {source.page_or_sheet}, "
        f"chunk {source.chunk_id}, score {source.score:.3f}. Excerpt: {source.excerpt}"
    )


def _safe_source_file(source_file: str) -> str:
    if PureWindowsPath(source_file).is_absolute() or PurePosixPath(source_file).is_absolute():
        return source_file.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return source_file
