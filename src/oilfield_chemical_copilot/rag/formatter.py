from __future__ import annotations

import re
from dataclasses import replace
from pathlib import PurePosixPath, PureWindowsPath

from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagAnswer, RagDraft, SourceEvidence


_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "and",
        "are",
        "before",
        "can",
        "check",
        "for",
        "from",
        "general",
        "how",
        "into",
        "only",
        "public",
        "review",
        "should",
        "the",
        "this",
        "use",
        "what",
        "which",
        "with",
    }
)


def format_answer(draft: RagDraft, sources: list[SourceEvidence], *, question: str) -> RagAnswer:
    selected_sources = select_supported_sources(question=question, draft=draft, sources=sources)
    selected_draft = replace(draft, cited_source_ids=[source.source_id for source in selected_sources])
    source_by_id = {source.source_id: source for source in sources}
    selected_draft.validate(set(source_by_id))
    cited_sources = [source_by_id[source_id] for source_id in selected_draft.cited_source_ids]
    evidence_lines = [_evidence_line(source) for source in cited_sources]
    if not evidence_lines:
        evidence_lines = ["- No retrieved source was cited."]
    checks = [f"{index}. {check}" for index, check in enumerate(draft.recommended_next_checks, start=1)]
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
    question_terms = _terms(question)
    answer_terms = _terms(
        " ".join(
            (
                draft.answer,
                draft.why_this_matters,
                *draft.recommended_next_checks,
                draft.limitations,
            )
        )
    )
    candidates: list[tuple[int, float, str, SourceEvidence]] = []
    for source in sources:
        source_terms = _terms(f"{source.source_file} {source.topic} {source.excerpt}")
        question_anchor_overlap = question_terms & _source_anchor_terms(source)
        answer_overlap = answer_terms & source_terms
        if not question_anchor_overlap or not answer_overlap:
            continue
        support_score = 3 * len(question_anchor_overlap) + 2 * len(answer_overlap)
        candidates.append((support_score, source.score, source.source_id, source))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [candidates[0][3]]


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) >= 3 and term not in _STOP_WORDS
    }


def _source_anchor_terms(source: SourceEvidence) -> set[str]:
    source_name = PurePosixPath(source.source_file.replace("\\", "/")).stem
    return _terms(f"{source_name} {source.topic}")


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
