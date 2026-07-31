from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagAnswer, RagDraft, SourceEvidence


def format_answer(draft: RagDraft, sources: list[SourceEvidence]) -> RagAnswer:
    source_by_id = {source.source_id: source for source in sources}
    draft.validate(set(source_by_id))
    cited_sources = [source_by_id[source_id] for source_id in draft.cited_source_ids]
    evidence_lines = [_evidence_line(source) for source in cited_sources]
    if not evidence_lines:
        evidence_lines = ["- No retrieved source was cited."]
    checks = [f"{index}. {check}" for index, check in enumerate(draft.recommended_next_checks, start=1)]
    text = (
        f"Answer:\n{draft.answer.strip()}\n\n"
        f"Why this matters:\n{draft.why_this_matters.strip()}\n\n"
        "Evidence from retrieved sources:\n"
        + "\n".join(evidence_lines)
        + "\n\nRecommended next checks:\n"
        + "\n".join(checks)
        + f"\n\nLimitations:\n{draft.limitations.strip()}"
    )
    return RagAnswer(text=text, sources=cited_sources, weak_evidence=False)


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


def _evidence_line(source: SourceEvidence) -> str:
    return (
        f"- {source.source_id}: {_safe_source_file(source.source_file)}, {source.page_or_sheet}, "
        f"chunk {source.chunk_id}, score {source.score:.3f}. Excerpt: {source.excerpt}"
    )


def _safe_source_file(source_file: str) -> str:
    if PureWindowsPath(source_file).is_absolute() or PurePosixPath(source_file).is_absolute():
        return source_file.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return source_file