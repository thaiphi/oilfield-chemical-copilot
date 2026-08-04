"""Run privacy-safe public grounded-answer evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oilfield_chemical_copilot.evaluation.answers import (  # noqa: E402
    AnswerEvaluationCase,
    DeterministicAnswerResult,
    evaluate_answer,
    load_answer_evaluation_cases,
)
from oilfield_chemical_copilot.evaluation.judge import (  # noqa: E402
    AnswerJudge,
    JudgeAvailable,
    JudgeResult,
)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Runtime-only answer material supplied to the evaluator."""

    question_id: str
    answer: str
    evidence: str
    cited_evidence_ids: tuple[str, ...]
    abstained: bool


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """Report-safe result that excludes answer and evidence text."""

    question_id: str
    deterministic: DeterministicAnswerResult
    judge: JudgeResult


def evaluate_cases(
    cases: list[AnswerEvaluationCase],
    answers: list[GeneratedAnswer],
    judge: AnswerJudge,
) -> list[AnswerEvaluationResult]:
    answers_by_id = {answer.question_id: answer for answer in answers}
    if len(answers_by_id) != len(answers):
        raise ValueError("generated answers must not contain duplicate question IDs")
    expected_ids = {case.question_id for case in cases}
    if set(answers_by_id) != expected_ids:
        raise ValueError("generated answer IDs must exactly match evaluation case IDs")

    results: list[AnswerEvaluationResult] = []
    for case in cases:
        answer = answers_by_id[case.question_id]
        results.append(
            AnswerEvaluationResult(
                question_id=case.question_id,
                deterministic=evaluate_answer(
                    case,
                    cited_evidence_ids=answer.cited_evidence_ids,
                    abstained=answer.abstained,
                ),
                judge=judge.judge(case, answer=answer.answer, evidence=answer.evidence),
            )
        )
    return results


def write_report(results: list[AnswerEvaluationResult], output_dir: Path) -> tuple[Path, Path]:
    report = _report_summary(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "answer_eval.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "answer_eval.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _report_summary(results: list[AnswerEvaluationResult]) -> dict[str, object]:
    ordered_results = sorted(results, key=lambda result: result.question_id)
    citation_counts = Counter(result.deterministic.citation_status for result in ordered_results)
    abstention_counts = Counter(result.deterministic.abstention_status for result in ordered_results)
    available = [result.judge for result in ordered_results if isinstance(result.judge, JudgeAvailable)]
    judge_status = Counter(result.judge.status for result in ordered_results)
    score_fields = (
        "groundedness",
        "relevance",
        "limitation_awareness",
        "operational_certainty",
    )
    scores = {
        field: mean(getattr(result.scores, field) for result in available)
        for field in score_fields
    } if available else {}
    return {
        "cases": [result.question_id for result in ordered_results],
        "deterministic": {
            "citations": dict(sorted(citation_counts.items())),
            "abstention": dict(sorted(abstention_counts.items())),
        },
        "judge": {
            "status": dict(sorted(judge_status.items())),
            "providers": sorted({result.judge.provider for result in ordered_results}),
            "models": sorted({result.judge.model for result in ordered_results}),
            "scores": scores,
        },
    }


def _markdown_report(report: dict[str, object]) -> str:
    deterministic = report["deterministic"]
    judge = report["judge"]
    assert isinstance(deterministic, dict)
    assert isinstance(judge, dict)
    return "\n".join(
        [
            "# Grounded answer evaluation",
            "",
            f"Cases: {len(report['cases'])}",
            "",
            "| deterministic check | pass | fail |",
            "| --- | ---: | ---: |",
            "| citations | {pass_count} | {fail_count} |".format(
                pass_count=deterministic["citations"].get("pass", 0),
                fail_count=deterministic["citations"].get("fail", 0),
            ),
            "| abstention | {pass_count} | {fail_count} |".format(
                pass_count=deterministic["abstention"].get("pass", 0),
                fail_count=deterministic["abstention"].get("fail", 0),
            ),
            "",
            "## Judge",
            "",
            f"Available: {judge['status'].get('available', 0)}",
            f"Unavailable: {judge['status'].get('unavailable', 0)}",
            "",
        ]
    )


def _load_generated_answers(path: Path) -> list[GeneratedAnswer]:
    fields = {"question_id", "answer", "evidence", "cited_evidence_ids", "abstained"}
    answers: list[GeneratedAnswer] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("generated answer records must be valid JSON objects") from error
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("generated answer records must have exactly the expected fields")
        if not all(isinstance(record[field], str) for field in ("question_id", "answer", "evidence")):
            raise ValueError("generated answer text fields must be strings")
        cited_ids = record["cited_evidence_ids"]
        if not isinstance(cited_ids, list) or not all(isinstance(value, str) for value in cited_ids):
            raise ValueError("cited_evidence_ids must be a list of strings")
        if not isinstance(record["abstained"], bool):
            raise ValueError("abstained must be boolean")
        answers.append(
            GeneratedAnswer(
                question_id=record["question_id"],
                answer=record["answer"],
                evidence=record["evidence"],
                cited_evidence_ids=tuple(cited_ids),
                abstained=record["abstained"],
            )
        )
    return answers


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate public grounded RAG answers.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/public_answer_evaluation.jsonl"))
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/evaluation/answers"))
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    results = evaluate_cases(
        load_answer_evaluation_cases(args.dataset),
        _load_generated_answers(args.answers),
        AnswerJudge(),
    )
    json_path, markdown_path = write_report(results, args.output_dir)
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
