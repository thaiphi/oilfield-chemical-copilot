"""Run privacy-safe public grounded-answer evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oilfield_chemical_copilot.evaluation.answers import (  # noqa: E402
    GeneratedAnswer,
    evaluate_cases,
    load_answer_evaluation_cases,
    write_report,
)
from oilfield_chemical_copilot.evaluation.judge import AnswerJudge  # noqa: E402


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