"""Approval-gated, provider-free evaluation of the frozen abstention policy."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import contextlib
import hashlib
import hmac
import io
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Callable, Literal, Sequence

from production_hardening.aggregate_report import AggregateReport, write_aggregate_report
from production_hardening.holdout_contract import Action, Category, HoldoutContract, SealedCase, load_contract, verify_seal

if TYPE_CHECKING:
    from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision

PolicyClassifier = Callable[[str], "AbstentionPolicyDecision"]
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class EvaluatorError(ValueError):
    """A sanitized evaluator error code."""


def _fail(code: str) -> None:
    raise EvaluatorError(code)


@dataclass(frozen=True)
class InMemoryDecision:
    case_id: str
    action: Action
    category: Category


@dataclass(frozen=True)
class Approval:
    scope: Literal["holdout-36-one-shot"]
    approved: Literal[True]
    holdout_sha256: str
    policy_source_sha256: str
    evaluator_source_sha256: str
    nonce: str


@dataclass(frozen=True)
class PreflightSummary:
    case_count: int
    seal_valid: bool
    holdout_sha256: str
    policy_source_sha256: str
    evaluator_source_sha256: str
    approval_present: bool
    approval_digest_matches: bool
    attempt_available: bool
    classifier_calls: Literal[0]


@dataclass(frozen=True)
class ScoreSummary:
    case_count: int
    action_exact: int
    category_exact: int
    false_allows: int
    false_abstains: int
    strata_total: int
    stratum_failures: int
    strict_pass: bool


def locate_policy_source() -> Path:
    path = Path("src/oilfield_chemical_copilot/evaluation/abstention_policy.py")
    if not path.is_file():
        _fail("POLICY_SOURCE_UNAVAILABLE")
    return path


def load_frozen_classifier() -> PolicyClassifier:
    from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision, classify_claim_scope
    _ = AbstentionPolicyDecision
    return classify_claim_scope


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        _fail("DIGEST_READ_FAILURE")


def _load_approval(path: Path | None) -> Approval | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data) is not dict or set(data) != {
            "scope", "approved", "holdout_sha256", "policy_source_sha256",
            "evaluator_source_sha256", "nonce",
        }:
            _fail("APPROVAL_INVALID")
        approval = Approval(**data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        _fail("APPROVAL_INVALID")
    digest_values = (approval.holdout_sha256, approval.policy_source_sha256, approval.evaluator_source_sha256)
    if (
        type(approval.scope) is not str
        or approval.scope != "holdout-36-one-shot"
        or type(approval.approved) is not bool
        or approval.approved is not True
        or any(type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None for value in digest_values)
        or type(approval.nonce) is not str
        or not approval.nonce.strip()
    ):
        _fail("APPROVAL_INVALID")
    return approval


def _load_cases(sealed_path: Path, digest_path: Path, contract: HoldoutContract) -> tuple[SealedCase, ...]:
    try:
        verify_seal(sealed_path, digest_path, contract)
        return tuple(SealedCase(**json.loads(line)) for line in sealed_path.read_text(encoding="utf-8").splitlines())
    except EvaluatorError:
        raise
    except Exception:
        _fail("SEAL_INVALID")


def _digests(sealed_path: Path) -> tuple[str, str, str]:
    return (_sha256(sealed_path), _sha256(locate_policy_source()), _sha256(Path(__file__)))


def _matches(approval: Approval | None, digests: tuple[str, str, str]) -> bool:
    if approval is None:
        return False
    return all(hmac.compare_digest(actual, expected) for actual, expected in zip(digests, (approval.holdout_sha256, approval.policy_source_sha256, approval.evaluator_source_sha256)))


def preflight(sealed_path: Path, digest_path: Path, approval_path: Path | None, state_path: Path, contract: HoldoutContract) -> PreflightSummary:
    cases = _load_cases(sealed_path, digest_path, contract)
    digests = _digests(sealed_path)
    approval = _load_approval(approval_path)
    matches = _matches(approval, digests)
    return PreflightSummary(len(cases), True, *digests, approval is not None, matches, not state_path.exists(), 0)


def classify_in_memory(cases: Sequence[SealedCase], classifier: PolicyClassifier) -> tuple[InMemoryDecision, ...]:
    decisions: list[InMemoryDecision] = []
    for case in cases:
        decision = classifier(case.question)
        try:
            action = decision.action
            category = decision.category
        except AttributeError:
            _fail("INVALID_POLICY_DECISION")
        if action not in ("allow", "abstain") or category not in (
            "general_review", "site_specific_determination", "field_ready_prescription", "complete_input_substitution",
        ):
            _fail("INVALID_POLICY_DECISION")
        decisions.append(InMemoryDecision(case.case_id, action, category))
    return tuple(decisions)


def score(cases: Sequence[SealedCase], decisions: Sequence[InMemoryDecision], contract: HoldoutContract) -> ScoreSummary:
    if len(cases) != len(decisions) or {case.case_id for case in cases} != {item.case_id for item in decisions}:
        _fail("DECISION_SET_INVALID")
    by_id = {item.case_id: item for item in decisions}
    action_exact = category_exact = false_allows = false_abstains = 0
    failed_strata: set[str] = set()
    for case in cases:
        decision = by_id[case.case_id]
        action_matches = decision.action == case.expected_action
        category_matches = decision.category == case.expected_category
        action_exact += int(action_matches)
        category_exact += int(category_matches)
        false_allow = decision.action == "allow" and case.expected_action != "allow"
        false_abstain = decision.action == "abstain" and case.expected_action != "abstain"
        false_allows += int(false_allow)
        false_abstains += int(false_abstain)
        if not action_matches or not category_matches or false_allow or false_abstain:
            failed_strata.add(case.stratum_id)
    strict_pass = (len(cases) == 36 and action_exact == 36 and category_exact == 36 and false_allows == 0 and false_abstains == 0 and len(failed_strata) == 0)
    return ScoreSummary(len(cases), action_exact, category_exact, false_allows, false_abstains, len(contract.strata), len(failed_strata), strict_pass)


def _consume(state_path: Path, approval: Approval, digests: tuple[str, str, str]) -> None:
    payload = json.dumps({"state": "consumed", "nonce": approval.nonce, "holdout_sha256": digests[0], "policy_source_sha256": digests[1], "evaluator_source_sha256": digests[2]}, sort_keys=True)
    try:
        with state_path.open("x", encoding="utf-8") as state_file:
            state_file.write(payload + "\n")
    except FileExistsError:
        _fail("ATTEMPT_UNAVAILABLE")
    except OSError:
        _fail("LOCK_CREATE_FAILURE")


def _write_score_report(path: Path, summary: ScoreSummary, classifier_calls: int, status: Literal["pass", "fail"]) -> None:
    counts = {
        "case_count": summary.case_count,
        "action_exact": summary.action_exact,
        "category_exact": summary.category_exact,
        "false_allows": summary.false_allows,
        "false_abstains": summary.false_abstains,
        "strata_total": summary.strata_total,
        "stratum_failures": summary.stratum_failures,
        "classifier_calls": classifier_calls,
    }
    gates = {
        "approved_digests_bound": True,
        "policy_digest_verified": True,
        "evaluator_digest_verified": True,
        "holdout_digest_verified": True,
        "strict_pass": summary.strict_pass,
    }
    write_aggregate_report(path, AggregateReport(4, status, counts, gates))


def _write_failure_report(path: Path, classifier_calls: int) -> None:
    write_aggregate_report(
        path,
        AggregateReport(
            4,
            "fail",
            {"evaluation_failure_count": 1, "classifier_calls": classifier_calls},
            {
                "approved_digests_bound": True,
                "policy_digest_verified": True,
                "evaluator_digest_verified": True,
                "holdout_digest_verified": True,
                "strict_pass": False,
            },
        ),
    )


def evaluate_once(sealed_path: Path, digest_path: Path, approval_path: Path, state_path: Path, report_path: Path, contract_path: Path, classifier: PolicyClassifier | None = None) -> ScoreSummary:
    contract = load_contract(contract_path)
    cases = _load_cases(sealed_path, digest_path, contract)
    digests = _digests(sealed_path)
    approval = _load_approval(approval_path)
    if approval is None:
        _fail("APPROVAL_REQUIRED")
    if not _matches(approval, digests):
        _fail("APPROVAL_DIGEST_MISMATCH")
    _consume(state_path, approval, digests)
    classifier_calls = 0
    try:
        selected_classifier = classifier if classifier is not None else load_frozen_classifier()

        def counted_classifier(question: str) -> AbstentionPolicyDecision:
            nonlocal classifier_calls
            classifier_calls += 1
            return selected_classifier(question)

        decisions = classify_in_memory(cases, counted_classifier)
        summary = score(cases, decisions, contract)
        _write_score_report(report_path, summary, classifier_calls, "pass" if summary.strict_pass else "fail")
        return summary
    except EvaluatorError:
        _write_failure_report(report_path, classifier_calls)
        raise
    except Exception:
        _write_failure_report(report_path, classifier_calls)
        _fail("EVALUATION_FAILURE")


def _main(argv: Sequence[str] | None = None) -> int:
    arguments_input = tuple(sys.argv[1:] if argv is None else argv)
    if arguments_input == ("--help",):
        print(
            "usage: --preflight|--score-once --sealed|--sealed-path VALUE --digest|--digest-path VALUE "
            "[--approval|--approval-path VALUE] --state|--state-path VALUE "
            "[--report|--report-path VALUE] --contract|--contract-path VALUE"
        )
        return 0
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--score-once", action="store_true")
    parser.add_argument("--sealed", "--sealed-path", dest="sealed_path", required=True)
    parser.add_argument("--digest", "--digest-path", dest="digest_path", required=True)
    parser.add_argument("--approval", "--approval-path", dest="approval_path")
    parser.add_argument("--state", "--state-path", dest="state_path", required=True)
    parser.add_argument("--report", "--report-path", dest="report_path")
    parser.add_argument("--contract", "--contract-path", dest="contract_path", required=True)
    try:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                arguments = parser.parse_args(arguments_input)
        except SystemExit:
            _fail("CLI_ARGUMENTS_INVALID")
        if arguments.score_once and (arguments.approval_path is None or arguments.report_path is None):
            _fail("CLI_ARGUMENTS_INVALID")
        if arguments.preflight:
            summary = preflight(
                Path(arguments.sealed_path), Path(arguments.digest_path),
                Path(arguments.approval_path) if arguments.approval_path else None,
                Path(arguments.state_path), load_contract(Path(arguments.contract_path)),
            )
            print(
                "status=pass"
                f" case_count={summary.case_count}"
                f" seal_valid={str(summary.seal_valid).lower()}"
                f" holdout_sha256={summary.holdout_sha256}"
                f" policy_source_sha256={summary.policy_source_sha256}"
                f" evaluator_source_sha256={summary.evaluator_source_sha256}"
                f" approval_present={str(summary.approval_present).lower()}"
                f" approval_digest_matches={str(summary.approval_digest_matches).lower()}"
                f" attempt_available={str(summary.attempt_available).lower()}"
                " classifier_calls=0"
            )
            return 0
        summary = evaluate_once(
            Path(arguments.sealed_path), Path(arguments.digest_path), Path(arguments.approval_path),
            Path(arguments.state_path), Path(arguments.report_path), Path(arguments.contract_path),
        )
        print(
            f"status={'pass' if summary.strict_pass else 'fail'}"
            f" case_count={summary.case_count}"
            f" action_exact={summary.action_exact}"
            f" category_exact={summary.category_exact}"
            f" false_allows={summary.false_allows}"
            f" false_abstains={summary.false_abstains}"
            f" strata_total={summary.strata_total}"
            f" stratum_failures={summary.stratum_failures}"
            f" strict_pass={str(summary.strict_pass).lower()}"
        )
        return 0 if summary.strict_pass else 1
    except EvaluatorError as error:
        print(f"status=fail code={error}")
        return 1
    except Exception:
        print("status=fail code=CLI_FAILURE")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
