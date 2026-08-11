from __future__ import annotations

from dataclasses import asdict
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from production_hardening.holdout_contract import HoldoutContract, RequiredPair, SealedCase


def _contract() -> HoldoutContract:
    return HoldoutContract(
        ("S01", "S02", "S03"), 12, 6, 6,
        (
            RequiredPair("allow", "general_review", 18),
            RequiredPair("abstain", "site_specific_determination", 6),
            RequiredPair("abstain", "field_ready_prescription", 6),
            RequiredPair("abstain", "complete_input_substitution", 6),
        ),
    )


def _cases() -> tuple[SealedCase, ...]:
    categories = ("site_specific_determination", "field_ready_prescription", "complete_input_substitution")
    cases: list[SealedCase] = []
    for stratum_number in range(1, 4):
        stratum = f"S{stratum_number:02d}"
        for number in range(6):
            cases.append(SealedCase(f"T{len(cases) + 1:02d}", stratum, f"toy-{len(cases) + 1}", "allow", "general_review", "author", "reviewer", True))
        for number in range(6):
            category = categories[(stratum_number * 6 + number) % 3]
            cases.append(SealedCase(f"T{len(cases) + 1:02d}", stratum, f"toy-{len(cases) + 1}", "abstain", category, "author", "reviewer", True))
    return tuple(cases)


def _write_seal(directory: Path, cases: tuple[SealedCase, ...]) -> tuple[Path, Path]:
    sealed_path = directory / "sealed.jsonl"
    payload = "".join(json.dumps(asdict(case), sort_keys=True, separators=(",", ":")) + "\n" for case in cases).encode()
    sealed_path.write_bytes(payload)
    digest_path = directory / "sealed.sha256"
    digest_path.write_text(hashlib.sha256(payload).hexdigest(), encoding="ascii")
    return sealed_path, digest_path


def _write_contract(directory: Path) -> Path:
    path = directory / "contract.json"
    path.write_text(json.dumps(asdict(_contract())), encoding="utf-8")
    return path


class _Decision:
    def __init__(self, action: str, category: str) -> None:
        self.action = action
        self.category = category


class _CountingDecision:
    def __init__(self, action: str, category: str) -> None:
        self._action = action
        self._category = category
        self.action_reads = 0
        self.category_reads = 0

    @property
    def action(self) -> str:
        self.action_reads += 1
        return self._action

    @property
    def category(self) -> str:
        self.category_reads += 1
        return self._category


class _Spy:
    def __init__(self, cases: tuple[SealedCase, ...], fail_at: int | None = None) -> None:
        self.cases = cases
        self.fail_at = fail_at
        self.calls: list[str] = []

    def __call__(self, question: str) -> _Decision:
        self.calls.append(question)
        if self.fail_at == len(self.calls):
            raise RuntimeError("secret exception detail")
        case = next(item for item in self.cases if item.question == question)
        return _Decision(case.expected_action, case.expected_category)


def _approval(directory: Path, sealed: Path, evaluator_source: Path) -> Path:
    from production_hardening.offline_evaluator import locate_policy_source

    path = directory / "approval.json"
    path.write_text(json.dumps({
        "scope": "holdout-36-one-shot", "approved": True,
        "holdout_sha256": hashlib.sha256(sealed.read_bytes()).hexdigest(),
        "policy_source_sha256": hashlib.sha256(locate_policy_source().read_bytes()).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(evaluator_source.read_bytes()).hexdigest(),
        "nonce": "test-nonce",
    }), encoding="utf-8")
    return path


def test_score_requires_all_exact_matches_for_strict_pass() -> None:
    from production_hardening.offline_evaluator import InMemoryDecision, score

    cases = _cases()
    decisions = tuple(InMemoryDecision(case.case_id, case.expected_action, case.expected_category) for case in cases)
    summary = score(cases, decisions, _contract())
    assert summary == summary.__class__(36, 36, 36, 0, 0, 3, 0, True)


@pytest.mark.parametrize(("action", "category", "false_allows", "false_abstains"), [
    ("abstain", "general_review", 0, 1),
    ("allow", "site_specific_determination", 1, 0),
])
def test_score_counts_wrong_actions_and_fails_affected_stratum(action: str, category: str, false_allows: int, false_abstains: int) -> None:
    from production_hardening.offline_evaluator import InMemoryDecision, score

    cases = _cases()
    decisions = [InMemoryDecision(case.case_id, case.expected_action, case.expected_category) for case in cases]
    target = 0 if action == "abstain" else 6
    decisions[target] = InMemoryDecision(cases[target].case_id, action, category)
    summary = score(cases, decisions, _contract())
    assert (summary.false_allows, summary.false_abstains, summary.stratum_failures, summary.strict_pass) == (false_allows, false_abstains, 1, False)


def test_score_category_miss_fails_its_stratum() -> None:
    from production_hardening.offline_evaluator import InMemoryDecision, score

    cases = _cases()
    decisions = [InMemoryDecision(case.case_id, case.expected_action, case.expected_category) for case in cases]
    decisions[0] = InMemoryDecision(cases[0].case_id, "allow", "site_specific_determination")
    summary = score(cases, decisions, _contract())
    assert (summary.action_exact, summary.category_exact, summary.stratum_failures, summary.strict_pass) == (36, 35, 1, False)


def test_preflight_is_classifier_free_and_does_not_consume_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from production_hardening import offline_evaluator

    sealed, digest = _write_seal(tmp_path, _cases())
    state = tmp_path / "state.json"
    monkeypatch.setattr(offline_evaluator, "load_frozen_classifier", lambda: (_ for _ in ()).throw(AssertionError("loader called")))
    sys.modules.pop("oilfield_chemical_copilot.evaluation.abstention_policy", None)
    summary = offline_evaluator.preflight(sealed, digest, None, state, _contract())
    assert (summary.approval_present, summary.classifier_calls, state.exists(), "oilfield_chemical_copilot.evaluation.abstention_policy" in sys.modules) == (False, 0, False, False)
    assert (summary.holdout_sha256, summary.policy_source_sha256, summary.evaluator_source_sha256) == (
        hashlib.sha256(sealed.read_bytes()).hexdigest(),
        hashlib.sha256(offline_evaluator.locate_policy_source().read_bytes()).hexdigest(),
        hashlib.sha256(Path(offline_evaluator.__file__).read_bytes()).hexdigest(),
    )


def test_preflight_reports_a_valid_but_mismatched_approval_digest(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    sealed, digest = _write_seal(tmp_path, _cases())
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    data = json.loads(approval.read_text(encoding="utf-8"))
    data["policy_source_sha256"] = "0" * 64
    approval.write_text(json.dumps(data), encoding="utf-8")
    summary = offline_evaluator.preflight(sealed, digest, approval, tmp_path / "state.json", _contract())
    assert (summary.approval_present, summary.approval_digest_matches, summary.attempt_available) == (True, False, True)


def test_evaluate_once_consumes_approval_before_exactly_36_calls(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    state, report = tmp_path / "state.json", tmp_path / "report.json"
    spy = _Spy(cases)
    summary = offline_evaluator.evaluate_once(sealed, digest, approval, state, report, _write_contract(tmp_path), spy)
    assert (summary.strict_pass, len(spy.calls), tuple(spy.calls), state.exists()) == (True, 36, tuple(case.question for case in cases), True)
    serialized = state.read_text() + report.read_text()
    assert all(case.question not in serialized for case in cases)
    with pytest.raises(offline_evaluator.EvaluatorError, match="ATTEMPT_UNAVAILABLE"):
        offline_evaluator.evaluate_once(sealed, digest, approval, state, report, _write_contract(tmp_path), spy)
    assert len(spy.calls) == 36


def test_lock_exists_before_the_first_classifier_call(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    state = tmp_path / "state.json"
    observed_lock: list[bool] = []

    def classifier(question: str) -> _Decision:
        observed_lock.append(state.exists())
        case = next(item for item in cases if item.question == question)
        return _Decision(case.expected_action, case.expected_category)

    offline_evaluator.evaluate_once(sealed, digest, approval, state, tmp_path / "report.json", _write_contract(tmp_path), classifier)
    assert observed_lock == [True] * 36


def test_classifier_decision_attributes_are_read_once(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    decisions: list[_CountingDecision] = []

    def classifier(question: str) -> _CountingDecision:
        case = next(item for item in cases if item.question == question)
        decision = _CountingDecision(case.expected_action, case.expected_category)
        decisions.append(decision)
        return decision

    offline_evaluator.evaluate_once(sealed, digest, approval, tmp_path / "state.json", tmp_path / "report.json", _write_contract(tmp_path), classifier)
    assert {(item.action_reads, item.category_reads) for item in decisions} == {(1, 1)}


def test_successful_evaluation_writes_strict_pass_as_a_boolean_gate(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    report = tmp_path / "report.json"
    offline_evaluator.evaluate_once(
        sealed, digest, approval, tmp_path / "state.json", report,
        _write_contract(tmp_path), _Spy(cases),
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert (payload["task"], payload["status"]) == (4, "pass")
    assert payload["gates"]["strict_pass"] is True
    assert "strict_pass" not in payload["counts"]
    assert all(type(value) is int for value in payload["counts"].values())
    assert payload["counts"]["classifier_calls"] == 36
    assert payload["gates"] == {
        "approved_digests_bound": True,
        "evaluator_digest_verified": True,
        "holdout_digest_verified": True,
        "policy_digest_verified": True,
        "strict_pass": True,
    }


def test_evaluate_once_retains_lock_after_classifier_failure(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    state, report = tmp_path / "state.json", tmp_path / "report.json"
    spy = _Spy(cases, fail_at=17)
    with pytest.raises(offline_evaluator.EvaluatorError, match="EVALUATION_FAILURE"):
        offline_evaluator.evaluate_once(sealed, digest, approval, state, report, _write_contract(tmp_path), spy)
    assert (len(spy.calls), state.exists()) == (17, True)


def test_attribute_decisions_are_required_and_fail_sanitized(tmp_path: Path) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    with pytest.raises(offline_evaluator.EvaluatorError, match="INVALID_POLICY_DECISION"):
        offline_evaluator.evaluate_once(sealed, digest, approval, tmp_path / "state.json", tmp_path / "report.json", _write_contract(tmp_path), lambda _: {"action": "allow", "category": "general_review"})


@pytest.mark.parametrize("field", ["holdout_sha256", "policy_source_sha256", "evaluator_source_sha256"])
def test_each_approval_digest_mismatch_is_rejected_before_classifier_calls(tmp_path: Path, field: str) -> None:
    from production_hardening import offline_evaluator

    cases = _cases()
    sealed, digest = _write_seal(tmp_path, cases)
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    data = json.loads(approval.read_text(encoding="utf-8"))
    data[field] = "0" * 64
    approval.write_text(json.dumps(data), encoding="utf-8")
    spy = _Spy(cases)
    with pytest.raises(offline_evaluator.EvaluatorError, match="APPROVAL_DIGEST_MISMATCH"):
        offline_evaluator.evaluate_once(sealed, digest, approval, tmp_path / "state.json", tmp_path / "report.json", _write_contract(tmp_path), spy)
    assert (spy.calls, (tmp_path / "state.json").exists()) == ([], False)


@pytest.mark.parametrize("field,value", [
    ("scope", 1), ("approved", "true"), ("holdout_sha256", "A" * 64),
    ("policy_source_sha256", 1), ("evaluator_source_sha256", "0" * 63), ("nonce", "   "),
])
def test_malformed_approval_values_are_sanitized(tmp_path: Path, field: str, value: object) -> None:
    from production_hardening import offline_evaluator

    sealed, digest = _write_seal(tmp_path, _cases())
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    data = json.loads(approval.read_text(encoding="utf-8"))
    data[field] = value
    approval.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(offline_evaluator.EvaluatorError, match="APPROVAL_INVALID"):
        offline_evaluator.preflight(sealed, digest, approval, tmp_path / "state.json", _contract())


def test_source_import_boundary_and_no_mapping_access() -> None:
    source = Path("production_hardening/offline_evaluator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    policy_imports = [node for node in imports if node.module == "oilfield_chemical_copilot.evaluation.abstention_policy"]
    assert all({alias.name for alias in node.names} == {"AbstentionPolicyDecision", "classify_claim_scope"} or {alias.name for alias in node.names} == {"AbstentionPolicyDecision"} for node in policy_imports)
    assert "decision[" not in source and "decision.get(" not in source and "dict(decision" not in source
    forbidden_modules = {"docker", "ollama", "openai", "requests", "httpx", "urllib.request", "socket", "subprocess", "importlib"}
    imports_by_name = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    from_modules = {node.module for node in imports if node.module}
    assert imports_by_name <= {"argparse", "contextlib", "hashlib", "hmac", "io", "json", "re", "sys"}
    assert from_modules <= {
        "__future__", "dataclasses", "pathlib", "typing",
        "production_hardening.aggregate_report", "production_hardening.holdout_contract",
        "oilfield_chemical_copilot.evaluation.abstention_policy",
    }
    assert not forbidden_modules.intersection(imports_by_name | from_modules)
    assert all(not module.startswith("oilfield_chemical_copilot") or module == "oilfield_chemical_copilot.evaluation.abstention_policy" for module in from_modules)
    assert not any(
        isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))
        for node in ast.walk(tree)
    )
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_frozen_classifier")
    type_guard = next(node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
    type_only_imports = [node for node in ast.walk(type_guard) if isinstance(node, ast.ImportFrom) and node.module == "oilfield_chemical_copilot.evaluation.abstention_policy"]
    lazy_imports = [node for node in ast.walk(loader) if isinstance(node, ast.ImportFrom) and node.module == "oilfield_chemical_copilot.evaluation.abstention_policy"]
    assert len(policy_imports) == 2
    assert len(type_only_imports) == len(lazy_imports) == 1
    assert [alias.name for alias in type_only_imports[0].names] == ["AbstentionPolicyDecision"]
    assert [alias.name for alias in lazy_imports[0].names] == ["AbstentionPolicyDecision", "classify_claim_scope"]
    assert {id(node) for node in policy_imports} == {id(type_only_imports[0]), id(lazy_imports[0])}


def test_locate_policy_source_resolves_the_frozen_source_file() -> None:
    from production_hardening.offline_evaluator import locate_policy_source

    assert locate_policy_source() == Path("src/oilfield_chemical_copilot/evaluation/abstention_policy.py")


def test_private_values_never_escape_sanitized_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from production_hardening import offline_evaluator

    cases = list(_cases())
    cases[0] = SealedCase("T01", "S01", "unique-question /private/sentinel https://example.invalid token=credential-sentinel label-sentinel", "allow", "general_review", "author", "reviewer", True)
    sealed, digest = _write_seal(tmp_path, tuple(cases))
    approval = _approval(tmp_path, sealed, Path(offline_evaluator.__file__))
    state, report = tmp_path / "state.json", tmp_path / "report.json"
    with pytest.raises(offline_evaluator.EvaluatorError) as error:
        offline_evaluator.evaluate_once(sealed, digest, approval, state, report, _write_contract(tmp_path), _Spy(tuple(cases), fail_at=1))
    captured = capsys.readouterr()
    exposed = str(error.value) + captured.out + captured.err + state.read_text() + report.read_text()
    assert all(value not in exposed for value in ("unique-question", "/private/sentinel", "https://example.invalid", "credential-sentinel", "label-sentinel", "secret exception detail"))


def test_preflight_cli_prints_only_aggregate_fields(tmp_path: Path) -> None:
    sealed, digest = _write_seal(tmp_path, _cases())
    command = [
        sys.executable, "-m", "production_hardening.offline_evaluator", "--preflight",
        "--sealed-path", str(sealed), "--digest-path", str(digest),
        "--state-path", str(tmp_path / "state.json"), "--contract-path", str(_write_contract(tmp_path)),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    from production_hardening import offline_evaluator

    assert result.stdout.strip() == (
        "status=pass case_count=36 seal_valid=true "
        f"holdout_sha256={hashlib.sha256(sealed.read_bytes()).hexdigest()} "
        f"policy_source_sha256={hashlib.sha256(offline_evaluator.locate_policy_source().read_bytes()).hexdigest()} "
        f"evaluator_source_sha256={hashlib.sha256(Path(offline_evaluator.__file__).read_bytes()).hexdigest()} "
        "approval_present=false approval_digest_matches=false attempt_available=true classifier_calls=0"
    )
    assert str(tmp_path) not in result.stdout + result.stderr


def test_score_once_cli_returns_sanitized_approval_error(tmp_path: Path) -> None:
    sealed, digest = _write_seal(tmp_path, _cases())
    command = [
        sys.executable, "-m", "production_hardening.offline_evaluator", "--score-once",
        "--sealed-path", str(sealed), "--digest-path", str(digest), "--approval-path", str(tmp_path / "missing.json"),
        "--state-path", str(tmp_path / "state.json"), "--report-path", str(tmp_path / "report.json"), "--contract-path", str(_write_contract(tmp_path)),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert (result.returncode, result.stdout.strip(), result.stderr) == (1, "status=fail code=APPROVAL_REQUIRED", "")
    assert str(tmp_path) not in result.stdout


def test_score_once_cli_routes_documented_short_arguments_without_loading_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from production_hardening import offline_evaluator

    captured_paths: dict[str, Path] = {}

    def fake_evaluate_once(sealed: Path, digest: Path, approval: Path, state: Path, report: Path, contract: Path) -> offline_evaluator.ScoreSummary:
        captured_paths.update({"sealed": sealed, "digest": digest, "approval": approval, "state": state, "report": report, "contract": contract})
        return offline_evaluator.ScoreSummary(36, 36, 36, 0, 0, 3, 0, True)

    monkeypatch.setattr(offline_evaluator, "evaluate_once", fake_evaluate_once)
    result = offline_evaluator._main((
        "--score-once", "--sealed", "sealed", "--digest", "digest", "--approval", "approval",
        "--state", "state", "--report", "report", "--contract", "contract",
    ))
    assert result == 0
    assert captured_paths == {name: Path(name) for name in ("sealed", "digest", "approval", "state", "report", "contract")}
    assert capsys.readouterr().out.startswith("status=pass case_count=36")


def test_cli_invalid_arguments_return_only_a_sanitized_code(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "production_hardening.offline_evaluator", "--preflight", "--sealed-path", str(tmp_path / "secret.jsonl")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout.strip(), result.stderr) == (1, "status=fail code=CLI_ARGUMENTS_INVALID", "")


def test_cli_help_documents_arguments_without_user_values() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "production_hardening.offline_evaluator", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert all(flag in result.stdout for flag in ("--preflight", "--score-once", "--sealed-path", "--digest-path", "--approval-path", "--state-path", "--report-path", "--contract-path"))
    assert result.stderr == ""
