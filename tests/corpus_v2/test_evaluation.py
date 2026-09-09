"""Synthetic contracts only: no evaluation, services, or private corpus access."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from oilfield_chemical_copilot.corpus_v2 import evaluation as ev


class Publication:
    """Authenticated publication transport double with immutable named trees."""
    def __init__(self):
        self.trees = {}
        self.members = {}
        self.name = ".evaluation.synthetic.tmp"

    def ensure_no_staging(self, *args):
        if self.members:
            raise ValueError

    def final_exists(self, name):
        return name in self.trees

    def create_staging(self, *args):
        return self

    def mkdir(self, name):
        pass

    def write_exclusive(self, name, content):
        assert name not in self.members
        self.members[name] = content

    def sync_directory(self, name):
        pass

    def sync_root(self):
        pass

    def publish_no_replace(self, staging, name):
        if name in self.trees:
            raise ValueError
        self.trees[name] = dict(self.members)
        self.members.clear()

    def sync_parent(self):
        pass

    def read_exact_tree(self, name, layout):
        tree = self.members if name == self.name else self.trees[name]
        if set(tree) != {f"{folder}/{item}" for folder, items in layout.items() for item in items}:
            raise ValueError
        return dict(tree)


def specification():
    return dict(release_id="synthetic-v2", index_sha256="a" * 64,
                development_ids=["dev-new"], unseen_ids=["unseen-new"],
                historical_ids={"E1a-3": ["old-3"], "E1a-4": ["old-4"]},
                scoring_protocol="fixed-protocol-v1", regression_families=["chemistry"],
                minimum_accuracy=0.8, minimum_no_answer_safety=1.0,
                no_answer_ids=["unseen-new"], no_answer_expected_count=1,
                no_answer_provenance_sha256="c" * 64,
                no_answer_baseline_sha256="d" * 64, no_answer_baseline_safety=1.0,
                no_answer_nonregression="no-decrease",
                maximum_latency_ms=1000, latency_measurement="end-to-end-p95-ms",
                no_tuning=True, steward_approved=True)


def aggregates():
    return dict(accuracy=0.9, no_answer_safety=1.0, latency_ms=900,
                development_count=1, unseen_count=1, no_answer_count=1,
                no_answer_cohort_sha256=sha256(ev._json(["unseen-new"])).hexdigest(),
                no_answer_provenance_sha256="c" * 64, no_answer_baseline_sha256="d" * 64,
                regression_families={"chemistry": 0.9})


def frozen():
    publication = Publication()
    digest = ev.freeze_evaluation_spec(specification(), publication=publication,
        verify_safety=lambda spec: spec == specification(),
        verify_historical=lambda cohorts: cohorts == {"E1a-3": ["old-3"], "E1a-4": ["old-4"]})
    return publication, digest


@pytest.mark.parametrize("field,value", [
    ("unseen_ids", ["dev-new"]), ("development_ids", ["old-3"]),
    ("unseen_ids", ["old-4"]), ("unseen_ids", []), ("no_tuning", False),
    ("steward_approved", False), ("minimum_accuracy", float("nan")),
    ("maximum_latency_ms", float("inf")), ("minimum_accuracy", True),
    ("minimum_no_answer_safety", -1), ("scoring_protocol", ""),
    ("latency_measurement", ""), ("regression_families", []),
])
def test_spec_rejects_unfrozen_or_reused_cohorts(field, value):
    spec = specification()
    spec[field] = value
    with pytest.raises(ev.CorpusV2EvaluationError, match="C2_EVALUATION_SPEC_INVALID"):
        ev.freeze_evaluation_spec(spec, publication=Publication(), verify_historical=lambda _: True,
                                  verify_safety=lambda _: True)


def test_spec_is_immutable_and_no_result_can_precede_it():
    pub, digest = frozen()
    changed = specification()
    changed["minimum_accuracy"] = 0.1
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.freeze_evaluation_spec(changed, publication=pub, verify_historical=lambda _: True,
                                  verify_safety=lambda _: True)
    assert ev.verify_evaluation_spec(publication=pub, release_id="synthetic-v2",
                                     expected_sha256=digest)["minimum_accuracy"] == 0.8
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.publish_evaluation_result(aggregates(), publication=Publication(),
            release_id="synthetic-v2", spec_sha256=digest, verify_index=lambda _: True)


def test_index_verification_precedes_run_permission_and_result():
    pub, digest = frozen()
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.authorize_evaluation(publication=pub, release_id="synthetic-v2",
            spec_sha256=digest, verify_index=lambda _: False)
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.publish_evaluation_result(aggregates(), publication=pub,
            release_id="synthetic-v2", spec_sha256=digest, verify_index=lambda _: False)
    assert "synthetic-v2-result" not in pub.trees


@pytest.mark.parametrize("field,value", [("source_text", "secret"), ("accuracy", 1.1),
    ("accuracy", float("nan")), ("no_answer_safety", None), ("no_answer_count", 0),
    ("latency_ms", -1), ("unseen_count", True), ("regression_families", {"other": 1.0})])
def test_result_rejects_source_fields_and_invalid_metrics(field, value):
    pub, digest = frozen()
    result = aggregates()
    result[field] = value
    with pytest.raises(ev.CorpusV2EvaluationError, match="C2_EVALUATION_RESULT_INVALID"):
        ev.publish_evaluation_result(result, publication=pub, release_id="synthetic-v2",
                                     spec_sha256=digest, verify_index=lambda _: True)


def evaluated(result=None):
    pub, spec = frozen()
    digest = ev.publish_evaluation_result(result or aggregates(), publication=pub,
        release_id="synthetic-v2", spec_sha256=spec, verify_index=lambda _: True)
    return pub, spec, digest


def gates(result_digest):
    return {key: (result_digest if key == "evaluation_result" else
                  "a" * 64 if key == "index_contract" else "b" * 64)
            for key in ("legacy_guard", "source_accounting", "critical_source_evidence",
                        "index_contract", "evaluation_result", "canary_plan", "rollback_rehearsal")}


def ready(pub, spec, result, **changes):
    kwargs = dict(publication=pub, release_id="synthetic-v2", spec_sha256=spec,
                  result_sha256=result, sealed_gates=gates(result),
                  verify_gate=lambda *args: True, canary_passed=True, rollback_passed=True)
    kwargs.update(changes)
    return ev.record_promotion_recommendation(**kwargs)


@pytest.mark.parametrize("change", ["missing", "unsealed", "canary", "rollback", "wrong-result",
                                    "wrong-index"])
def test_every_sealed_gate_and_successful_rehearsal_is_required(change):
    pub, spec, result = evaluated()
    kwargs = {}
    if change == "missing":
        evidence = gates(result)
        del evidence["critical_source_evidence"]
        kwargs["sealed_gates"] = evidence
    elif change in ("wrong-result", "wrong-index"):
        evidence = gates(result)
        evidence["evaluation_result" if change == "wrong-result" else "index_contract"] = "c" * 64
        kwargs["sealed_gates"] = evidence
    elif change == "unsealed":
        kwargs["verify_gate"] = lambda *args: False
    else:
        kwargs[f"{change}_passed"] = False
    assert ready(pub, spec, result, **kwargs)["status"] == "BLOCKED"


def test_failed_threshold_blocks_and_results_are_aggregate_only():
    result = aggregates()
    result["no_answer_safety"] = 0.5
    pub, spec, digest = evaluated(result)
    assert ready(pub, spec, digest)["status"] == "BLOCKED"
    content = pub.trees["synthetic-v2-result"]["evidence/payload.bin"]
    assert "unseen-new" not in content.decode()
    assert json.loads(content)["aggregates"]["no_answer_safety"] == 0.5


def test_only_explicit_owner_action_promotes_once():
    pub, spec, result = evaluated()
    recommendation = ready(pub, spec, result)
    assert recommendation["status"] == "PROMOTION_READY"
    for owner, explicit in [(False, True), (True, False)]:
        with pytest.raises(ev.CorpusV2EvaluationError):
            ev.record_owner_promotion(publication=pub, release_id="synthetic-v2",
                recommendation_sha256=recommendation["sha256"], action_id="action-1",
                authenticate_owner=lambda: owner, explicit_action=explicit,
                verify_gate=lambda *args: True)
    assert ev.record_owner_promotion(publication=pub, release_id="synthetic-v2",
        recommendation_sha256=recommendation["sha256"], action_id="action-1",
        authenticate_owner=lambda: True, explicit_action=True,
        verify_gate=lambda *args: True) == "PROMOTED"
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.record_owner_promotion(publication=pub, release_id="synthetic-v2",
            recommendation_sha256=recommendation["sha256"], action_id="action-2",
            authenticate_owner=lambda: True, explicit_action=True, verify_gate=lambda *args: True)


def test_mutated_sealed_spec_is_rejected():
    pub, digest = frozen()
    tree = deepcopy(pub.trees["synthetic-v2-spec"])
    tree["evidence/payload.bin"] += b" "
    pub.trees["synthetic-v2-spec"] = tree
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.authorize_evaluation(publication=pub, release_id="synthetic-v2",
            spec_sha256=digest, verify_index=lambda _: True)


def test_preexisting_result_prevents_spec_freeze():
    pub = Publication()
    pub.trees["synthetic-v2-result"] = {}
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.freeze_evaluation_spec(specification(), publication=pub, verify_historical=lambda _: True,
                                  verify_safety=lambda _: True)


def test_unverified_historical_inventory_cannot_be_frozen():
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.freeze_evaluation_spec(specification(), publication=Publication())
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.freeze_evaluation_spec(specification(), publication=Publication(),
                                  verify_historical=lambda cohorts: False)


@pytest.mark.parametrize("change", ["extra", "gates", "index", "metrics", "unauthenticated"])
def test_owner_rejects_forged_canonical_recommendation(change):
    metrics = aggregates()
    if change == "metrics":
        metrics["accuracy"] = 0.1
    pub, spec, result = evaluated(metrics)
    payload = dict(release_id="synthetic-v2", spec_sha256=spec, result_sha256=result,
                   index_sha256="a" * 64, sealed_gates=gates(result), status="PROMOTION_READY")
    if change == "extra":
        payload["unexpected"] = True
    elif change == "gates":
        payload["sealed_gates"] = {}
    elif change == "index":
        payload["index_sha256"] = "f" * 64
    digest = ev._publish(pub, "synthetic-v2-recommendation", payload)
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.record_owner_promotion(publication=pub, release_id="synthetic-v2",
            recommendation_sha256=digest, action_id="action-1", explicit_action=True,
            authenticate_owner=lambda: True, verify_gate=lambda *args: change != "unauthenticated")
    assert "synthetic-v2-owner-action" not in pub.trees


@pytest.mark.parametrize("field,value", [("no_answer_count", 2),
    ("no_answer_cohort_sha256", "f" * 64), ("no_answer_provenance_sha256", "f" * 64),
    ("no_answer_baseline_sha256", "f" * 64)])
def test_result_requires_frozen_safety_population(field, value):
    result = aggregates()
    result[field] = value
    with pytest.raises(ev.CorpusV2EvaluationError):
        evaluated(result)


def test_absolute_safety_pass_cannot_hide_baseline_regression():
    spec = specification()
    spec["minimum_no_answer_safety"] = 0.95
    pub = Publication()
    frozen_digest = ev.freeze_evaluation_spec(spec, publication=pub,
        verify_historical=lambda _: True, verify_safety=lambda _: True)
    metrics = aggregates()
    metrics["no_answer_safety"] = 0.97
    result = ev.publish_evaluation_result(metrics, publication=pub, release_id="synthetic-v2",
        spec_sha256=frozen_digest, verify_index=lambda _: True)
    assert ready(pub, frozen_digest, result)["status"] == "BLOCKED"


@pytest.mark.parametrize("field,value", [("minimum_no_answer_safety", 0),
    ("minimum_no_answer_safety", 0.5), ("no_answer_ids", ["other"]),
    ("no_answer_expected_count", 2), ("no_answer_nonregression", "allow-decrease"),
    ("no_answer_baseline_sha256", ""), ("no_answer_provenance_sha256", "")])
def test_safety_policy_cannot_be_weakened(field, value):
    spec = specification()
    spec[field] = value
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.freeze_evaluation_spec(spec, publication=Publication(), verify_historical=lambda _: True,
                                  verify_safety=lambda _: True)


@pytest.mark.parametrize("verifier", [None, lambda _: False])
def test_safety_provenance_must_be_authenticated_before_freeze(verifier):
    pub = Publication()
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.freeze_evaluation_spec(specification(), publication=pub,
                                  verify_historical=lambda _: True, verify_safety=verifier)
    assert not pub.trees


def test_owner_reauthenticates_exact_context_and_requires_verifier():
    pub, spec, result = evaluated()
    recommendation = ready(pub, spec, result)
    kwargs = dict(publication=pub, release_id="synthetic-v2",
        recommendation_sha256=recommendation["sha256"], action_id="action-1",
        authenticate_owner=lambda: True, explicit_action=True)
    with pytest.raises(ev.CorpusV2EvaluationError):
        ev.record_owner_promotion(**kwargs)
    calls = []

    def verify(name, digest, context):
        calls.append((name, digest, context))
        return True

    assert ev.record_owner_promotion(**kwargs, verify_gate=verify) == "PROMOTED"
    assert len(calls) == 7
    for name, digest, context in calls:
        assert digest == gates(result)[name]
        assert context == dict(release_id="synthetic-v2", index_sha256="a" * 64,
                               spec_sha256=spec, result_sha256=result)
