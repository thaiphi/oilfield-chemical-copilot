"""Private, immutable evaluation and promotion contracts; no operational adapters.

Publication must be the already authenticated, locked private publication capability.
Index/gate verifiers authenticate sealed evidence and its release/index bindings, not
merely check digest syntax. Owner authentication is supplied by the operator boundary.
These interfaces record decisions only; they never run a model or change production.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import re


class CorpusV2EvaluationError(ValueError):
    """Fixed-code failure without private input values."""


_LAYOUT = {"evidence": frozenset({"payload.bin"})}
_SPEC_KEYS = frozenset({"release_id", "index_sha256", "development_ids", "unseen_ids",
    "historical_ids", "scoring_protocol", "regression_families", "minimum_accuracy",
    "minimum_no_answer_safety", "maximum_latency_ms", "latency_measurement", "no_tuning",
    "steward_approved"})
_RESULT_KEYS = frozenset({"accuracy", "no_answer_safety", "latency_ms", "development_count",
    "unseen_count", "no_answer_count", "regression_families"})
REQUIRED_PROMOTION_GATES = frozenset({"legacy_guard", "source_accounting",
    "critical_source_evidence", "index_contract", "evaluation_result", "canary_plan",
    "rollback_rehearsal"})


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value)


def _number(value, *, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (maximum is None or value <= maximum))


def _identifiers(values):
    return (type(values) is list and bool(values) and all(_identifier(v) for v in values)
            and len(set(values)) == len(values))


def _validate_spec(spec):
    if (type(spec) is not dict or set(spec) != _SPEC_KEYS
            or not _identifier(spec["release_id"]) or not _digest(spec["index_sha256"])
            or not all(_identifiers(spec[k]) for k in
                       ("development_ids", "unseen_ids", "regression_families"))
            or type(spec["historical_ids"]) is not dict
            or set(spec["historical_ids"]) != {"E1a-3", "E1a-4"}
            or not all(_identifiers(v) for v in spec["historical_ids"].values())
            or not _identifier(spec["scoring_protocol"])
            or not _identifier(spec["latency_measurement"])
            or spec["no_tuning"] is not True or spec["steward_approved"] is not True
            or not _number(spec["minimum_accuracy"], maximum=1)
            or not _number(spec["minimum_no_answer_safety"], maximum=1)
            or not _number(spec["maximum_latency_ms"]) or spec["maximum_latency_ms"] <= 0):
        raise ValueError
    dev, unseen = set(spec["development_ids"]), set(spec["unseen_ids"])
    historical = set().union(*map(set, spec["historical_ids"].values()))
    if dev & unseen or (dev | unseen) & historical:
        raise ValueError


def _read(publication, name, digest):
    if not _digest(digest):
        raise ValueError
    publication.ensure_no_staging(".evaluation.", ".tmp")
    content = publication.read_exact_tree(name, _LAYOUT)["evidence/payload.bin"]
    value = json.loads(content)
    if sha256(content).hexdigest() != digest or _json(value) != content:
        raise ValueError
    return value


def _publish(publication, name, payload):
    content = _json(payload)
    digest = sha256(content).hexdigest()
    publication.ensure_no_staging(".evaluation.", ".tmp")
    if publication.final_exists(name):
        raise ValueError
    staging = publication.create_staging(".evaluation.", ".tmp")
    staging.mkdir("evidence")
    staging.write_exclusive("evidence/payload.bin", content)
    staging.sync_directory("evidence")
    staging.sync_root()
    if publication.read_exact_tree(staging.name, _LAYOUT) != {"evidence/payload.bin": content}:
        raise ValueError
    publication.publish_no_replace(staging, name)
    publication.sync_parent()
    if _read(publication, name, digest) != payload:
        raise ValueError
    return digest


def freeze_evaluation_spec(spec, *, publication, verify_historical=None):
    """Publish steward-attested cohort exclusions before any run/result exists.

    verify_historical must authenticate the sealed E1a-3/E1a-4 artifacts and
    compare their COMPLETE identifier inventories with the supplied mapping.
    No historical reader is installed; absent verification fails closed.
    """
    try:
        spec = json.loads(_json(spec))
        _validate_spec(spec)
        if verify_historical is None or verify_historical(spec["historical_ids"]) is not True:
            raise ValueError
        release_id = spec["release_id"]
        if any(publication.final_exists(f"{release_id}-{suffix}")
               for suffix in ("result", "recommendation", "owner-action")):
            raise ValueError
        return _publish(publication, f"{release_id}-spec", spec)
    except Exception:
        raise CorpusV2EvaluationError("C2_EVALUATION_SPEC_INVALID") from None


def verify_evaluation_spec(*, publication, release_id, expected_sha256):
    try:
        if not _identifier(release_id):
            raise ValueError
        spec = _read(publication, f"{release_id}-spec", expected_sha256)
        _validate_spec(spec)
        if spec["release_id"] != release_id:
            raise ValueError
        return spec
    except Exception:
        raise CorpusV2EvaluationError("C2_EVALUATION_SPEC_INVALID") from None


def authorize_evaluation(*, publication, release_id, spec_sha256, verify_index):
    """Pre-initialization guard; operator must call before creating run resources.

    verify_index receives the frozen specification and must authenticate the
    exact sealed V2 index contract against its release_id and index_sha256.
    No model/retrieval initializer is installed in this contract-only module.
    """
    try:
        spec = verify_evaluation_spec(publication=publication, release_id=release_id,
                                      expected_sha256=spec_sha256)
        if publication.final_exists(f"{release_id}-result") or verify_index(spec) is not True:
            raise ValueError
        return spec
    except Exception:
        raise CorpusV2EvaluationError("C2_EVALUATION_RUN_BLOCKED") from None


def _validate_aggregates(result, spec):
    if (type(result) is not dict or set(result) != _RESULT_KEYS
            or not _number(result["accuracy"], maximum=1)
            or not _number(result["no_answer_safety"], maximum=1)
            or not _number(result["latency_ms"])
            or any(type(result[k]) is not int or result[k] <= 0 for k in
                   ("development_count", "unseen_count", "no_answer_count"))
            or result["development_count"] != len(spec["development_ids"])
            or result["unseen_count"] != len(spec["unseen_ids"])
            or result["no_answer_count"] > result["unseen_count"]
            or type(result["regression_families"]) is not dict
            or set(result["regression_families"]) != set(spec["regression_families"])
            or not all(_number(v, maximum=1) for v in result["regression_families"].values())):
        raise ValueError


def publish_evaluation_result(aggregates, *, publication, release_id, spec_sha256,
                              verify_index):
    """Record supplied aggregate metrics only; never perform an evaluation."""
    try:
        spec = authorize_evaluation(publication=publication, release_id=release_id,
                                    spec_sha256=spec_sha256, verify_index=verify_index)
        _validate_aggregates(aggregates, spec)
        return _publish(publication, f"{release_id}-result", {
            "release_id": release_id, "spec_sha256": spec_sha256,
            "index_sha256": spec["index_sha256"], "aggregates": aggregates})
    except Exception:
        raise CorpusV2EvaluationError("C2_EVALUATION_RESULT_INVALID") from None


def _result(publication, release_id, spec_sha256, result_sha256):
    spec = verify_evaluation_spec(publication=publication, release_id=release_id,
                                  expected_sha256=spec_sha256)
    result = _read(publication, f"{release_id}-result", result_sha256)
    if (set(result) != {"release_id", "spec_sha256", "index_sha256", "aggregates"}
            or result["release_id"] != release_id or result["spec_sha256"] != spec_sha256
            or result["index_sha256"] != spec["index_sha256"]):
        raise ValueError
    _validate_aggregates(result["aggregates"], spec)
    return spec, result["aggregates"]


def record_promotion_recommendation(*, publication, release_id, spec_sha256, result_sha256,
                                    sealed_gates, verify_gate, canary_passed, rollback_passed):
    """Authenticate each sealed gate with (name, digest, context) before readiness.

    Context binds release, index, spec, and result. The injected verifier must
    validate gate semantics, including complete source/critical accounting and
    successful canary/rollback evidence. Booleans alone never grant readiness.
    """
    try:
        spec, result = _result(publication, release_id, spec_sha256, result_sha256)
        context = dict(release_id=release_id, index_sha256=spec["index_sha256"],
                       spec_sha256=spec_sha256, result_sha256=result_sha256)
        valid = (type(sealed_gates) is dict and set(sealed_gates) == REQUIRED_PROMOTION_GATES
                 and all(_digest(v) for v in sealed_gates.values())
                 and sealed_gates["evaluation_result"] == result_sha256
                 and sealed_gates["index_contract"] == spec["index_sha256"]
                 and canary_passed is True and rollback_passed is True)
        if valid:
            valid = all(verify_gate(name, digest, dict(context)) is True
                        for name, digest in sorted(sealed_gates.items()))
        passed = (result["accuracy"] >= spec["minimum_accuracy"]
                  and result["no_answer_safety"] >= spec["minimum_no_answer_safety"]
                  and result["latency_ms"] <= spec["maximum_latency_ms"]
                  and all(v >= spec["minimum_accuracy"]
                          for v in result["regression_families"].values()))
        status = "PROMOTION_READY" if valid and passed else "BLOCKED"
        # Reject arbitrary gate fields instead of reflecting them into an artifact.
        gates = sealed_gates if valid else {}
        payload = dict(context, status=status, sealed_gates=gates)
        digest = _publish(publication, f"{release_id}-recommendation", payload)
        return {"status": status, "sha256": digest}
    except Exception:
        raise CorpusV2EvaluationError("C2_PROMOTION_BLOCKED") from None


def validate_evaluation_artifacts(*, artifacts, release_id, critical_register_sha256):
    """Reconcile canonical evaluation evidence within an enclosing release seal.

    Canary and rollback digests refer to external private attestations verified by
    the recommendation producer. This is consistency validation, not a replacement
    for authenticating those attestations through record_promotion_recommendation.
    """
    try:
        spec, result, recommendation = [json.loads(artifacts[key]) for key in
            ("evaluation_specification", "evaluation_result", "promotion_state")]
        for key, payload in zip(("evaluation_specification", "evaluation_result", "promotion_state"),
                                (spec, result, recommendation), strict=True):
            if _json(payload) != artifacts[key]:
                raise ValueError
        _validate_spec(spec)
        _validate_aggregates(result["aggregates"], spec)
        index_digest = sha256(artifacts["index_contract"]).hexdigest()
        context = dict(release_id=release_id, index_sha256=index_digest,
                       spec_sha256=sha256(artifacts["evaluation_specification"]).hexdigest())
        if (spec["release_id"] != release_id or spec["index_sha256"] != index_digest
                or result != dict(context, aggregates=result["aggregates"])):
            raise ValueError
        context["result_sha256"] = sha256(artifacts["evaluation_result"]).hexdigest()
        gates = recommendation["sealed_gates"]
        if (type(gates) is not dict or set(gates) != REQUIRED_PROMOTION_GATES
                or not all(_digest(v) for v in gates.values())
                or gates["legacy_guard"] != sha256(artifacts["legacy_guard"]).hexdigest()
                or gates["source_accounting"] != sha256(artifacts["dispositions"]).hexdigest()
                or gates["critical_source_evidence"] != critical_register_sha256
                or gates["index_contract"] != index_digest
                or gates["evaluation_result"] != context["result_sha256"]
                or recommendation != dict(context, status="PROMOTION_READY", sealed_gates=gates)):
            raise ValueError
        metrics = result["aggregates"]
        if (metrics["accuracy"] < spec["minimum_accuracy"]
                or metrics["no_answer_safety"] < spec["minimum_no_answer_safety"]
                or metrics["latency_ms"] > spec["maximum_latency_ms"]
                or any(v < spec["minimum_accuracy"]
                       for v in metrics["regression_families"].values())):
            raise ValueError
    except Exception:
        raise CorpusV2EvaluationError("C2_PROMOTION_BLOCKED") from None


def record_owner_promotion(*, publication, release_id, recommendation_sha256, action_id,
                           authenticate_owner, explicit_action):
    """Record one authenticated explicit action; does not switch a live release."""
    try:
        if (not _identifier(release_id) or not _identifier(action_id)
                or explicit_action is not True or authenticate_owner() is not True):
            raise ValueError
        recommendation = _read(publication, f"{release_id}-recommendation",
                               recommendation_sha256)
        if (recommendation["release_id"] != release_id
                or recommendation["status"] != "PROMOTION_READY"):
            raise ValueError
        _result(publication, release_id, recommendation["spec_sha256"],
                recommendation["result_sha256"])
        _publish(publication, f"{release_id}-owner-action", dict(
            release_id=release_id, recommendation_sha256=recommendation_sha256,
            action_id=action_id, status="PROMOTED"))
        return "PROMOTED"
    except Exception:
        raise CorpusV2EvaluationError("C2_PROMOTION_BLOCKED") from None
