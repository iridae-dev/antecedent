"""Use a causal analysis as a component in an application.

A retention service asks how many more weekly sessions a reminder message produces.
The service is the host: it decides what to ask, whether to run it and what to do with
the answer. Antecedent supplies the computation and the contracts the host reads.

The data are simulated so the known effect (2 sessions) can be asserted. The reminder
is assigned at random and prior activity is a precision covariate, so the simulated
effect is a fixture truth of this model and not an empirical claim.

Run: ``python examples/python/application_component.py``
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import antecedent as ant
import numpy as np
from antecedent import host

TRUE_EFFECT = 2.0


@dataclass(frozen=True)
class Request:
    """What a caller submits: prepared data, a declared graph and one question."""

    data: dict
    graph: object
    query: object


@dataclass(frozen=True)
class AcceptancePolicy:
    """The host's own rule for acting on an answer. Antecedent does not supply one."""

    accepted_support: frozenset = frozenset({"licensed"})


def make_data():
    """Simulated users: reminder assigned at random, true effect on sessions is 2."""
    rng = np.random.default_rng(17)
    prior = rng.normal(size=160)
    reminder = rng.binomial(1, 0.5, size=160).astype(float)
    return {
        "reminder": reminder,
        "sessions": TRUE_EFFECT * reminder + prior + rng.normal(size=160),
        "prior_sessions": prior,
    }


def observed_request(data):
    """The caller declares that prior activity is the only other driver of sessions."""
    graph = [("reminder", "sessions"), ("prior_sessions", "sessions")]
    return Request(data, graph, ant.AverageEffect("reminder", "sessions"))


def confounded_request(data):
    """The caller declares an unmeasured common cause of reminder and sessions."""
    graph = ant.Admg.from_edges(
        ["prior_sessions", "reminder", "sessions"],
        [
            ("prior_sessions", "reminder"),
            ("prior_sessions", "sessions"),
            ("reminder", "sessions"),
        ],
        [("reminder", "sessions")],
    )
    return Request(data, graph, ant.AverageEffect("reminder", "sessions"))


def prepare_request(request):
    """Compile the request. Nothing is fitted; the study is what was compiled."""
    return ant.prepare(
        request.data,
        graph=request.graph,
        query=request.query,
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=19),
    )


def read_plan(study):
    """What the host reads before it spends any compute."""
    if study.refusal is not None:
        return {"stage": "refused", "code": study.refusal.code}
    view = study.plan_view()
    return {
        "stage": view.stage,
        "route": view.route,
        "support_status": view.support.status,
        "identification": view.identification.status,
        "program": view.identity,
    }


def execute(study):
    """Run a prepared study. A scientific refusal and an answer are both returned."""
    outcome = ant.estimate(study)
    if not outcome.accepted:
        try:
            outcome.refusal.raise_error()
        except ant.CausalError as error:
            return {
                "status": "unsupported",
                "refusal": host.refusal_record(error),
            }, outcome
    report = outcome.inspect()
    claim = host.claim_record(report.contract["claim"])
    response = {
        "status": claim["status"],
        "claim": claim,
        "value": float(outcome.answer.value),
        "support_status": report.support.payload["matrix_status"],
        "calibration": report.calibration.status,
    }
    return response, outcome


def decide(response, policy):
    """The host's policy: Antecedent reports fields, the host chooses what they permit."""
    if response["status"] == "unsupported":
        return "reject"
    if response["status"] == "partial":
        return "relay_with_limits"
    if response["support_status"] not in policy.accepted_support:
        return "send_to_review"
    return "act"


def consume(encoded, retained_identity):
    """An independent consumer: it has only the bytes and the identity it retained."""
    expected = json.loads(retained_identity)
    loaded = ant.load(encoded, expected=expected)
    if loaded.acceptance.identity_bound and loaded.acceptance.answer_available:
        return loaded
    raise AssertionError(f"not accepted as a claim: {loaded.acceptance}")


def main():
    data = make_data()
    policy = AcceptancePolicy()

    # 1. Prepare, then inspect the plan before running.
    study = prepare_request(observed_request(data))
    plan = read_plan(study)
    print("plan:", json.dumps(plan))
    assert plan["stage"] == "prepared"
    assert plan["support_status"] == "licensed"

    # 2. Execute and read the claim, the support status and the calibration status.
    response, outcome = execute(study)
    print("response:", json.dumps({k: v for k, v in response.items() if k != "claim"}))
    assert response["status"] == "supported"
    assert response["claim"]["kind"] == "point"
    assert abs(response["value"] - TRUE_EFFECT) < 0.6
    assert isinstance(response["calibration"], str)
    decision = decide(response, policy)
    assert decision == "act"
    # The same answer under a stricter host policy is not acted on.
    assert (
        decide(response, AcceptancePolicy(accepted_support=frozenset()))
        == "send_to_review"
    )

    # 3. A typed refusal is scientific information, not an error to retry.
    refused_study = prepare_request(confounded_request(data))
    assert read_plan(refused_study)["stage"] == "refused"
    refused, refused_outcome = execute(refused_study)
    print("refusal:", json.dumps(refused["refusal"]))
    assert not refused_outcome.accepted and refused_outcome.answer.kind == "refused"
    assert refused["refusal"]["reason_code"] == "effect_not_identified"
    assert refused["refusal"]["reason_code"] in host.refusal_codes()
    assert decide(refused, policy) == "reject"

    # 4. Branch onto changed data. The earlier result is preserved.
    changed = {**data, "sessions": data["sessions"] + data["reminder"]}
    branch = ant.rebind(study, data=changed)
    branch_response, _ = execute(branch)
    assert branch is not study
    assert abs((branch_response["value"] - response["value"]) - 1.0) < 1e-6
    assert ant.estimate(study).answer.value == outcome.answer.value
    assert branch.plan_view().identity != study.plan_view().identity
    assert study.plan_view().identity == plan["program"]

    # 5. Export, then consume as an independent consumer.
    artifact = outcome.export()
    retained = json.dumps(artifact.expectation.to_dict()["identity"])
    loaded = consume(bytes(artifact), retained)
    print("loaded:", loaded.acceptance)
    assert loaded.answer.value == outcome.answer.value
    # Loading restores a recorded claim. It does not recreate the live study.
    assert loaded.study is None
    assert "refresh" not in loaded.capabilities and "explore" not in loaded.capabilities

    # Bytes that were altered, or a different request's identity, are not accepted.
    try:
        consume(b"not an artifact", retained)
    except ant.CausalError as error:
        altered = host.refusal_record(error)
        assert altered["family"] in {family.code for family in host.error_families()}
    else:
        raise AssertionError("altered bytes were accepted")
    flags = np.array([0.0, 1.0] * 80)
    other = Request(
        {"reminder": flags, "sessions": 3 * flags},
        [("reminder", "sessions")],
        ant.AverageEffect("reminder", "sessions"),
    )
    other_artifact = ant.estimate(prepare_request(other)).export()
    try:
        consume(
            bytes(artifact),
            json.dumps(other_artifact.expectation.to_dict()["identity"]),
        )
    except ant.CausalError:
        pass
    else:
        raise AssertionError("a different identity accepted these bytes")


if __name__ == "__main__":
    main()
