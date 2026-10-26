"""A minimal host with no agent logic: compile, read the status, tell a supported
claim from a refusal, and load an artifact.

A host is whatever sits between a caller (a service, a notebook tool, an LLM tool
layer) and the library. This one only enforces the contract in ``antecedent.host``:
it reads the status before running, runs only what is open, reports the claim with
its status, reports a failure as a coded refusal, and accepts an artifact only
against the identity it retained. It plans nothing and chooses nothing.

Run: ``python examples/python/reference_host.py``
"""

from __future__ import annotations

import json

import antecedent as ant
import numpy as np
from antecedent import errors, host


def make_request():
    """What the caller submits: data, a declared graph and one question."""
    rng = np.random.default_rng(17)
    z = rng.normal(size=160)
    treatment = rng.binomial(1, 0.5, size=160).astype(float)
    data = {
        "treatment": treatment,
        "outcome": 2 * treatment + z + rng.normal(size=160),
        "z": z,
    }
    graph = [("treatment", "outcome"), ("z", "outcome")]
    return data, graph, ant.AverageEffect("treatment", "outcome")


def serve(data, graph, query):
    """Compile, read the status, run only when it is open, and answer with the claim.

    Returns the host response and the artifact the host would store beside the
    identity it retains independently of the bytes.
    """
    # 1. Compile. The prepared study is what was compiled, not an answer.
    prepared = ant.prepare(
        data,
        graph=graph,
        query=query,
        execution=ant.Execution(seed=19),
        inference=ant.Frequentist(bootstrap=0),
    )
    before = host.inspection_status(prepared.inspect())
    print("before running:", before)
    if before == "unsupported":
        # The report already names the blockers; a host relays them.
        return {"status": before, "report": prepared.inspect().to_dict()}, None

    # 2. Run, then read the claim. A partial claim is relayed as partial.
    result = prepared.estimate()
    report = result.inspect()
    claim = host.claim_record(report.contract["claim"])
    response = {"status": claim["status"], "claim": claim, "value": result.answer.value}
    return response, result.artifact()


def accept(artifact_bytes, expected):
    """Accept bytes only as a claim bound to the identity the host retained."""
    try:
        loaded = ant.load(artifact_bytes, expected=expected)
    except ant.CausalError as error:
        return host.refusal_record(error)
    if loaded.acceptance.identity_bound and loaded.acceptance.answer_available:
        return {"status": "supported", "value": loaded.answer.value}
    return {"status": "partial", "restriction": str(loaded.acceptance)}


def main():
    # The host first checks it speaks the contract version it was written for.
    assert host.CONTRACT_VERSION == 1

    data, graph, query = make_request()

    # Supported claim.
    response, artifact = serve(data, graph, query)
    print("response:", json.dumps({k: v for k, v in response.items() if k != "claim"}))
    assert response["status"] == "supported", response
    assert response["claim"]["kind"] == "point"
    assert abs(response["value"] - 2.0) < 0.6, response["value"]

    # Load the artifact against what the host retained, not against its own claim.
    accepted = accept(artifact.data, artifact.expectation)
    print("loaded:", accepted["status"])
    assert accepted["status"] == "supported", accepted
    assert accepted["value"] == response["value"]

    # A refusal is a coded record, never a number. This one is what a route that
    # cannot answer raises; the host relays code, family and remedy unchanged.
    refusal = host.refusal_record(
        errors.CausalUnsupportedError(
            "the attestation cannot be re-verified",
            reason_code="attested_not_reverifiable",
        )
    )
    print("refusal:", json.dumps(refusal))
    assert refusal["status"] == "unsupported"
    assert refusal["reason_code"] == "attested_not_reverifiable"
    assert refusal["family"] == "unsupported"
    assert refusal["reason_code"] in host.refusal_codes()

    # Altered bytes are refused; they are never a claim.
    altered = accept(b"not an artifact", artifact.expectation)
    print("altered bytes:", altered["family"])
    assert altered["status"] == "unsupported"
    assert altered["family"] in {family.code for family in host.error_families()}

    # A different request's identity does not accept these bytes.
    other_t = np.array([0.0, 1.0] * 80)
    other = ant.analyze(
        {"t": other_t, "y": 3 * other_t},
        graph=[("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    ).artifact()
    wrong = accept(artifact.data, other.expectation)
    print("wrong identity:", wrong["status"])
    assert wrong["status"] != "supported", wrong


if __name__ == "__main__":
    main()
