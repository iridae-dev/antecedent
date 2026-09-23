"""The uncontracted and contracted `analysis_result` artifacts of one execution must
agree on whether a scalar effect was reported, and both must actually encode.

`PreparedAnalysis.export_artifact(payload="result")` used to hand-build its own
`AnalysisResultWire` (a near-duplicate of the facade's `body_for`/`body_frame`),
including its own `estimate` rule (`ate.is_finite().then_some(ate)`) and its own
`identification_variables` fallback that only consulted the certificate's temporal
envelope, not the query's structural mixture. For a DAG-posterior response result
that duplicate `identification_variables` lookup left `identification.query`
inconsistent with the wire's own `query`, so the uncontracted artifact failed to
encode at all (`CausalSerializationError: identification.query does not match the
enclosing query`) while the contracted artifact, built from the facade's own
`analysis_result_wire`/`body_for`, encoded fine. Routing `export_artifact` through
the same facade method fixes the crash and keeps `estimate` consistent between the
two artifacts of one execution (a function-valued response result has no scalar
effect, so both must report `None`, never one an overclaimed number).
"""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent import artifacts
from antecedent.estimation import PreparedAnalysis


def test_export_artifact_matches_contracted_artifact_for_dag_posterior_response():
    n = 80
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = z + 0.1
    y = 1.0 + 2.0 * t + z
    data = {"t": t, "y": y, "z": z}
    prepared = PreparedAnalysis.prepare(
        data,
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=antecedent.ResponseCurve("t", "y", grid=[0.0, 0.5, 1.0]),
        inference=antecedent.Bayesian(n_draws=32, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    result = prepared.estimate(data, seed=1)
    assert result.response is not None

    # This used to raise `CausalSerializationError` from the binding's own
    # duplicated wire construction; it must now encode like any other export.
    uncontracted = artifacts.loads(prepared.export_artifact(payload="result"))
    contracted = artifacts.loads(prepared.export())
    assert uncontracted.payload_kind == "analysis_result"
    assert contracted.payload_kind == "analysis_result"

    # A function-valued response result has no single scalar effect: the contracted
    # artifact (built from the facade's canonical `executed_scalar`) reports `None`.
    assert contracted.payload["estimate"] is None
    # The uncontracted artifact of the very same execution must say the same thing,
    # not a leftover `ate` from a hand-rolled rule that never checked for a response.
    assert uncontracted.payload["estimate"] is None
