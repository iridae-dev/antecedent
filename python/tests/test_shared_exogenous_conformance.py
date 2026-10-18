"""A shared exogenous term in a fully declared model: the public example, replay and refusals.

The example's model is `x = u`, `y = x xor u` with `u ~ (0.7, 0.3)`: observing `x = 1` fixes
`u = 1`, so `y` under `do(x = 0)` is 1 with certainty. Independent noise would answer 0.3.
"""

from __future__ import annotations

import runpy

import antecedent as ant
import numpy as np
import pytest
from antecedent import cross_world_execution as cw

from _repo_text import REPO_ROOT

R = cw.CrossWorldExecutionRefusal
EXAMPLE = REPO_ROOT / "examples" / "python" / "shared_exogenous_counterfactual.py"


def _request() -> cw.CrossWorldRequest:
    scm = cw.FiniteSCM(
        population="clinic",
        history="current_regime",
        exogenous=(cw.ExogenousTerm("u", (0.7, 0.3)),),
        variables=(
            cw.FiniteVariable("x", 2, (), "u", (0, 1)),
            cw.FiniteVariable("y", 2, ("x",), "u", (0, 1, 1, 0)),
        ),
    )
    return cw.CrossWorldRequest(
        scm=scm,
        units=(cw.FactualUnit("patient", {"x": 1}),),
        worlds=(cw.World("untreated", {"x": 0}), cw.World("treated", {"x": 1})),
        quantity=cw.probability("untreated", "y", 1),
    )


def test_the_public_example_executes_and_its_artifact_replays_the_values():
    # The example asserts its own numbers, seals its execution and replays it.
    runpy.run_path(str(EXAMPLE), run_name="__main__")

    first = cw.seal(_request())
    assert first.value == pytest.approx(1.0, abs=1e-12)
    for _ in range(2):
        replayed = cw.consume_cross_world_execution_artifact(
            first.to_bytes(), expected_identity=first.identity
        )
        assert np.float64(replayed.result.value).tobytes() == np.float64(first.value).tobytes()
    # A fresh execution seals to the same identity and bytes.
    again = cw.seal(_request())
    assert again.identity == first.identity and again.to_bytes() == first.to_bytes()


def test_observational_nonidentification_is_not_reported_as_unsupported_execution():
    # A bow-arc graph: no observational data identifies the effect of t on y.
    bow = ant.Admg.from_edges(["t", "y"], [("t", "y")], [("t", "y")])
    rng = np.random.default_rng(2003140)
    data = {"t": rng.normal(size=300), "y": rng.normal(size=300)}
    with pytest.raises(ant.errors.EffectNotIdentified) as not_identified:
        ant.analyze(data, graph=bow, query=ant.AverageEffect("t", "y"))
    assert not_identified.value.reason_code == "effect_not_identified"
    assert not_identified.value.identification_status == "not_identified"

    # A declared model whose joint enumeration is too large is unsupported execution.
    terms = tuple(cw.ExogenousTerm(f"u{k}", (0.125,) * 8) for k in range(6))
    oversized = cw.FiniteSCM(
        population="clinic",
        history="current_regime",
        exogenous=terms,
        variables=tuple(
            cw.FiniteVariable(f"v{k}", 2, (), f"u{k}", (0, 1, 0, 1, 0, 1, 0, 1)) for k in range(6)
        ),
    )
    with pytest.raises(R) as unsupported:
        cw.CrossWorldExecutor().execute(
            cw.CrossWorldRequest(
                scm=oversized,
                units=(cw.FactualUnit("u"),),
                worlds=(cw.World("w", {"v0": 0}),),
                quantity=cw.probability("w", "v1", 1),
            )
        )
    assert unsupported.value.reason_code == "route_not_supported"
    # The two refusals are different classes with different codes.
    assert unsupported.value.reason_code != not_identified.value.reason_code
