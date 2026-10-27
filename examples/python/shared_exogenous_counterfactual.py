"""A counterfactual through a shared hidden cause, computed because the model is fully declared.

A hidden frailty ``u`` (frail with probability 0.3) drives both who is treated and how the
outcome responds: ``x = u`` and ``y = x xor u``. Because the mechanisms and the law of ``u`` are
all declared, a patient observed treated (``x = 1``) is known to be frail, and that same frailty
is read in the counterfactual world ``do(x = 0)``: ``y = 0 xor 1 = 1`` with certainty. The
model that gives ``y`` its own independent noise (same law, no shared cause) answers 0.3, because
the factual treatment then says nothing about the outcome noise. The two answers differ, which is
why a shared term must not be replaced by independent noise.

The execution is sealed, replayed by a consumer that recomputes it, and a model whose mechanisms
are not declared is refused as not identified rather than guessed at."""

from __future__ import annotations

import math

from antecedent import cross_world_execution as cw
from antecedent.outcomes import require_success


def scm(*, shared: bool) -> cw.FiniteSCM:
    terms = [cw.ExogenousTerm("u", (0.7, 0.3))]
    if not shared:
        terms.append(cw.ExogenousTerm("u_y", (0.7, 0.3)))
    return cw.FiniteSCM(
        population="clinic",
        history="current_regime",
        exogenous=tuple(terms),
        variables=(
            cw.FiniteVariable("x", 2, (), "u", (0, 1)),
            # table[u * 2 + x]: u = 0 gives y = x, u = 1 gives y = 1 - x.
            cw.FiniteVariable("y", 2, ("x",), "u" if shared else "u_y", (0, 1, 1, 0)),
        ),
    )


def ask(model: cw.FiniteSCM) -> cw.CrossWorldRequest:
    return cw.CrossWorldRequest(
        scm=model,
        units=(cw.FactualUnit("patient", {"x": 1}),),
        worlds=(cw.World("untreated", {"x": 0}), cw.World("treated", {"x": 1})),
        quantity=cw.probability("untreated", "y", 1),
    )


executor = cw.CrossWorldExecutor()
shared = require_success(executor.execute(ask(scm(shared=True))))
independent = require_success(executor.execute(ask(scm(shared=False))))
print(f"P(Y_x0 = 1 | X = 1), shared frailty:      {shared.value:.3f}")
print(f"P(Y_x0 = 1 | X = 1), independent noise:   {independent.value:.3f}")
assert shared.matrix_row == "counterfactual.finite_shared_exogenous_scm.marginal"
assert math.isclose(shared.value, 1.0, abs_tol=1e-12)
assert math.isclose(independent.value, 0.3, abs_tol=1e-12)

# Seal and replay: the consumer recomputes the enumeration and refuses any edit.
sealed = require_success(cw.seal(ask(scm(shared=True))))
replayed = require_success(
    cw.consume_cross_world_execution_artifact(sealed.to_bytes(), expected_identity=sealed.identity)
)
assert replayed.result.value == sealed.value

# Mechanisms that are not declared do not identify the counterfactual.
try:
    cw.decide(quantity="counterfactual_joint", model="semi_markovian")
except cw.CrossWorldExecutionRefusal as refusal:
    print(f"undeclared latent confounding: {refusal.reason_code} / {refusal.detail}")
    assert refusal.reason_code == "cross_world_not_identified"
