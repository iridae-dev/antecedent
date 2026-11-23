"""Which upstream noise term drives a target's variance? Intrinsic causal influence.

Data: 2,000 simulated rows of ``x -> y -> z`` with ``y = 2x + e_y`` (unit noise) and
``z = 1.5y + e_z`` (noise scale 0.5). The total coefficients of the three noise terms on ``z`` are
3, 1.5 and 1, so their Shapley shares of ``Var(z)`` are 9, 2.25 and 0.25.

The estimate is nested Monte Carlo through batch noise replay on mechanisms fitted to the rows.
The script asserts each share is near its closed form within the reported simulation and
mechanism-fit error, that a component that was not measured reads ``unquantified`` instead of
zero, and that undeclared or correlated noise is refused. Independence of the noise terms is the
caller's declaration; the graph and that declaration are not tested.

Run: ``python examples/python/intrinsic_influence.py``."""

from __future__ import annotations

import numpy as np
from antecedent import Execution
from antecedent.gcm import intrinsic_influence

EDGES = [("x", "y"), ("y", "z")]
TRUTH = {"x": 9.0, "y": 2.25, "z": 0.25}

rng = np.random.default_rng(4)
x = rng.normal(size=2000)
y = 2.0 * x + rng.normal(size=2000)
z = 1.5 * y + 0.5 * rng.normal(size=2000)
data = {"x": x, "y": y, "z": z}

result = intrinsic_influence(
    data,
    EDGES,
    "z",
    independent_noise=True,
    outer_draws=400,
    inner_draws=20,
    fit_bootstrap=20,
    execution=Execution(seed=3),
)
# Three sources allow exact Shapley values over all orderings, so no permutation-sampling error
# arises; the remaining error is Monte Carlo simulation and mechanism fit (the bootstrap).
assert result.method["shapley"] == {"kind": "exact"}
for source in result.sources:
    parts = {u.component: u for u in source.uncertainty}
    assert parts["permutation_sampling"].status == "not_applicable"
    assert parts["simulation"].status == "quantified"
    assert parts["mechanism_fit"].status == "quantified"
    # Components add in quadrature here only because every component that arises is quantified.
    stderr = np.hypot(parts["simulation"].stderr, parts["mechanism_fit"].stderr)
    truth = TRUTH[source.source]
    assert abs(source.contribution - truth) <= 6 * max(stderr, 0.05 * truth), source
assert result.budget["refits"] == 20

# The shares should add up to Var(z) = 11.5; the reported residual is the estimation error that
# keeps the estimated shares from summing exactly to the estimated variance.
assert abs(result.total_variance - sum(TRUTH.values())) <= 0.1 * sum(TRUTH.values())
total = sum(s.contribution for s in result.sources)
assert np.isclose(result.allocation_residual, result.total_variance - total)

# Without a bootstrap the mechanism-fit component is unquantified, which is not zero.
plain = intrinsic_influence(data, EDGES, "z", independent_noise=True, execution=Execution(seed=1))
for source in plain.sources:
    fit = source.component("mechanism_fit")
    assert fit.status == "unquantified" and fit.stderr is None
    assert fit.reason == "bootstrap_not_requested"

# The noise-independence assumption is the caller's to state; latent causes are refused.
for kwargs, detail in (
    ({}, "intrinsic_influence.noise_independence_undeclared"),
    ({"bidirected_edges": [("x", "z")]}, "intrinsic_influence.correlated_noise"),
):
    try:
        intrinsic_influence(data, EDGES, "z", **kwargs)
    except Exception as refusal:  # noqa: BLE001 - the refusal's detail is what is checked
        assert detail in str(refusal), refusal
        assert getattr(refusal, "reason_code", None) == "route_not_supported"
    else:
        raise AssertionError(f"{detail} must be refused")
print({s.source: round(s.contribution, 2) for s in result.sources})
