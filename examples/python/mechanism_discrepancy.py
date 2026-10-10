"""Has the mechanism of a node changed between a source study and a target population?

``diagnose_mechanism_discrepancy`` is a Wald test that ``E[V | parents]`` is one linear mechanism
in both populations, given comparable measurements. A rejection says the mechanism of the node
is not invariant (so a selection node on it cannot be excluded). A non-rejection never certifies
invariance: the result states the smallest differences it could have detected.

Hand oracle (four rows, one parent ``x``): the source fits ``[1.3, 0.8]``; a target equal to the
source plus ``0.5 + 0.5 x`` gives ``W = 25 / 6`` on 2 degrees of freedom, ``p = exp(-25 / 12)``."""

from __future__ import annotations

import math

from antecedent import mechanism_discrepancy as md

X = [0.0, 1.0, 2.0, 3.0]
SOURCE_Y = [1.0, 3.0, 2.0, 4.0]
SHIFTED_Y = [1.5, 4.0, 3.5, 6.0]  # SOURCE_Y + 0.5 + 0.5 x
LARGE_SHIFT_Y = [1.0, 8.0, 12.0, 19.0]  # SOURCE_Y + 5 x

# Both populations declare the same node, units, parents and protocol: comparability is stated,
# never inferred.
measurement = md.Measurement("V", "mg", parents=[("x", "cm")], protocol_id="protocol-1")


def sample(label: str, y: list[float]) -> md.Sample:
    return md.Sample.from_rows(label, [(yi, {"x": xi}) for yi, xi in zip(y, X, strict=True)])


source = sample("source", SOURCE_Y)

small = md.diagnose_mechanism_discrepancy(
    source=source, target=sample("target", SHIFTED_Y), measurement=measurement
)
print(small.explain())
assert math.isclose(small.statistic, 25.0 / 6.0, abs_tol=1e-9)
assert small.degrees_of_freedom == 2
assert math.isclose(small.p_value, math.exp(-25.0 / 12.0), abs_tol=1e-9)
# Not rejected at 0.05, and that is not evidence of invariance.
assert small.conclusion == "not_rejected"
assert small.non_rejection_certifies_invariance is False
assert set(small.minimal_detectable_differences) == {"(intercept)", "x"}

large = md.diagnose_mechanism_discrepancy(
    source=source, target=sample("target", LARGE_SHIFT_Y), measurement=measurement
)
print(large.explain())
assert large.rejected
assert large.informs_selection_on == ("V",)

# The result is a durable artifact; a consumer that retained its identity recomputes it and
# refuses a different one.
again = md.MechanismDiscrepancyResult.consume(small.export(), expected_identity=small.identity)
assert again.statistic == small.statistic
try:
    md.MechanismDiscrepancyResult.consume(large.export(), expected_identity=small.identity)
except md.MechanismDiscrepancyRefusal as refusal:
    assert refusal.detail == "mechanism_discrepancy.wrong_contract"
else:
    raise AssertionError("a different result must not consume under the retained identity")
