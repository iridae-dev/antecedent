"""Why does each transport scenario answer the way it does? (selection differences and invariances)

A transport scenario is a graph plus the selection targets: the variables whose mechanism may
differ between source and target. After a scenario set is estimated, ``invariance_report`` says
what each scenario assumed shared, which source factors its identified formula relies on, which
factors come from the target instead, and, for a scenario that cannot be transported, the
structural obstruction. The report is derived from the decided scenarios; nothing is stored.

Setup: every scenario shares ``z -> x``, ``z -> y``, ``x -> y`` and ``x <-> y``. The source holds
the experiment ``do(x)`` over ``(z, y)`` and the target its observational joint over ``(z, x, y)``.

* selection on ``z``: standardize, ``sum_z P_s(y | do(x), z) P*(z)``: one source factor and one
  target factor;
* no selection: ``P_s(y | do(x))``, one source factor, rule ``transport.direct``;
* selection on ``y``: the outcome's own mechanism differs (an s-hedge), so an obstruction."""

from __future__ import annotations

from antecedent import Admg
from antecedent.scenario_invariance import invariance_report
from antecedent.transport import advanced as transport

NAMES = ["z", "x", "y"]
SOURCE = (0.32, 0.08, 0.12, 0.48)  # do(x=1) over (z, y)
TARGET = (0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1)  # (z, x, y)


def scenario(name: str, selection: list[str]) -> transport.TransportScenario:
    graph = Admg.from_edges(NAMES, [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])
    return transport.TransportScenario(name, graph, selection)


scenarios = transport.TransportScenarioSet(
    [scenario("standardize", ["z"]), scenario("direct", []), scenario("outcome_shift", ["y"])],
    [transport.VariableCoordinate(name, "binary") for name in NAMES],
)
catalog = transport.EvidenceCatalog(
    regimes=[
        transport.EvidenceRegime(
            "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
        ),
        transport.EvidenceRegime("obs", "target", measured=["z", "x", "y"]),
    ]
)
laws = transport.ExactTransportData(
    (
        transport.ExactDiscreteLaw(
            "source",
            "trial",
            (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
            SOURCE,
            "trial",
            interventions=(("x", 1.0),),
        ),
        transport.ExactDiscreteLaw(
            "target",
            "obs",
            (("z", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
            TARGET,
            "target",
        ),
    )
)

stage = transport.prepare_transport_scenarios(
    scenarios,
    outcomes=["y"],
    treatments=["x"],
    source="source",
    target="target",
    catalog=catalog,
    laws=laws,
    at={"x": 1.0},
)
stage.estimate()  # the report reads the decided scenarios of an estimated stage

report = invariance_report(stage)
print(report.explain())

standardize = report.scenario("standardize").report
assert standardize.kind == "identified" and standardize.selection.targets == ("z",)
assert standardize.rules == ("transport.pretreatment_standardize",)
(source_factor,) = standardize.invariances
assert (source_factor.variables, source_factor.conditioned_on) == (("y",), ("z",))
(target_factor,) = standardize.target_factors  # P*(z) is a target law, not an invariance
assert target_factor.variables == ("z",)

direct = report.scenario("direct").report
assert direct.selection.targets == () and direct.rules == ("transport.direct",)

shifted = report.scenario("outcome_shift").report
assert shifted.kind == "obstructed" and shifted.invariances is None
assert shifted.obstruction is not None and shifted.obstruction.kind == "s_hedge"
