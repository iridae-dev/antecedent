"""When a causal contract does not identify, what evidence is owed and which study would repair it?

A back-door contract ``t <- z1, z2 -> y`` with ``t -> y``, observed only on ``t`` and ``y``, owes
one joint law over ``t, y, z1, z2`` (the only admissible adjustment set is ``{z1, z2}``).
``repair.obligations`` states that debt in machine-readable form; ``repair.repair`` checks
candidate studies, singly or in bounded subsets, against the contract's own identification
theorem. Antecedent never conducts a study: candidates are declarations with cost semantics.

Two registries that each measure one confounder do not combine into a joint law, so only the
cohort that measures both is verified sufficient. The result is a portable artifact that an
independent consumer replays."""

from __future__ import annotations

from antecedent import Dag, repair

NAMES = ["t", "y", "z1", "z2", "w1", "w2", "w3"]
EDGES = [("z1", "t"), ("z1", "y"), ("z2", "t"), ("z2", "y"), ("t", "y")]

contract = repair.BackdoorContract(
    graph=Dag.from_edges(NAMES, EDGES),
    treatment="t",
    outcome="y",
    population="clinic",
    observed=["t", "y"],
)

(obligation,) = repair.obligations(contract)
print(f"{obligation.kind} over {sorted(obligation.variables)} in {obligation.population!r}")
assert obligation.kind == "provide_joint_law" and obligation.joint is True
assert set(obligation.variables) == {"t", "y", "z1", "z2"}
assert obligation.satisfiable_by_study is True


def observation(label: str, measured: list[str], cost: float) -> repair.StudyCandidate:
    return repair.StudyCandidate.observation(
        label,
        population="clinic",
        measured=measured,
        cost=cost,
        sample_size=500,
        recruitment="consecutive patients",
        timing="baseline",
        unit="patient",
        cost_unit="USD",
    )


studies = [
    observation("registry_z1", ["t", "y", "z1"], 5),
    observation("registry_z2", ["t", "y", "z2"], 5),
    observation("cohort", ["t", "y", "z1", "z2"], 20),
]
result = repair.repair(contract, studies)
print(result.explain())

assert result.outcome == "repaired"
assert result.best is not None and result.best.labels == ("cohort",)
assert result.classification("cohort") == "verified_sufficient"
# Each confounder alone leaves the other back-door path open, and two separate studies are not
# one joint law.
assert result.classification("registry_z1") == "insufficient"
assert result.classification("registry_z1", "registry_z2") == "insufficient"
assert result.inference_claim == "none"

# Without the cohort nothing is certified, and that is all it says.
partial = repair.repair(contract, studies[:2])
assert partial.outcome == "none_certified" and partial.best is None

replayed = repair.consume(result.export())
assert replayed.outcome == "repaired" and replayed.best == result.best
