"""Withdraw a source: historical replay still succeeds, current decision use refuses.

One decision rests on three separately fitted routes: a cluster-randomized trial (the margin), a
checked two-step temporal history (the period effect) and a conjugate Gaussian cost model. The
three published means combine in one linear combination, ``benefit * margin + temporal - cost``,
and the decision compares acting with declining at the combined mean. Every route is fitted from
the library's own simulated fixture, so the numbers are fixture numbers of that declared model and
make no empirical claim.

The example does four things in order: build the decision, change the utility without refitting
anything, export the decision, then withdraw the trial. Afterwards the exported numbers still
replay exactly, and every operation that asks for current permission refuses. This is the
supported behaviour of this mixed-evidence workflow; it is not a general revocation mechanism.
"""

from __future__ import annotations

from antecedent import mixed_decision as md
from antecedent.execution import Execution

SEED = 0xF1A6


def refused(call) -> md.MixedDecisionRefusal:
    """Run ``call`` and return the typed refusal it raises; anything else is a failure."""
    try:
        call()
    except md.MixedDecisionRefusal as refusal:
        return refusal
    raise AssertionError("the call should have been refused")


# -- 1. Build the decision from three routes ------------------------------------------------
# The dependence between the routes is declared, never defaulted: here they are independent.
workflow = md.run_mixed_decision(
    md.Dependence.independent(), execution=Execution(seed=SEED)
)
routes = workflow.routes
decision = workflow.decision()
print(
    decision.evaluation.verdict,
    decision.evaluation.chosen,
    decision.mean,
    decision.variance,
)

assert routes.cluster_trials == 1 and routes.temporal_means == 2
assert decision.dependence == "independent"
# The combined mean is the sum of the three published moments (benefit one).
assert (
    abs(
        decision.mean
        - (decision.margin_mean + routes.temporal_effect - routes.cost_mean)
    )
    < 1e-9
)
assert decision.evaluation.verdict == "uniquely_optimal"
assert decision.evaluation.chosen == (("act",) if decision.mean > 0.0 else ("decline",))

# -- 2. Change the utility: the decision is recomputed and nothing is refitted --------------
receipt = workflow.utility_only(2.0)
# The receipt counts every fit, identification and score recomputation the change performed.
assert receipt == md.UtilityReplay(0, 0, 0, 0, 0)
after = workflow.decision()
current = workflow.routes
assert after.margin_mean == decision.margin_mean
assert (
    abs(
        after.mean
        - (2.0 * after.margin_mean + current.temporal_effect - current.cost_mean)
    )
    < 1e-9
)
assert current.cluster_effect == routes.cluster_effect
assert current.temporal_effect == routes.temporal_effect
assert current.cost_mean == routes.cost_mean

# -- 3. Export the decision, bound to the current standing of its sources -------------------
exported = workflow.export_decision()
# Keep the identity apart from the bytes: a consumer checks the bytes against it.
loaded = workflow.load(exported.artifact, exported.identity)
assert (loaded.mean, loaded.variance) == (after.mean, after.variance)
# An independent consumer replays the numbers from the bytes alone and is granted no authority.
replayed = md.replay_decision(
    exported.artifact, exported.identity, execution=Execution(seed=SEED)
)
assert replayed == (after.mean, after.variance)

# -- 4. Withdraw the trial ------------------------------------------------------------------
verification = workflow.withdraw_cluster()
# The margin and the decision depended on the trial; the temporal and cost claims did not.
for key in ("cluster_trial", "margin_bridge", "decision"):
    assert verification.classify(key).klass == "withdrawn", key
for key in ("temporal_drift", "bayesian_cost"):
    assert verification.classify(key).klass == "still_valid", key
assert verification.classify("decision").authority_retained is False

# Historical numerical replay still succeeds, and gives exactly the sealed numbers.
assert (
    md.replay_decision(
        exported.artifact, exported.identity, execution=Execution(seed=SEED)
    )
    == replayed
)

# Current use of the decision refuses. Each refusal names the ledger standing, not a numerical fault.
for current_use in (
    workflow.decision,
    workflow.export_decision,
    lambda: workflow.load(exported.artifact, exported.identity),
    lambda: workflow.utility_only(3.0),
):
    refusal = refused(current_use)
    assert refusal.detail.startswith("evidence_ledger."), refusal.detail
    assert refusal.reason_code
    print("refused:", refusal.reason_code, refusal.detail)

# The surviving routes keep their estimates; only the claims that depended on the trial are gone.
assert workflow.routes.cost_mean == routes.cost_mean
assert workflow.routes.temporal_effect == routes.temporal_effect
