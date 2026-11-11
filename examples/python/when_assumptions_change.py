#!/usr/bin/env python3
"""Follow one causal claim from the graph that supported it to the data that stopped fitting.

Question: a subscription business asks how much weekly engagement changes cancellations. In
January a discovered and reviewed graph supports an estimate. In April a new table arrives.
Is the January claim still justified, and if not, what evidence would make the question
answerable again?

Data: simulated from two declared mechanisms over the same five measured columns (`price`,
`tenure`, `engagement`, `support`, `cancellation`, all standardized scores). In both, one unit of
engagement lowers the cancellation score by 0.8. In April an unmeasured cause is switched on
that raises engagement, support contacts and cancellations together (customers who depend on an
integration that has started failing). The causal effect of engagement is the same in both
months; what changes is whether the measured columns can recover it.

The assertions check: January's estimate is within 0.1 of -0.8; the same analysis on April's
table returns a different number; the staleness diagnostic recommends `reuse` on a second
January sample and `reidentify` on April's, naming the independence of engagement and support
that the graph implies and April contradicts; the whole-graph check agrees on that statement; a
resampled search almost never joins engagement and support in January and almost always does in
April; a search that allows hidden causes no longer orients engagement and cancellation; three
readings of April's table give two different estimates and one refusal; the evidence plan
certifies an experiment and not more rows of the same table; in the simulation more rows
keep the wrong answer while the experiment recovers -0.8; and with only three of the columns
the estimate moves just as far, the staleness diagnostic recommends `reuse`, and it names the
check it could not make.

What this does not claim: the checks here found the change only because the unmeasured cause
left traces among the other measured columns. A changed discovery result is not proof
that the earlier graph was wrong, the evidence plan is hypothetical identification, not an
estimate, and nothing here launches discovery, replaces an accepted graph or runs a study by
itself: each transition below is a call the host makes.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent import discovery as disc
from antecedent import drift, repair
from antecedent.gcm import evaluate_graph
from antecedent.outcomes import require_success

ROWS = 4000
TRUE_EFFECT = -0.8
NAMES = ["price", "tenure", "engagement", "support", "cancellation"]
QUERY = ant.AverageEffect("engagement", "cancellation")
EXEC = ant.Execution(seed=1)
ESTIMATE = {
    "query": QUERY,
    "inference": ant.Frequentist(bootstrap=0),
    "refute": False,
    "execution": EXEC,
}
FALSIFY = {"alpha": 0.01, "n_permutations": 500, "permutation_seed": 7}
record: list[tuple[str, str, str]] = []  # the host's history: (month, step, what came back)


def customers(
    seed: int, *, hidden: bool = False, randomized: bool = False, rows: int = ROWS
) -> dict[str, np.ndarray]:
    """One table of customers. `hidden` switches on the unmeasured cause; it is never returned."""
    rng = np.random.default_rng(seed)
    price, tenure = rng.normal(size=rows), rng.normal(size=rows)
    integration = rng.normal(size=rows) if hidden else np.zeros(rows)
    engagement = -0.6 * price + 0.5 * tenure + 0.6 * integration + rng.normal(size=rows)
    if randomized:  # an experiment sets engagement, whatever its usual causes are
        engagement = 1.3 * rng.normal(size=rows)
    support = 0.5 * price + 0.8 * integration + rng.normal(size=rows)
    cancellation = (
        TRUE_EFFECT * engagement
        + 0.5 * support
        + 0.3 * price
        + 0.8 * integration
        + rng.normal(size=rows)
    )
    return {
        "price": price,
        "tenure": tenure,
        "engagement": engagement,
        "support": support,
        "cancellation": cancellation,
    }


def marks(result, a: str, b: str) -> tuple[str, str] | None:
    """Endpoint marks (at a, at b) of the edge between a and b, or None if not adjacent."""
    for edge in result.graph_edges:
        if (edge.source, edge.target) == (a, b):
            return edge.at_source, edge.at_target
        if (edge.source, edge.target) == (b, a):
            return edge.at_target, edge.at_source
    return None


def implication(result, node: str, other: str):
    """The old graph's implied independence of `node` and `other`, as tested on one table."""
    return next(t for t in result.evidence if (t.node, t.other) == (node, other))


january = customers(1)
april = customers(2, hidden=True)

# %% [markdown]
# ## January: discover, review, estimate
#
# PC discovery returns what the data fix. Five edges come back directed and one, between
# `price` and `support`, does not. The direction of that edge is the reviewer's to supply,
# and the accepted graph records that it was reviewed.

# %%
discovered = disc.discover(
    january, method=disc.PC(alpha=0.01), accept_discovered=True, execution=EXEC
)
pending = [(edge.source, edge.target) for edge in discovered.pending]
print("left open by discovery:", pending)
assert pending == [("price", "support")]

accepted = discovered.review({("price", "support"): ("tail", "arrow")})
assert not accepted.pending and accepted.version > discovered.version
GRAPH = [
    ("price", "engagement"),
    ("tenure", "engagement"),
    ("price", "support"),
    ("price", "cancellation"),
    ("engagement", "cancellation"),
    ("support", "cancellation"),
]

study = ant.prepare(january, graph=accepted, **ESTIMATE)
first = require_success(ant.estimate(study))
print(f"January effect {first.answer.value:.3f} (simulated truth {TRUE_EFFECT})")
print("calibration:", first.calibration.status)
assert abs(first.answer.value - TRUE_EFFECT) < 0.1
record.append(("January", "estimate under the reviewed graph", f"{first.answer.value:.2f}"))

# %% [markdown]
# ## April: the same analysis, a different number
#
# Rebinding the retained study to April's table repeats January's analysis exactly: same
# graph, same adjustment, new rows. It runs without complaint and returns a much smaller
# effect. Nothing in that computation is wrong, and it will reproduce.

# %%
rerun = require_success(ant.estimate(ant.rebind(study, data=april)))
print(f"April, January's analysis: {rerun.answer.value:.3f}")
assert abs(rerun.answer.value - first.answer.value) > 0.3
record.append(("April", "January's analysis on new rows", f"{rerun.answer.value:.2f}"))

# %% [markdown]
# ## Is the January claim stale?
#
# The staleness diagnostic compares the new table with what the plan was prepared on. Most of
# what it checks is coverage: covariate distributions, the range of the treatment and the
# adjustment set, missingness, schema. April passes all of that. It also repeats, on the new
# rows, the test of every conditional independence the plan's graph implies. January's graph
# says engagement and support are independent once price is held fixed. A second January
# sample agrees. April's table does not, and the recommendation is to identify again.

# %%
policy = drift.StalenessPolicy(alpha=0.01)
steady = require_success(drift.staleness(study, customers(3), policy=policy))
stale = require_success(drift.staleness(study, april, policy=policy))
print("second January sample:", steady.recommendation)
print("April:", stale.recommendation)
for finding in stale.findings:
    print(f"  {finding.kind}: {finding.subject} (statistic {finding.statistic:.3f})")
print(stale.next_attempt)
assert steady.recommendation == "reuse" and not steady.findings
assert stale.recommendation == "reidentify"
assert {finding.kind for finding in stale.findings} == {"implied_independence"}
assert "engagement _||_ support | price" in {finding.subject for finding in stale.findings}
record.append(("April", "staleness diagnostic", stale.recommendation))

# %% [markdown]
# ## How wrong is the graph as a whole?
#
# The staleness finding is about single statements. `evaluate_graph` asks a different
# question of the whole graph: does it explain the table better than random relabellings of
# its own nodes? In April it still does. "Not rejected" only says the graph is better than
# chance, which a partly wrong graph easily is. Its list of implications shows the same
# failure the staleness diagnostic found.

# %%
control = evaluate_graph(customers(3), GRAPH, **FALSIFY)
checked = evaluate_graph(april, GRAPH, **FALSIFY)
for label, result in (("second January sample", control), ("April", checked)):
    failed = [(t.node, t.other) for t in result.evidence if t.violated]
    print(
        f"{label}: {result.outcome}, {result.n_violations} of {result.n_implications}",
        failed,
    )
assert control.outcome == checked.outcome == "not_rejected"
assert control.n_violations == 0
assert not implication(control, "engagement", "support").violated
broken = implication(checked, "engagement", "support")
assert broken.violated and broken.p_value < 1e-6
record.append(("April", "old graph's implications", f"{checked.n_violations} violated"))

# %% [markdown]
# ## Is that sampling noise?
#
# Two tables can give two graphs with nothing having changed. Re-running the search on
# resamples of each table shows how often an edge appears at all. Engagement and support are
# almost never joined in January and almost always joined in April, which sampling variation
# does not explain.

# %%
stability = disc.BootstrapStability(disc.PC(alpha=0.01), replicates=200)


def joined(table, a: str, b: str) -> float:
    edges = stability.run(table, execution=EXEC).edges
    return next(e.frequency for e in edges if {e.source, e.target} == {a, b})


january_rate = joined(january, "engagement", "support")
april_rate = joined(april, "engagement", "support")
print(f"engagement and support joined: January {january_rate:.3f}, April {april_rate:.3f}")
assert january_rate < 0.05 and april_rate > 0.9

# %% [markdown]
# ## Discover again, allowing for what is not measured
#
# FCI does not assume every common cause is in the table. On January's rows it still reads
# engagement as a cause of cancellation. On April's rows it leaves both ends of that edge as
# circles: the data no longer say which way it points, or whether something unmeasured sits
# behind both.

# %%
fci = disc.FCI(alpha=0.01, fdr=False)
before, after = fci.run(january, execution=EXEC), fci.run(april, execution=EXEC)
print("January engagement-cancellation:", marks(before, "engagement", "cancellation"))
print("April   engagement-cancellation:", marks(after, "engagement", "cancellation"))
print("April   engagement-support:     ", marks(after, "engagement", "support"))
assert marks(before, "engagement", "cancellation") == ("tail", "arrow")
assert marks(before, "engagement", "support") is None
assert marks(after, "engagement", "cancellation") == ("circle", "circle")
assert marks(after, "engagement", "support") is not None
record.append(("April", "rediscovery", "engagement-cancellation direction open"))

# %% [markdown]
# ## Three readings of April, three answers
#
# Rediscovery does not hand back one graph. A reviewer can read the new link in three ways,
# and each is a different causal claim:
#
# - engaged customers contact support more (`engagement -> support`);
# - support contacts change how much customers use the product (`support -> engagement`);
# - something unmeasured drives engagement, support and cancellation together.
#
# The first two are complete graphs, so they can be tested and estimated. The third declares
# an unmeasured common cause, and the question is put to it as it stands.

# %%
readings = {
    "engagement -> support": [*GRAPH, ("engagement", "support")],
    "support -> engagement": [*GRAPH, ("support", "engagement")],
}
answers = {}
for label, edges in readings.items():
    fit = evaluate_graph(april, edges, **FALSIFY)
    effect = require_success(ant.analyze(april, graph=edges, **ESTIMATE)).answer.value
    answers[label] = effect
    print(f"{label}: {fit.outcome}, {fit.n_violations} violated, effect {effect:.3f}")
    record.append(("May", f"reading: {label}", f"{fit.outcome}, {effect:.2f}"))

hidden_cause = ant.Admg.from_edges(
    NAMES,
    GRAPH,
    [
        ("engagement", "support"),
        ("engagement", "cancellation"),
        ("support", "cancellation"),
    ],
)
refused = ant.analyze(april, graph=hidden_cause, **ESTIMATE)
print("unmeasured common cause:", refused.refusal.code)
assert not refused.accepted and refused.refusal.code == "effect_not_identified"
record.append(("May", "reading: unmeasured common cause", refused.refusal.code))

# The two testable readings disagree with each other, and neither is the simulated truth.
low, high = sorted(answers.values())
assert high - low > 0.15
assert all(abs(value - TRUE_EFFECT) > 0.15 for value in answers.values())

# %% [markdown]
# ## What would settle it
#
# Under the third reading the refusal is the answer to the causal question. Asked to
# describe, the same call also returns what evidence the failure determines. Two studies are
# declared with their costs: ten thousand more rows of the table already in hand, and an
# experiment that sets engagement. Each is checked against the identification theorem of its
# route.
#
# The unmeasured cause appears in the plan under a placeholder name. Nothing here can say what
# it is.

# %%
experiment = repair.StudyCandidate.experiment(
    "engagement experiment",
    population="study",
    interventions=["engagement"],
    measured=["cancellation"],
    cost=40,
    sample_size=2000,
    recruitment="random assignment",
    timing="next quarter",
    unit="customer",
    cost_unit="usd_thousand",
    sample_budget=2000,
    feasible=True,
)
more_rows = repair.StudyCandidate.observation(
    "ten thousand more rows",
    population="study",
    measured=NAMES,
    cost=5,
    sample_size=10_000,
    recruitment="billing export",
    timing="next quarter",
    unit="customer",
    cost_unit="usd_thousand",
)
described = ant.analyze(
    april,
    graph=hidden_cause,
    on_refusal="describe",
    evidence_candidates=[experiment, more_rows],
    **ESTIMATE,
)
needs = described.evidence_plan
actions = [action["id"] for action in needs.next_actions]
print("next actions:", actions)
assert "run_experiment:engagement" in actions
assert "measure_variable:common_cause_of_engagement_cancellation" in actions

plan = needs.plan
print(plan.explain())
assert plan.outcome == "closing" and plan.best is not None
assert plan.best.labels == ("engagement experiment",)
assert plan.best.gate == "verified_sufficient"
assert plan.unlocks.unverified == ("ten thousand more rows",)
assert (plan.ranking_guarantee, plan.calibration) == ("point_only", "not_measured")
assert described.associational.answer.kind == "associational"
record.append(("May", "evidence plan", f"certified: {plan.best.labels[0]}"))

# %% [markdown]
# ## Run both studies in the simulation
#
# Because the mechanism is declared, both candidates can be carried out. More rows of the
# same table sharpen the wrong number. The experiment assigns engagement at random, so in its
# table engagement has no causes to adjust for, measured or not, and the estimate returns to
# the simulated effect.

# %%
larger = require_success(
    ant.analyze(customers(6, hidden=True, rows=14_000), graph=GRAPH, **ESTIMATE)
)
print(f"14,000 observational rows: {larger.answer.value:.3f}")
assert abs(larger.answer.value - TRUE_EFFECT) > 0.3

trial = customers(5, hidden=True, randomized=True, rows=2000)
randomized = require_success(
    ant.analyze(
        {name: trial[name] for name in ("engagement", "cancellation")},
        graph=[("engagement", "cancellation")],
        **ESTIMATE,
    )
)
print(f"2,000 randomized rows:     {randomized.answer.value:.3f} (simulated truth {TRUE_EFFECT})")
assert abs(randomized.answer.value - TRUE_EFFECT) < 0.1
record.append(("Next quarter", "engagement experiment", f"{randomized.answer.value:.2f}"))

# %% [markdown]
# ## The same change with nothing to catch it
#
# The checks above had something to test because the unmeasured cause also moved `support`,
# and because `tenure` moves engagement without touching cancellation. Keep only `price`,
# `engagement` and `cancellation` and the graph over them implies no independence at all.
# The estimate moves exactly as before. The staleness diagnostic recommends `reuse`, and
# lists the graph's implications among what it could not evaluate, with the reason.

# %%
KEPT = ("price", "engagement", "cancellation")
SMALL = [
    ("price", "engagement"),
    ("price", "cancellation"),
    ("engagement", "cancellation"),
]
small_january = {name: january[name] for name in KEPT}
small_april = {name: april[name] for name in KEPT}
small_study = ant.prepare(small_january, graph=SMALL, **ESTIMATE)
small_before = require_success(ant.estimate(small_study)).answer.value
small_after = require_success(ant.estimate(ant.rebind(small_study, data=small_april))).answer.value
small_stale = require_success(drift.staleness(small_study, small_april, policy=policy))
small_check = evaluate_graph(small_april, SMALL, **FALSIFY)
print(f"three columns: January {small_before:.3f}, April {small_after:.3f}")
print("staleness:", small_stale.recommendation, "| graph check:", small_check.outcome)
untested = dict(small_stale.not_evaluated)
print("not evaluated:", untested)
assert abs(small_before - TRUE_EFFECT) < 0.1 and abs(small_after - TRUE_EFFECT) > 0.3
assert small_stale.recommendation == "reuse"
assert "implies no conditional independence" in untested["implied_independence"]
assert small_check.outcome == "uninformative"
assert small_check.uninformative_reason == "no_testable_implication"

# %% [markdown]
# ## The record the host kept
#
# None of these steps triggered the next one. A host application decided when to check, when
# to search again and which reading to put the question to, and kept this history.

# %%
for month, step, outcome in record:
    print(f"{month:<13}{step:<36}{outcome}")
