#!/usr/bin/env python3
"""Compare structure-learning methods on one simulated system whose graph is known.

Question: given only a table of five measured variables, which causal graph does each
discovery method return, how much of it is fixed by the data and how much by what the analyst
asserts, and how uncertain is the graph itself?

Data. Five variables with uniform (non-Gaussian) noise and a declared graph
``w -> x <- u``, ``u -> y -> z``. The only collider is at ``x``, so the Markov equivalence
class compels ``w -> x`` and ``u -> x`` and leaves the chain ``u - y - z`` undirected. The
script writes that class down by hand and scores each result against it with a structural
Hamming distance (SHD): one point for every pair whose adjacency or orientation differs.

What is checked. Score-based GES recovers the class (SHD 0 to the class, SHD 2 to the fully
directed DAG, because it cannot orient the chain). Background knowledge (tiers, a forbidden
link) orients the chain; a contradictory statement is refused with a typed reason. LiNGAM uses
the non-Gaussian noise to recover every direction (SHD 0 to the DAG). With ``u`` dropped from
the table, FCI returns a PAG that puts arrowheads at ``x`` and never claims ``x -> y`` or
``y -> x``, where GES reads the same table as ``y -> x``. A Bayesian posterior over graphs
(order MCMC) is checked against the exact enumeration at five variables: the skeleton and the
compelled colliders get posterior mass near 1, and the exact posterior splits the chain's
three equivalent DAGs one third each.

What this does NOT claim. Every result is conditional on the declared assumptions of its
method (linear dependence for the partial-correlation tests, no latent variables for GES,
LiNGAM and the posterior, acyclicity). Agreement with the truth on this simulation says
nothing about a real data set, where the true graph is unknown. No interval is reported.
"""

from __future__ import annotations

# %%
import itertools

import antecedent as ant
import numpy as np
from antecedent import discovery as disc
from antecedent.errors import CausalUnsupportedError

N = 3000
NAMES = ["w", "u", "x", "y", "z"]
TRUE_DAG = [("w", "x"), ("u", "x"), ("u", "y"), ("y", "z")]
COMPELLED = {("w", "x"), ("u", "x")}  # the collider at x; the chain u - y - z is free
PAIRS = list(itertools.combinations(sorted(NAMES), 2))
EXEC = ant.Execution(seed=7)


def simulate(seed: int = 11) -> dict[str, np.ndarray]:
    """Uniform unit-variance noise, so the system is linear but not Gaussian."""
    rng = np.random.default_rng(seed)
    noise = lambda: rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), N)  # noqa: E731
    w, u = noise(), noise()
    x = 0.8 * w + 0.8 * u + noise()
    y = 0.9 * u + noise()
    z = 0.9 * y + noise()
    return {"w": w, "u": u, "x": x, "y": y, "z": z}


def truth_states(directed: bool) -> dict[tuple[str, str], str]:
    """Pair -> '->', '<-', '--' or 'none' for sorted pairs; ``directed`` fixes every edge."""
    states = {}
    for a, b in PAIRS:
        state = "none"
        for s, t in TRUE_DAG:
            if (s, t) == (a, b) or (s, t) == (b, a):
                forward = (s, t) == (a, b)
                if directed or (s, t) in COMPELLED:
                    state = "->" if forward else "<-"
                else:
                    state = "--"
        states[(a, b)] = state
    return states


def result_states(result) -> dict[tuple[str, str], str]:
    """The same encoding for a discovery result's ``graph_edges`` (sorted pair order)."""
    symbol = {
        ("tail", "arrow"): "->",
        ("arrow", "tail"): "<-",
        ("tail", "tail"): "--",
        ("arrow", "arrow"): "<->",
    }
    states = {pair: "none" for pair in PAIRS}
    for e in result.graph_edges:
        marks = (e.at_source, e.at_target)
        if e.source > e.target:
            marks, a, b = (marks[1], marks[0]), e.target, e.source
        else:
            a, b = e.source, e.target
        states[(a, b)] = symbol.get(marks, "/".join(marks))
    return states


def shd(found: dict, truth: dict) -> int:
    return sum(found[p] != truth[p] for p in truth)


def show(label: str, states: dict) -> None:
    edges = [f"{a}{s}{b}" for (a, b), s in states.items() if s != "none"]
    print(f"{label:<28}", " ".join(edges))


data = simulate()
CLASS, DAG = truth_states(directed=False), truth_states(directed=True)
show("truth (class)", CLASS)

# %% [markdown]
# ## Score-based search: GES
#
# GES searches over equivalence classes, so it can only return what the data fix. The
# collider at `x` is compelled; the chain `u - y - z` has three equivalent orientations and
# stays undirected. Distance to the class is 0, distance to the fully directed DAG is 2.

# %%
ges = disc.GES(alpha=0.01, fdr=False).run(data, execution=EXEC)
ges_states = result_states(ges)
show("GES", ges_states)
print("SHD to class:", shd(ges_states, CLASS), "| SHD to DAG:", shd(ges_states, DAG))
assert shd(ges_states, CLASS) == 0
assert shd(ges_states, DAG) == 2

# %% [markdown]
# ## Background knowledge
#
# Tiers say a later tier cannot cause an earlier one: `w, u` come first, then `x, y`, then
# `z`. That orients the chain without any further data. A single forbidden link `y -> u`
# does the same for the pair it names. A statement that contradicts another is refused with
# a typed reason, never silently dropped.

# %%
tiers = disc.BackgroundKnowledge(tiers=(("w", "u"), ("x", "y"), ("z",)))
tiered = result_states(disc.GES(alpha=0.01, fdr=False, knowledge=tiers).run(data, execution=EXEC))
show("GES + tiers", tiered)
assert shd(tiered, DAG) == 0

forbid = disc.BackgroundKnowledge(forbidden=(disc.Link("y", "u"),))
forbidden = result_states(
    disc.GES(alpha=0.01, fdr=False, knowledge=forbid).run(data, execution=EXEC)
)
assert forbidden[("u", "y")] == "->"
print("forbidding y -> u orients u - y as:", forbidden[("u", "y")])

contradiction = disc.BackgroundKnowledge(required=(disc.Link("y", "u"),), tiers=(("u",), ("y",)))
try:
    disc.GES(alpha=0.01, fdr=False, knowledge=contradiction).run(data, execution=EXEC)
except CausalUnsupportedError as error:
    assert error.reason_code == "invalid_argument", error
    print("refused:", error.reason_code or type(error).__name__, "-", error)
else:
    raise AssertionError("a contradictory statement was accepted")

# %% [markdown]
# ## Functional-form search: LiNGAM
#
# The noise here is uniform, hence non-Gaussian, which is the assumption LiNGAM uses to tell
# `u -> y` from `y -> u`. Gaussian-score methods cannot make that distinction; LiNGAM can, and
# it recovers every direction of the DAG.

# %%
lingam = result_states(disc.LiNGAM().run(data, execution=EXEC))
show("LiNGAM", lingam)
print("SHD to DAG:", shd(lingam, DAG))
assert shd(lingam, DAG) == 0

# %% [markdown]
# ## Latent confounding: FCI against GES
#
# Now `u` is not measured. The table `w, x, y, z` has `x <- u -> y`, a latent common cause.
# FCI allows for that and returns a PAG: circles mark what the data leave open. GES assumes
# no latent variables and reads the collider at `x` as `y -> x`. The script asserts the FCI
# facts that hold for the true PAG: the true skeleton, an arrowhead at `x` on both of its
# edges, and no tail at `y` (which would say `y` causes `x`).

# %%
observed = {k: v for k, v in data.items() if k != "u"}
fci = disc.FCI(alpha=0.01, fdr=False).run(observed, execution=EXEC)
ges_hidden = disc.GES(alpha=0.01, fdr=False).run(observed, execution=EXEC)


def marks_at(result, a: str, b: str) -> tuple[str, str] | None:
    """Marks (at a, at b) of the edge between a and b, or None if they are not adjacent."""
    for e in result.graph_edges:
        if (e.source, e.target) == (a, b):
            return e.at_source, e.at_target
        if (e.source, e.target) == (b, a):
            return e.at_target, e.at_source
    return None


skeleton = {frozenset((e.source, e.target)) for e in fci.graph_edges}
assert skeleton == {frozenset(p) for p in [("w", "x"), ("x", "y"), ("y", "z")]}, skeleton
print("FCI  w-x :", marks_at(fci, "w", "x"), "| x-y :", marks_at(fci, "x", "y"))
print("FCI  y-z :", marks_at(fci, "y", "z"))
assert marks_at(fci, "w", "x")[1] == "arrow"
assert marks_at(fci, "y", "x")[1] == "arrow"
assert marks_at(fci, "x", "y")[1] != "tail"  # no claim that y is an ancestor of x
print("GES  y-x :", marks_at(ges_hidden, "y", "x"), "(tail at y, arrow at x reads as y -> x)")
assert marks_at(ges_hidden, "y", "x") == ("tail", "arrow")

# %% [markdown]
# ## A posterior over graphs
#
# Instead of one graph, order MCMC returns weights over DAGs and the probability of each
# edge. At five variables the exact enumeration is also available, so it serves as the
# reference. Under the Gaussian-BIC posterior the three DAGs of the chain are equivalent, so
# about one third of the mass has `u -> y`. Non-edges keep a little mass at finite n: `w` and
# `u` are independent parents of the collider `x`, so the posterior puts some weight on `w - u`
# (up to 0.15 is allowed here).

# %%
exact = disc.ExactDagPosterior().run(data, execution=EXEC)
mcmc = disc.OrderMcmc(n_chains=4, n_warmup=6000, n_draws=24_000, thin=2, max_parents=3).run(
    data, execution=EXEC
)
assert exact.names == mcmc.names
index = {name: i for i, name in enumerate(exact.names)}
p_exact = np.array(exact.edge_marginal_matrix())
p_mcmc = np.array(mcmc.edge_marginal_matrix())
print("order MCMC converged:", mcmc.converged, "| max edge MCSE:", max(mcmc.edge_mcse))
for s, t in TRUE_DAG:
    i, j = index[s], index[t]
    print(f"P({s}->{t}): exact {p_exact[i, j]:.3f}  mcmc {p_mcmc[i, j]:.3f}")
assert mcmc.converged

true_pairs = {frozenset(e) for e in TRUE_DAG}
for a, b in itertools.combinations(exact.names, 2):
    i, j = index[a], index[b]
    for p in (p_exact, p_mcmc):
        adjacency = p[i, j] + p[j, i]
        if frozenset((a, b)) in true_pairs:
            assert adjacency >= 0.95, (a, b, adjacency)
        else:
            assert adjacency <= 0.15, (a, b, adjacency)
for s, t in COMPELLED:
    assert p_exact[index[s], index[t]] >= 0.95
    assert p_mcmc[index[s], index[t]] >= 0.9
# The chain is not fixed by the data: one DAG in three has u -> y, two in three y -> z.
assert abs(p_exact[index["u"], index["y"]] - 1 / 3) < 0.06
assert abs(p_exact[index["y"], index["z"]] - 2 / 3) < 0.06
assert 0.1 < p_mcmc[index["u"], index["y"]] < 0.9
