# Interpreting graph uncertainty and unidentified mass

Discovered structure is evidence, not a finished causal model. Antecedent keeps
compatible graphs in play, reports when an effect is not identified on part of
that belief, and refuses to invent orientations the data do not support. This
page is a practitioner reading guide; the [support matrix](support-matrix.md)
remains the license for which combinations may run.

Related mental-model pages: [a result is a claim](result-is-a-claim.md),
[refusal and partial knowledge](refusal-and-partial-knowledge.md), and the
discover → [`AcceptedGraph`](artifacts.md) spine.

## Exact DAG posterior vs PAG / partial graphs

Two common ways structural uncertainty shows up:

| Route | What the object means | How mass is handled |
| --- | --- | --- |
| Exact DAG posterior (`ExactDagPosterior`, MCMC posteriors) | A probability distribution over fully oriented DAGs consistent with the data and prior | Each atom is identified or not for the query; posterior weight on non-identified atoms is **`unidentified_mass`**, retained and not renormalized away |
| CPDAG / PAG (and temporal class graphs) | An equivalence class: undirected edges or circle marks stand for many completions | Completions are envelope atoms. Enumeration weight is **not** posterior probability unless you supply a class prior. Unidentified completion mass stays visible on the envelope |

Practical distinction:

- A **DAG posterior** answers “under these scored graphs, how much belief lands where the estimand is identified?”
- A **PAG / CPDAG envelope** answers “across completions of this partial graph, what identified set and residual unidentified mass remain?” Completing the class (review) is a separate scientific act from averaging over it.

Priors never upgrade an unidentified atom into an identified one. Failed estimation on an identified atom is **unevaluable** mass, not unidentified mass—keep those axes separate when reading diagnostics.

## Completion search budgets

Completions of a CPDAG or PAG are enumerated under a search budget, not under a count of
undirected edges, circle endpoints or nodes. The walk over endpoint assignments prunes every
subtree whose already-decided edges cannot be ancestral, so a CPDAG with more than 64
undirected edges, or a PAG with more than 16 circle endpoints, completes whenever its class
is structured enough for the pruning to keep the visited states within the budget (an
undirected tree on `n` nodes has exactly `n` members however large `n` is).

The budget bounds the states visited (and, for a PAG, the m-separation queries of the
equivalence audit), the live memory, and observes cancellation. Exhausting it is a typed
partial result, never a statement about the unvisited completions: the sampler reports the
bound that stopped it (`search_stop`) with a receipt of the limits in force, the work done
and the region left unexamined. A partial enumeration withdraws every class-wide claim: the
envelope is at most partially identified and carries the stop in its critical graph
features, and PAG class-wide m-separation answers `Undetermined`. The same holds when the
equivalence audit is skipped because a single member's m-separation signature (one query
per node pair and conditioning subset) would not fit the budget.

## Graph posteriors above six variables

`ExactDagPosterior` enumerates every labeled DAG and stops at **6** variables. For more,
`OrderMcmc` and `StructureMcmc` (and `CiScreenedPosterior`, which runs structure MCMC on the
pairs its screen kept) sample the same Gaussian-BIC pseudo-posterior over DAGs and accept up to
**16** variables. The posterior carries each atom's directed edges as a packed edge set whose
first word is the earlier 64-bit encoding unchanged: a graph on at most eight variables is
exactly the integer it was before, and a stored one decodes as it is. `GraphPosterior.adjacency`
in Python holds one Python `int` per atom, so graphs of up to 16 variables decode with ordinary
integer arithmetic (bit `from * (n_vars - 1) + (to if to < from else to - 1)` is the edge
`from -> to`). The DBN posterior keeps its own smaller bound, set by its lag-edge space. The
identity of a frozen posterior is built from each atom's canonical graph, not from the packed
integer, so it does not depend on the packing.

Conditions of the validated 10- and 12-variable workflow:

| Condition | Value |
| --- | --- |
| Variables | 10 or 12 (up to 16 run; cost below) |
| Parents per node | at most 3 (`max_parents` in the prior's constraints) |
| Chains | 4; above 8 variables each runs 8 replicas at inverse temperatures 1, 0.75, 0.56, 0.42, 0.32, 0.24, 0.18, 0.13 (the default 4-replica ladder 1, 1/2, 1/4, 1/8 is kept at 8 or fewer); replica exchange every 10 iterations |
| Warmup per chain | structure MCMC 3 000 iterations; order MCMC 6 000 |
| Retained draws per chain | structure MCMC 12 000, order MCMC 24 000, thinned by 2 |
| Publication gate | split R-hat below 1.2 over every edge indicator, bulk ESS above 10, every edge's Monte-Carlo standard error at most 0.05, every chain made a structural move, every chain's edge marginals agree with the pooled ones (at most 6 standard errors, with the excursion error for edges a chain rarely flips), and rank-normalized split R-hat of the log target at most 1.1 |

A schedule far shorter than this is refused by the gate rather than published.

Single-edge add/delete/reverse moves alone get trapped on interacting structures: in a ring of
colliders where each parent is shared by two children, chains settle in local modes 100 nats
below the truth and split R-hat stays far above 1.2. Both samplers therefore run replica
exchange, and structure MCMC adds a paired parent-set move:

- **Replica exchange.** Each chain runs one replica per entry of `temperatures`; replica `k`
  targets the posterior raised to `b_k`, and neighbouring temperatures exchange replicas with the
  Metropolis probability `min(1, exp((b_k - b_{k+1})(f_{k+1} - f_k)))`, so the `b = 1` replica
  (the only one recorded) has exactly the target as its stationary law. `temperatures = [1]` is
  a single chain. Order MCMC keeps its correction for the number of topological orders inside
  the tempered density, `(exp(score) / orders)^b`, so the order-modular bias stays removed.
- **Paired parent-set resampling** (structure MCMC, probability 0.1 per iteration; a
  generalisation of the new-edge-reversal move of Grzegorczyk and Husmeier, 2008). Two nodes are
  drawn uniformly and their parent sets are redrawn jointly from the exact conditional of the
  target given everybody else's parents. It contains an edge reversal together with the parent
  changes the reversal needs, is always accepted, and keeps the target invariant. It is disabled
  when a node has more than 4 096 allowed parent sets (set `max_parents`).
- **Equivalence-class and order moves** (above 8 variables). Both samplers reverse a covered
  edge with probability 0.25 per iteration, walking the Markov equivalence class whose members
  share a score; the Metropolis–Hastings ratio carries the ratio of covered-edge counts. Order
  MCMC also relocates one node to a uniformly drawn position in the order, a symmetric move
  that reshapes the order in one step where adjacent transpositions random-walk.
- **Start states.** Every chain after the first begins with a short random walk at inverse
  temperature 0.1, so chains start dispersed but biased toward the target, and R-hat compares
  chains that began in different places.

Acceptance cases at the published conditions: independent colliders and the interacting ring,
each at 10 and 12 variables, with both samplers clearing the gate and agreeing with each other,
and the interacting ring's twelve-edge skeleton compared with an exhaustive enumeration of its
3^12 states. On sparse random graphs at 10 and 12 variables, every posterior the gate publishes
matches the exact dynamic-programming marginals of the same score, prior and constraints within
its Monte Carlo error. These marginals are the posterior of the Gaussian BIC score; they are not
claimed to be calibrated against the true graph.

Chain diagnostics are reported per edge, not only as a worst case:
`GraphPosterior.edge_rhat`, `edge_ess` and `edge_mcse` (Python and Rust) hold the split R-hat,
bulk ESS and Monte-Carlo standard error of each `n_vars x n_vars` edge indicator across the
chains, aligned with `edge_marginals`. A cell whose indicator never varied reports R-hat 1.
`diagnostics.rhat_max` and `ess_bulk_min` are the worst of them. Read the cells before trusting
a marginal near 0.5: the gate bounds the worst edge, and it is the ambiguous edges that mix
slowest.

What is checked, and what is not:

- The samplers are compared with exhaustive enumeration at five variables (all 29 281 DAGs) and,
  at twelve, with an exhaustive enumeration of the 3^12 states of a twelve-pair candidate
  skeleton scored by a separate Gaussian-BIC implementation. Their proposal kernels are
  compared with a closed-form law with no data: the uniform prior on DAGs with at most one
  parent per node has `(n + 1)^(n - 1)` members and an exact edge-count law.
- Four independent, strongly identified 12-variable colliders are recovered through the multi-word encoding.
- `antecedent_discovery::exact_dag_marginals` computes the exact `P(i -> j)` of the same posterior (Gaussian BIC, the uniform or Bernoulli edge prior and every constraint, including the parent cap) for up to 15 variables by dynamic programming over node subsets, without enumerating DAGs: the sum over DAGs of a product of per-node factors is a sink recursion, and the sum over the DAGs containing an edge is the same recursion with that node's factor restricted. It agrees with an independent enumeration of every DAG to 1e-9 at five, six and seven variables (1.1 billion DAGs), and the published-condition samplers agree with it at 10 and 12 variables on 2,000 rows. This is the route behind `ExactDagPosterior().run(...)` from 7 to 15 variables (`discover_exact_dag_marginals`): it returns `ExactDagMarginals` (exact `P(i -> j)` and the log normalizing constant) and no graph atoms, so the list of graphs, weights, the MAP graph and the mixture that `analyze` takes over graphs are refused (`option_not_applicable`) rather than approximated; up to 6 variables the enumerating engine and its `GraphPosterior` are unchanged. More than 15 variables are refused naming `EXACT_MARGINAL_MAX_NODES`; larger problems are the samplers' alone.
- The paired move is checked against the exact conditional from every one of the 543 four-variable DAGs, and the whole samplers (paired moves alone, then the default ladder) against all 543 DAGs of four variables and all 29 281 of five.
- **No calibration is claimed.** A reported skeleton marginal of 0.7 does not mean that 70% of
  such pairs are adjacent: the weights are a BIC pseudo-posterior, not an integrated Bayesian
  posterior, and against simulated truths its low marginals are overstated or understated
  depending on how the true effect sizes are distributed (the printed diagnostic
  `exact_marginals_locate_the_skeleton_overstatement` in `antecedent-discovery`). What is
  checked is computational: the exact marginals against enumeration, and the published samplers
  against the exact marginals. The marginals are conditional on the prior and the parent cap.

## Uncertainty over a discovered structure

A frequentist search (PC, FCI, GES, LiNGAM, NOTEARS, PCMCI) returns one structure and no
sampling error. Two compiled routes carry the missing uncertainty. Both execute unlicensed
and neither has a coverage record.

**Posterior over Markov equivalence classes.** `EquivalenceClassPosterior(base)` (Rust:
`DiscoverySpec::equivalence_classes`) runs an `ExactDagPosterior`, `OrderMcmc`,
`StructureMcmc` or `CiScreenedPosterior` unchanged, replaces every DAG it lists by its
CPDAG and sums the mass of the DAGs that share one. The answer is a `GraphPosterior` whose
`atom_kind` is `"Cpdag"`, the Bayesian counterpart of PC and GES: a class the DAG posterior
favours is not split across its members, because the Gaussian BIC score cannot tell them
apart. Both directed bits of an unordered pair mark an undirected edge, so
`edge_marginals` count it in both directions. The class posterior is accepted as a
posterior and never as one DAG. `OrderMcmc(noise="laplace")` scores parent sets under Laplace noise, which is not
equivalent within a class, so the posterior can orient an edge the Gaussian posterior leaves
undirected (the Bayesian counterpart of LiNGAM and NOTEARS orientation); it is a
pseudo-posterior under that declared noise family. A `DbnPosterior` base collapses onto the
classes of each template's one-step window. `ExactPagPosterior` is the Bayesian counterpart of FCI and RFCI for at most four
variables: every maximal ancestral graph is fitted by residual iterative conditional fitting,
scored by BIC, and Markov-equivalent graphs are merged into their PAG, so the result is a
`GraphPosterior` with `atom_kind` `"Pag"` and `mark_masks` flagging the circle endpoints.
`PagMcmc` is its Metropolis counterpart for two to eight variables (single-pair link moves from
the empty graph, with ancestrality and maximality checks); `converged` records only that the
chains' class frequencies agree. Larger graphs refuse.

**Stability of the search.** `BootstrapStability(base, replicates)` (Rust:
`DiscoverySpec::bootstrap_stability`) re-runs the search on resamples drawn under the
declared sampling design: rows for independent subjects, circular blocks for a dependent
series (and each environment of a multi-environment study), whole units for a panel. LPCMCI and
J-PCMCI+ are re-run too, and regime PCMCI is re-run on block resamples that carry their regime labels, one table per regime. The result tabulates how often every node pair was
adjacent, how often it carried every pair of endpoint marks, and how often the search
returned every whole structure; `reference_edges` is the structure on the full sample. The
frequencies are a bootstrap distribution of the search, not a posterior. A resample on
which the search fails is counted in `replicates_failed` and never redrawn, and fewer than
half succeeding is refused. A table with a missing searched value, a masked or weighted
table and a declared design that does not sample independent rows are refused with the next
attempt named.

## Reading `unidentified_mass` and envelope cases

After `analyze(...)` with a graph posterior or class envelope, inspect the result
before projecting a scalar:

```python
result = analyze(...)
posterior = result.posterior  # may be None on some Frequentist routes
mass = posterior.unidentified_mass if posterior is not None else None
```

What to look for:

1. **`posterior.unidentified_mass`** (or `result` / HTML callouts that mirror it) — posterior (or class) weight where the query has **no** identified estimand. A value of `0.5` means half of the structural belief says the observational data alone cannot answer the question under those graphs.
2. **`posterior.effect_mean` / envelope means** — typically **`E[τ | identified]`**: a mixture only over atoms that identify the effect. That number is *not* “the ATE under all plausible graphs.”
3. **Answer shape `partial` / limitation `unidentified_mass`** — Antecedent withholds a single point mean when unidentified mass is material; the HTML banner says so explicitly.
4. **Envelope fields** (`posterior.envelope`, structural response identified sets) — lower/upper over identified atoms, plus the same unidentified mass axis. Informative when you need a set; still not a license to drop the unidentified share.

When is an envelope useful vs “don’t publish a point”?

- **Publish the split:** report identified-conditional effect *and* unidentified mass (and, if present, the identified set).
- **Do not publish a lone ROI / ATE** when unidentified mass is large enough that a decision would change if that mass resolved against you—unless you state the conditioning (“among structures where the effect is identified…”).
- **Envelope alone** is still incomplete if you hide the unidentified share; the set only covers identified completions.

Pinned contracts for the math and refusal behavior:

- [Bayesian graph effect envelope](conformance/bayesian__graph_effect_envelope.md) (`conformance/bayesian/graph_effect_envelope`) — weighted ensemble with known unidentified fraction; `renormalize_identified_only` refuses a 100% mixture after dropping mass.
- [PAG envelope unidentified mass](conformance/pag__envelope_unidentified_mass.md) (`conformance/pag/envelope_unidentified_mass`) — generalized-adjustment PAG envelope preserves unidentified completion mass.
- [Known-truth mixtures](conformance/bayesian__known_truth_mixtures.md) — analytic `E[τ | identified]` with retained unidentified mass (e.g. static mixture mean `2.625` with mass `0.2`).

## When `CausalReviewError` is raised

Discovery algorithms often return a CPDAG or PAG with pending marks. Estimation
that would require treating that partial graph as a finished DAG raises
`CausalReviewError` (also available as `ReviewRequired`). Structured attributes
include `kind`, `algorithm`, `pending_edge_count`, `hint`, and pending edge
records—see `python/tests/test_review_required_ux.py`.

Typical triggers:

- `analyze(..., discovery=FCI(...), accept_discovered=False)` (or incomplete
  review) when circle / undirected marks remain.
- Forcing a point estimate on a TemporalPag / MAG completion path with no
  identified mass (compile refusal rather than a silent NaN).

How to complete or accept a graph:

1. **Review pending marks** with subject-matter knowledge:
   `accepted.review({(u, v): ("tail", "arrow"), ...})` on an
   [`AcceptedGraph`](artifacts.md) handle (version bumps; the old handle stays).
2. **Supply an explicit DAG** you are willing to defend
   (`AcceptedGraph.from_graph([...])` or `graph=[(u, v), ...]`), as in
   `examples/python/discover_then_estimate.py`.
3. **Keep the class and use an envelope / posterior route** when you are not
   ready to orient—do not silently pick a MAP orientation to “unblock” the
   pipeline.
4. Use `antecedent.errors.next_action(err)` / `pending_edges(err)` for an
   actionable orientation hint in UIs.

Acceptance records review provenance; it does **not** prove the real-world
graph is true.

## Worked vignette: marketing channel structural uncertainty

Full walkthrough:
[`examples/notebooks/marketing_channel_structural_uncertainty.ipynb`](https://github.com/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/marketing_channel_structural_uncertainty.ipynb).

Synthetic weekly spend / demand / pipeline data are generated so the true effect
of paid-search spend on qualified pipeline is **1.5** (£k pipeline per £k
spend). PC recovers a fully undirected three-edge skeleton. An
`ExactDagPosterior` + Bayesian `analyze` then scores compatible DAGs.

On that pinned notebook run, the result is graph-dependent with:

**`posterior.unidentified_mass == 0.5` (50%).**

So half of the structural posterior sits on graphs where the spend→pipeline
effect is not identified from these observations. The reported
`posterior.effect_mean` (about **2.29** in the notebook) averages **only** the
identified half; it is not a 50/50 blend with “zero” or with the unidentified
region. A reviewed marketing DAG that adjusts for demand recovers a point near
the true **1.5**; a naive spend–pipeline slope (~**2.68**) ignores confounding.

### How to talk about this in a paper or dashboard

Prefer language that keeps the two uncertainties separate:

> Under the observational equivalence class scored by the exact DAG posterior,
> **50%** of posterior mass does not identify the paid-search effect. Conditional
> on the identified structures, the mixture mean is approximately **2.29** £k
> pipeline per £k spend. That figure is not the effect under all plausible
> graphs; the unidentified half must be reduced by design, measurement, or
> justified orientation before a single ROI is decision-grade.

Dashboard anti-copy: a single “incremental ROI” tile with a wider error bar.
Dashboard-appropriate: two readouts—**identified-conditional effect** and
**unidentified structural mass**—plus a link to the pending graph / review
state.

## Anti-patterns

- **Treating the MAP CPDAG (or MAP DAG) as truth.** The highest-scoring graph is
  still one atom. Reporting only its ATE discards competing orientations and
  hides unidentified mass.
- **Dropping or renormalizing away `unidentified_mass`.** Rescaling identified
  weights to sum to one manufactures a 100% mixture the data did not earn.
  Conformance explicitly refuses `renormalize_identified_only` for that reason.
- **Calling an envelope a confidence interval.** Identified-set / envelope width
  is structural disagreement among completions, not sampling noise alone.
- **Using `accept_discovered=True` as a substitute for review** when marks remain
  that the auto-accept path cannot resolve—you still owe orientations or an
  envelope-aware claim.
- **Equating “partial answer” with “no finding.”** A high unidentified mass is a
  positive scientific result: it tells you what experiment or assumption would
  move the decision.

## Further reading

- Notebook: [marketing channel structural uncertainty](https://github.com/iridae-dev/antecedent/blob/v2.4.1/examples/notebooks/marketing_channel_structural_uncertainty.ipynb)
- Conformance: [graph effect envelope](conformance/bayesian__graph_effect_envelope.md),
  [PAG envelope unidentified mass](conformance/pag__envelope_unidentified_mass.md),
  [known-truth mixtures](conformance/bayesian__known_truth_mixtures.md)
- UX pins: `python/tests/test_review_required_ux.py`
- Workflow: [Python workflow](python-workflow.md), [artifacts](artifacts.md)
