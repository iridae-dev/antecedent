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
[`examples/notebooks/marketing_channel_structural_uncertainty.ipynb`](https://github.com/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_channel_structural_uncertainty.ipynb).

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

- Notebook: [marketing channel structural uncertainty](https://github.com/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_channel_structural_uncertainty.ipynb)
- Conformance: [graph effect envelope](conformance/bayesian__graph_effect_envelope.md),
  [PAG envelope unidentified mass](conformance/pag__envelope_unidentified_mass.md),
  [known-truth mixtures](conformance/bayesian__known_truth_mixtures.md)
- UX pins: `python/tests/test_review_required_ux.py`
- Workflow: [Python workflow](python-workflow.md), [artifacts](artifacts.md)
