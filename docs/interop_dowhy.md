# DoWhy ↔ Antecedent interoperability

Use this page to move a causal graph between DoWhy and Antecedent and compare
what each library does with it. DoWhy is a model–identify–estimate–refute
surface with a large Python estimator ecosystem; Antecedent is
identification-first and refuses to publish an estimate on an unreviewed or
unidentified structure. This page is a **handoff cookbook**, not a wrapper:
Antecedent does not import or call DoWhy estimators, and each library keeps its
own claim.

Use the same small backdoor SCM in both libraries:

- confounder `z`, binary treatment `t`, continuous outcome `y`
- edges `z → t`, `z → y`, `t → y`
- structural average treatment effect \(+2\) (linear-Gaussian noise; propensity
  selection on `z`)

The runnable recipe is
[`examples/python/dowhy_handoff.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/dowhy_handoff.py).
`dowhy` is a **soft dependency**: the Antecedent half always runs; the DoWhy
half is skipped when `dowhy` is not installed.

Graph interchange uses Antecedent’s DOT / NetworkX codecs on `Dag` (see
[artifacts](artifacts.md#graph-interchange-non-artifact)). Do not wrap DoWhy
inside Antecedent APIs.

## Antecedent → DoWhy

1. Build a reviewed `Dag` (or wrap an `AcceptedGraph` after review).
2. Identify under Antecedent (`identify` / `analyze`) and record the certified
   adjustment set.
3. Export `Dag.to_dot()` (or `Dag.to_networkx_node_link()` / `Dag.to_networkx_adjacency()`, which
   return NetworkX node-link or adjacency JSON as a string).
4. Construct a DoWhy `CausalModel` with that graph string and the same table.
5. Call DoWhy `identify_effect` / `estimate_effect` for a familiar estimand
   (for example `backdoor.linear_regression`).

Antecedent owns identification support; DoWhy receives a graph string and
runs its own estimand/estimator stack. Compare adjustment sets and effect
**sign** (and, on this toy SCM, magnitudes near \(+2\)), not byte-identical
intervals.

## DoWhy → Antecedent

1. Start from a DoWhy-style DOT string for the same backdoor DAG, or a
   NetworkX digraph serialized as node-link or adjacency JSON.
2. Import with `Dag.from_dot(...)` (or `Dag.from_networkx_node_link(...)` /
   `Dag.from_networkx_adjacency(...)`).
3. Review, then wrap it: `AcceptedGraph.from_graph(dag, algorithm_id=...)`, where
   `algorithm_id` records where the graph came from.
4. Run `identify` / `analyze` under Antecedent’s publication rules
   (for example `estimator="propensity.weighting"`).

Unreviewed discovery stays outside the estimate path: discover once, accept a
DAG, then estimate (see
[`discover_then_estimate.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/discover_then_estimate.py)). Passing
`discovery=` into `analyze` with `accept_discovered=False` raises a review
error rather than inventing an adjustment set.

## Semantic differences

| Concern | DoWhy (typical) | Antecedent |
| ------- | --------------- | ---------- |
| Identification | Often advisory: `proceed_when_unidentifiable=True` continues into estimation | Hard gate: unidentified queries refuse to publish a point estimate |
| Discovery | Graph may be assumed or loosely attached | Unreviewed discovery refuses; estimation uses an `AcceptedGraph` |
| Partial structure | Easy to collapse to a single numeric answer | Retains unidentified / graph-dependent mass (for example `graph_dependent` identification status, structural unidentified mass on class envelopes) |

Where Antecedent's results are compared with DoWhy's, the comparison is pinned
to DoWhy 0.14 (`parity/baselines/dowhy.toml`). Four task guides state the
differences route by route: [intervention programs](guides/intervention-programs.md#differences-from-dowhy),
[mechanism selection](guides/mechanism-selection.md#differences-from-dowhy),
[intrinsic influence](guides/intrinsic-influence.md#differences-from-dowhy) and
[graph falsification](guides/graph-falsification.md#differences-from-dowhy).

## Soft dependency

```bash
python -m pip install antecedent          # required
python -m pip install dowhy               # optional, for the DoWhy half
python examples/python/dowhy_handoff.py
```

If `import dowhy` fails, the example prints a skip notice and still asserts the
Antecedent adjustment set `{z}` and an ATE within 0.5 of the true +2.
