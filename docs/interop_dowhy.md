# DoWhy ↔ Antecedent interoperability

DoWhy and Antecedent are complementary: DoWhy is a familiar
model–identify–estimate–refute surface with a large Python estimator ecosystem;
Antecedent is identification-first and refuses to publish an estimate on an
unreviewed or unidentified structure. This page is a **handoff cookbook**, not a
wrapper. Antecedent does not import or call DoWhy estimators.

Use the same small backdoor SCM in both libraries:

- confounder `z`, binary treatment `t`, continuous outcome `y`
- edges `z → t`, `z → y`, `t → y`
- structural average treatment effect \(+2\) (linear-Gaussian noise; propensity
  selection on `z`)

The runnable recipe is
[`examples/python/dowhy_handoff.py`](https://github.com/iridae-dev/antecedent/blob/1.11/examples/python/dowhy_handoff.py).
`dowhy` is a **soft dependency**: the Antecedent half always runs; the DoWhy
half is skipped when `dowhy` is not installed.

Graph interchange uses Antecedent’s DOT / NetworkX codecs on `Dag` (see
[artifacts](artifacts.md#graph-interchange-non-artifact)). Do not wrap DoWhy
inside Antecedent APIs.

## Antecedent → DoWhy

1. Build a reviewed `Dag` (or wrap an `AcceptedGraph` after review).
2. Identify under Antecedent (`identify` / `analyze`) and record the certified
   adjustment set.
3. Export `Dag.to_dot()` (or NetworkX adjacency / node-link JSON).
4. Construct a DoWhy `CausalModel` with that graph string and the same table.
5. Call DoWhy `identify_effect` / `estimate_effect` for a familiar estimand
   (for example `backdoor.linear_regression`).

Antecedent owns identification licensing; DoWhy receives a graph string and
runs its own estimand/estimator stack. Compare adjustment sets and effect
**sign** (and, on this toy SCM, magnitudes near \(+2\)), not byte-identical
intervals.

## DoWhy → Antecedent

1. Start from a DoWhy-style DOT string (or NetworkX digraph) for the same
   backdoor DAG.
2. Import with `Dag.from_dot(...)` (or the NetworkX codecs).
3. Review, then wrap `AcceptedGraph.from_graph(...)`.
4. Run `identify` / `analyze` under Antecedent’s publication rules
   (for example `estimator="propensity.weighting"`).

Unreviewed discovery stays outside the estimate path: discover once, accept a
DAG, then estimate (see `examples/python/discover_then_estimate.py`). Passing
`discovery=` into `analyze` with `accept_discovered=False` raises a review
error rather than inventing an adjustment set.

## Semantic differences

| Concern | DoWhy (typical) | Antecedent |
| ------- | --------------- | ---------- |
| Identification | Often advisory: `proceed_when_unidentifiable=True` continues into estimation | Hard gate: unidentified queries refuse to publish a point estimate |
| Discovery | Graph may be assumed or loosely attached | Unreviewed discovery refuses; estimate clicks use an `AcceptedGraph` |
| Partial structure | Easy to collapse to a single numeric answer | Retains unidentified / graph-dependent mass (for example `GraphDependent` status, structural unidentified mass on class envelopes) |

Frozen DoWhy 0.14 conformance fixtures remain the authoritative record for
scoped parity pins (`parity/baselines/dowhy.toml`). Prefer this exercised
backdoor ATE recipe over unexercised general-ID oracles.

## Soft dependency

```bash
python -m pip install antecedent          # required
python -m pip install dowhy               # optional, for the DoWhy half
python examples/python/dowhy_handoff.py
```

If `import dowhy` fails, the example prints a skip notice and still asserts the
Antecedent adjustment set `{z}` and a positive ATE.
