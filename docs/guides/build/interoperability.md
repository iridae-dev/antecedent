# Work with other causal libraries

You can use Antecedent's identification beside another library's estimators, or bring a graph or a fitted estimate from another library into an Antecedent result. Both directions are handoffs: Antecedent does not import, call or wrap the other library, and each side keeps its own scientific claim.

The rule that keeps a handoff honest: whatever crosses the boundary keeps the status it had where it was produced. An adjustment set Antecedent identified stays an identification under the graph you declared. An estimate another library computed enters Antecedent as an attested, uncalibrated external result, and Antecedent does not give it a native licence or a calibration it did not measure.

## Which direction do you need?

| You have | You want | Use |
| --- | --- | --- |
| A graph and a question | A reviewed adjustment set to give a learner you fit yourself | `handoff.adjustment(result)` (the same function as `handoff.econml`). |
| A graph | The same graph in DoWhy, or a DoWhy graph in Antecedent | Graph interchange (DOT and NetworkX codecs), then identify under Antecedent. |
| An estimate another library computed | A labelled effect with lineage and support | `spec.attach(...)`, which binds through `external.bind`, or `external.bind` directly. |
| A Python package that must run | A record of exactly what ran, optionally verified | The provider registry, described in [connect scientific components and bring in outside results](scientific-interfaces.md). |

## Hand an adjustment set to EconML

Antecedent identifies the adjustment set and estimates the effect. You then fit the learner and own that fit. The handoff supplies the identified columns, so the learner never chooses the adjustment set.

```python
import antecedent as _ant_api
from antecedent import AverageEffect, Dag, analyze, identify
from antecedent.handoff import econml

graph = Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
query = AverageEffect(treatment="t", outcome="y")

identified = identify(graph=graph, query=query)
print(identified.status, identified.method, list(identified.adjustment_set))

result = analyze(
    data,
    graph=graph,
    query=query,
    refute=False,
    inference=_ant_api.Frequentist(bootstrap=0),
    execution=_ant_api.Execution(seed=11),
)
spec = econml(result.value)
assert spec.confounders == ("z",)
cols = spec.columns(data)  # Y, T and W (the adjustment columns) for the learner you fit
```

Here `data` is a table with columns `z`, `t` and `y`; the maintained example, [`examples/python/econml_cate_handoff.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/econml_cate_handoff.py), generates one with a known effect of 2 and fits `LinearDML` on `cols` when `econml` is installed. The learner and its CATE are yours; Antecedent supplies the adjustment columns and the identification status.

The handoff is licensed for point-identified backdoor and generalized-adjustment estimands on an explicit or accepted graph. Front-door, instrumental-variable, general-identification, partial-identification and graph-posterior results refuse instead of being flattened into a single adjustment set. Temporal columns are lag-aligned when the identification carries offsets.

To bring a fitted estimate back, `spec.attach(...)` records it as attested evidence with no native calibration. A scalar effect labelled `"ate"` is bound through `external.bind`, so the learner, its declared version, its support labels and its lineage travel with the result as `.claim`.

## Share a graph with DoWhy

Antecedent owns identification licensing, and DoWhy receives a graph and runs its own estimand and estimator stack. The two libraries do not produce byte-identical intervals, so compare the adjustment sets and the sign and rough size of the effect on a model where you know the answer.

From Antecedent to DoWhy:

1. Build a reviewed `Dag`, or wrap an `AcceptedGraph` after review.
2. Identify under Antecedent and record the adjustment set.
3. Export the graph with `Dag.to_dot()`, or the NetworkX codecs.
4. Build the DoWhy `CausalModel` from that graph string and the same table.

From DoWhy to Antecedent:

1. Start from a DoWhy-style DOT string or a NetworkX digraph.
2. Import with `Dag.from_dot(...)`.
3. Review it, then wrap it with `AcceptedGraph.from_graph(...)`.
4. Run `identify` or `analyze` on the accepted graph.

Discovery stays outside the estimate path: discover once, accept a graph, then estimate. Passing `discovery=` into `analyze` with `accept_discovered=False` raises a review error instead of inventing an adjustment set.

The runnable recipe, [`examples/python/dowhy_handoff.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/dowhy_handoff.py), runs the Antecedent half always and the DoWhy half when `dowhy` is installed. The full cookbook and the semantic differences are in [DoWhy interoperability](../../interop_dowhy.md). Four task guides compare an Antecedent route with the matching DoWhy function at the version recorded in `parity/baselines/dowhy.toml`: [intervention programs](../intervention-programs.md#differences-from-dowhy), [mechanism selection](../mechanism-selection.md#differences-from-dowhy), [intrinsic influence](../intrinsic-influence.md#differences-from-dowhy) and [graph falsification](../graph-falsification.md#differences-from-dowhy).

## Choose a library for the question

[Compare Antecedent with other libraries](../../comparison.md) describes where each library's strengths and scope lie, including estimators Antecedent does not ship and routes where Antecedent's native result carries a stated calibration or licence status. Read each result's `inspect().calibration` and support status rather than inferring them from the library that computed the number.
