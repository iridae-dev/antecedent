# Check a graph against data

Use this page when you have drawn a DAG by hand, or accepted one from discovery, and want to
know whether the data contradict the conditional independencies it asserts. Do this before
fitting anything to it. Call `antecedent.gcm.evaluate_graph`. It compares the graph with random
relabellings of its own nodes. A graph that explains the independencies in the data no better
than a relabelled copy of itself is rejected. It needs only the graph and a table with one
finite numeric column per node. **No mechanism is fitted. No likelihood is computed.**

```python
from antecedent.gcm import evaluate_graph

result = evaluate_graph(
    data,                                   # every column is a node
    [("z", "t"), ("z", "y"), ("t", "y")],   # (parent, child) edges
    ci_test="parcorr",                      # or "gpdc", "gsquared", "knn_dependence", ...
    alpha=0.01,                             # level for one implication (default 0.05)
    n_permutations=500,                     # relabellings to compare with (default 200)
    permutation_seed=7,                     # default: the execution's seed
)
result.outcome            # "rejected" | "not_rejected" | "uninformative" | "incomplete"
result.p_value, result.n_violations, result.evidence
result.permutation_seed, result.execution_seed, result.tests_run, result.max_ci_tests
result.limitations
```

In Rust, call `antecedent::gcm::evaluate_graph(&dag, &data, &config, &ctx)`. Both calls use the
engine `antecedent_model::falsify_graph`.

Read `result.outcome` first. Then read `result.limitations`, which states in words what the
verdict does not establish. `p_value` is `None` when the outcome is `"incomplete"` or
`"uninformative"`.

## What is tested

The local Markov property says each node is independent of its non-descendants given its
parents. The call lists it as pairwise statements. For every node `i` and every node `j` that
is neither a parent nor a descendant of `i`, the statement is `i ⟂ j | parents(i)`. Each one is
tested with the conditional-independence test you name. Any name a discovery call accepts is
allowed, nonlinear tests included. An implication counts as violated when its p-value is below
`alpha`. `result.evidence` lists every implication. Each implication has `node`, `other`, and
`conditioning`. It has `statistic`, `p_value`, and whether it was `violated`. `n_implications`
and `n_violations` are the totals.

## The permutation comparison

A violation count means little alone. A graph claiming many independencies is violated by
dependent data. Almost any graph is violated by dependent data. The same graph is therefore
relabelled at random. Those are `n_permutations` draws from the `permutation_seed` stream.
When `n!` does not exceed the request, every non-identity relabelling is used. The violations
are recounted on the same data. Write `v` for the violation count of your graph. Write
`v_1, ..., v_m` for the counts of the `m` relabellings (`result.permutation_violations`). Then

```
p_value = (1 + #{k : v_k <= v}) / (1 + m)
```

A small `p_value` means few relabellings do as well as your graph. It is compared with
`significance` (default 0.05). That level is separate from the per-implication `alpha`.

- **rejected**: `p_value` exceeds `significance`. The graph is not better than random
  relabellings of itself.
- **not_rejected**: significantly fewer violations than relabellings. This does not show the
  graph correct. Markov-equivalent graphs are indistinguishable.
- **uninformative**: nothing separates the graph from its relabellings. Either the graph
  implies no testable independence (a complete graph), or the share of relabellings Markov
  equivalent to it exceeds `max_equivalent_share`. Equivalence means the same skeleton and the
  same unshielded colliders, counting the graph itself. `max_equivalent_share` defaults to
  `significance`. Above that share the p-value cannot fall below `significance`. Tiny graphs
  are uninformative at the default level. Three nodes have only five other relabellings, so the
  equivalent share is at least 1/6.
- **incomplete**: a budget or cancellation stopped the run before `min_permutations`
  relabellings finished. No claim is made either way. `stop` and `search_receipt` name the
  bound. `tests_run` and `tests_skipped` account for the work.
  A stop after `min_permutations` still gives a verdict from the relabellings completed.
  `stop` and `search_receipt` are set.

## Tests, caching and budgets

Distinct `(x, y | z)` tests are cached across relabellings, since relabelled graphs repeat
queries. `max_ci_tests` caps the distinct tests as a search-budget operation. Cancellation is
observed at every charge. Tests that report permutation p-values draw `ci_replicates` null
replicates. Those tests are `gpdc` and `knn_dependence`. Their smallest attainable p-value is
`1 / (ci_replicates + 1)`. It must be below `alpha`, or the run refuses.

## Refusals

A budget or cancellation stop is the `"incomplete"` outcome. It is not an error. These requests
are refused before any comparison is made. The codes are described in the
[refusal reference](../reference/refusal-codes.md#structured-refusals-and-their-remedies).

| `reason_code` | Raised when | What to change |
| --- | --- | --- |
| `invalid_argument` | `alpha` or `significance` outside (0, 1); `max_equivalent_share` outside (0, 1]; `n_permutations`, `min_permutations` or `ci_replicates` zero, or `min_permutations` above `n_permutations`; fewer than two nodes; no rows; a graph column with a missing or non-finite value; two nodes on one variable | correct the named argument; drop or impute incomplete rows yourself, because rows are never dropped for you |
| `route_not_supported` | the named test reports a posterior probability, not a frequentist p-value; a node that is not a static variable | name a frequentist conditional-independence test; pass a static DAG |
| `execution_resource_limit` | the relabellings times the implications exceed the work limit | request fewer relabellings |
| `diagnostic_not_available` | the test returned a NaN p-value for an implication | use a test that is defined on these columns |

A permutation-null test whose smallest attainable p-value is not below `alpha` is refused as
unsupported. Raise `ci_replicates` or `alpha`.

## What it does not establish

- The violation count is not corrected for multiple testing and the tests are dependent. It is
  a comparison statistic, not a count of established violations.
- Failing to reject an implication is not evidence of independence. Partial correlation
  detects only linear dependence. On `y = x² + noise` it sees nothing. `gpdc` does.
- The null relabels the graph; it cannot distinguish Markov-equivalent graphs.
- Causal sufficiency and no selection are assumed.
- Every graph column must be observed and finite; rows are never dropped.
- A test with no power against the real dependence structure rejects every graph, correct or
  not, because no graph then beats its relabellings.

## Not the shuffled-outcome likelihood diagnostic

`ModelEvaluator::evaluate` on a fitted model reports a *permutation baseline*. One outcome
column is shuffled. The fitted model's mean log-likelihood is recomputed. That compares a
fitted model with noise. Graph permutation falsification fits nothing. It relabels the graph's
nodes. It compares the independencies the graph asserts with the data. The two answer
different questions. They share no statistic.

## Examples

[`examples/python/graph_falsification.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/graph_falsification.py)
and
[`examples/rust/graph_falsification.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/graph_falsification.rs)
compare a graph of four dependent pairs with 300 relabellings. The matching graph is not
rejected. A graph that pairs independent variables is rejected. A complete graph is
uninformative. In Python, a nonlinear dependence needs the nonlinear test. A test budget too
small to cover the implications gives no verdict. Both assert their results and run in the
example suites.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py` (skipped when DoWhy is not installed; every DoWhy value is
computed at test time).

Overlapping semantics. `gcm.falsify.falsify_graph` also tests the local Markov implications of the
graph and of its node relabellings. It asks whether the graph violates fewer of them. With the same
partial-correlation test, the same 5% level, and every relabelling, the two libraries give the same
verdict on a correct graph (not rejected). They give the same verdict on a graph whose nodes are
relabelled (rejected). They give the same verdict on a graph missing one true edge (still far
better than its relabellings, not rejected). DoWhy's `falsified` and `falsifiable` map to
`rejected`, `not_rejected`, and `uninformative`.

Where they differ.

- DoWhy compares the *fraction* of implications violated by each relabelling. This call compares
  the *count*. The two coincide when relabellings imply as many implications as the graph. They
  can differ otherwise.
- DoWhy calls a graph uninformative when the share of relabellings in its Markov equivalence class
  exceeds the significance level. Here the threshold is `max_equivalent_share`. It defaults to the
  same level.
- A budget or cancellation stop is a first-class `incomplete` outcome with a receipt. Every
  test is cached across relabellings. Every test is capped by `max_ci_tests`.
- The conditional-independence test is any name a discovery call accepts. A permutation-null
  test is refused when its smallest attainable p-value cannot reach `alpha`.
- Seeds (`permutation_seed`, `execution_seed`), the implications with their statistics, and the
  limitations of the verdict are part of the result.

## Related

- [Fit and query a structural causal model](gcm-estimands.md)
- [Give discovery background knowledge](discovery-background-knowledge.md)
- [Read graph uncertainty and unidentified mass](../graph_uncertainty.md)
