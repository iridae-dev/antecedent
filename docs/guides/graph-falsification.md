# Falsifying a graph by node permutation

`antecedent.gcm.evaluate_graph` (Rust: `antecedent::gcm::evaluate_graph`, engine
`antecedent_model::falsify_graph`) asks whether a DAG is better than random relabellings of
itself at explaining the conditional independencies in your data. It needs only the graph and
the data. **No mechanism is fitted and no likelihood is computed.**

```python
from antecedent.gcm import evaluate_graph

result = evaluate_graph(
    data,                                   # every column is a node
    [("z", "t"), ("z", "y"), ("t", "y")],   # (parent, child) edges
    ci_test="parcorr",                      # or "gpdc", "gsquared", "knn_dependence", ...
    alpha=0.01,                             # level for one implication
    n_permutations=500,
    permutation_seed=7,
)
result.outcome            # "rejected" | "not_rejected" | "uninformative" | "incomplete"
result.p_value, result.n_violations, result.evidence
result.permutation_seed, result.execution_seed, result.tests_run, result.max_ci_tests
result.limitations
```

## What is tested

The local Markov property says each node is independent of its non-descendants given its
parents. The route lists it as pairwise statements: for every node `i` and every node `j` that
is neither a parent nor a descendant of `i`, the statement `i ⟂ j | parents(i)`. Each one is
tested with the conditional-independence test you name (any name the discovery routes accept,
nonlinear tests included) and counts as violated when its p-value is below `alpha`.
`result.evidence` lists every implication with its statistic and p-value.

## The permutation comparison

A violation count means little alone: a graph claiming many independencies is violated by
dependent data, but so is almost any graph. The same graph is therefore relabelled at random
(`n_permutations` draws from the `permutation_seed` stream; when `n!` does not exceed the
request, every non-identity relabelling is used) and the violations are recounted on the same
data. `p_value` is `(1 + #relabellings with at most as many violations) / (1 + #relabellings)`.

- **rejected**: `p_value` exceeds `significance`; the graph is not better than random
  relabellings of itself.
- **not_rejected**: significantly fewer violations than relabellings. This does not show the
  graph correct: Markov-equivalent graphs are indistinguishable.
- **uninformative**: nothing separates the graph from its relabellings. Either the graph
  implies no testable independence (a complete graph) or the share of relabellings Markov
  equivalent to it (same skeleton and unshielded colliders, counting the graph itself) exceeds
  `max_equivalent_share`, which defaults to `significance`: above that share the p-value cannot
  fall below `significance`. Tiny graphs are uninformative at the default level: three nodes have only five other
  relabellings, so the equivalent share is at least 1/6.
- **incomplete**: a budget or cancellation stopped the run before `min_permutations`
  relabellings finished. No claim is made either way; `stop` and `search_receipt` name the
  bound, and `tests_run` and `tests_skipped` account for the work.
  A stop after `min_permutations` still gives a verdict from the relabellings completed, with `stop`
  and `search_receipt` set.

## Tests, caching and budgets

Distinct `(x, y | z)` tests are cached across relabellings, since relabelled graphs repeat
queries. `max_ci_tests` caps the distinct tests as a search-budget operation, and cancellation
is observed at every charge. Tests that report permutation p-values (`gpdc`, `knn_dependence`)
draw `ci_replicates` null replicates; their smallest attainable p-value is
`1 / (ci_replicates + 1)` and must be below `alpha`, or the run refuses.

## What it does not establish

- The violation count is not corrected for multiple testing and the tests are dependent. It is
  a comparison statistic, not a count of established violations.
- Failing to reject an implication is not evidence of independence. Partial correlation
  detects only linear dependence: on `y = x² + noise` it sees nothing, `gpdc` does.
- The null relabels the graph; it cannot distinguish Markov-equivalent graphs.
- Causal sufficiency and no selection are assumed.
- Every graph column must be observed and finite; rows are never dropped.
- A test with no power against the real dependence structure rejects every graph, correct or
  not, because no graph then beats its relabellings.

## Not the shuffled-outcome likelihood diagnostic

`ModelEvaluator::evaluate` on a fitted model reports a *permutation baseline*: one outcome
column is shuffled and the fitted model's mean log-likelihood is recomputed. That compares a
fitted model with noise. Graph permutation falsification fits nothing, relabels the graph's
nodes, and compares the independencies the graph asserts with the data. The two answer
different questions and share no statistic.

## Examples

`examples/rust/graph_falsification.rs` and `examples/python/graph_falsification.py` compare a graph
of four dependent pairs with 300 relabellings: the matching graph is not rejected, a graph that
pairs independent variables is rejected, a complete graph is uninformative, and (Python) a
nonlinear dependence needs the nonlinear test and a test budget too small to cover the
implications gives no verdict. Both assert their results and run in the example suites.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py` (skipped when DoWhy is not installed; every DoWhy value is
computed at test time).

Overlapping semantics. `gcm.falsify.falsify_graph` also tests the local Markov implications of the
graph and of its node relabellings and asks whether the graph violates fewer of them. With the same
partial-correlation test, the same 5% level and every relabelling, the two libraries give the same
verdict on a correct graph (not rejected), on a graph whose nodes are relabelled (rejected) and on
a graph missing one true edge (still far better than its relabellings, not rejected). DoWhy's
`falsified` and `falsifiable` map to `rejected`, `not_rejected` and `uninformative`.

Where they differ.

- DoWhy compares the *fraction* of implications violated by each relabelling; this route compares
  the *count*. The two coincide when relabellings imply as many implications as the graph and can
  differ otherwise.
- DoWhy calls a graph uninformative when the share of relabellings in its Markov equivalence class
  exceeds the significance level; here the threshold is `max_equivalent_share`, defaulting to the
  same level.
- A budget or cancellation stop is a first-class `incomplete` outcome with a receipt, and every
  test is cached across relabellings and capped by `max_ci_tests`.
- The conditional-independence test is any name the discovery routes accept, with permutation-null
  tests refused when their smallest attainable p-value cannot reach `alpha`.
- Seeds (`permutation_seed`, `execution_seed`), the implications with their statistics, and the
  limitations of the verdict are part of the result.
