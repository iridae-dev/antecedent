# Give discovery background knowledge

State what you already know about the graph so that discovery respects it. Background knowledge is what you assert about the graph before the data are consulted:
links that must be there, links that cannot be, an ordering of the variables, and a limit
on how many parents a variable may have. The ordinary discovery classes (`PC`, `FCI`,
`RFCI`, `GES`, `PCMCI`, `PCMCIPlus`) and the selective-discovery contracts read the same statement, through one
parser, and give it one digest.

```python
from antecedent.discovery import PC, BackgroundKnowledge, Link

knowledge = BackgroundKnowledge(
    required=(Link("dose", "response"),),          # dose -> response (a direct cause)
    forbidden=(Link("response", "age"),),          # response does not cause age
    tiers=(("age", "sex"), ("dose",), ("response",)),  # later tiers cannot cause earlier ones
)
result = PC(knowledge=knowledge).run(data)
result.background_knowledge        # hex digest; None when no knowledge was given
```

A `Link` is `source[t - source_lag] -> target[t - target_lag]` by variable name, with both
lags 0 in a static search. A statement an algorithm cannot honour is **refused**, never
ignored: the refusal carries a reason code (`invalid_argument` for statements that
contradict each other, `unknown_variable` for a name the search does not cover,
`option_not_applicable` for a statement with no meaning for the algorithm) and a detail that
names the statement.

## What each algorithm does with each statement

| Statement | PC | FCI and RFCI | GES | PCMCI | PCMCI+ |
| --- | --- | --- | --- | --- | --- |
| required `a -> b` | adjacency kept without a test (also past FDR); edge oriented `a -> b` before colliders; Meek rules propagate it | adjacency kept without a test; edge `a -> b` (tail at `a`, arrow at `b`: a direct cause among the observed variables is an ancestor); Zhang rules propagate it | edge seeded, never deleted or reversed; oriented `a -> b` on the final CPDAG, then Meek rules | `a[t-lag] -> b[t]` is kept through parent selection, exempt from the parent limit, and reported with its own MCI statistic and p-value even when the threshold would drop it | as PCMCI; a contemporaneous link is also oriented `a -> b` before the majority collider rule, and Meek rules propagate it |
| forbidden `a -> b` | with `b -> a` also forbidden, no edge; with only `a -> b` forbidden the pair may be adjacent and an adjacent pair is oriented `b -> a` | both directions forbidden: no edge; only `a -> b` forbidden: an arrowhead at `a`, the source of the forbidden link (no tail there), and nothing at `b` | the Insert or Reverse that makes `a` a parent of `b` is skipped; if the pair still ends up adjacent, it is oriented `b -> a` | the link is not a candidate and is not scored | as PCMCI; a contemporaneous pair with one direction forbidden stays an edge and is oriented in the allowed direction |
| tiers | `tier(a) > tier(b)` forbids `a -> b` at every lag, so any adjacent pair across tiers is oriented from the earlier to the later tier | the same prohibition read as an arrowhead only: an arrowhead at the later-tier end and no tail, because a latent common cause may still connect the pair | as PC | as forbidden links, at every lag | as forbidden links |
| `max_parents` | a search restriction (below): no variable keeps more than `max_parents` adjacencies that could be its parents | the same restriction | enforced: an Insert or Reverse whose scored parent set (directed parents, the undirected neighbours scored with them, and the new parent) exceeds the limit is skipped | bounds the parent set of each target kept by parent selection, strongest first, required links exempt; MCI still scores every allowed candidate, so the reported links may be more than the limit | as PCMCI |
| contradiction (`a -> b` required and forbidden, forbidden by a tier, required links forming a directed cycle, or more required parent adjacencies than `max_parents`) | refused | refused | refused | refused | refused |
| a name the search does not cover | refused | refused | refused | refused | refused |
| a lagged link in a search over a table | refused | refused | refused | n/a | n/a |
| a required link outside the lag window (or not at the target's own time) | n/a | n/a | n/a | refused | refused |

**The parent limit in PC, FCI and RFCI.** An equivalence class leaves the direction of most
edges open, so the only bound that holds for every graph in the class is on the adjacencies
that could be parents: those not fixed as children by knowledge (`v -> m` required, or the
reverse forbidden). After the adjacency search, while a variable exceeds the limit, its
removable adjacency with the weakest dependence evidence (largest retained p-value) is dropped;
required adjacencies are never dropped, and more required parent adjacencies than the limit is a
contradiction. *What this changes in the guarantees:* a dropped adjacency was not found
independent, so the output no longer follows from the CI decisions alone. Where the limit binds
the skeleton can lose true edges, the usual soundness of the orientations does not extend to the
triples around the dropped pairs (they carry no separating set, so the collider rule treats them
as undecided), and every removal is reported in a `*.parent_limit_removed_adjacencies`
diagnostic. Where the limit never binds nothing changes. The bound is a degree bound on possible
parents, so it can be stricter than a bound on the parents of the true DAG; the guarantee it gives
is that every DAG in the returned class respects the limit.

Two notes on reading the table.

* **Direction knowledge and the data can disagree.** Knowledge orients the edges the
  data leave undirected. When the data compel the opposite direction (a collider, or an
  orientation that would close a cycle with the knowledge), PC, GES and FCI do not choose
  silently: the edge is marked as a conflict and the result carries the diagnostic
  `pc.orientation_conflict`, `ges.orientation_conflict` or `fci.orientation_conflict`.
* **PCMCI+ orients by knowledge as PC does.** Fixed contemporaneous directions are applied
  before the majority collider rule, so a collider that disagrees is reported as an orientation
  conflict instead of overriding them. A pair with only one direction forbidden is kept as an
  edge (the usual both-directions-must-survive rule would otherwise discard it).
* **LiNGAM and NOTEARS (Rust only; the Python classes take no `knowledge=`).** Both return a DAG and enforce
  every statement exactly: a forbidden direction (or a later tier) is excluded from the
  candidate edges, a required link is kept, and `max_parents` caps each variable's parents
  (required links exempt in LiNGAM, re-added in NOTEARS). They run the same checks as GES and
  record the digest in their algorithm configuration.
* **A forbidden link is a statement about the output graph.** In FCI, "no `a -> b`"
  removes only that tail; it does not claim that `a` has no causal path to `b` through
  latent variables. Only a required link, which states a direct cause, puts a tail in the
  PAG.

## Identity

The digest of the statements (`DiscoveryResult.background_knowledge`,
`AcceptedGraph.background_knowledge`, and the `background_knowledge` input of a selective
contract, which declares the parent limit under its search limits) is a function of the required and forbidden links as sets, the tiers, the vector
groups and the parent limit. Reordering the links leaves it unchanged; any other change
alters it. A result that was given knowledge also records the digest in its algorithm
configuration, and an accepted graph serializes it in `to_json`, so a graph discovered under
different knowledge is a different artifact. `AcceptedGraph.rediscover` and `accept()` run
the configuration they are handed, knowledge included. A study prepared on a discovery that ran under knowledge states it in its contract:
`discovery_algorithm` reads `pc+knowledge:<digest>` (the bare algorithm id without knowledge), and a
refresh or a replayed preparation keeps it. A refusal that names a statement names its variables
by the names you wrote, not by position.

## Splitting rows for discovery and analysis

`selective_discovery.discover_on_split` filters first and splits second. The order is part
of the method:

1. rows outside the analysis mask (`mask=`) are excluded;
2. of the rest, rows with a missing value in a searched variable are excluded;
3. the excluded rows are recorded in the receipt by original index and belong to neither
   part;
4. a seeded permutation of the *eligible* rows assigns `round(fraction * n_eligible)` of
   them to discovery. An explicit rule names original row indices and is refused if it names
   an excluded row.

Weights (`weights=`) travel with their rows, unchanged and unnormalized. Discovery on a
weighted table needs the weighted partial-correlation test (`ci_test="weighted_parcorr"`),
which reads the discovery part's weights; weights under any other test, or that test without
weights, are refused. The analysis part's weights, its original row ids and the Kish
effective sample size `(sum w)^2 / sum w^2` are on the `SampleSplit`
(`analysis_weights`, `analysis_row_ids`, `analysis_effective_sample_size`) and in the
receipt. See [Selective discovery](../selective-discovery.md#splitting-rows-filter-then-split)
for the estimands, normalization and what the receipt verifies.

## Related

- [Rerun discovery without recomputing it](../selective-discovery.md)
- [Read graph uncertainty and unidentified mass](../graph_uncertainty.md)
- [Discovery, attribution and simulation as compiled routes](compiled-structure-routes.md)
