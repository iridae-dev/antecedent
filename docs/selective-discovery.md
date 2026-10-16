# Selective discovery

Discovery is the expensive first stage of many analyses, and it is rerun for reasons
that leave most of its work unchanged: a different threshold, added background
knowledge, a different orientation budget, a different lag window. Selective discovery
reuses the conditional-independence statistics a run computed, at exactly the input
coordinates where a later request needs them, and reruns the decisions and the
orientation on top. It makes one claim: a selective run returns what a full run of the
same request returns, bit for bit. It makes no claim that the result is the true graph.

Code: `antecedent_discovery::selective` (contracts, stages, store, adapters, portable
state), `antecedent_io_core::selective_discovery_artifact` (artifact kind
`selective_discovery_v1`), `antecedent::analysis::selective_discovery` (stage planner,
claim graph, ledger and transport integration).

## Frozen contracts

A static contract (`StaticDiscoveryContract`: PC, FCI over a table) and a temporal
contract (`TemporalDiscoveryContract`: PCMCI, PCMCI+ over a series) are separate types.
Each fixes everything about a run except the data:

| Input | Static | Temporal |
|---|---|---|
| searched variables and their order | yes | yes |
| declared population (exactly one) | yes | yes |
| CI test and its settings (named, or declared with an identity) | yes | yes |
| significance method, mask type | yes | yes |
| threshold, multiplicity control | yes | yes |
| required, forbidden, tier and vector-group background knowledge | yes | yes |
| conditioning-set limit, parent limit | yes | yes |
| Possible-D-Sep and discriminating-path budgets | FCI | n/a |
| lag bounds and the lagged-frame depth | n/a | yes |
| random seed (the context must carry it) | yes | yes |

The data enter a run as a *data identity*: the content of each searched column as bit
patterns, its name and type, the analysis mask and weights, and (temporal) the sampling
index.

## Stages and their identities

| Stage | Depends on | Reused when |
|---|---|---|
| `statistics` | data, variable order, population, test, significance, seed, kernel policy, mask, lag-frame depth | all of those are identical |
| `search` | statistics, algorithm, threshold, multiplicity, background knowledge, search limits | its own inputs and statistics are identical |
| `orientation` | search, orientation settings | likewise |
| `graph_output` | orientation | likewise |

A stage's identity is the digest of its own declared inputs and the identity of the
stage it reads (`StageIdentity` of the stage planner), so a change anywhere upstream is
visible downstream, and the plan names the changed inputs.

## What a request reuses

* Every stage identity equal to the retained one: the retained outcome is returned. No
  statistic is computed and no search runs.
* Otherwise the algorithm runs again in full, on a test that answers a request from a
  retained statistic exactly when the test, its settings, the seed, the kernel policy,
  the column indexes and the content of every column read are identical. The saved work
  is the statistics.
  * A threshold, multiplicity or background-knowledge change reruns the decisions and
    the adaptive test schedule over retained statistics.
  * A change of the data, the test, the seed, the kernel policy, the variable order,
    the mask or the lag-frame depth retains nothing. New data invalidates the
    statistics; a previous graph is never the start of a search.
* A changed value at an unchanged shape is a different coordinate: nothing is served
  from the old value.
* A candidate runs on a fork of the store and is published only on success. A refusal,
  an algorithm failure or a cancellation leaves the previous state, outcome and
  identities exactly as they were.

The skeleton state between runs is not retained: the algorithms interleave orientation
with the final search passes and expose no resumable state. That stage is a closed row.

## Observed work units

Counted where the work happens, not at a wrapper call:

* CI tests handed to the test implementation (attempted, completed), and requests
  answered from a retained statistic (reused, never counted as executed work);
* orientation fixed-point drives, rounds and applied orientations, counted inside the
  shared fixed-point driver every orientation scheduler runs through;
* search passes, from the per-depth records the search loops emit;
* score evaluations: always zero. Selective reuse is constraint based.

Speculative batch work that the search later discards is completed work and is counted;
it is not in the algorithm's own consumed count.

## Graph uncertainty is preserved

The outcome keeps the structure in its class (CPDAG, PAG, temporal DAG, temporal CPDAG),
each edge's marks, the unresolved edges with their reasons (undirected within the class,
circle endpoint, conflict, bidirected, search state not definite), orientation
conflicts, and the diagnostics of every exhausted search bound. A representative member
of an equivalence class is never reported as the discovered graph.

### What reaches identification, prepared fits, transport bindings and decisions

* `accepted_graph_stage_identity` is the only way a discovered structure becomes the
  `Graph` input of the stage planner. It requires a structure with no unresolved edge or
  an explicitly declared completion (a label and an identity; recorded as a declaration,
  not a discovery), and it refuses a search that exhausted a bound.
* `add_discovery_claims` places the discovery stages in the claim-level invalidation
  graph. New data recomputes the graph and everything resting on it (the structure is
  reused and the fit is not retained); a threshold change keeps the statistics and
  recomputes what rests on the graph; a changed structure or completion recomputes; an
  exhausted search bound makes the graph unsupported and withdraws every claim resting
  on it.
* The discovery rows are declared in the evidence ledger by dataset snapshot and row
  index (`discovery_data_source`). Reusing them in a later analysis is the ledger's
  shared-observation overlap; a disjoint split is independent.

### Transport under graph uncertainty (decision)

A discovered structure cannot supply a selection diagram. `request_selection_diagram`
refuses and names every unresolved edge, the exhausted bounds, and what would license
the request. Three decisions are recorded:

1. A CPDAG/PAG or search-incomplete graph returns a **refusal with the unresolved
   edges**. An envelope of transport answers across the members of the equivalence
   class is **not offered**: it multiplies identification work by the class size, says
   nothing about which member is true, and would be read as one answer.
2. A resolved, search-complete structure is still a discovered estimate. The caller
   accepts it as the causal graph (or declares a completion) and then declares the
   selection diagram.
3. **Multi-population discovery is out.** A pooled or per-population search carries its
   own invariance and identification assumptions, which belong to the transport
   contract and are not inferable from data. The contract refuses a second population or
   multi-dataset constraints with `route_not_supported`.

## Portable state and resume

`export_state` produces a configuration and stage-identity record, the statistic
receipts (coordinate digest, statistic, p-value, degrees of freedom and interval as bit
patterns) and the graph and result identities. It carries no raw data. The artifact
(`selective_discovery_v1`) is a checksummed container; a consumer recomputes the receipt,
graph, contract and artifact identities and refuses a stored identity that differs, an
unknown feature or label, and a retained artifact identity that does not match.

Resume needs the raw data and the same contract and kernel policy. The algorithm is
replayed over the receipts only, a statistic the state does not hold is a refusal, and
the replayed result must have the recorded identity. Missing raw data, other data, a
different contract, an unusable receipt (non-finite, a p-value outside `[0, 1]`, a
repeated coordinate) and a replay that reaches another result are each refused with
`discovery_statistics_invalid`. A receipt the replay never consumes is retained but is
not re-derived.

Limits: Portable receipts: 250000. Structure items: 65536.

## Algorithmic match, and what is not claimed

* **Published:** selective equals full on masks, variable permutations, CI-test changes,
  lag changes, thresholds, background knowledge and exhausted budgets, through the
  result identity and the per-edge evidence bits, against independent known-graph
  oracles (a d-separation oracle over hand-written DAGs).
* **Not published, and not licensable per dataset:** recovery of the true graph, a
  false-discovery or type-I error rate for the output. Whether a discovered graph is the
  true graph rests on faithfulness and causal sufficiency, and neither can be tested from
  the data discovery ran on, so no calibration of the procedure turns the statement into
  one about a given dataset. `request_recovery_claim` refuses with `cell_not_licensed`.
* **Not published:** an interval for an estimate computed on the rows discovery used.
  `request_downstream_interval` refuses with `cell_not_licensed`; selecting the graph and
  estimating on the same rows is not a construction with a measured coverage.

### The interval that is licensed: the whole method on a row split

`antecedent::analysis::sample_split_effect::estimate_effect_on_split` (Python:
`selective_discovery.sample_split_effect`) runs one fixed construction: a seeded half
split, PC with the partial-correlation test (alpha 0.01) on the discovery half, acceptance
of the structure only when the search finished and it is one fully directed graph, and the
linear effect of a treatment on an outcome estimated on the analysis half by adjusting for
the treatment's parents, with a 0.95 least-squares interval. A structure with any
unresolved edge reports no interval (`split_effect.structure_unresolved`); no completion
is part of the construction.

The coverage statement is about the *method*: over linear-Gaussian, causally sufficient,
faithful generating processes of five to seven variables whose equivalence class is fully
orientable, at the measured analysis row counts, how often the interval contains the true
structural coefficient. It is keyed `structure = discovered_sample_split` and never
inherits a known-graph record. `SplitEffect.measure` replays the source and attests the
interval from the current coverage record; without one the interval is a computed value,
and `measure` refuses with `cell_not_licensed`.

The measured family, the gated test and the named boundaries (weak edges, a cancelling
edge, a test that cannot see the confounder, and the reuse of the discovery rows) are in
`crates/antecedent/tests/sample_split_effect_calibration.rs`. Each boundary is recorded
under its own key and licenses nothing.

### Recovery as a measured table

How often PC returns exactly the true graph over that family is printed by the same
harness (`info recovery` lines) and published here as a table, not as a license. A row
states the variable count, the rows given to discovery and the share of replicates whose
discovered graph equals the true one. The table is filled from the measurement at the
release cut and carries the commit it was measured at (`4ae60f8b5`). It is a description of
how the search behaves, not a license: the license is the coverage record of the interval.

| Variables | Discovery rows | Replicates | Graph recovered | Interval reported |
| --- | --- | --- | --- | --- |
| 5 to 7 | 300 | 400 | 0.950 | 0.960 |
| 5 to 7 | 600 | 400 | 0.975 | 0.978 |
| 5 to 7 | 1,200 | 400 | 0.975 | 0.975 |

## Lifecycle

* **prepare / estimate:** a contract and data produce an outcome and a plan; accepting
  a graph feeds the stage planner.
* **refresh with new outcome-only data:** keeps the contract, recomputes the statistics
  (new data), and recomputes every claim reading the data; no fit or interval is
  carried across.
* **export / load:** the portable state and its artifact; resume needs the raw data.

## Closed rows

| Row | Reason | Refusal |
|---|---|---|
| recovery or error-control claim for the output graph of one dataset | faithfulness and causal sufficiency are untestable | `cell_not_licensed` |
| interval for an estimate computed on the rows discovery used | no construction with a measured coverage | `cell_not_licensed` |
| score-based search reuse | no score-cache injection seam | `route_not_supported` |
| skeleton-state resume between runs | algorithms expose no resumable state | not offered |
| multi-population discovery | assumptions belong to the transport contract | `route_not_supported` |
| transport of a discovered structure | an estimate is not a declared graph | `discovered_structure_unresolved` |
| derived sufficient-statistic updates | a separate row with its own validation | not offered |

## Reach

Inference mode: none (a discovered structure and its reuse accounting; no estimate, no
interval, no posterior). Paradigm: neither Frequentist nor Bayesian output is produced.

| Consumer | State | Why |
|---|---|---|
| decision | ADAPTER | through an accepted graph into identification and the decision, point-only; the claim graph reports when it must be recomputed or is withdrawn |
| inverse query | REFUSED | a discovered structure supplies no response to invert |
| sensitivity | REFUSED | sensitivity needs a declared graph; accept or complete the structure first |
| design ranking | REFUSED | no quantity to rank |
| counterfactual | ADAPTER | through an accepted graph, under the counterfactual route's own premises |
| transport | REFUSED | decision recorded above; unresolved edges named |
| temporal | DIRECT | the temporal adapter and contract |
| composition | DIRECT | the claim-level invalidation graph and the evidence ledger |

## Python surface

`antecedent.selective_discovery`: `StaticContract`, `TemporalContract`, `SelectiveDiscovery`
(`run`, `plan`, `outcome`, `export_state`, `resume`, `request_recovery_claim`,
`request_downstream_interval`, `request_selection_diagram`), `sample_split_effect` and `SplitEffect` (with `measure`), `Outcome` (structure with
`unresolved_edges`, `directed_edges`, evidence, bounded-search receipts), `RunReport` and
`WorkUnits`, `inspect_selective_discovery_artifact`, `consume_selective_discovery_artifact`,
`request_score_search_reuse` and `SelectiveDiscoveryRefusal`. Contract seeds default to 0.
