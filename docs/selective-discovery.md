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
* **Not published:** recovery of the true graph, a false-discovery or type-I error rate
  for the output, or an interval for a downstream estimate. `request_recovery_claim` and
  `request_downstream_interval` refuse with `cell_not_licensed`.

### Recorded calibration design (not measured)

The licensable claim is a statement about the whole method: discovery, graph acceptance,
and the downstream analysis, with the reuse of the discovery rows (or a sample split)
inside the procedure. The first calibration targets the sample-split design, because it
needs no post-selection adjustment. DGPs: linear-Gaussian SCMs with known graphs over the
declared variable count and density, including near-cancelling (weakly faithful)
parameters and a misspecified test. Probes: nominal downstream interval level, the
probability the discovered class contains the true graph, and the probability a declared
completion is a member of the true class, each with Monte Carlo error. Support range:
the measured sample-size and variable-count grid only.

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
| recovery or error-control claim for the output graph | no whole-method calibration | `cell_not_licensed` |
| interval for a downstream estimate on account of discovery | as above | `cell_not_licensed` |
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
