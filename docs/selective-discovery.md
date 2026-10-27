# Rerun discovery without recomputing it

Discovery is the expensive first stage of many analyses, and it is rerun for reasons
that leave most of its work unchanged: a different threshold, added background
knowledge, a different orientation budget, a different lag window. Selective discovery
reuses the conditional-independence statistics a run computed, at exactly the input
coordinates where a later request needs them, and reruns the decisions and the
orientation on top. It makes one claim: a selective run returns what a full run of the
same request returns, bit for bit. It makes no claim that the result is the true graph.

The portable state is exported as the artifact kind `selective_discovery_v1`. The Python surface is `antecedent.selective_discovery`; the Rust implementation is in `antecedent_discovery::selective`.

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

## Background knowledge

Background knowledge (`required`, `forbidden`, `tiers`, `max_parents`) is read by the
contracts and by the ordinary classes (`PC`, `FCI`, `RFCI`, `GES`, `PCMCI`, `PCMCIPlus` take
`knowledge=BackgroundKnowledge(...)`) through one wire parser, and has one digest: the
`background_knowledge` input of the contract is the same value as
`DiscoveryResult.background_knowledge`. A result's digest also covers the parent limit when one is given (a contract declares it under its search limits). What each
algorithm does with each statement, and which statements it refuses, is one table, in
[Background knowledge in discovery](guides/discovery-background-knowledge.md#what-each-algorithm-does-with-each-statement).
In short:

| Statement | PC | FCI | PCMCI | PCMCI+ |
|---|---|---|---|---|
| required `a -> b` | adjacency kept untested, oriented `a -> b`, Meek propagates | adjacency kept untested, tail at `a` and arrow at `b`, Zhang rules propagate | kept through parent selection, exempt from the parent limit, reported with its own statistic | as PCMCI, and a contemporaneous link is oriented `a -> b` before the collider rule |
| forbidden `a -> b` | no edge if both directions are forbidden, else the pair is oriented `b -> a` | no edge if both are forbidden, else an arrowhead at `a` only | not a candidate | not a candidate |
| tiers | forbid the later tier causing the earlier; orient across tiers | arrowheads only | forbid at every lag | forbid at every lag |
| `max_parents` | restricts the adjacency search to at most `max_parents` possible-parent adjacencies per variable, reported when it binds | the same | caps the parent set parent selection keeps | as PCMCI |

A contract refuses, with a `selective_discovery.background_knowledge_*` detail: statements
that contradict each other (`invalid_argument`), a variable the contract does not search
(`unknown_variable`), and a statement with no meaning for the algorithm
(`option_not_applicable`: a lagged link in a static contract,
a required link outside the lag window).

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

### Splitting rows: filter, then split

`antecedent::analysis::sample_split_analysis::discover_on_split` (Python:
`selective_discovery.discover_on_split`) partitions the rows of a static table into a
discovery part and an analysis part. The order of operations is fixed:

1. **Filter by the analysis mask.** Rows outside the mask are excluded.
2. **Filter to complete cases** over the searched variables. A row of the mask with a missing
   cell in any of them is excluded. Covariates outside the searched variables keep their
   missing cells in the analysis part.
3. **Record the exclusions.** The receipt lists the masked-out rows and the incomplete rows
   by original index; they belong to neither part.
4. **Split the eligible rows.** A seeded permutation permutes only the eligible rows and
   gives `round(fraction * n_eligible)` of them to discovery. An explicit rule names original
   row indices; naming an excluded row is refused (`selective_discovery.split_row_excluded`).

Filtering happens before the split so that completeness and the mask cannot decide a row's
part, and so that the discovery part is a table the discovery algorithm can read without
dropping rows (a weighted test needs one weight per row).

*Preserved through the split.* Each part holds the original row id of every row
(`SplitRows::discovery_row_ids` / `analysis_row_ids`, `SampleSplit.analysis_row_ids`), the
weights of its rows, and the all-included analysis mask when the source had a mask. The
receipt records the exclusions, the identity of the source mask and of the source,
discovery and analysis weights, and the content identity of the source and both parts. The identity version is unchanged: exclusions, mask and weights enter the receipt identity only when present, so a split of a complete, unmasked, unweighted table keeps the identity it always had.

*Weights.* Weights are gathered with their rows, bit for bit and unnormalized, so a part's
weights sum to whatever its rows held. Zero weights are legitimate and stay in their part (they contribute nothing to a weighted
statistic and the downstream estimate counts them as excluded); a part without positive weight
is refused (`split_part_weight_invalid`).

* **Estimand.** The weights define the population the rows represent. Weighted discovery
  (`ci_test="weighted_parcorr"`, read from the discovery part's own weights; weights under any
  other test, or that test without weights, are refused) targets the conditional-independence
  structure of that weighted population. The split is exchangeable in the weighted sense only
  when the weights are fixed design weights rather than functions of the outcomes, which is
  the caller's assertion.
* **Normalization** belongs to the consumer, because it depends on what the weights mean
  (a weighted mean divides by the sum of the analysis weights; a population total does not).
  The analysis weights are exposed unscaled, with the source digest to show they were not
  altered.
* **Effective sample size.** A weighted interval uses the Kish size `(sum w)^2 / sum w^2`
  of the analysis weights (`antecedent_kernels::kish_ess`), not the row count; both parts'
  sizes are recorded in the receipt (`SplitWeights`).
* **Downstream estimation.** `SplitDiscovery::estimate_analysis_effect` (Rust) estimates the
  linear effect of a treatment given a declared adjustment set on the analysis part under that
  part's own weights: the effect in the weighted population the weights represent, conditional
  on the selected graph. Weights are normalized to mean one over the rows used, so rescaling
  them moves neither the estimate nor its sampling-weighted HC1 standard error. Rows of weight
  zero are excluded from the estimate and counted (`SamplingWeightRecord::zero_weight_rows`),
  and the effective sample size of the rows used is reported beside the row count.
  `SplitAnalysisEffect::interval` is the Student interval at the effective sample size less
  the regression's coefficients; its whole-method coverage is `unmeasured` until
  `tests/sample_split_weighted_effect_calibration.rs` (10 variables, 3,000 participants,
  weights of coefficient of variation 0.8) has been measured.

*What verification checks.* `SampleSplitReceipt::verify` re-derives the partition from the
rule and the recorded exclusions and checks every stored identity (rows, exclusions, mask,
weights, source and part contents). `SampleSplitReceipt::verify_against_source` replays the
split on the table it was made from and refuses a changed mask
(`split_mask_mismatch`), changed weights (`split_weights_mismatch`) or any other change
(`split_source_mismatch`). `SplitRows::verify` checks the parts themselves: their original
row ids (`split_rows_mismatch`), masks, weights and content. `check_analysis_table` accepts
only the recorded analysis part, weights included.

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

## Effect after discovery on a treatment parent set

`antecedent::analysis::sample_split_parent_set_effect::estimate_parent_set_effect` (Python:
`selective_discovery.sample_split_parent_set_effect`) is the construction for sparse studies of
ten to twenty connected variables whose equivalence class is not fully orientable. It differs
from the fully directed construction above in one step, the acceptance rule, and adds the
declared analysis scale:

1. **Declare the scale first.** `linear`, or `log_linear`. Anything else is refused before the
   data are read (`parent_set_effect.transformation_undeclared`). A log-linear study requires
   every searched column to be positive (`parent_set_effect.log_linear_input_not_positive`,
   naming the column), and the natural logarithm is applied to every column before the split,
   so discovery and estimation both run on the declared scale.
2. **Split and discover.** A seeded split assigns three quarters of the participants to discovery;
   PC with the partial-correlation test (alpha 0.01) on the discovery rows, with a conditioning-set size and a parent limit equal to
   the declared maximum degree. A degree above four is refused
   (`parent_set_effect.degree_above_supported`, naming the declared and the supported degree). If
   the parent limit removes an adjacency no test separated, the orientations rest on the limit and
   the run is refused (`parent_set_effect.parent_limit_bound`).
3. **Accept on the treatment's parent set.** The class is accepted when the search finished and
   every edge incident to the treatment is directed. A directed edge of a completed partially
   directed graph is directed in every member of the class, so the treatment's parents are the
   same in every member. Edges elsewhere may stay undirected: the result lists them
   (`ParentSetEffect.unresolved_edges`, with `treatment_parents`, `treatment_children` and the
   acceptance reason) and nothing completes them. An undirected edge at the treatment is refused
   with the edges named (`parent_set_effect.treatment_parents_ambiguous`): the class then holds
   graphs with different parent sets, and no member is chosen. PC marks an edge as a conflict
   when two colliders disagree about it, and the orientation applied first has already
   propagated; such a conflict anywhere in the graph is evidence that some colliders are wrong,
   not a statement about the treatment. The skeleton is therefore oriented again under every
   choice among the implicated colliders (each kept, or declared a non-collider; at most eight),
   and the class is accepted when the treatment's edges come out the same in every choice that
   leaves no conflict. Conflicting edges elsewhere are listed as unresolved with the reason
   `orientation_conflict`. A choice-dependent orientation of the treatment's edges is refused
   (`parent_set_effect.treatment_orientation_depends_on_conflict`), as are more than eight
   implicated colliders (`parent_set_effect.conflict_resolution_budget`) and a conflict no choice
   removes (`parent_set_effect.search_not_finished`).
4. **Estimate on the independent quarter.** Adjust for the treatment's parents. A linear study reports
   the total effect with the least-squares Student interval. A log-linear study reports the
   intervention mean ratio `exp(shift * beta)` of multiplying the treatment by `exp(shift)`
   (shift 1 reports `exp(beta)`), its interval the exponential image of the log-scale Student
   interval, and its delta-method standard error.

Each configuration is its own calibration key: structure
`discovered_sample_split.treatment_parent_set`, the estimator of the declared transformation and
a functional naming the variable count, the declared degree, the discovery allocation, any
background knowledge and the topology, so no record is borrowed from the fully directed five-to-seven-variable construction
or from another count. The measured grid is ten, fifteen and twenty variables at 1,000, 2,000
and 4,000 participants (three quarters to discovery and one quarter to analysis), gated separately for each transformation and each
topology, in `crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs`.

Three quarters go to discovery because resolving the graph needs more information than the
held-out coefficient fit; all reporting, coverage and width targets remain unchanged. New
sources use protocol version two and a `discovery_fraction_0_75` calibration key. Version-one
sources replay their original half split and retain their original key; evidence from one
allocation cannot license the other.

The topologies are connected graphs with confounding, mediators and nuisance variables:
`hub_confounded_multi_path` (a hub confounds the treatment and the outcome; the effect runs
through a mediator), `layered_multi_mediator_nuisance` (two mediators in series and a
mediator-outcome confounder) and `confounded_direct_and_mediated` (a direct edge and a mediated
path under confounding; maximum degree four). A table does not announce its topology, so
`ParentSetEffect.measure` licenses a cell only when the records of every topology of its
declared degree attest it at that variable count, transformation, knowledge and analysis row
count: the conservative intersection (a family declared at degree three has the first two
topologies; the third has degree four). Until the records exist `measure` refuses with
`cell_not_licensed`.

Every replicate of a design reruns discovery, acceptance or refusal and estimation. The truth
is computed independently of the estimator by simulating the intervention (the treatment's
equation replaced by a constant, on common noise draws), for linear and log-linear studies
alike, and is cross-checked against the path-product total effect. Each design prints the
reporting denominator and share, every refusal reason with its count, the median interval
width, the share of replicates whose treatment parents were recovered, and the skeleton and
orientation precision and recall of the discovered class against the true graph. The frozen
usefulness targets are asserted: reporting at least 0.9, median width at most 0.6, 0.42 and
0.3 at 1,000, 2,000 and 4,000 participants, and, with the 2,000 replicates the acceptance
manifest freezes, coverage of at least 0.93. Named boundary designs at fifteen variables (weak
signal below the published margin, an exactly cancelling path, and skewed noise the
partial-correlation test cannot see through) are recorded under their own keys and license
nothing.

*Background knowledge.* `estimate_parent_set_effect_with_knowledge` (Python: `required=`,
`forbidden=`, `tiers=`) hands required and forbidden `(cause, effect)` links and tiers to
discovery. The statements are the caller's assertions: discovery honours them and does not test
them, they are part of the source, and they enter the calibration key
(`.knowledge_required+tiers`), so an interval is attested only from records measured with the
same kinds of statements. Each regime (required edges, forbidden edges, tiers, and a declared
maximum degree of three as the parent limit) is validated end to end at ten variables with
statements that are true of the model. Knowledge naming a column that is not searched is
refused (`parent_set_effect.knowledge_invalid`).

### Linear family

* **Signal strength.** Every adjacent pair, and every pair with a common neighbour, keeps a
  population partial correlation of at least 0.15 in magnitude given each conditioning set PC can
  form from the true neighbours (at most four of one endpoint's) that does not d-separate it and
  is no larger than the smallest set that does (strong faithfulness). The measured family draws
  standardized coefficients of magnitude 0.4 to 0.7 and redraws until this holds
  (`crates/antecedent/tests/common/sparse_scm.rs`).
* **Faithfulness.** The only conditional independences of the population are those the graph
  implies, with the margin above; cancelling paths are outside the family.
* **Causal sufficiency.** No unmeasured common cause of any two searched variables.
* **Distribution.** Independent rows from a linear Gaussian structural model with
  homoskedastic noise; the partial-correlation test and the Student interval rely on it.
* **Maximum degree.** No variable has more than the declared number of neighbours, at most four.
* **Intervention support.** The treatment varies given its parents (a positive conditional
  variance), and the effect is a total effect per unit change over the observed range.

### Log-linear family

* **Signal strength.** The same margin, on the logarithmic scale of the searched columns.
* **Faithfulness.** As above, on the logarithmic scale.
* **Causal sufficiency.** As above.
* **Distribution.** Every searched column is positive, and its logarithm follows the linear
  Gaussian model above (the columns are jointly log-normal). The transformation is declared
  before analysis and frozen in the source; positivity is checked, never repaired.
* **Maximum degree.** As above.
* **Intervention support.** The prespecified intervention multiplies the treatment by
  `exp(shift)` and the treatment's observed support covers that move given its parents; the
  reported ratio compares the outcome's mean under that intervention to its mean without it, and
  is not extrapolated to other multipliers.

The graph recovered on one table is an estimate: sufficiency and faithfulness cannot be tested
from the table discovery ran on, so these conditions are the family the coverage was measured
over, never a per-table certificate.

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

`antecedent.selective_discovery`: `StaticContract`, `TemporalContract`, `Link` (shared with
`antecedent.discovery`, which also holds `BackgroundKnowledge`), `discover_on_split` (with `weights=` and
`mask=`) and `SampleSplit` (`receipt`, `verify`, `verify_source`, `analysis_row_ids`,
`analysis_weights`, `analysis_effective_sample_size`), `SelectiveDiscovery`
(`run`, `plan`, `outcome`, `export_state`, `resume`, `request_recovery_claim`,
`request_downstream_interval`, `request_selection_diagram`), `sample_split_effect` and `SplitEffect` (with `measure`), `sample_split_parent_set_effect` and `ParentSetEffect` (with `measure`), `Outcome` (structure with
`unresolved_edges`, `directed_edges`, evidence, bounded-search receipts), `RunReport` and
`WorkUnits`, `inspect_selective_discovery_artifact`, `consume_selective_discovery_artifact`,
`request_score_search_reuse` and `SelectiveDiscoveryRefusal`. Contract seeds default to 0.
