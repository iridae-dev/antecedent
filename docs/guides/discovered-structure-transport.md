# Transport through uncertain or discovered structure

You can transport an effect when the graph is itself an estimate: a CPDAG or PAG from discovery, a temporal class from a lagged discovery method, or a set of candidate graphs you enumerate. A discovered structure is a graph estimate, not a selection diagram, so Antecedent keeps the two apart. It checks the transport question in every member of the class and reports what the members agree on, never picking one member for you.

You declare the selection targets, the variables whose mechanisms may differ between populations. They are never inferred from differences between per-population discoveries.

## Routes

| You hold | Python entry (`antecedent.transport.discovered` unless noted) | You get |
| --- | --- | --- |
| A CPDAG or PAG from discovery | `transport_across_class`, `discover_and_transport` | A structural envelope over the class's members, or a typed refusal |
| A temporal class (lagged DAG, temporal CPDAG or PAG) and a horizon | `transport_across_temporal_class`, `transport_temporal_graph` | A temporal envelope at that horizon |
| A class plus a source trial and a target sample | `estimate_discovered_transport`, `verify_discovered_transport_estimate`, `estimate_discovered_temporal_transport` | A point, a structural range or a refusal, with a replayable artifact |
| A maximally oriented CPDAG and supplied exact laws | `antecedent.transport.advanced.cpdag_completion_scenarios` | Every DAG completion decided with its own evidence |
| Rows from several populations | `antecedent.selective_discovery.PerPopulationReport` | One structure per population and an edge-by-edge comparison |
| One table you want to discover on and analyse | `antecedent.selective_discovery.{discover_on_split, sample_split_effect, sample_split_parent_set_effect}` | Disjoint discovery and analysis tables with a receipt |

Tests that exercise these paths: `python/tests/test_discovered_transport.py`, `test_per_population_discovery.py` and `test_sample_split_analysis.py`. Rust entry points live in `antecedent_identify::discovered_transport` (`transport_across_class`, `transport_across_pag`), `discovered_transport_temporal`, and the facade modules `analysis::discovered_transport*`, `analysis::per_population_discovery` and `analysis::sample_split_analysis`.

## Structural envelope

A CPDAG stands for every DAG in a Markov equivalence class, and which member is true is not determined by data. The route enumerates the class within a member bound and returns each member's transport verdict side by side. This structural envelope is not a point, interval or bound, and it is not weighted. Read its status:

- `IdentifiedInEveryMember`: enumeration finished and every member is certified. Check `formulas_agree`: if the formulas differ, the class does not determine the estimand's expression and the graph must be resolved before estimating.
- `NotIdentifiedInEveryMember`: no member is certified. This is the conservative not-certified reading, not a proof of non-transportability.
- `MembersDisagree`: transportability depends on which member is true.
- `Partial`: the member bound stopped enumeration; the remainder is not certified empty and no "every member" statement is made.

## PAG classes and latent variables

A PAG (from FCI or RFCI) stands for every maximal ancestral graph of its Markov class. In a maximal ancestral graph a directed edge states ancestry and a bidirected edge stands for a latent common cause. Each member is read as the most-confounded graph it is compatible with: a directed edge that is not *visible* (Zhang 2008) also gets a bidirected edge, because only a visible edge excludes a latent common cause of its endpoints. A derivation on that graph holds in every causal model the member represents. A circle endpoint is never read as the absence of confounding; the completion walk enumerates it under an operation budget, and a stop makes the envelope `Partial`. Each member lists its `bidirected_edges` and `confounded_edges`.

A selected cause of the outcome whose edge to the outcome is not visible is the usual reason a class is not identified: where it may share a latent cause with the outcome, no standardizer repairs the shift. The members then disagree.

## Temporal classes at a declared horizon

A lagged DAG (PCMCI), temporal CPDAG (PCMCI+) or temporal PAG (LPCMCI) is a stationary template over lagged nodes. A request names a treatment, an outcome and the number of steps between the intervention and the outcome. Every stationary completion is unrolled over the window the request needs; completions never point backward in time. A selection target applies at every slice. The history before the window is replaced by a bidirected edge between every pair of window nodes the template links to it, which may refuse an effect the exact projection would identify but never the reverse. The lag assumption is the template's largest lag, reported as `max_lag`.

## Estimates: a point, a structural range or a refusal

`estimate_discovered_transport` checks every member structurally, then searches for a checked formula against the evidence you actually hold: one source treatment trial and one target observational sample, integer-coded and complete. It never promotes another observed column to an experimental intervention, and its choice of formula uses the graph and declared evidence availability, not observed outcomes or cell counts.

- **Point.** Every member is identified and they agree.
- **Structural range.** Every member is identified but they differ. The smallest and largest member estimate per level are reported as `structural_range`: a statement about which values the class allows, with no probability over members and no interval.
- **Refusal.** Some member is not identified (`class_not_identified`, naming the members), some member cannot be estimated from the evidence (`member_not_estimable`), or enumeration did not finish (`class_enumeration_partial`).

The artifact replays with `verify_discovered_transport_estimate`: every member's grid is consumed with its embedded proof and tables and the answer is recomputed. A point answer on independent rows can carry a pointwise percentile interval on the contrast (`bootstrap_replicates`). It is conditional on the discovered class, and its calibration is `not_bound_to_this_execution` until a complete-procedure coverage record (discovery, enumeration, identification and estimation in every replicate) is bound; read `result.inspect().calibration`. A temporal estimate's interval uses a circular block bootstrap of each table's rows, with `block_length` rows per block (default twice the window span), and carries the same label.

**Trial refinement.** For independent, completely randomized binary source assignments and outcomes, `estimate_trial_compatible_discovered_transport` offers an explicit refinement. You declare `TrialCompatibilityOptions { alpha, randomization: CompletelyRandomized }` before inspecting rows. A member with no directed treatment-to-outcome path implies independence in that trial; a two-sided Fisher exact test rejects that shared implication at the declared level, and every no-path member is then explicitly rejected. The other members remain, and a point still requires agreement. The test does not force an adjacency or modify the original report, and its artifact retains the tested rows, counts, p-value and every retention decision. Stratified or adaptive allocation needs a conditional randomization test and is not asserted here.

**Bounded outcomes.** In Rust, `estimate_trial_compatible_transport_confidence_set` accepts an explicit `TransportOutcomeSupport { lower, upper }`, which asserts potential-outcome bounds in both populations (observed minima and maxima cannot establish that). Identified results keep the percentile-bootstrap contrast interval. Scientific nonidentification, rejection of every member by the trial, or a structural range instead returns the logical contrast bounds `[lower - upper, upper - lower]`, so binary outcomes give `[-1, 1]`; `kind()` and `logical_reason()` distinguish these from identified results. Replay requires the original inputs and reruns estimation, including the bootstrap.

## Enumerate scenario graphs

`antecedent.transport.advanced.cpdag_completion_scenarios` asks whether a target effect stays identified, and what range it spans, across every DAG completion of a maximally oriented CPDAG when each completion receives evidence bound to its own assumptions. It takes fully observed nodes (no selection or latent structure) and supplied exact laws; counted laws refuse. Enumeration and per-completion decisions share one search budget; a stopped search keeps the completions found, marks undecided ones `unevaluated` and counts never-enumerated ones with an upper bound (`counts.exact` is false then). Every completion is kept whatever its status and nothing is renormalized. The output is an unweighted structural envelope over the identified completions: no graph probability, no interval across completions. Refusals use the `cpdag_scenarios.*` namespace (`selection_or_latent`, `not_a_cpdag`, `exact_laws_only`). `consume_cpdag_scenarios_artifact` replays it.

Estimates from identified completions that reuse the same observed units are correlated. `scenario_shared_covariance` reports that covariance by `shared_row_bootstrap` (seeded whole-row resampling on one complete-row snapshot; a replicate where any estimator fails is dropped for all scenarios and counted) or `exact_enumeration` (tables of at most 24 rows). Scenarios on different snapshots, declared independent or with repeated unit ids refuse (`scenario_covariance.*`); a diagonal-only matrix is never a fallback. The matrix is point-only, not an interval.

## Per-population discovery

One declared discovery contract is run on each population's rows separately, with its own statistic store and identity. Nothing is pooled. A difference between two discovered structures is a difference between two estimates; it is neither a mechanism-difference claim nor an invariance test, and no error rate is attached to agreement. Each population's discovery stages enter the composition graph under the population's coordinate, so one population's changed rows recompute that population's claims and leave the others valid.

## Discover on one half, analyse on the other

`discover_on_split` partitions exchangeable rows into a discovery table and an analysis table by a declared rule, with a `SampleSplitReceipt` a consumer re-derives. The binding licenses a statement for an analysis that treats the selected graph as given. A table that is not the receipt's analysis rows is refused.

`sample_split_effect` is the whole-method construction with a measured coverage property: split, PC with the partial-correlation test, a resolved structure accepted, and the treatment's effect adjusted for its parents on the analysis half. Its interval is attested by `SplitEffect.measure` from a coverage record and is a statement about the method over the measured family (linear-Gaussian, causally sufficient, faithful, fully orientable class, five to seven variables), not about one dataset's graph. Faithfulness and causal sufficiency cannot be tested from the table discovery ran on, so a per-dataset recovery claim stays closed.

For ten to twenty variables with undirected edges elsewhere in the class, `sample_split_parent_set_effect` accepts the class when every edge incident to the treatment is oriented (the same way under every choice among conflicting colliders), lists the other unresolved edges without completing them, and reports the total effect (linear) or the intervention mean ratio (log-linear). See [selective discovery](../selective-discovery.md) for the conditions it is measured under.

## What is refused, and what to do

| Refused request | Reason code or detail | Next step |
| --- | --- | --- |
| Class not enumerable (search unfinished, conflicting orientation, target not in structure, too large) | typed `DiscoveredTransportRefusal` naming unresolved pairs | Add background knowledge, an intervention, or declare a graph |
| PAG marks of more than one Markov class, or a selection-bias edge | `pag_class_ambiguous`, `pag_selection_bias_edge` | Rerun discovery, remove the edge with background knowledge, or declare the graph |
| Temporal structure without a horizon | `temporal_structure_unsupported` | Use the temporal entry with the treatment, outcome and horizon |
| Temporal query outside the structure | `temporal_query_invalid` | State a mean response curve among the structure's variables |
| Effect not identified in some member | `class_not_identified` | Resolve the marks that separate the members, or declare the graph |
| Evidence that cannot supply a formula's factor | `member_not_estimable` | Supply covering evidence |
| Unfinished enumeration | `class_enumeration_partial` | Raise the member bound and completion budget |
| Tables not integer-coded or complete, a series shorter than its window | `evidence_invalid` | Supply complete integer-coded tables and valid temporal windows |
| Pooled search over stacked populations | `selective_discovery.pooled_population_search` | Use the pooled algorithm directly and declare cross-population assumptions |
| Content-identical populations | `selective_discovery.pooled_population_data` | Supply disjoint population rows |
| Interval after reusing the same rows to select and analyse | `cell_not_licensed` (`selective_discovery.downstream_interval_not_licensed`) | Split the rows and use `sample_split_effect` |
| Effect after discovery with an unresolved structure, an unfinished search, or outside five to seven variables | `cell_not_licensed` (`split_effect.structure_unresolved`, `split_effect.search_not_finished`, `split_effect.variable_count_not_measured`) | Use a fully orientable table, or analyse under a declared graph |
| Effect after discovery with an undirected edge at the treatment, a non-positive log-linear column, or a declared degree above four | `cell_not_licensed` (`parent_set_effect.*`) or `invalid_argument` (`parent_set_effect.log_linear_input_not_positive`) | Orient the named edges, supply positive columns, or declare a degree of at most four |
| Attestation without a current coverage record, or outside its measured row range | `cell_not_licensed` (`measured_inference.record_not_attesting`) | Measure the construction at this coordinate |
| Temporal, panel or clustered rows in a split, or a non-float64 column | `selective_discovery.split_unsupported_rows` | Materialize rows, or use a design for dependent rows |

## Related

- [Carry an effect to another population](transport/carry-an-effect.md)
- [Graph uncertainty](../graph_uncertainty.md) and [selective discovery](../selective-discovery.md)
- [Discovery background knowledge](discovery-background-knowledge.md)
- [When transport fails](transport-failure.md)
