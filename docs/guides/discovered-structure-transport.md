# Discovered-structure transport and per-population discovery

A discovered structure is a graph estimate, not a selection diagram. Three
routes keep the two apart.

| Route | Rust entry point | What it gives |
| --- | --- | --- |
| Transport over a discovered CPDAG class | `antecedent_identify::discovered_transport::transport_across_class` | A `StructuralTransportEnvelope` or a typed refusal |
| Transport over a discovered PAG class | `antecedent_identify::discovered_transport::transport_across_pag` | The same envelope over the PAG's maximal ancestral graphs |
| Transport at a declared horizon over a discovered temporal class | `antecedent_identify::discovered_transport_temporal::transport_across_temporal_class` | A `TemporalTransportEnvelope` |
| A point, a structural range or a refusal from the envelope and evidence | `analysis::discovered_transport_estimate::estimate_discovered_transport` | A `DiscoveredTransportEstimate` with a replayable artifact |
| Experimental compatibility refinement of a static class | `analysis::discovered_transport_trial::estimate_trial_compatible_discovered_transport` | The unchanged original report, an explicit binary-trial test proof and an estimate over every retained member |
| A contrast confidence set with declared bounded outcomes | `analysis::discovered_transport_trial::estimate_trial_compatible_transport_confidence_set` | An identified sampling interval or a labelled logical support bound, with replay against the original inputs |
| Discovery run per population | `antecedent_discovery::selective::populations::PerPopulationDiscovery` | One structure per population and an edge-by-edge comparison |
| Discovery then analysis on a row split | `antecedent_discovery::selective::reuse::split_rows` | Disjoint discovery and analysis tables with a receipt |

Provenance records: `transport.cpdag_class_envelope`,
`discovery.per_population`, `discovery.sample_split_analysis`.

Facade entries: `analysis::discovered_transport::{transport_discovered_structure,
discover_and_transport}` (a retained discovery outcome or a table, plus the
declared selection targets and query),
`analysis::discovered_transport_estimate::{estimate_discovered_transport,
verify_discovered_transport_estimate}`,
`analysis::discovered_transport_temporal::{transport_discovered_temporal_structure,
transport_temporal_class, estimate_discovered_temporal_transport}`,
`analysis::per_population_discovery::{discover_per_population,
add_per_population_claims}` (each population's discovery stages enter the
composition graph under the population's coordinate, so one population's changed
rows recompute that population's claims and leave the others valid) and
`analysis::sample_split_analysis::{discover_on_split,
SplitDiscovery::check_analysis_table}` (a table that is not the receipt's
analysis rows is refused). Python entries: `antecedent.transport.discovered.{transport_across_class,
discover_and_transport, estimate_discovered_transport, verify_discovered_transport_estimate}` (`python/tests/test_discovered_transport.py`),
`antecedent.selective_discovery.PerPopulationReport`
(`python/tests/test_per_population_discovery.py`) and
`antecedent.selective_discovery.{discover_on_split,sample_split_effect,sample_split_parent_set_effect}`
(`python/tests/test_sample_split_analysis.py`).

## Structural envelope

A CPDAG stands for every DAG in a Markov equivalence class, and which member is
true is not determined by data. The route never picks a member. It enumerates
the class within a member bound and returns each member's structural transport
verdict side by side. This is a **structural envelope**: not a point, interval
or bound, and not weighted. Read its status:

- `IdentifiedInEveryMember`: enumeration finished and every member is certified.
  Check `formulas_agree`; if the formulas differ the class does not determine
  the estimand's expression and the graph must be resolved before estimating.
- `NotIdentifiedInEveryMember`: no member is certified. This is the conservative
  not-certified reading, not a proof of non-transportability.
- `MembersDisagree`: transportability depends on which member is true.
- `Partial`: the member bound stopped enumeration; the remainder is not
  certified empty and no "every member" statement is made.

Selection targets are declared by you. They are never inferred from differences
between per-population discoveries.

## PAG classes and latent variables

A PAG (FCI, RFCI) stands for every maximal ancestral graph (MAG) of its Markov
class, and a MAG is not a DAG: a directed edge states ancestry, a bidirected edge
stands for a latent common cause. Each member is read as the most-confounded graph
it is compatible with: every directed and bidirected edge is kept, and a directed
edge that is not *visible* (Zhang 2008) also gets a bidirected edge, because only
a visible edge excludes a latent common cause of its endpoints. A derivation on
that graph holds in every causal model the member represents. A circle endpoint
is never read as the absence of confounding: the completion walk enumerates it
under an operation budget, and a stop or a skipped equivalence audit makes the
envelope `Partial`. Each member lists its `bidirected_edges` and `confounded_edges`.

A selected cause of the outcome whose edge to the outcome is not visible is the
usual reason a class is not identified: where it may share a latent cause with the
outcome no standardizer repairs the shift, and where it only shares a cause the shift
never reaches the outcome. The members then disagree.

## Temporal classes at a declared horizon

A lagged DAG (PCMCI), a temporal CPDAG (PCMCI+) or a temporal PAG (LPCMCI) is a
stationary template over lagged nodes. A request names a treatment, an outcome and
the number of steps between the intervention at the origin and the outcome. Every
member (the stationary completions) is unrolled over the window the request needs;
completions never point backward in time and never create a collider the class
does not have. A selection target applies at every slice. The history before the
window is replaced by a bidirected edge between every pair of window nodes the
template links to it, a supergraph of the latent projection, so an autoregressive
history is answered rather than refused (it may refuse an effect the exact projection
would identify, never the reverse). The lag assumption is the template's largest lag,
reported as `max_lag`.

## Estimates: a point, a structural range or a refusal

`estimate_discovered_transport` checks every member structurally, then searches for
a checked formula against the actual available evidence: one source treatment trial
and one target observational sample, integer-coded and complete. It never promotes
another observed column to an experimental intervention. The catalog-aware search
prefers a certified minimal pretreatment standardization, when available, over a
larger target-joint factorization. This choice uses the graph and declared evidence
availability, not observed outcomes or sample-cell counts. Each member retains its
causal proof and transport grid.

For independent, completely randomized binary source assignments and outcomes,
`estimate_trial_compatible_discovered_transport` offers an explicit refinement.
Declare `TrialCompatibilityOptions { alpha, randomization:
TrialRandomization::CompletelyRandomized }` before inspecting rows. Assignments must
be uniform given the treated count; common-probability Bernoulli assignment also
qualifies after conditioning on that count. A member with
no directed treatment-to-outcome path implies independence in that trial. A
two-sided Fisher exact test rejects this one shared implication at the declared
level; every no-path member is then explicitly rejected. All other members remain,
and a point still requires agreement. This does not force an adjacency or modify
the original discovered report.

The proof also retains `estimation_strategy`. The current
`declared_selection_joint_pretreatment_standardization` strategy preserves
declared population-shift variables and prefers their joint standardization set
only when every member's pretreatment and identification checks permit it.
This depends on structure and supplied evidence, not fitted effect values.
Missing legacy strategy fields retain the original minimal-standardizer identity.

The refinement artifact retains the exact tested assignment/outcome rows, their
canonical source identity, contingency counts, p-value, level, original member
indices, every original graph and structural verdict, and every retention decision.
`verify_trial_compatible_transport_estimate` rebuilds the original report identity,
recomputes the test and graph-path decisions, and consumes every retained member
grid. Its estimate identity binds the refinement, so it cannot be substituted for
an estimate over the original unrefined class. Stratified or adaptive allocation
needs a conditional randomization test and is not asserted by this entry point.

For outcomes with known support, the Rust endpoint
`estimate_trial_compatible_transport_confidence_set` accepts an explicit
`TransportOutcomeSupport { lower, upper }`. This asserts potential-outcome bounds
in both populations; observed minima and maxima cannot establish that assumption.
The query must request the mean curve at binary treatment levels `[0,1]`, with
at least 40 bootstrap replicates.
Identified results retain the existing percentile-bootstrap contrast interval.
Completed scientific nonidentification, rejection of every member by the trial,
or a structural range instead returns the logical contrast bounds
`[lower - upper, upper - lower]`. Thus binary outcomes yield `[-1,1]`.
The returned `kind()` and `logical_reason()` distinguish these sets from identified
point results. Cancellation, incomplete enumeration, resource limits and
unestimable members remain errors.

The confidence-set receipt binds the support, branch, trial proof, query, both
datasets and execution seed. Replay with
`verify_trial_compatible_transport_confidence_set_against` requires those original
inputs and reruns estimation, including bootstrap; its cost can approach the
original computation. Neither export nor replay grants calibration authority.

The static complete-procedure design uses FCI's nominal independence-test cutoff
`0.05` and 1023 within-stratum permutations when the analytic CI test lacks
support. A smaller cutoff can lose indirect dependencies and introduce incorrect
collider orientations; dividing by the number of variable pairs does not establish
a graph-wide error bound. The design measures the complete procedure, including
discovery and logical support sets, on its original DGP. At every static grid point,
it separately requires identified points on at least 90% of attempts and a median
confidence-set width at most 1.5 times the effect magnitude, including the full
width of logical sets. The six-variable temporal manifest point retains its
separate reporting and width requirements.

- **Point**: every member is identified and they agree (the same expression, or equal
  estimates).
- **Structural range**: every member is identified but they differ; the smallest and
  largest member estimate per level, labelled `structural_range`. It is a statement
  about which values the class allows, with no probability over members and no interval.
- **Refusal**: some member is not identified (`class_not_identified`, naming the
  members), some member cannot be estimated from the evidence (`member_not_estimable`),
  or the enumeration did not finish (`class_enumeration_partial`).

The export replays with `verify_discovered_transport_estimate`: every member grid is
consumed with its embedded proof and tables and the answer is recomputed. A point
answer on independent rows can carry a pointwise percentile interval on the contrast
(`bootstrap_replicates`); it is conditional on the discovered class and its calibration
is `not_bound_to_this_execution` until the complete-procedure design in
`crates/antecedent/tests/discovered_transport_calibration.rs` (discovery, enumeration,
identification and estimation in every replicate) is measured. A temporal estimate's interval (rows of overlapping windows are dependent) is a circular
block bootstrap of each table's rows, with `block_length` rows per block (default twice
the window span); it is conditional on the class, labelled `not_bound_to_this_execution`,
and its complete-procedure design has a separately measured three-variable scope
(`discovered_temporal_transport_contrast_complete_procedure_coverage`) with the
full allowed-edge Holm discovery protocol. This does not turn an arbitrary raw
class-conditional interval into an attested complete-procedure result; static
trial refinement and the six-variable procedure have distinct identities. Python: `transport.discovered.{transport_across_temporal_class,
transport_temporal_graph, estimate_discovered_temporal_transport}`.

## Per-population discovery

For PCMCI+, opting into an adjustment that includes contemporaneous links uses the
predeclared allowed directed-link family `(source, source_lag, target)`. It includes
every allowed lagged link and both contemporary directions, excluding self-links
at lag zero. A tested link uses its maximum MCI p-value; a PC1-removed or untested
MCI link contributes p=1. Thus three unconstrained variables with lag one have 15
family entries, and six with lag two have 102. The algorithm record and selective
contract identity name `all_allowed_edges_mci_max_p_missing_p1.v1`. Historical
lagged-only settings retain their original protocol.

The discovered-transport temporal designs declare Holm adjustment over that family.
This identifies the adjustment operation and its denominator; adaptive CI queries,
orientation and class selection do not acquire an unconditional family-wise error
theorem from that setting. Complete-procedure coverage includes all those steps and
must be measured. Ordered bootstrap inference freezes the chosen causal derivation
on the original evidence, then rebinds it to each resampled catalog and recomputes
its empirical laws; it never reuses fitted probabilities across resamples.

One declared contract is run on each population's rows separately, with its own
statistic store and identity. Nothing is pooled. A difference between two
discovered structures is a difference between two estimates; it is neither a
mechanism-difference claim nor an invariance test, and no error rate is
attached to agreement.

## Sample split

`split_rows` partitions exchangeable rows into a discovery table and an
analysis table by a declared rule, with a `SampleSplitReceipt` that a consumer
re-derives with `verify`. The receipt's binding licenses a statement for an
analysis that treats the selected graph as given.

`sample_split_effect::estimate_effect_on_split` is the one whole-method
construction with a measured coverage property: split, PC with the
partial-correlation test, a resolved structure accepted, and the treatment's
effect adjusted for its parents on the analysis half. Its interval is attested
by `SplitEffect.measure` from a coverage record keyed
`discovered_sample_split` and is a statement about the method over the
measured family (linear-Gaussian, causally sufficient, faithful, fully
orientable class, five to seven variables), not about one dataset's graph.
Faithfulness and causal sufficiency are not testable from the table discovery
ran on, so a per-dataset recovery or error-rate claim stays closed.

For ten to twenty variables with undirected edges elsewhere in the class,
`sample_split_parent_set_effect::estimate_parent_set_effect` accepts the class when every edge
incident to the treatment is oriented (the same way under every choice among conflicting
colliders), lists the other unresolved edges without completing them, and reports the total effect (linear) or the intervention mean ratio (log-linear). See
[selective discovery](../selective-discovery.md) for the conditions it is measured under.

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Class not enumerable (search unfinished, conflicting orientation, target not in structure, too large) | typed `DiscoveredTransportRefusal` naming unresolved pairs | Add background knowledge, an intervention, or declare a graph |
| PAG marks of more than one Markov class, or a selection-bias edge | `pag_class_ambiguous`, `pag_selection_bias_edge` | Rerun discovery, remove the edge with background knowledge, or declare the graph |
| Temporal structure without a horizon | `temporal_structure_unsupported` | Request transport through the temporal entry with the treatment, outcome and horizon |
| Temporal query outside the structure | `temporal_query_invalid` | State a mean response curve among the structure's variables |
| Effect not identified in some member | `class_not_identified` | Resolve the marks that separate the members, or declare the graph |
| Evidence that cannot supply a formula's factor | `member_not_estimable` | Supply covering evidence |
| Unfinished enumeration | `class_enumeration_partial` | Raise the member bound and completion budget |
| Tables not integer-coded or complete, a series shorter than its window | `evidence_invalid` | Supply complete integer-coded tables and valid temporal windows/block-bootstrap settings |
| Pooled search over stacked populations | `selective_discovery.pooled_population_search` | Use the pooled algorithm directly and declare cross-population assumptions |
| Content-identical populations | `selective_discovery.pooled_population_data` | Supply disjoint population rows |
| Interval after reusing the same rows to select and analyse | `cell_not_licensed` (`selective_discovery.downstream_interval_not_licensed`) | Split the rows and use `sample_split_effect` |
| Effect after discovery with an unresolved structure, an unfinished search, or outside five to seven variables | `cell_not_licensed` (`split_effect.structure_unresolved`, `split_effect.search_not_finished`, `split_effect.variable_count_not_measured`) | Use a fully orientable table, or analyse under a declared graph |
| Effect after discovery with an undirected edge at the treatment, a log-linear column that is not positive, or a declared degree above four | `cell_not_licensed` (`parent_set_effect.treatment_parents_ambiguous`, `parent_set_effect.treatment_orientation_depends_on_conflict`, `parent_set_effect.conflict_resolution_budget`, `parent_set_effect.degree_above_supported`) or `invalid_argument` (`parent_set_effect.log_linear_input_not_positive`) | Orient the named edges, supply positive columns, or declare a degree of at most four |
| Attestation without a current coverage record, or outside its measured row range | `cell_not_licensed` (`measured_inference.record_not_attesting`) | Measure the construction at this coordinate |
| Temporal, panel or clustered rows in a split, or a non-float64 column | `selective_discovery.split_unsupported_rows` | Materialize rows, or use a design for dependent rows |
