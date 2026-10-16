# Discovered-structure transport and per-population discovery

A discovered structure is a graph estimate, not a selection diagram. Three
routes keep the two apart.

| Route | Rust entry point | What it gives |
| --- | --- | --- |
| Transport over a discovered equivalence class | `antecedent_identify::discovered_transport::transport_across_class` | A `StructuralTransportEnvelope` or a typed refusal |
| Discovery run per population | `antecedent_discovery::selective::populations::PerPopulationDiscovery` | One structure per population and an edge-by-edge comparison |
| Discovery then analysis on a row split | `antecedent_discovery::selective::reuse::split_rows` | Disjoint discovery and analysis tables with a receipt |

Provenance records: `transport.cpdag_class_envelope`,
`discovery.per_population`, `discovery.sample_split_analysis`.

Facade entries: `analysis::discovered_transport::{transport_discovered_structure,
discover_and_transport}` (a retained discovery outcome or a table, plus the
declared selection targets and query),
`analysis::per_population_discovery::{discover_per_population,
add_per_population_claims}` (each population's discovery stages enter the
composition graph under the population's coordinate, so one population's changed
rows recompute that population's claims and leave the others valid) and
`analysis::sample_split_analysis::{discover_on_split,
SplitDiscovery::check_analysis_table}` (a table that is not the receipt's
analysis rows is refused). Python entries: `antecedent.transport.discovered.{transport_across_class,
discover_and_transport}` (`python/tests/test_discovered_transport.py`),
`antecedent.selective_discovery.PerPopulationReport`
(`python/tests/test_per_population_discovery.py`) and
`antecedent.selective_discovery.{discover_on_split,sample_split_effect}`
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

## Per-population discovery

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

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Class not enumerable (search unfinished, conflicting orientation, target not in structure, too large) | typed `DiscoveredTransportRefusal` naming unresolved pairs | Add background knowledge, an intervention, or declare a graph |
| PAG members | `pag_members_not_causal_graphs` | Declare a causal graph or use a CPDAG under causal sufficiency |
| Temporal structure | outside this module | Use the temporal transport route with a declared diagram |
| Pooled search over stacked populations | `selective_discovery.pooled_population_search` | Use the pooled algorithm directly and declare cross-population assumptions |
| Content-identical populations | `selective_discovery.pooled_population_data` | Supply disjoint population rows |
| Interval after reusing the same rows to select and analyse | `cell_not_licensed` (`selective_discovery.downstream_interval_not_licensed`) | Split the rows and use `sample_split_effect` |
| Effect after discovery with an unresolved structure, an unfinished search, or outside five to seven variables | `cell_not_licensed` (`split_effect.structure_unresolved`, `split_effect.search_not_finished`, `split_effect.variable_count_not_measured`) | Use a fully orientable table, or analyse under a declared graph |
| Attestation without a current coverage record, or outside its measured row range | `cell_not_licensed` (`measured_inference.record_not_attesting`) | Measure the construction at this coordinate |
| Temporal, panel or clustered rows in a split, or a non-float64 column | `selective_discovery.split_unsupported_rows` | Materialize rows, or use a design for dependent rows |
