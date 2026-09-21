# Complementary sources and retained finite transport responses

The theoretical contract is Bareinboim and Pearl (2013),
[*Meta-Transportability of Causal Effects: A Formal Approach*](https://proceedings.mlr.press/v31/bareinboim13a.pdf),
Figure 5 (μsID) and its μs-hedge obstruction theorem. Only the paper's unrestricted
source-experiment setting licenses completeness. The
[limited-experiment setting](https://ftp.cs.ucla.edu/pub/stat_ser/r419.pdf) is
distinct. Arbitrary finite catalogs and general limited-experiment completeness
remain outside this claim.

## Checked identification

`MetaTransportQuery` supplies a shared ADMG, target, outcomes, treatments, and a
canonical collection of source-specific selection targets. `identify_meta` in
Python obtains this collection from the evidence catalog. It does not inspect
observed outcomes. Ordinary target-only ID runs first. Recursive source choices
use canonical population order; catalog binding uses canonical regime order.
A source dependency names a population and an actual joint regime, never a bag
of separately available experiments.

The common checked derivation retains a population-tagged expression DAG,
original variable coordinates, external conditioning parameters, immutable
source assumptions, and typed local premises. Single-source entrypoints remain
compatibility facades. The inherited rule identifiers remain stable; their
mapping to the two papers differs:

| μsID Figure 5 | Typed rule / stable identifier | Consuming test |
| --- | --- | --- |
| 1 | Marginal / `sid.line1` | `ordered_three_node_meta_graphs_match_every_target_atom_and_contrast` |
| 2 | Ancestors / `sid.line2` | same independent SCM enumeration |
| 3 | Enlarge / `sid.line3` | same test, including middle-node intervention |
| 4 | Districts / `sid.line4` | same test, including joint outcome requests |
| 6–7 | DirectTransport / `transport.direct` (the source answers the state as posed; Figure 5 line 10 proper is `sid.line10`, used only by single-source recursion) | same test and `complementary_sources_require_both_and_proof_is_checked` |
| 8 | independently checked common μs-hedge | `common_negative_witness_cannot_drop_an_admissible_source` |
| 9 | Factor / `sid.line7` | independent SCM enumeration |
| 10 | Recurse / `sid.line8` | independent SCM enumeration with multi-node districts |

The positive checker replays each local graph/kernel transformation against its
premises, not the identification search. S-admissibility is decided again by a second
implementation (Richardson augmented-graph criterion over the original graph) that
shares no code with the search; ancestor and c-component structure is shared, and the
latent-SCM enumeration tests are the end-to-end independent evidence. The negative checker verifies the same
nested rooted forests against every source, including selection, ancestry,
treatment intersection, graph, query and evidence scope. A catalog gap or an
exhausted computation never becomes a proven negative. Certificates serialize
positive, proven-negative and `NotCertified` outcomes separately.

Finite-catalog search tries target-only ID, canonical μsID, and catalog-aware
μsID. The last strategy searches admissible sources for available factors.
Selected leaves, attempted strategies and missing factors remain inspectable.
Exhausting these strategies means bounded search exhaustion, not completeness
for arbitrary catalogs. Existing catalog diagnostics distinguish evidence that
already exists from a proposed experiment.

## Provider and response execution

`TransportResponseGridQuery` and `StudyBuilder::transport_grid` use the common
prepared lifecycle with exact or empirical providers. A finite request must
assign precisely the certified treatment coordinates within declared discrete
domains. A joint intervention requires an actual joint regime. Every requested
point survives in order, with either the full target distribution and mean, or
a typed missing-evidence/support result. Failures retain expression coordinates,
population/regime dependencies, conditioning assignments and intervention worlds
when available. Numerical errors, cancellation and exhausted budgets fail the
whole execution; they are not relabeled as scientific support failures.

Preflight freezes the executable subset. A bounded provider-owned factor cache
reuses conditional probabilities across plans and points; cache keys include the
immutable provider's law index and assignments. Replaced data creates new cache
authority. Execution reserves memory for retained points and divides operation
budgets conservatively across points and execution phases.

Empirical providers fit one joint per dataset/world. An explicit
`RegimeBinding.dataset_identity` declares forwarded aliases. Aliases must agree
on raw columns, measured domains, sampling and dependence contracts; one fitted
joint and resampling stream serves all aliases of the same world. Conflicting
aliases fail validation. Unknown, linked or clustered dependence retains point
estimates and withholds IID uncertainty. Alternative formulas are never averaged
and disagreeing sources are never silently pooled.

T6's outer bootstrap uses the same successful replicate IDs across executable
points. A support failure invalidates the entire replicate family. A contrast
references two points and the parent grid identity and uses paired draws.
Requested coordinates and the executable family are both identity inputs.
Intervals are pointwise and nominal. Quantile inference, derivatives,
simultaneous bands, and new calibrated-coverage claims remain closed.

## Durable authority and migration

Grid and standalone certificate payloads have independent schema version 1,
required feature lists, and sections in the existing artifact container.
Statistical child results retain statistical-v2 semantics. Their identities bind
queries, graph, selections, checked functionals, catalog, snapshots, sample
provenance, inference options, seed, requested family and point outcomes.
All four reasoning slots and located support findings travel with the artifact.

Semantic loading checks the container and feature contract, independently checks
proofs/witnesses, binds every leaf, validates tables and sample summaries, and
recomputes numerical point claims. It neither fetches providers nor refits or
resamples. Exact artifacts with embedded tables can execute again. Statistical
artifacts support proof and embedded-table numerical verification, but require
explicit sample refresh for estimator replay. Missing raw samples are not
fabricated. Scalar projection supplies a loss receipt and cannot impersonate a
full response family. Unknown required features prevent semantic acceptance.

Compatible snapshot replacement keeps the checked structural derivation and
invalidates execution, support and uncertainty. Refresh prepares and evaluates a
candidate before atomically replacing the old state. Structural/evidence changes
require preparation. Python display objects never authorize execution or export.
Catalog entry order and Python regime labels are canonicalized; new multi-source
sample collections also have canonical ordering. Historical single-source
identities and bootstrap ordering remain unchanged. Historical conservative
negatives stay `NotCertified`; flat experiments acquire no invented joint regime.
Earlier insufficient statistical artifacts still require regeneration.

## Evidence and deferred calibration

`crates/antecedent-identify/tests/meta_transport_scm.rs` independently enumerates
64 ordered three-node ADMGs, seven source-selection pairs, two parameterizations,
both intervention values, and scalar/joint outcomes. It compares every supported
atom, mean and contrast and requires a checked positive or checked obstruction
on every uncapped query. A separate four-node conformance test checks 20,480
graph/source configurations and permits no unchecked obstruction outcomes. The complementary-source fixture in
`python/tests/test_transport_meta_grid.py` uses
`sum_z P_a(z | do(x)) P_b(y | do(z))`, checks source-alone obstructions, and
executes means 0.26 and 0.74 with contrast 0.48.

That Python file also supplies reproducible unequal-size sampling fixtures,
provider aliases, partial grids, missing joint regimes, support failures,
artifact consumption and mutation, atomic refresh and resource controls.
`test_bounded_multisource_calibration_fixture` checks deterministic execution and
paired bookkeeping only. It is **not a coverage measurement**.

The explicitly opt-in `test_deferred_multisource_grid_calibration` records
candidate mean and contrast interval hit counts across two parameterizations.
It is skipped by default. Running it requires
`ANTECEDENT_RUN_META_CALIBRATION=1`; its outputs still require scientific review
and registry binding before any coverage license changes. No hours-long
calibration was run for T7–T9 implementation, and the calibration-dependent T7
and T8 completion gates remain open.
