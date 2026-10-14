# Mechanism transport and transported synthesis

Three routes decide what a source-population fit may contribute to a target
population's composed model, an evidence synthesis, or a surrogate bridge. Each
states which estimates may be used; none turns that statement into a calibrated
claim about the target.

| Route | Rust entry point | Result |
| --- | --- | --- |
| Fitted linear-Gaussian mechanisms into a target composed model | `antecedent_model::mechanism_transport::transport_linear_gaussian` | `TransportedLinearGaussian` with a per-component disposition |
| Native plus transported summaries pooled, borrowed or used as a prior | `antecedent_design::transported_synthesis::synthesize_transported` | `TransportedSynthesis`: point and model variance |
| Surrogate bridge carried to another population | `antecedent_design::transported_bridge::TransportedSurrogateBridge` | A bridge contract checked against a selection diagram |

Provenance records: `transport.fitted_linear_gaussian_mechanism`,
`transport.synthesis`, `transport.surrogate_bridge_premise`.

## Mechanism transport

A fitted additive-noise linear-Gaussian mechanism is carried only when the
declared selection diagram has no selection node pointing into its variable or
into a latent cause it loads on. A mechanism that loads on a shared latent cause
also needs the analyst's explicit population-invariance declaration. A natively
refitted target mechanism always replaces a carried one for the same variable;
the two are never averaged. Every component of the result records whether it was
carried or refitted. The carried parameter covariance is the source fit's; the
selection assumption adds no variance and the result says so.

Finite tables are decided by the finite counterfactual transport derivation,
which gives an exact obstruction witness; a table here is refused with a pointer
to it. The integration tests check this route and the composition checker
against one shared fixture.

## Transported synthesis

A transported summary is `y = theta + b_t + e`, with a transport error `b_t` of
declared variance `tau_t^2` shared by every summary carried through the same
transport `t`. The three routes are generalized least squares under that
covariance, hierarchical normal borrowing across exchangeable studies, and a
transported claim used as the prior of a native likelihood. The evidence ledger
decides overlap before any number is combined, and the route is atomic: the
ledger is changed only when the numbers were computed. The result is a point and
a model variance; no interval, posterior coverage or calibration is claimed, and
the dependence between source and native estimates is your declaration.

## Surrogate bridge premise

The premise is that the relation between the effect on the surrogate and on the
terminal outcome, validated across trials in population `P`, is the same in `Q`.
It holds only when no selection node points into the terminal outcome or into a
declared modifier of the relation. A node into the surrogate does not contradict
it. The module checks the premise against the diagram; it does not discharge it,
and it stays an unresolved obligation of the derived claim. The bridge remains
point-only.

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Selection node into a mechanism or its latent cause | `mechanism_composition.population_selection_arrow_into_mechanism`, `mechanism_transport.selection_arrow_into_latent` | Refit the mechanism in the target |
| Diagram not between the stated source and target | `mechanism_transport.diagram_source_mismatch` / `diagram_target_mismatch` | Declare the diagram for these two populations |
| Declared or externally bound mechanism, or a non-Gaussian family | `mechanism_transport.origin_not_fitted`, `family_unsupported` | Fit the mechanism, or use the finite transport route for tables |
| Transport declared with no uncertainty | `transported_synthesis.transport_variance_undeclared` | Declare `tau_t^2` for the transport |
| One transport serving several claims without a shared covariance | `transported_synthesis.shared_transport_needs_covariance` | Declare the shared transport once |
| Source that would be both native target data and a transported claim, or unknown overlap | `joint_law_required` / ledger refusal | Keep the sources disjoint in the ledger |
| Selection node into the terminal outcome or a modifier | `bridge_assumption_missing` (`scientific_bridge.transport_premise_contradicted_by_selection_diagram`) | Validate the surrogate relation in the target |
| Interval on a synthesis or bridge | none issued | Use the point and model variance as labelled |
