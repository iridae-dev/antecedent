# Identities of composed measurement, mechanism, evidence and transport results

A composed result is only as honest as the identities it carries. Five
families of object that compose (a measurement model, a set of mechanisms, an
evidence ledger, a counterfactual transport derivation and its sampled
evidence, and the scientific quantities they all refer to) each freeze their
own identities, so that a change in one thing is visible as that change and a
consumer can hold one identity independently of the bytes it reads.

This page states what the code does today: which identity types exist, the
digest domain or tag and version of each, what each binds, what is
deliberately not bound, and how shared latents, aligned draws and overlapping
observations stay coupled from execution into artifacts. Every identity is a
length-delimited canonical encoding (`CanonicalWriter`, or the canonical CBOR
of a typed wire structure under an `IdentityDomain`). None is a hash of a
`Debug`, `repr` or JSON rendering. Floats enter as IEEE bit patterns.

## Scientific quantity coordinates

`antecedent_contract::ScientificQuantity` is the coordinate every composed
object is mapped through.

| Part | Binds | Excludes |
| --- | --- | --- |
| variable identity (`variable_id`) | the stable, schema-qualified variable | the display `variable_name` |
| role | treatment, outcome, covariate, mediator, selection, utility (a tag word in the digest form) | |
| units, transform | declared unit and scale; no implicit conversion | |
| population, regime | `population_id`, `regime_id` (`observational` is explicit) | |
| horizon, functional, conditioning | ordered causal steps, functional identity, canonical conditioning | |

Two spellings exist. `write_canonical` writes the digest form (ids as text,
the role as a tag word, then horizon and conditioning). `canonical_identity`
is a length-framed string used for matching and pinned by a golden-vector
test; its role field is spelled from the role's enum name, so renaming a
role variant is a visible, test-pinned identity change. Composition refuses
a mismatch in any coordinate with its own reason
(`QuantityMismatch::{Variable, Units, Population, Regime, Horizon, Functional,
Conditioning, Transform}`); it never infers an equivalence.

## Measurement model

Declared in `antecedent_core::measurement_model`; format word
`MEASUREMENT_MODEL_FORMAT = 1`.

| Identity | Tag | Binds |
| --- | --- | --- |
| measurement | `antecedent.measurement_model.measurement.v1` | format word, route, indicator and construct (id, kind tag, units), error structure (non-differential, or differential with its axis) and every parameter as bit patterns |
| population | `antecedent.measurement_model.population.v1` | population id |
| regime | `antecedent.measurement_model.regime.v1` | regime id |
| arm | `antecedent.measurement_model.arm.v1` | arm scope (all arms, or one named arm) |
| representation | `antecedent.measurement_model.representation.v1` | coordinate-system id and integer version |
| evidence | `antecedent.measurement_model.evidence.v1` | where the parameters came from: declared (declarer), validation sample (id, units), replicate sample (id) or external provider (id) |
| model | `antecedent.measurement_model.model.v1` | the six digests above (measurement, population, regime, arm, representation, evidence), in that order |
| data | `antecedent.measurement_model.data.v1` | the named numeric blocks (name, count, bit patterns) a corrected functional was computed from |

The display label `model_id` is excluded from every digest. Variable
identities are bound inside the measurement digest; population, regime,
arm scope, representation version and evidence each have a digest of their
own, so changing one moves that digest and the model digest and no other. The
coordinates population and regime come from are the quantity coordinates above, which the contract layer matches
(`measurement_binding_mismatch` for a different construct, population,
regime or arm). A representation mapping is a request-time check
(`representation_mapping_required`): it is not part of the model's
representation digest, so using a mapping never changes which model it is.

Execution keeps parameters coupled: shared sensitivity and specificity are
one parameter set whose uncertainty enters both arms, and the arm covariance
that split rates would miss is retained. A provider posterior is consumed
only as its exact object, with the construct declared in the model's
population and regime and never with native trust; its joint draws are kept
as draws (per-imputation fits combined by Rubin's rules), never reduced to a
marginal before the outcome model is fitted.

The artifact (`measurement_model_v1`, version 1) stores the model, its four
identities and an optional corrected result with its replay inputs. A
consumer holding an independently retained model identity recomputes the
identities, then recomputes the stored result from the stored inputs and
refuses any difference. A swapped sensitivity and specificity, population,
regime or representation version is a different identity.

## Mechanism composition

Declared in `antecedent_contract::mechanism_composition` and
`antecedent_io_core::mechanism_composition_artifact`.

| Identity | Domain / tag and version | Binds |
| --- | --- | --- |
| plan | `antecedent.mechanism_composition_plan`, `MECHANISM_PLAN_IDENTITY_VERSION = 1` | target population and regime; per mechanism the variable, units, transform, population, regime, origin (fitted, external binding, declared), fit artifact id, evidence id, parents with units and transform, conditioning set and latent causes with the mechanisms that share each; the edges; every admission (coordinate, source, target, selection-diagram targets or invariance assumption); the dependence declaration |
| premises and query | `IdentityDomain::Program` (`antecedent.identity.program.v1`) over the canonical wire encoding with the label `mechanism_composition_v1` and artifact version 1 | the plan's content plus every number: coefficients, intercepts, scales, tables, fitted domains, loadings, parameter covariances and cross-covariances, and the query (interventions, outcome, arms, sample size). The stored result is excluded: it is an output replayed from the premises |
| fit and evidence identities | caller supplied strings | which fit produced a mechanism and which data it was estimated from, so shared evidence is visible to the dependence check |

Population, regime, units and transform are separate coordinates on each
mechanism and parent, so a mismatch is reported as that coordinate; a
declared admission (selection diagram with no arrow into the variable, or an
explicit invariance assumption) is part of the plan identity and therefore of
the result's identity. The plan identity is also what a refresh recomputes:
refitted numbers and their fit and evidence identities change it.

Shared latents: a latent is declared by id on every mechanism that depends on
it and lists the others that share it. Absent, unreciprocated, undeclared or
unloaded sharing is refused; it is never treated as independent. In
execution the latent's draw is keyed by its name (`noise_stream_index(1,
name)` under `COMPOSITION_STREAM_DOMAIN`), so every mechanism reading the
latent reads the same draw and the draws of one source do not depend on how
many other variables the composition has. Idiosyncratic noise and finite
categorical uniforms use their own roles (0 and 2) of the same name key.

Aligned draws and dependence: separately fitted mechanisms are not
independent by default. The dependence declaration is one of unknown (an
uncertainty claim over several mechanisms refuses with `joint_law_required`),
independent by design with a justification and distinct evidence identities,
a named aligned joint law (the plan holds its id), or explicit
cross-covariance blocks. The artifact stores the declared law's parameter
order and covariance, or each block, inside the premises, so they are covered
by the premises digest. The plan holds an aligned law by id; the artifact
carries its numbers.

Replay: the consumer recomputes the premises digest, recompiles the premises
under the coherence checker (a premises that cannot compose is refused even
when resealed) and replays the query to identical result bits.

## Evidence ledger

Declared in `antecedent_contract::evidence_ledger` and
`antecedent_io_core::evidence_ledger_artifact`; every identity writes
`LEDGER_IDENTITY_VERSION = 1` first.

| Identity | Tag | Binds |
| --- | --- | --- |
| observation (`ObservationId`) | `antecedent.evidence_ledger.observation` | a producer-chosen domain and key (for example a snapshot digest and row index); two sources that used the same observation carry the same id |
| source (`SourceId`) | `antecedent.evidence_ledger.source` | kind, label, observation basis (declared sorted rows, or undeclared), lineage and transport identity |
| transport identity | `antecedent.evidence_ledger.transport` | sorted source populations, target population, selection-diagram digest, sorted assumptions |
| result (`ResultId`) | `antecedent.evidence_ledger.result` | label, the canonical combination (operation and inputs) and the result it replaces |
| operation (`OperationId`) | `antecedent.evidence_ledger.operation`, chained from `antecedent.evidence_ledger.genesis` | the previous head, the operation kind and its targets, so the head identity binds the whole history |
| observation union | `antecedent.evidence_ledger.observations` | the sorted distinct observation ids behind a result (present only when no source is unknown) |
| state | `antecedent.evidence_ledger.state` | every source and its standing, every result and its standing and accounting, and the head |
| covariance | `antecedent.evidence_ledger.covariance` | study count and the row-major covariance bits |
| artifact | `antecedent.evidence_ledger.artifact`, artifact version 1, feature `evidence_ledger_v1` | head, state digest, ordered operations and every stored synthesis (estimates, variances, covariance, tau support and probabilities) |

Variable, measurement, population, regime and representation identities are
not ledger identities. The ledger holds evidence provenance only; a
transported claim carries its populations and selection diagram through the
transport identity, and the scientific coordinates of what a source estimates
belong to the quantity and the producing route.

Overlapping observations: overlap is computed from row-level observation ids,
never from a provenance label. A result counts each observation once (declared
slots, distinct observations and repeated slots are all reported). A source
with no declared observations is unknown, not independent: independent
pooling and a diagonal covariance refuse, and only a covariance with cross
terms or an explicit assumption range is licensed. A transported claim is
never counted as native target data, even with a declared covariance. A
provenance or lineage label proves bookkeeping, not independence.

Replay: a consumer replays the operations through the ledger, recomputes the
head, state and each synthesis, and refuses a stored identity, effect, value,
covariance or range that does not replay, including a consistently resealed
edit and an operation the ledger itself refuses.

## Counterfactual transport

Declared in `antecedent_io_core::counterfactual_transport_wire` and, for the
sampled route, `antecedent_estimate::counterfactual_transport_sampled` and
`antecedent_io::sampled_counterfactual_transport_artifact`. Both artifact
kinds are at version 1 and are hashed canonical CBOR of typed wire structures
under `IdentityDomain::TransportCertificate`
(`antecedent.identity.transport_certificate.v1`).

| Identity | Tag | Binds |
| --- | --- | --- |
| premises digest (exact) | `counterfactual_transport_v1` | status, problem (graph, levels, selection targets), query (worlds, interventions as bits, routes, event, conditioning, sources), variable names, search limits, evaluation limits, derivation and obstruction |
| data digest (exact) | `counterfactual_transport_data_v1` | the evidence catalog (environments, regimes and bindings each sorted, and the target sampling contract) and every law table, sorted |
| count table | `antecedent.estimate.counterfactual_transport_sampled.counts.v1` | population, regime, intervention levels as bits, axes and levels, every count, snapshot identity and sampling design |
| sampled evidence | `antecedent.estimate.counterfactual_transport_sampled.evidence.v1` | every table's identity in canonical order, the overlap declaration, the unseen-cell rule and its parameter, minimum table total, sparsity and envelope limits and evaluation limits |
| sampled derivation | `antecedent.estimate.counterfactual_transport_sampled.derivation.v1` | the stored derivation's canonical text |
| premises digest (sampled) | the artifact kind marker | problem, query, names, search, derivation, overlap, unseen rule, limits and the stored estimate |
| data digest (sampled) | `sampled_counterfactual_transport_data_v1` | catalog and sorted tables |

Population and regime are part of every table's and every law's key, so
evidence of one population or regime never stands in for another. A consumer
holds the premises digest and the data digest independently
(`CounterfactualTransportExpectedIdentity`) and the artifact fails when
either differs from the recomputed value. A refresh with changed numbers
changes the data digest. Recomputing a digest grants nothing: a consumer
still re-decides and re-evaluates.

Overlapping observations in the sampled route: tables are independent samples
unless the declaration says they share units, and shared units are modelled
only for tables that are the same observations (one dataset group, one
multinomial sample). An undeclared relation refuses with
`counterfactual_transport_sampled.overlap_unknown`; two tables declared
independent that hold identical observations refuse; two tables declared to
share units that are not the same observations refuse because counts alone
carry no covariance. The overlap declaration is part of the evidence
identity, so changing it changes the result's identity. Table order is
canonical, so permuting the request leaves every identity unchanged.

## What is not frozen separately

- A standalone digest for a variable exists only as the quantity coordinate
  (and, in a measurement model, as fields inside the measurement digest);
  population, regime and arm have their own measurement-model digests.
- A measurement model's representation mapping is not part of its
  representation digest.
- The plan identity of a composition holds an aligned joint law by id; the
  numbers are in the artifact's premises.
- The ledger's source identity includes the label, so a rename is a new
  source identity; overlap is decided by observation ids, not labels.
