# Composing independently fitted mechanisms

Individually valid conditional mechanisms do not imply a valid joint model.
Composition compiles *declared* mechanisms into one executable joint law and
refuses, with a structured finding, whenever the declarations do not establish
a coherent one. Nothing is inferred from names or fits.

## Boundary

| Item | Supported | Refused |
| --- | --- | --- |
| Structure | explicitly mapped acyclic graphs | any directed cycle (before execution) |
| Family | additive-noise linear-Gaussian with declared parents and declared unit-variance Gaussian shared latents; or finite conditional tables | mixed families, nonlinear or non-additive mechanisms |
| Execution | closed-form joint mean/covariance and `do()` law; exact enumeration and truncated factorization; structural sampling | conditioning on observations (a separate query) |
| Uncertainty | first-order delta-method variance of a linear contrast of interventional means, from a declared parameter law | intervals, posteriors, variance functionals, any claim over several mechanisms with unknown dependence |

Residual `sigma` of a linear-Gaussian mechanism is the *idiosyncratic* part;
a declared latent adds `loading^2` to the total residual variance and
`loading_i * loading_j` to the covariance of the mechanisms that share it.

## What is checked

Each defect is a finding with a closed kind, a `mechanism_composition.<slot>`
detail, the offending subject, expected and supplied semantics, and the
declaration that would license the composition.

- **Variable identity**: one mechanism per variable; every parent has a mechanism.
- **Causal parents / conditioning set**: a mechanism's parents equal the declared
  graph's parents, and the fit conditioned on exactly its parents.
- **Units / transforms**: a consumer's parent units and transform equal the
  producer's output; no implicit conversion.
- **Population / regime**: a mechanism fitted elsewhere is admitted only through a
  declared selection diagram with no arrow into its variable, or an explicit
  invariance assumption (regime: invariance only). The transport theory itself is
  owned elsewhere; here only the declaration is represented.
- **Support / overlap**: the producer's output support lies inside the domain
  the consumer was fitted on (finite tables: level sets and complete rows);
  interventions must lie inside every consumer's domain.
- **Shared latents**: a latent shared with a mechanism that is absent,
  unreciprocated, undeclared by its peers or without a loading is refused. It is
  never silently treated as independent.
- **Provenance**: owning fit or binding identity and evidence identity; a
  mechanism that was declared rather than estimated needs acknowledgement.
- **Uncertainty alignment**: one meaning (posterior or sampling), matching
  parameter sets, symmetric positive semidefinite covariances.
- **Posterior dependence**: separately fitted mechanisms are *not* independent by
  default. An uncertainty claim over more than one mechanism needs an aligned
  joint law, an explicit cross-mechanism covariance block for every pair (a zero
  block is a declared independence for that pair), or independence by design with
  a justification and distinct evidence identities. Otherwise the refusal is
  `joint_law_required`. Point execution is unaffected.

## Layers

- `antecedent_model::compose`: descriptors, `check_composition`,
  `ComposedLinearGaussian`, `ComposedFinite`, `assemble_parameter_covariance`.
  No claim vocabulary and no contract dependency.
- `antecedent_contract::mechanism_composition`: the semantic mapping from
  `ScientificQuantity` coordinates (a parent must be a *realization*, not a mean,
  and the same coordinate as the producer), admissions, and registered refusals
  (`quantity_semantics_mismatch`, `graph_invalid`, `joint_law_required`,
  `mechanism_composition_incoherent`).
- `antecedent_io_core::mechanism_composition_artifact`: kind
  `mechanism_composition_v1`; `lower_plan` binds a checked plan to its numbers;
  `consume` recomputes the premises digest, recompiles under the coherence
  checker and replays the query to identical result bits.

## Claim labels

Composition is exact given the declared mechanisms. The delta-method variance is
a parameter-covariance variance only: it omits second-order terms, holds residual
scales fixed and carries no coverage claim. The interval route is closed with
`cell_not_licensed` until whole-method coverage is measured.

## Facade and lifecycle

`StudyBuilder::mechanism_composition` takes the semantic declarations and the
numbers together: `check_mechanism_composition` checks the coordinates, the
plan is bound to the numbers (`lower_plan`) and the result is compiled under
the coherence checker, so every defect of a request is returned with its
code, detail and remedy before anything executes. A mechanism built from a
fitted registry slot is labelled `fitted`; coefficients or tables written down
by the analyst are labelled `declared`, need the acknowledgement above and are
listed by `declared_mechanisms()`. The facade sets the label from the source
of the numbers. The prepared composition answers an intervention law, a mean
contrast with its delta-method variance and explicit cross-covariance label,
structural samples, or an exact finite law, and exports one query as
`mechanism_composition_v1`.

`refresh_mechanisms` takes mechanisms refitted on new outcome-only data:

| | |
|---|---|
| keeps | the declared graph, units, transforms, populations, regimes, admissions, latent declarations, fitted domains and supports, the kind of declared dependence |
| recomputes | the refitted numbers, their fit and evidence identities (so the plan identity), the reported marginal covariances, the joint model |
| refuses | a refit of an unknown variable, of another family or with other parents; a joint law or cross-covariance declared for the earlier fit (it is `unknown` again until redeclared); any prepare refusal on the new numbers |

The call takes `&self` and returns a new value; a refusal leaves the prepared
composition unchanged.

## Python surface

`antecedent.mechanism_composition` exposes the same route with typed
declarations: `Quantity`, `MechanismDeclaration`, `Mechanism.declared_linear`,
`Mechanism.declared_table` and `Mechanism.fitted_linear` (which reads the plain
linear-Gaussian slot of a fitted `FittedGcm`; written-down numbers are labelled
declared and need `admit_declared_mechanisms=True`), `Dependence` and
`Admission`, then `compose_mechanisms`. The prepared object answers
`intervention_law`, `mean_contrast`, `mean_contrast_uncertainty`, `sample` and
`finite_law`, refreshes with `refresh_mechanisms`, exports one query with
`export` and is consumed by `consume_mechanism_composition_artifact`. A refusal
is a `MechanismCompositionRefusal` listing every defect (`refusals`,
`details`); `interval()` raises `cell_not_licensed`.

The `composed_model` producer is a row of
[the producer and consumer matrix](producer-consumer-matrix.md): a composed model
is a point mean source for the declared-adapter consumers and is refused by the
rest with typed errors. The design composition-bundle has no node kind for a composed model.

### Checked Python handoffs

`prepared.as_mean_source(quantity=..., contrast=...)` selects a checked interventional
mean contrast and requires its compatible scientific coordinate. The canonical
`results.MeanSource` carries the point, identity and lineage into decision and inverse
queries. It does not carry the model's law or independent marginal variances as joint
decision uncertainty.

`prepared.export(query, ...)` returns a typed `ExportedArtifact` with the query-selected
bytes, expectation and immutable export metadata. Shared
`ant.load(...)` invokes `consume_mechanism_composition_artifact`, preserving its checks
and typed consumed result.
