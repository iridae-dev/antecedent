# 2.3 joint distribution artifact contract

Joint distribution artifacts provide checked quantity meaning, aligned joint draws and
independent artifact replay. Their numerical operations retain the meaning of the
supplied law; semantic validation does not establish causal identification or
sampling coverage. Method-specific measured scalar intervals use the separate
`antecedent.inference.MeasuredInference` carrier.

## Layout and identity

`DistributionArtifact` is a distinct, versioned artifact kind. It does not
reinterpret the existing `CausalPosterior` wire. The metadata section records
the distribution meaning, ordered scientific quantities, alignment, source,
provider, RNG, snapshot, causal contract, support mask, weights, calibration
state and provider trust. Each quantity records stable variable ID, name, role,
units, population, regime, horizon, functional, conditioning and transform.
The numeric section is little-endian f64 in draw-major order, with named axes
`[draw, quantity]` and shape `[n_draws, n_quantities]`. Row `i` is one joint
realization only when `alignment = joint`.
The Rust value exposes `semantic`, `axes`, `shape`, `n_draws`, `quantities`
and a read-only draw-major f64 slice. Python exposes typed
`antecedent.artifacts.ScientificQuantity`, `DistributionIdentity` and
`JointDistributionArtifact` values. The Python constructor accepts only a
two-dimensional float64 NumPy array; `export(artifact_id)` returns bytes and
`load(bytes, expected_identity=...)` requires a separate consumer identity.
`numpy.asarray(artifact)` makes one bounded copy into NumPy-owned row-major
storage. Mutating that array cannot change the validated artifact; requesting
`copy=False` raises because the bridge does not expose its Rust buffer.
Python `mean`, `covariance` and `joint_product_expectation` call the checked
Rust operations, including the refusal for independently sampled marginals.

A consumer supplies the expected identity from its own causal contract on
load. Artifact checksums detect corruption; they do not establish that a
resealed artifact still belongs to the original population, regime, provider,
quantity order or data snapshot. Direct composition compares declared units
exactly. No unit is inferred or converted.

The initial Rust reader accepts at most 16 MiB, 100,000 draws and 1,024 scalar
coordinates. It validates declared section sizes before loading payloads and
checks actual shape, finite draws, weights, masks and quantity identities
before numerical evaluation. Covariance and nonlinear expectations require
joint alignment; independent marginal samples are insufficient even when their
array lengths match. The finite calculations are descriptive point values,
not a posterior coverage or sampling-interval claim.

## Distribution meanings

The wire distinguishes parameter posterior, causal-functional posterior,
observational posterior predictive, interventional predictive, estimator
sampling, bootstrap and empirical outcome draws. An interventional outcome
threshold accepts only `interventional_predictive` draws for an outcome
quantity under the named intervention. A posterior over a mean effect is not
an outcome distribution. The checked observational conditional
`P(Y | A=a, Z=z)` can be bound to `P(Y^do(a) | Z=z)` only if a separate
identification result establishes that equality for the same graph, population,
selection, regime, conditioning, time and support. The distribution tag or an
external provider's attestation cannot establish the equality by itself.

`native_licensed`, `external_attested`, `verified_extension` and `unverified`
are separate trust labels. Loading preserves the label; it never upgrades one
label to another. The artifact does not by itself certify a provider, an
identified causal estimand or an inferential guarantee.

## 2.2 posterior conversion

The 2.2 `CausalPosterior` remains a separate artifact kind. The checked Rust
converter accepts its column-major draws only when the source artifact ID
matches the independently retained expected identity and every coefficient
has a stable name with an explicit one-to-one target quantity binding. It
refuses summary-only artifacts, mixed parameter/effect meanings, failed fits,
unidentified mass and unevaluated structural mass. It reorders by named
bindings into draw-major rows, retaining cross-coordinate covariance and the
complete legacy posterior metadata (including backend and treatment contrast)
and the exact source-to-target bindings as a receipt. The reader reconstructs
the source column order from those bindings and checks the old summaries
against the stored draws on every load. It also refuses changes to weights,
support, alignment, distribution meaning or trust on converted artifacts.
For an unweighted parameter or causal-functional posterior, an equal-tailed
interval can be recomputed from loaded draws with the 2.2 type-7 quantile
rule. That interval describes posterior draw mass, not coverage.
The converted artifact is marked `unverified` and `unmeasured`; the conversion
cannot inherit a 2.2 provider or interval license. The 2.2 wire does not carry
population, regime or snapshot identities, so the caller must establish those
in the independently retained causal contract before conversion.

## Mapped Gaussian posterior transfer

`antecedent::inference::mapped_posterior_transfer` consumes a 2.3 aligned
parameter-posterior artifact under a separately supplied expected source
identity. The caller gives a complete mapping from stable source variable IDs
to stable target variable IDs, plus target snapshot and causal-contract IDs;
target quantities are in design-column order. The resulting prior retains a
mapping and source/target identity receipt in its restrictions.
It checks units, population, regime, horizon, conditioning, role and transform
for every mapped coefficient. It requires one constant residual-variance
coordinate equal to the target prior's known variance, no unsupported or
weighted coordinates, and no extra parameters. It computes the source's full
finite-draw covariance and passes it to the existing 2.2 dense covariance
hydration, permuted into target order. An old posterior must first pass the
explicit checked conversion above. Positional or partial mappings refuse.

A checked transfer example uses batch A design row `(1, 1)` with response `1` and batch B
row `(1, -1)` with response `3`, under `N(0, I)` and known residual variance
`1`. A four-draw quadrature artifact has the exact first two moments of the
batch A Gaussian posterior. With target columns reversed, sequential and
pooled conjugate fits both produce mean `(-2/3, 4/3)` and covariance `I/3`.
This licenses a point-only transfer calculation. Upstream posterior trust and
causal identification still come from the independently checked source
contract; the artifact's trust label is not such proof.
