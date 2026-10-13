# Composition-aware uncertainty

Unrelated posterior or bootstrap draws are never paired by index. A row pairing
is the statement that the rows are realizations of one joint law, and one rule
decides whether that statement is licensed:
`antecedent_contract::check_propagation`. Everything that composes uncertain
inputs goes through it.

## The rule

| Operation | Needs | Refusal without it |
| --- | --- | --- |
| linear transform, contrast, sum, weighted sum | a covariance when evidence is shared or an explicit joint moment exists; a declared independence only for inputs with distinct evidence; or an aligned joint law | `sampling_dependence_unknown` (`dependence_unknown`); `evidence_double_counted` (`shared_evidence_declared_independent`) |
| product, nonlinear functional, chained Monte Carlo transformation | an **aligned joint law**: draws that all carry one alignment identity. A covariance is not enough | `joint_law_required` (`nonlinear_needs_aligned_joint_law`, `covariance_insufficient_for_nonlinear`, `index_pairing_refused`, `alignment_mismatch`) |
| mixture | known weights; the components need no joint law | `invalid_argument` (`mixture_weights_unknown`) |
| transported interval, or its source-sampling, target-sampling and transport-functional pieces, composed with another claim | an aligned joint law or a supplied covariance; independence is never assumed for these pieces | `joint_law_required` (`transported_dependence_required`) |
| functional of unpaired marginals | nothing, but the result is a labelled envelope | `cell_not_licensed` if read as a point (`envelope_is_not_a_point`) |
| interval or posterior of any composed quantity | a measured coverage record at the composed coordinate; none exists | `cell_not_licensed` (`interval_route_closed`, `posterior_route_closed`) |

Each refusal carries a registered reason code, a namespaced
`joint_law_propagation.*` detail, the offending input and a remedy that names
the dependence information that would license the request.

### Alignment identity

Two columns of one distribution artifact share an alignment identity (a digest
of the study, provider, random stream, snapshot, checked contract and draw
count). Two artifacts never do, even when both declare their rows joint, so
the rows of different artifacts are never paired. An artifact declared as
independent marginals has no alignment identity at all.

## Routes

Each route is a separate support row with its own refusal.

* **Linear moments.** `Var(sum a_i X_i) = a' Sigma a`, exact. The covariance is
  supplied, computed from aligned draws, or diagonal for distinct evidence.
  A covariance that is not symmetric positive semidefinite is refused, not
  repaired.
* **Aligned joint law.** The row mean of a closed functional (product, ratio,
  minimum, maximum, squared difference) of two aligned columns. For a
  correlated pair `E[XY] = rho s_x s_y + mu_x mu_y`. The result is a labelled
  point value of the draws; Monte Carlo error is not reported as an interval.
* **Known-weight mixture.** Law of total variance: `within + between`.
* **Dependence envelope.** The extremes over every coupling of two marginals:
  * `P(X + Y <= t)` is between `sup_x max(F(x) + G(t - x) - 1, 0)` and
    `inf_x min(F(x) + G(t - x), 1)` (Makarov 1982; Rueschendorf 1982);
    differences by negation of one marginal.
  * `E[g(X, Y)]` for a function with a mixed second difference of one sign is
    extremized by the comonotone and countermonotone couplings of the
    marginals' quantile functions; `Cov`, `E[XY]` and `Var(aX + bY)` follow.
    The sign structure is verified on the whole atom lattice before any
    coupling extreme is reported as a bound, and a function that fails the
    check is refused.
  * A declared correlation range gives the endpoints of quantities linear in
    the correlation, labelled `assumption_range`; the pooled estimate over an
    equicorrelated range reuses the evidence ledger's envelope.

  The envelope is labelled `structural_envelope`. It is never a point, an
  interval or a posterior and never a substitute for an aligned joint law.
  Marginals are finite atoms (draws or weighted atoms); the bounds are exact
  for those atoms, not for the continuous laws they were drawn from.

## Artifact

`joint_law_propagation_v1` binds the inputs' identities and alignment, the
rule (operation, dependence declaration, reading), the numbers consumed (IEEE
bit patterns), the route, the result label and the result. The identity
excludes the stored result. A consumer recomputes the identity, re-runs the
rule on the stored declaration, requires the stored route, label and alignment
identity to be what the rule yields, and accepts only identical result bits.
A resealed edit of the operation, the dependence declaration, the alignment
identity, a number or the label is refused. No interval or calibration field
exists.

## Bounds

Marginal atoms: 100000. Structure-verification lattice: 4000000 adjacent
cells. Linear terms and mixture components: 256. Aligned draws:
100000 per column. Work observes cancellation at bounded boundaries.

## Calibration

Closed and unmeasured. A component interval does not cover the composition,
and no composed coordinate has a measured coverage record. The design recorded
for a later measurement (in `joint_law.composed_interval`): the whole method is
the component estimators, their uncertainty and the licensed route, and the
interval built on the composed value; the nominal target is coverage of a stated
interval at 0.95 at the composed coordinate; DGPs are correlated and
independent Gaussian pairs and a skewed pair with shared evidence between
components; probes are a covariance estimated from the same draws and a
dependence declared but violated.

## References

Makarov, G. D. (1982), Theory of Probability and its Applications 26(4),
803-806, doi:10.1137/1126086. Rueschendorf, L. (1982), Advances in Applied
Probability 14(3), 623-632, doi:10.2307/1426677. Frechet (1951), Hoeffding
(1940), Lorentz (1953) and Tchen (1980) are cited by name; no DOI was verified
for them.
