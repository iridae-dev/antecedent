# Two-point response curve versus average effect

**Suite path:** `conformance/response/two_point_curve_average_effect`

This is the **claimed** Kennedy shared-linear-contract conformance fixture for
0.5.0. On one fixed DAG and one deterministic linear-Gaussian-style design, it
compares the contrast between two points of a `ResponseCurve` with an
`AverageEffect` using the same numeric intervention levels.

The shared scientific contract is
`E[Y | T=t, Z=z] = 1 + 2t + 0.8z`, with `Z` as the identified adjustment set.
The response path uses the documented Kennedy-style doubly robust curve
estimator and local-quadratic smoother. The average-effect path uses the
existing linear backdoor-adjustment estimator. Therefore the fixture checks
agreement within a declared statistical tolerance; it does not require or
imply bitwise estimator identity.

This is **not** an `ehkennedy/npcausal::ctseff` black-box claim. That package
remains an unclaimed candidate until a shared nuisance/bandwidth contract is
pinned.

## Expected summary

Top-level keys: `claim, contract, estimator_contract, fixture_id, generation, tolerance` (6 fields).
