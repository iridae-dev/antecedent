# Original-source projections

The independently declared native data follow Y = 1 + 2A + 0.2X. Each of 80
symmetric X values in [-1,1] occurs once at A=0,1,2,3, so the target mean of X
is exactly zero and the intervention means at A=1,2 are 3 and 5. The existing
checked response-grid estimator executes on these rows; its regularization and
numerics are checked within the explicit tolerance in expected.json. An external
supplier declares the same two means as a separate attested grid.

The named affine utility transformations are U_A=2*mean_1-1 and
U_B=0.5*mean_2, independently giving 5 and 2.5. Coefficients and output utility
units are caller declarations; this is no automatic physical unit conversion.
The original affine decision evaluator supplies all transformed numbers. Mean
sources cannot supply nonlinear utility, probabilities or aligned draws.

Portable projections retain the entire original source, exact coordinates,
contract/request identities and unresolved dependencies. Actual native execution
resolution requires an opaque receipt produced by re-executing the existing
checked preparation on its actual data and matching the whole original source.
Matching portable hashes alone does not authenticate data or issue authority.
External attestation remains external. No calibration or interval license is
claimed; these deterministic acceptance fixtures are not calibration runs.
