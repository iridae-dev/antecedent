# Recordless manual interval recipes

`parity/calibration_recipe_receipts.json` binds five acceptance cases to their
original passed gate receipts. The original harness used `CoverageTally::new`,
which emits no typed coverage record. These files preserve that receipt type;
they do not create `cov.*` IDs, a sample-size range, or additional grid runs.

Each case retains its raw gate log, original recordless gate row, raw tally state,
source SHA and source hashes. Coverage is `covered / (scored + skipped)`:
refusals are observed misses. Reporting is `with_interval / (scored + skipped)`.
In particular, the older held-out policy experiment has 2,000 observed outcomes,
1,985 reported intervals and 15 refusals. The reader does not drop those refusals.

The c30 and e15 receipts keep their original legacy decision semantics. The fresh
70 held-out policy receipt keeps and revalidates its original frozen family,
validation domain, planned look and actual execution point 1. It is one fixed
n = 3,000 experiment, not a three-point sample-size grid. The source-audited
numerical dependency footprint must still match the measured and current
production paths. Recipe differences are explicit: the c30 policy constructor
uses its default propensity floor 0.01; e15/70 set 0.001 in the harness.

For n = 3,000, the saved mean widths prove the existing median-width target:
for nonnegative widths and positive truth magnitude at least b, the upper-middle
median of width/|truth| is at most twice the mean width divided by b. The policy
truth is bounded below by `1 - 1.5 / sqrt(2*pi)`; the survival contrast is the
original finite sum over six confounder cells. This is a conservative bound,
not a claim that the original process saved individual widths or a median.

For the two n = 1,000 cases that this bound could not resolve, `width_n1000/`
preserves an independently declared width-only supplement. Its exact c30 recipe,
source audit, standalone helper, runner, clean-checkout build provenance,
compiler/dependency fingerprints, fixed domain and 2,048 attempts per case were
sealed before outcomes. The initial pre-outcome plan and the reviewed amendment
remain available. The amendment restores the c30 propensity floor and uses the
repository's upper-middle median. The runner never calculates coverage.

| Width supplement | Reported / attempted | Median width / median truth magnitude |
| --- | ---: | ---: |
| Held-out policy, n = 1,000 | 2,028 / 2,048 | 0.233966 |
| Covariate-censored survival, n = 1,000 | 2,048 / 2,048 | 0.981183 |

Both satisfy the unchanged 2,000-reported-interval minimum and maximum ratio 1.5.
Every width/truth row and refusal, raw helper summary, independently recomputed
summary, source/plan/helper/binary hash and controlled build receipt is retained.
The compiler inputs came from the clean da811 checkout's existing warm rlibs;
this is controlled build provenance, not an embedded source-SHA signature.
The executable itself is reproducible build output and is not checked in.
`plan.json` preserves the original absolute build paths; a replay must remap those
paths to its own checkout/cache while keeping the source, parameters and domain.

These receipts own the exact manual `estimate +/- Z95 * se` construction under
the measured generating family and evaluation strategy. The public policy and
survival APIs still publish estimates and standard errors, with their existing
`calibration = "unmeasured"` fields. No new interval endpoint or automatic facade
calibration binding is implied.
