# Recordless manual interval recipes

`parity/calibration_recipe_receipts.json` binds five policy/survival acceptance cases to their
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

## KNN source-asserted receipts

The two additional KNN cases retain their e15 original recordless gate rows,
protocol marker, raw logs and tally states. Their recorded sizes are total
generated rows: n = 10,000 at the actual execution point 0, and n = 100,000 at
point 2, with eight covariates and three neighbors. The manifest's previous
`donors` label was corrected to `participants`, preserving the original dataset
sizes; these receipts do not claim 100,000 observed control-arm donors.

Each original gate reported all 2,000 intervals. The covered counts were 1,899
and 1,882, respectively. The original source also asserted the unrounded median
width target and printed medians 0.1026 and 0.0322 against effect 0.5. The reader
uses an upper rounding bound rather than pretending the printed four-decimal
value is an exact retained median. The currently collected original rows remain
subject to the ordinary calibration attestation, including their reviewed KNN,
distance-wrapper, shared matching-kernel and measurement-suite dependencies.


## Original GPDC and sensitivity assertion counts

`original_assertion_counts/` retains the two original c30 passed gate receipts and
sealed same-protocol witnesses at 767659966, with raw output, plan, binary hashes,
controller and postprocessing reader. The reviewed source proof admits only each
qualified terminal print of already computed primitive locals. No seed, loop,
scientific branch or 2000-attempt budget changed.

GPDC reported 2000 of 2000, rejected 78 at alpha 0.05, and retained conditional
non-rejection 0.961. It is a conditional-null statistical-test calibration;
interval width is explicitly inapplicable. Sensitivity reported and covered all
2000, with upper-middle median bracket width 0.1 and effect 0.5 (ratio 0.2).
Both scientific harnesses printed one passed and zero failed. The original
controller's line-start-only parser failed after GPDC because libtest printed a
prefix on the same line. The preserved recovery reader extracted only existing
unchanged raw bytes; neither scientific test was rerun. The sensitivity OS exit
code was not directly observed after the controller failed and remains null;
its completed harness result and terminal scientific assertions are retained.

These supplement the original gate receipts. They are not synthetic coverage
records, fresh-family decisions or retrospective FWER claims. Current numerical
dependencies still need the ordinary current-source attestation. Build/run
provenance is controlled and hash-bound, not an embedded cryptographic source
attestation.
