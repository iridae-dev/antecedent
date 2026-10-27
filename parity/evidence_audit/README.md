# Coordinated evidence and test audit

This is an implementation and audit progress record, not closure of the three
Before 7 TODO items. Calibration, numerical replay and performance acceptance
campaigns have not run on this branch. Existing coverage records and licenses
have not been re-stamped or promoted.

## Reproduce the inventory

```sh
python3 scripts/evidence_audit.py report
python3 scripts/evidence_audit.py report --json
python3 scripts/verify_provenance_citations.py --check
python3 scripts/check_v3_test_registry.py
```

The inventory joins existing registries; it is not another source of licenses.
It records exact assertions, scenarios/axes, fixture origins and tolerances,
source hashes, method/provenance links, calibration references, required target
features, doctest source candidates and coverage dispositions. Source review
and executing-test discovery are explicitly distinct. Bibliographic title
identity does not certify theorem sections or correspondence with code.

All 455 active DOI citations now match retained primary registry titles. The
original title/bibliographic wording remains in `citation_note` where it changed.
The nonresolving ACM identifier on Zhang's ancestral-graph paper was replaced
with the primary JMLR publication URL. Offline gates reject a missing cache,
unresolved DOI or title mismatch; explicit `--fetch` uses Crossref/DataCite.

## Suite and reference changes

- The complete registry now includes inline Rust unit tests and transitive
  integration modules. Stale suite maps and split-module citations are repaired.
  Comment/string decoys and disabled includes do not establish discovery.
- `bash scripts/test.sh integration` selects integration binaries;
  `bash scripts/test.sh scientific` executes exact cited scientific assertions.
  Statistical smoke checks remain ordinary software tests and are labelled
  separately. Measurements remain with `scripts/measure_calibration.sh`.
- Python assertions requiring currently attesting measurements carry the explicit
  `requires_calibration` marker. `python-software` reports those deselections;
  `python-calibration-acceptance` selects them after lockdown and calibration.
  Default Python commands and `all` retain them as required assertions: no skip,
  xfail or fabricated attestation turns an unmeasured branch green.
  The audit inventories these assertions from their source markers.
- The learner ATE 200-replicate/four-method coverage study is explicitly ignored
  by ordinary tests and registered in the calibration driver for Q08. Its
  replicate count, truth, bias/coverage bounds and influence assertions remain.
  A new one-scenario/four-method point check also excludes the biased unadjusted
  contrast; it cannot substitute for that deferred measurement.
- Local Python test reuse verifies native source inputs, ABI, binary bytes and
  relevant build settings. Inputs must stay unchanged during a build. Build
  receipts live under `target/`, outside the distributed Python package. The
  runner handles inherited Conda prefixes and coexisting interpreter ABIs.
- Retained, locked statsmodels tooling reproduces 34 fixed scenarios: six original
  OLS/WLS cases, eleven sandwich covariance comparisons, five GLM families and
  twelve new continuous/categorical/weighted designs, including HC3 uncertainty.
  Context-only outputs are distinguished from fields the product test asserts.
- Oversized Newey–West lags use different Bartlett denominators in the two
  implementations. The product test now compares matched effective lags and
  separately asserts the difference from the original raw lag-20 output.
- The existing DoWhy public-API comparison runs in its own locked environment:
  `uv run --frozen --script scripts/run_dowhy_comparison.py`. Its explicit runner
  rejects missing, skipped and failed required assertions and retains an execution
  receipt under `target/scientific-conformance/`. Ordinary CI need not install it.
- Fixed-reference regeneration checks use declared tolerances. Fault controls
  exercise corrupted numerical expectations, widened tolerances, empty grids,
  wrong origins/inference, missing assertions, skipped comparator cases, stale
  native inputs/binaries and edits during a build. Rust fault controls invoke the
  actual point/prediction/RSS/HC3 comparisons.

No numerical case or acceptance budget was deleted. The assertion rename,
measurement separation, repaired citations and duplicated documentation check
have explicit coverage dispositions in `parity/test_refactor_dispositions.toml`.

## Feedback measurements

On this macOS arm64 host, the registry checker on the 1,594-row intermediate
inventory took 82.90 seconds before per-crate target-map reuse and 3.42 seconds
afterward (43.95/3.03 user seconds, 38.06/0.35 system seconds). The latter run
used about 73 MB peak RSS. These are individual local checks, not release
performance measurements. The baseline had one stale suite-map entry; its
repair accounts for the changed verdict, not the speedup. The final inventory
adds hundreds of inline-source registrations; do not treat the intermediate
timings as a full-workspace build/runtime result. The final 2,388-row inventory
also has a separate timing/RSS receipt in `registry_timings.json`.

The focused ordinary statistical target passes ten tests, retaining its existing
ignored fixture writer. The new learner point assertion passes. Thirty-nine
compiled evidence-gate fault controls and twenty-one audit/tooling controls pass.
The eight existing independent reference-generator checks pass. The explicit
DoWhy environment passes all seven public-API comparison tests. These results
do not grant coverage or license the deferred calibration study.

The retained statsmodels environment now also reproduces conditional-effect,
binomial-adjustment, temporal-mediation, multiplicity, descriptive-reduction and
GAM-basis fixtures. The three estimate assertions and three multiplicity
assertions pass; the ten foundations assertions pass. These are fixed-input
comparisons. Temporal Sobel uncertainty is checked only under explicit iid
opt-in, and GAM fit outputs remain product regression pins.

Ten retained reference reviews bind generator, transitive comparison
helper, lock, assertion source and fixture payload by digest. Gates reject
changed inputs, expected outputs, tolerances, origins, assertions and helpers
until reviewed again. Reviews record limited independence, redistribution
sources and uncertainty about historical exposure. They do not certify primary
theorem correspondence. A mixed fixture must explicitly name independent
`evidence_fields`; a GAM fitted-output claim cannot borrow its basis comparison.

Inspection also repaired stale conditional-estimator provenance: the current
implementation supports joint modifiers and adds modifier-mean sampling
variance. The fixture checks its coefficient-variance part after removing that
explicit term; the one-modifier fixture grants no joint-modifier calibration.

## Work still required before closing the TODO items

The generated report retains every unresolved finding. In particular, other
external generators remain deleted/unretained, fixtures still need origin
classification, and algorithm/theorem sections and transitive oracle helpers
still need method-by-method independence/implementation review. The route
evidence ledger now accounts for all 711 owned routes, including the previously
missing continuous-conditioning, parameter-spread and measurement bindings.
Their references distinguish exact conditional laws, uncalibrated posterior
spread and refusal contracts; no interval status was promoted. The calibration surface
check also exposes existing cross-facet/stale-allowance findings; no blanket
allowance or replay waiver was introduced to hide them.

Complete those reviews and repairs, broaden the remaining method-family grids,
and repeat complete ordinary/feature/CI verification on a stable revision with
comparable build/runtime/resource reports. The documentation pass is being
handled separately. Q08/E14 and Q11 remain deferred until the frozen candidate.
