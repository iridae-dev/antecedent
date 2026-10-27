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
- Thirteen Rust measured-record consumers are classified by exact assertion in
  the existing test registry. `rust-software` explicitly deselects them;
  `rust-calibration-acceptance` retains the separate post-lockdown selection.
  Default `rust` and `all` still require them. The shared report uses the same
  classification, and the runner verifies each exact, non-ignored assertion in
  its registered binary before applying a selection.
- Six shared calibration-decision controls executed in 126 places each in the
  recorded full run. Their complete assertion bodies now execute once in
  `calibration_decision_contract`; all six pass. Every one of the 756 baseline
  occurrences maps to its replacement in the disposition ledger. No integer
  enumeration, family/domain/prefix case, budget or tolerance was removed.
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

Every `scripts/test.sh` invocation now writes a local JSON receipt under
`target/test-performance/`: exact command, host/build settings, before/after
checkout identities including dirty-file hashes, exit status, wall/CPU costs
and available child-process peak RSS. Changed checkouts are explicit. These
ordinary feedback measurements are not performance acceptance evidence; Cargo
build timings and nextest/pytest durations still separate build and execution
costs. A failed-child control confirms that receipt generation preserves failure.

The schema gate distinguishes explanatory `reason` prose from typed
`calibration_reason`/`reason_code` fields, including within prose-bearing
registries. Device fallback codes remain a separate vocabulary, scoped to the
exact `FallbackReason` implementation; other code methods in that module still
require scientific reason-code validation. Comment/string decoys and formatting
changes have fault controls.

The mechanism dependency correction widened derived facet tags on 17 historical
records. All 1,472 record identities, measurements, original source SHAs, surface
blobs and attestations are unchanged. This adds dependencies; it grants no current
attestation. Retagging no longer builds an unused current-source reference graph:
the tags derive from the record fields and surface keys alone. Twenty-nine
classification controls and 62 collector controls pass.

`bash scripts/test.sh scripts` discovers all script unit tests in a locked,
isolated Python environment. The full run executed 517 controls successfully in
322.91 seconds, with two explicitly pending external-performance-probe consumers.
CI records the three documentation-publication assertions as deselected here;
the documentation job executes them after notebook rendering. Empty selections,
unknown exclusions, failed assertions and unexpected skips fail the runner.

Historical source reads now cache only full Git object identities. Live files,
HEAD, tags and abbreviated references are reread, with mutation controls. The
21 historical receipt controls passed in 49.43 seconds versus the earlier
178.13-second run. Both are ordinary local feedback timings with different
contention and checkout state, not a controlled release performance comparison.


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
compiled evidence-gate fault controls and twenty-four audit/tooling controls pass.
The eight existing independent reference-generator checks pass. The explicit
DoWhy environment passes all seven public-API comparison tests. These results
do not grant coverage or license the deferred calibration study.

The retained statsmodels environment now also reproduces conditional-effect,
binomial-adjustment, temporal-mediation, multiplicity, descriptive-reduction and
GAM-basis fixtures. The three estimate assertions and three multiplicity
assertions pass; the ten foundations assertions pass. These are fixed-input
comparisons. Temporal Sobel uncertainty is checked only under explicit iid
opt-in, and GAM fit outputs remain product regression pins.

Twenty-five retained reference reviews bind generator, transitive comparison
helper, lock, assertion source and fixture payload by digest. Gates reject
changed inputs, expected outputs, tolerances, origins, assertions and helpers
until reviewed again. Reviews record limited independence, redistribution
sources and uncertainty about historical exposure. They do not certify primary
theorem correspondence. A mixed fixture must explicitly name independent
`evidence_fields`; a GAM fitted-output claim cannot borrow its basis comparison.

The linear ATE fixture now has a retained DoWhy 0.14 generator and frozen
environment. Its original CSV, estimate and tolerances are preserved; corrupted
inputs, outputs, methods, scenario coordinates and tolerances fail regeneration.
The treatment is a deterministic function of the confounder: this is a linear
model contrast with parametric extrapolation, not a nonparametric overlap claim.

Three incorrectly external-labelled fixtures now explicitly establish software
contracts: covariance routing labels, sensitivity-grid counts and descriptive
drop-covariate policy thresholds. Their original metadata is retained. The two
affected known-truth declarations instead name the exact executing analytic
positivity counterexample and confounded-effect inequality; no policy label is
used as an independent numerical oracle. Numeric covariance comparisons remain
with the separate retained statsmodels fixture.

Retained isolated Tigramite environments reproduce six LPCMCI motifs, five
pooled J-PCMCI+ cases and four caller-fixed-regime cases, including the complete
upstream endpoint-marked graphs and numeric matrices. Corrupted inputs,
outputs, marks, tolerances and empty/missing selections fail the controls.
Product assertions compare canonical skeletons; they do not establish endpoint
or statistic parity. The fixed-regime comparator runs PCMCI+ independently on
declared segments and establishes no regime-inference claim. Context/dummy-node
tests, weights, missingness, categorical inputs and calibration are outside these
matrices. Original values, pins and deleted-harness histories remain preserved.

Inspection also repaired stale conditional-estimator provenance: the current
implementation supports joint modifiers and adds modifier-mean sampling
variance. The fixture checks its coefficient-variance part after removing that
explicit term; the one-modifier fixture grants no joint-modifier calibration.

The MCMC audit repaired two numerical defects: positive-lag covariance used N-1
instead of N, and the upper-tail indicator used >=q95 rather than the CDF event
<=q95 at tied cutoffs. The first failed an independent 19/48 hand calculation;
the second reported about 184 rather than 100 ESS on the new tied-tail case.
The previous final-pair omission was already fixed; its 8/7 regression remains.
All 189 ordinary probability-library unit tests and both reference tests pass.

Pinned ArviZ tooling reproduces all nine original outputs without replacing
them. A separate 60-digit Decimal/public-SciPy reference checks the full-pair
IPS/IMS equations on the three original chains plus tied-tail and odd-draw cases
at absolute 1e-6. Compatible original 5% ESS comparisons remain. The shifted-chain
truncation and odd-split discrepancies are explicit counterexamples: full-pair
truncation differs from ArviZ's averaged endpoint convention, and the product
keeps its documented drop-last rule. These are limits on comparator parity.

Two more reference reviews and one bounded primary implementation review bind
the exact Vehtari/Geyer sections, assumptions, source, assertions and fixtures.
Citation/deviation-content edits also make that method review stale. There are
two reviewed method records among 495; other primary reviews remain open.

The second bounded primary review covers BH/BY/Bonferroni/Holm adjusted p-values
and the explicitly declared temporal correction family. It verifies the original
BH procedure/theorem, BY harmonic/PRDS results and Holm's ordered rejection
scheme/union-bound theorem, and repairs the missing Holm/Bonferroni citation.
All six primitive controls, three comparator/routing assertions and the five
pinned reference cases pass. Underlying p-value validity and adaptive discovery
families remain outside this review; raw contemporaneous tests do not inherit
the lagged family's FDR control. The Holm citation retains its publisher stable
identifier without asserting a DOI-registry verification that did not resolve.

Artifact migration and incremental particle-filter input fixtures now name their
actual software propositions and executing assertions. Current-format wire
preservation does not establish every historical migration, and agreement of
stepwise/batch paths sharing one filter and RNG is not independent filter
accuracy. Original fixture inputs and tolerances are unchanged; historical
authorship/exposure remains unknown.

The workspace doctest run passes 17 tests across 25 crate targets, with three
existing ignored illustrative snippets (facade GCM/prelude and attribution).
Those snippets provide no executing evidence. Its local execution receipt marks
the changing checkout explicitly; it is not a branch calibration attestation.
The prior-support reference now reproduces 24 domain cases in its own pinned
SciPy environment. Exact evidence fields cover support and coefficient precision;
upstream normalization checks are separate fixture context, not product density
calculations. A constant-only threshold test was removed with a disposition to
the existing executing publication-gate boundary and missing-metric tests. The
Bayesian-check configuration fixture is labelled software-only and retains its
historical external-generator disclosure.

Neither fixed-chain comparison nor source correspondence licenses sampling
calibration, convergence, causal identification or interval coverage.

## Work still required before closing the TODO items

The generated report retains every unresolved finding. In particular, other
external generators remain deleted/unretained, fixtures still need origin
classification, and algorithm/theorem sections and transitive oracle helpers
still need method-by-method independence/implementation review. The route
evidence ledger now accounts for all 711 owned routes, including the previously
missing continuous-conditioning, parameter-spread and measurement bindings.
Their references distinguish exact conditional laws, uncalibrated posterior
spread and refusal contracts; no interval status was promoted. The calibration surface
check now passes with 475 paths in 222 facets. Reviewed repairs remove stale
allowances, assign moved numerical counterfactual modules to mechanism, and
include observation table posteriors and counterfactual transport in that
dependency. Wire/result adapters and local-name collisions have exact reviewed
allowances. Twenty-two dependency/facet controls pass, including the corrected
SID caller inventory after its engine/derive split. No replay waiver or
attestation was created or changed; historical waiver records remain intact.

Five Bayesian configuration fixtures now explicitly name software-only roles,
their exact assertions and bounded propositions (tags, catalog compatibility,
finiteness and sensitivity direction). They supply no independent numerical
truth. Both the report and gates preserve that distinction; an external-origin
tag cannot promote a software-only fixture to known truth.

Complete those reviews and repairs, broaden the remaining method-family grids,
and repeat complete ordinary/feature/CI verification on a stable revision with
comparable build/runtime/resource reports. The documentation pass is being
handled separately. Q08/E14 and Q11 remain deferred until the frozen candidate.

The two original DoWhy identification fixtures now reproduce their complete
printed outputs in a retained locked environment. Product checks require an
estimand expression, since DoWhy prints method headings even when identification
fails. All 481 active identification unit tests pass; three existing tests remain
ignored. These graph-specific comparisons establish no effect-estimation or
uncertainty claim.

Three scalar Tigramite fixtures now retain independent PCMCI/PCMCI+ generators
for ordinary lagged, common-row-masked and contemporaneous-link scenarios. Their
original CSVs, parent sets and endpoint-marked comparator outputs reproduce
exactly. All three product assertions pass; the ordinary PCMCI assertion now
requires its comparator instead of conditionally bypassing it. The PCMCI+ product
assertion still canonicalizes contemporaneous adjacency, so it earns no endpoint
parity claim. Frozen CSV inputs are digest-bound artifacts; corruption and missing
inputs fail the ordinary audit without installing an upstream package.

The Gaussian absolute-scale prior audit found and repaired an input-shape panic:
residual-variance estimation read malformed slices before the delegated fit's
validation. Predictors, outcomes, weights and offsets now refuse with shape
errors before those reads, for both known and estimated residual variance. All
190 probability-library unit tests pass. The provenance record explicitly
separates conjugate scale parameters, power-likelihood weights and the
data-dependent residual-variance plug-in from calibrated uncertainty claims.
This repair does not close the outstanding primary-method review.

The bow-arc fixture now also has a retained DoWhy generator, preserving its
original verbatim output and deleted-harness history. Its product assertion
requires all four standard strategies to fail and no estimand expression to
appear. A single "not found" substring could previously pass alongside another
successful strategy. This comparison establishes the pinned standard-strategy
result; nonidentifiability and certificate correctness require separate evidence.
The strengthened product assertion and all three regenerated DoWhy references
pass.

The complete explicit Rust software selection passed 12,916 assertions in
903.397 seconds of test execution, with 877 default-filter/ignored/calibration
selection skips recorded by nextest. The run retains build, discovery, resource,
checkout and per-assertion receipts under `target/test-performance/`; its checkout
changed during concurrent work, so it is not final frozen-candidate verification
or a controlled before/after performance comparison. Measured-record acceptance
remains explicitly excluded pending lockdown and calibration.

`ordinary_rust_comparison.json` accounts for every absent baseline compiled test
identity using the existing disposition ledger and calibration registry: 756
copies become six controls, thirteen measured-record consumers remain explicit
post-lockdown selections, and four metadata assertions are renamed. Five added
assertions account for the resulting 12,916 executions. No baseline identity is
unaccounted for. Raw execution time increased from 744.162 to 903.397 seconds;
matching source/cache/contention and baseline resource settings were not retained,
so this is coverage accounting and an uncontrolled feedback record, not evidence
of a whole-workspace speedup. A controlled comparison remains open.

The graph interchange assertions now check complete three-node chain adjacency,
including absent extra edges, for each original parse and round trip. A fault
control rejects missing, reversed and extra edges; all three interchange tests
pass. The DOT/JSON and GML/NetworkX-style fixtures are software format contracts,
not independent scientific evidence. The parsing capability uses the same
classification. The PAG input-domain fixture is also software evidence; its
assertion now requires the expected compile-refusal variant and passes.

Optional ordinary feature verification passed 95 learner tests with
`ml-full,ml-neural` and 386 model tests with `gaussian-process,ml-full`, with no
skips or failures. Their doctest targets contain zero examples. These feature
runs remain ordinary verification and do not supply calibration evidence.

The y0 0.2.11 generator now regenerates all 257 counterfactual graph/event
records, including 171 expressions and 328 terms, 66 conflicts and 20 Zero
results. Original cases and disagreements are unchanged. The audit found that
historical `observational` sentinels and constructor-style conflict text were
adapter output, not verbatim package strings. These are disclosed explicitly.
Rendering also varies with set ordering and graph-implied redundant conditioning;
the retained comparison preserves all variables and uses separate NetworkX
d-separation checks on the latent-expanded graph to justify only those changes.
It passes with Python hash seeds 0, 17 and 42. Controls reject removal of a genuine
causal parent and the dependence created by conditioning on a collider.

All three existing product y0 assertions pass: verdict/disagreement accounting,
recorded-output mutations, and evaluation of the original 171 expression cases
against independently enumerated shared-exogenous models at the original 1e-10
scaled tolerance. The assertion name and promotion prose now say recorded rather
than verbatim. The audit discloses that API searches exposed upstream helper
source excerpts; the fixture also preserves its older source-inferred Zero
hypothesis as unverified. No algorithm/test code was copied or translated. These fixed
finite cases do not establish general completeness or calibration.

The next full Python software run completed 7,823 assertions successfully, with
159 explicit skips and 39 calibration deselections. Its sole failure was the
UTF-8 source-I/O guard: three newly retained harnesses omitted explicit encoding.
Those reads are repaired; all four source-only encoding checks pass, and the
scalar Tigramite, DoWhy identification and y0 generators pass again with refreshed
review digests. Later concurrent observation-source edits make the native build
stale, which the guard correctly reports; this run is not current frozen-source
verification of those later edits.
