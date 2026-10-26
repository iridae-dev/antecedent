# Strict measured-license failure disposition

Read-only source/hash audit, 2026-10-25. Examined `/private/tmp/engineering-facade-tests.log`, current source, shipped snapshot and historical Git objects. No Cargo/build, calibration run, record/test edit or attestation repin performed.

## Definitive initial-state evidence

Initial commit: `6ad678830b167f7c3bf398850e2ce2125c6c387c`. Audited HEAD: `41b057c5e9ad58bfe23e6bcf224550b489b88825` (working tree contains later engineering edits).

`crates/antecedent-io-core/data/coverage_records.json` is byte-identical at initial commit, audited HEAD and working tree: **1472 records, ZERO `attesting_record_ids`**. SHA-256 in all three: `263714e05b029a88e2e19ab883f53416d9c918f5114c4ee6cb6dd66199bcec2c`.

`crates/antecedent-io-core/src/coverage_records_data.rs` loads exactly that snapshot. `calibration.rs::record_attests_current_code` searches only the generated attesting-ID list. An empty list makes every shipped record non-attesting. Therefore the initial tree already could not satisfy these strict successful measured-license assertions. This is a source-level proof of the initial licensing predicate, not a claim to have run the initial numerical tests.

All governing records listed below carry measurement commit **`c30e10304f4bd85a01ad81dd133a360ddc95c2f1`**, not initial or current source. For a concrete shared-code hash comparison, `crates/antecedent-core/src/lib.rs` Git blob at measurement is `83cee9e97ff99646adade8830490ae5e8ff28ddf`; initial and audited HEAD are `89dd4b6a8b95dae12b6e199a4a16b99c5ecc3404`. `operator.rs` and `physical_operator.rs` are absent at that measurement commit and present at initial/current, explicitly classified core today. These examples establish actual source drift; they do not claim that every changed line alters statistical results. The generated empty membership is the decisive runtime fact. There is no single runtime executable-code hash field: `calibration_sha` names measured source; generated membership attests its relevant surface.

## Nine failures and exact governing historical records

IDs below use the full prefix shown per family. `calibration_slot_with` chooses lowest governing coverage among construction-matched covering records, ties by ID; its ordinary non-boundary rule uses the record's worst coverage, not the nearest grid-point coverage. Thus the governing scenario can differ from the fixture scenario. Every listed row has no attesting membership.

| Failing test file / assertion | Governing historical record ID | Source observation |
| --- | --- | --- |
| `adaptive_inference_facade.rs:381`, group sequential repeated interval | `cov.adaptive_experiment_contrast.not_applicable.frequentist.repeated_confidence_interval.l95.group_sequential_repeated_interval_null_effect_l95` | Three effect-size rows share construction; null row governs at 240 observations (worst .935). Typed `coverage_record_not_attesting`. |
| `adaptive_inference_facade.rs:326`, Thompson interval | `cov.adaptive_experiment_contrast.not_applicable.bayesian.posterior_quantile.l95.thompson_posterior_prior_predictive_stop_099_boundary_l95` | Stop .90/.95/.99 rows share construction. Despite its historical name, .99 row and its grid are not boundary-flagged in shipped data; it governs at 200 observations (.9325). Typed `coverage_record_not_attesting`. No record relabeling performed. |
| `adaptive_inference_facade.rs:351`, weighted contrast and sequence | `cov.adaptive_experiment_contrast.not_applicable.frequentist.analytic_se.l95.epsilon_greedy_repeated_measures_weighted_contrast_l95` | Weighted contrast at 400 observations fails first; later sequence assertion is not executed. |
| `composed_uncertainty_facade.rs:285`, composed interval | `cov.composed_quantity.not_applicable.frequentist.normal_approximation.l95.composed_contrast_aligned_bootstrap_gaussian_l95` | Refusal explicitly names this record; N=200, aligned-joint-law contrast. Error label says `coverage_record_unmeasured`; historical measurement exists but is unattested for current code. |
| `evidence_ledger_facade.rs:659`, pooled interval loop | `cov.pooled_effect.no_graph.frequentist.wald_z.l95.pooled_fixed_effect_twenty_studies` | First fixed-effect call: functional `pooled_mean;fixed_effect;studies=5-30`, smallest-study N=100; all six study-count rows share this key, twenty-study row has worst .935. Important error-masking detail below. |
| `scientific_bridge_interval_facade.rs:340`, measurement correction | `cov.bridged_claim_interval.not_applicable.frequentist.normal_delta_method_clipped.l95.latent_measured_correction_prevalence_0_2_l95` | Prevalence .2 and .5 rows share construction; .2 governs at validation N=400 (.93). |
| `scientific_bridge_interval_facade.rs:373`, trial prediction | `cov.bridged_claim_interval.not_applicable.frequentist.student_t_prediction_interval.l95.surrogate_trial_level_prediction_interval_l95` | 20 validation trials, level .95. |
| `predictive_score_interval_facade.rs:147`, CRPS mean | `cov.predictive_score_interval.not_applicable.frequentist.student_t_iid.l95.iid_crps_mean_l95` | Held-out Gaussian CRPS, iid, N=800, level .95. |
| `predictive_score_interval_facade.rs:173`, serial log score | `cov.predictive_score_interval.not_applicable.frequentist.student_t_bartlett_hac.l95.serial_log_score_mean_l95` | Held-out Gaussian log score, serial, N=1600, level .95; later score-difference and CRPS-boundary assertions are not reached. |

Pooling diagnostic nuance: `evidence_ledger.rs:1135` tries the family basis followed by legacy exact-study-count basis and uses `reduce(Result::or)`. The first correctly matched family basis cannot attest. The fallback uses `pooled_mean;fixed_effect;studies=5` and total N=500, which does not match these current family records; its `estimator_grid_not_measured` error replaces the first refusal. This is a preexisting diagnostic masking behavior, not evidence of a missing fixed-effect grid, nor a new numerical failure. The strict licensing failure remains inevitable with empty membership. A future diagnostics cleanup can preserve both refusal reasons; no source change made under this read-only assignment. Later Gaussian/random-effect branches of this test were not reached; historical Gaussian-study records at `70a71dd33bc55cbc2398b0f91c4a2c797afad618` must not be claimed validated by this run.

## Owners and closure

Runtime licensing: `antecedent-io-core/src/calibration.rs`; generated payload: `antecedent-io-core/data/coverage_records.json`; generator: `scripts/generate_support_matrix_docs.py::{attesting_record_ids,_live_attesting_record_ids}` delegates to `calibration_facets.assess` and explicitly refuses cached attestation restoration. Measurement registry/harness owners: `parity/coverage_records.toml` and each record's registered `test`/`dgp`. Facade owners retain the nine strict measured-claim assertions; they have not been weakened, ignored or repinned.

**Disposition: preexisting missing current-code measurement attestation, pending Q08/Q09; fail-closed refusal is correct.** TODO K1.5 freezes implementation after Q13/Q16; Q08 requires whole-procedure controlled-reference-hardware grids at that frozen candidate; Q09 collects records preserving measurement commits and attests facets. Smoke/component tests and facet-map success do not license these routes. Engineering must report these nine assertions as failing/pending measured evidence rather than green. This does not waive Q08/Q09 or any release gate, and does not establish scientific coverage of changed code.

The broad log also has independent implementation failures (retained static cross-world execution, refresh/configuration, smoothed artifact lifecycle, etc.). Parent/facade owners are fixing those separately. This disposition applies only to the nine measured-license failures above and cannot be used to dismiss other failures.

Reproduction used standard-library Python `json`, `hashlib`, `subprocess` with `git show REV:path`, plus `git rev-parse REV:path` and source reads; no build or competing measurement process.

## Two additional IO strict-license assertions in final integration batch

The expanded final run `/private/tmp/engineering-final-rust-tests.log` exposes two additional measured-license failures, bringing this disposition to **eleven failing strict-license assertions**. The same unchanged snapshot proof applies: initial `6ad678830b167f7c3bf398850e2ce2125c6c387c` and current source ship1472 records and zero attesting IDs, SHA-256 `263714e05b029a88e2e19ab883f53416d9c918f5114c4ee6cb6dd66199bcec2c`. New lifecycle assertions added in this pass can expose the preexisting empty licensing predicate; this does not claim these newly added tests existed or were executed at initial commit.

1. `crates/antecedent-io/src/measured_inference/transport.rs:340`, `native_lifecycle_tests::every_measured_transport_variant_matches_original_execution_basis_bit_for_bit`. First loop case is one source, varying Intercept, IndependentVaryingBlocks,150 rows,4096 draws. `JointBayesianArtifactWire::build_measured` refuses target_effect with `coverage_record_not_attesting`. Exact record: `cov.classical_transport.dag.bayesian.posterior_quantile.l95.joint_gaussian_one_source_intercept_l95`; measurement SHA `c30e10304f4bd85a01ad81dd133a360ddc95c2f1`; functional `target_average_effect.Intercept.IndependentVaryingBlocks.sources1`, measured N150–600. Measurement owner/test: `crates/antecedent-estimate/tests/joint_bayesian_transport_calibration.rs::joint_gaussian_one_source_intercept_l95`; runtime/artifact owner: `antecedent-io/src/measured_inference/transport.rs` and `joint_bayesian_transport_artifact.rs`. Later loop constructions (two independent intercepts, two shared intercepts, two varying covariates) have exact sibling records `joint_gaussian_two_independent_intercepts_l95`, `joint_gaussian_two_shared_intercepts_l95`, `joint_gaussian_two_varying_covariates_l95` under the same prefix and SHA, all un-attesting, but those strict lifecycle assertions are not reached by this failing test run. No independent replay success can be claimed from this failure.

2. `crates/antecedent-io/tests/sampled_counterfactual_transport_artifact.rs:481`, `a_measured_construction_reports_an_interval_bound_to_its_coverage_record`. `measure_at(1000, UnseenCellRule::Refuse)` selects catalog `transportable_latent_district`, target `Y_{x1}=1`, and `Wire::measure` refuses target_probability with `coverage_record_not_attesting`. Exact record: `cov.counterfactual_transport.admg.frequentist.delta_method_wald.l95.sampled_latent_district`; measurement SHA `c30e10304f4bd85a01ad81dd133a360ddc95c2f1`; estimator `sampled_counterfactual_transport`, measured units/table250–4000. Its full functional identifies shared-latent-exogenous coupling, do0=1, world0 variable2=1, source selection variable1. Measurement owner/test: `crates/antecedent-estimate/tests/counterfactual_transport_sampled.rs::sampled_latent_district`; artifact/runtime owner: `antecedent-io/src/sampled_counterfactual_transport_artifact.rs` and its integration test. Later measured consume/replay assertions are not reached.

Both remain explicit Q08/Q09 evidence work after K1.5, with strict assertions unchanged. No record, test expectation or attestation edit made. These failures cannot stand in for passing measured artifact lifecycle gates; software-only refusal/corruption/cleanup checks must be reported separately.

## Three design-suite failures exposed by the expanded context run

Read-only triage of `/private/tmp/engineering-final-broad-context-tests-v2.log`
identifies **two further strict measured-license assertions and one historical
promotion assertion**. Together with the eleven above, these are fourteen
measured-evidence-related test failures, not fourteen independent scientific or
software defects. No source, fixture, record or attestation was changed for this
triage.

The shipped JSON is still byte-identical to initial `6ad678830b167f7c3bf398850e2ce2125c6c387c`:
1472 records, zero attesting ids, SHA-256
`263714e05b029a88e2e19ab883f53416d9c918f5114c4ee6cb6dd66199bcec2c`.
The three test files, `decision_reliability/calibration.rs` and
`decision_reliability/artifact.rs` are also byte-identical to that initial commit.
The unchanged `decision_reliability_v1.bin` fixture has SHA-256
`ee66aab8ac62a99af56777d954fb11839ceaa7df518e83616221fb8f8d36eb78`.
These comparisons establish the original licensing/test premise from source and
fixture bytes; they do not claim an initial-checkout numerical test execution.

### Strict successful-publication assertions

- `crates/antecedent-design/tests/decision_reliability.rs:816`,
  `every_published_route_is_licensed_inside_a_measured_coverage_record`, fails the
  first `PublishedRoute::ALL` iteration, `RankingProbability`.
- `crates/antecedent-design/tests/decision_reliability_intervals.rs:509`,
  `a_derivation_is_published_inside_a_measured_record_and_refused_outside_it`,
  likewise fails the first `derived.publish(RankingProbability)` call.

Both coordinates are two candidates, 1000 draws, maximize expected utility,
shared simulation, known law, independent unweighted streams and no adaptive
candidate generation, at level .95. The banded key is
`candidates=2-4;rule=maximize_expected_utility;functional=expected_utility`.
The exact-count compatibility fallback has no matching shipped record. Runtime
`license_route_at` flattens unsuccessful calibration slots into `CellNotLicensed`;
the matching banded slot cannot calibrate because its governing record does not
attest current code.

The governing records for this coordinate are listed below. Only the first row's
publication assertion was reached in these two tests; the other rows document the
remaining loop obligations, not successful execution. All carry measurement SHA
`c30e10304f4bd85a01ad81dd133a360ddc95c2f1` and no attesting membership.

| Route | Exact governing record ID |
| --- | --- |
| Ranking probability | `cov.decision_reliability.not_applicable.frequentist.wilson_effective_sample.l95.ranking_probability_three_candidates_l95.shared_rho_0_5` |
| Optimality confidence set | `cov.decision_reliability.not_applicable.frequentist.bonferroni_confidence_set.l95.confidence_set_three_candidates_budget_500_l95.shared_rho_0_0` |
| Utility interval | `cov.decision_reliability.not_applicable.frequentist.wald_normal.l95.utility_interval_two_candidates_l95.shared_rho_0_5` |
| Selection-corrected interval | `cov.decision_reliability.not_applicable.frequentist.simultaneous_wald_normal.l95.selection_corrected_two_candidates_l95.shared_rho_0_0` |

The three-candidate ranking record legitimately governs a two-candidate fixture:
both use the measured 2–4 band; its worst coverage .945 ties the two-candidate
record, and the documented id tie-break selects `three` before `two`. The
confidence-set records all have coverage 1; the budget-500 three-candidate record
wins the same tie-break. This is not a missing two-candidate grid.

The first test does not reach its final empty-closure assertion or subsequent
other-level/dependent-stream refusals. The second does not reach its remaining
route publications or other-level, simultaneous-multiplicity and small-budget
refusal assertions. The measurement owner is
`crates/antecedent-design/tests/decision_reliability_interval_calibration.rs`,
with tests named by each record's scenario above; licensing owners are
`decision_reliability/calibration.rs` and the shared IO-core calibration runtime.

### Historical closure fixture: failed promotion premise, not failed decoding

`artifact_compat_sequential_reliability_golden::historical_closure_snapshot_is_conservative_and_cannot_change_the_report`
fails at `artifact_compat_sequential_reliability_golden.rs:248`, the strict
`original.report.closed_routes.len() > current.closed_routes.len()` assertion.
The preceding decode, current `assess`, and `original.verify` all completed in
this run. Independently reading the unchanged container's CBOR body after Zstd
decompression confirms three candidates, 300 draws, shared independent simulation,
expected utility and **all four historical closed routes**. Empty attesting
membership closes all four currently as well, so the assertion requires `4 > 4`.
Its comment assumes newly earned licenses that this shipped snapshot does not
supply.

At that 300-draw coordinate, ranking probability and utility interval are outside
the measured range. Confidence-set and selection-corrected records do cover
250–1000 draws but cannot attest. Their exact governing ids are:

- `cov.decision_reliability.not_applicable.frequentist.bonferroni_confidence_set.l95.confidence_set_three_candidates_budget_500_l95.shared_rho_0_0`
- `cov.decision_reliability.not_applicable.frequentist.simultaneous_wald_normal.l95.selection_corrected_three_candidates_budget_500_l95.shared_rho_0_5`

Both retain the same measurement SHA above. The second record is owned by
`decision_reliability_interval_calibration.rs::selection_corrected_three_candidates_budget_500_l95`;
the first by its corresponding `confidence_set_three_candidates_budget_500_l95`
test. The compatibility runtime owner is `decision_reliability/artifact.rs`:
`conservative_closure_snapshot` permits the canonical carried refusals to remain
closed and requires every current refusal to be present. That verification
succeeded; no artifact compatibility or numerical recomputation defect is
established by this failure.

The failing assertion prevents this test's later consumed digest/closure checks
and resealed-mutation rejection checks from executing. They must not be reported
as covered by this test run. The separate golden recompute/version-gating test
passed in the same log, but does not substitute for those unreached assertions.

**Disposition:** all three failures expose preexisting missing-current-code
measurement/promotion evidence. Keep strict assertions and fixture intact; Q08/Q09
must earn the relevant licenses after the implementation freeze. This evidence
neither waives those gates nor excuses independently identified software failures
such as the separately owned cell-AIPW context propagation defect.
