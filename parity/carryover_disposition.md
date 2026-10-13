# Interval dispositions

The records below are the checked-in calibration measurements from the v2.2.0 cut, not new measurements. Multi-source mz transport and smoothed-dose intervals remain closed. Joint sensitivity has since gained a promoted Rust/Python interval route returning a tuple; its version-3 assumption-range artifact remains range-only and withholds sampling endpoints. The historical measurements do not attest the current branch’s public wrapper.

`scripts/gate_calibration_attestation.sh` passed against those records: the multi-source mz transport and joint sensitivity facets matched their measurement commits; the smoothed-dose changes used the reviewed bit-identical replay waiver already recorded by that gate. This historical result attests the recorded measurement source at that cut, not the current branch’s public joint-sensitivity wrapper. A release still requires current source-surface attestation after implementation stabilizes.

## Versioned release evidence reports

`python3 scripts/release_evidence_report.py` reports every record of `parity/promotion.toml` including every record's declared status, inference claim, fixture and route citations, and allocated coverage ids. `--json` retains the complete fixture and route metadata for downstream evidence reports. The release inventory invokes this report, and the gate self-tests exercise missing dispositions, missing failed gates, stale dispositions and incorrectly opened carried routes. An inventory success proves metadata consistency; execution, calibration and release certification are explicitly unverified by this report.

`parity/promotion_carryovers.toml` is the enforced disposition contract for every carried-forward record. The report rejects omissions and stale entries, requires concrete failed gates and corrective actions, takes corrective owners directly from the promotion registry, and requires every carried route to retain its closed status, reason code and named refusal evidence. The promotion gate separately checks those routes against their owning support/stage registries and executes their refusal witnesses.

`decision_exit_report.py` reports the registered decision stories against `parity/promotion.toml`. `--promotion` accepts only that canonical registry in the selected tree; alternate paths and unknown flags refuse instead of silently reporting a different registry. `scripts/decision_exit_report_cli_selftest.py` exercises these CLI boundaries.

| Pending exact normal release registration | Remaining evidence obligation | Runtime boundary |
| --- | --- | --- |
| `transport.joint_bayesian` | Completed5c candidate measurement; final scientific-basis surface attestation and installed producer/consumer acceptance. | Exact Gaussian/prior/design guards and current scalar evidence; broader requests refuse. |
| `identify.binary_nested_markov` | Completed full11D Beta1/Beta2 measurement; final surface attestation and normal source replay acceptance. | Only three named95% scalars, original default fit and frozen sampler/diagnostic guards. |
| `identify.binary_nested_markov_fisher` | Completed six Fisher scalar measurements; final surface attestation and normal original-source acceptance. | Interior selected Verma model/default fit; adjacent scopes refuse. |
| `temporal.dependent_interval` | Completed three studentized measurements; final direct-panel and original checked-source consumer acceptance and current attestation. | Exact balanced whole-unit bootstrap-t; failed historical response methods remain separate. |
| `recovery.sampled_observation` | Completed original BCa measurement; corrected scientific-key basis requires real final remeasurement and normal acceptance. | BCa2000 only with original checked recovery proof; the previous key cannot authorize the corrected key. |
| `transport.learned_joint` | Completed four degree2 Gaussian measurements; final scientific-basis surface attestation and original-source replay acceptance. | Degree2 fixed prior/known variance/design; broader candidates remain unmeasured. |
| `counterfactual.transported_path_specific` | The general finite route's Rust producer and independent consumer exist (`counterfactual.transport_general`, with the sampled point route `counterfactual.transport_sampled`; interval closed); the remaining failed gates are the Python public producers with a fresh-process consumer and the calibration-free promotion of the exact finite route. | This path-specific route remains closed until that promotion; separately promoted affine-additive route retains its narrow scope. |

The joint-sensitivity route is now registered as promoted and is no longer a carryover disposition. This historical note preserves the scope limits of its measurements; it does not attest current branch code. The remaining carried disposition is listed in `parity/promotion_carryovers.toml`.

## Multi-source mz transport

Decision: retain `cell_not_licensed` / `mz_transport.interval_withheld` for the joint percentile-bootstrap interval. The estimator is compiled only with `calibration-internal`; the released artifact producer records a withheld interval and its consumer rejects an interval-bearing artifact. The promotion gate therefore lacks a released public producer and independent public consumer for either dependence design. Unknown or incompatible dependence additionally withholds the interval through `transport.unsupported_dependence`.

| Sampling design | n | Observed 95% coverage | Replicates |
| --- | ---: | ---: | ---: |
| Independent studies | 400 | 0.9470 | 2,000 |
| Independent studies | 800 | 0.9625 | 400 |
| Independent studies | 1,600 | 0.9455 | 2,000 |
| Shared units | 200 | 0.9575 | 400 |
| Shared units | 400 | 0.9500 | 400 |
| Shared units | 800 | 0.9425 | 400 |

Source: `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.multi_source_mz_independent_studies` and `...multi_source_mz_shared_units` in `coverage_records.toml`. Corrective owner: `antecedent-estimate`, `antecedent-io`, and facade/Python transport. Opening requires a normal-build estimator, a measured-design-specific public interval route, consumer replay under the same dependence declaration, and exact-coordinate remeasurement after any number-moving change. The existing point route stays licensed.

## Smoothed dose response

Decision: retain `cell_not_licensed` / `dose_response.interval_withheld` at the Rust and Python interval routes. The joint outer-refit percentile bootstrap is `calibration-internal`; the public v1 smoothed-dose artifact has no interval field. A public interval could not pass independent artifact consumption now. These records concern the pointwise smoothed functional `ψ_h` at the tested grid dose; they do not cover a simultaneous band, derivative or unsmoothed truth. Smoothing-bias diagnostics and quadrature differences remain separate from sampling coverage.

| Sampling design | n | Observed 95% pointwise coverage | Replicates | Recorded gate |
| --- | ---: | ---: | ---: | --- |
| Independent samples | 1,050 | 0.9475 | 400 | pass |
| Independent samples | 2,100 | 0.9375 | 400 | pass |
| Independent samples | 4,200 | 0.9410 | 2,000 | pass |
| Nested cohort | 1,050 | 0.9450 | 400 | pass |
| Nested cohort | 2,100 | 0.9475 | 400 | pass |
| Nested cohort | 4,200 | 0.9375 | 400 | pass |

Source: `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.smoothed_dose_transport_independent_samples_psi_h` and `...smoothed_dose_transport_nested_cohort_psi_h`. Corrective owner: `antecedent-estimate`, `antecedent-io`, and facade/Python transport. A new public route needs the exact method and design bound to an interval-bearing artifact with independent replay, followed by the full grid after stabilization. The point grid and its support/quadrature checks remain licensed.

## Joint sensitivity uncertainty (historical measurements; current public tuple route)

The promotion registry now licenses the named Rust and Python interval functions, which return a `JointSensitivityInterval` or `(lower, upper, coverage_target)` tuple at level 0.95 and 199–2000 replicates. This is not an interval artifact: the version-3 range artifact still withholds its interval descriptor and its consumer refuses a forged interval. The coverage figures below are historical measurements, and the current public wrapper has not been re-attested on this branch. At the zero box, the measured target is the two-sided percentile interval. At a positive box, the registered record is one-sided (`side = upper`) and targets the **true upper extremal bound** `U` at the maximizing vertex of the true law. It does not establish two-sided coverage at positive boxes or coverage for arbitrary interior effects. The assumption-range record remains separate.

| Sensitivity box | n | Observed coverage | Replicates | Registered role |
| --- | ---: | ---: | ---: | --- |
| Zero | 200 | 0.9325 | 400 | two-sided gated |
| Zero | 400 | 0.9375 | 400 | two-sided gated |
| Zero | 800 | 0.9525 | 400 | two-sided gated |
| Positive | 200 | 0.9725 | 400 | one-sided upper |
| Positive | 400 | 0.9700 | 400 | one-sided upper |
| Positive | 800 | 0.9775 | 400 | one-sided upper |

Source: `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.z_joint_sensitivity_zero_box` and `...z_joint_sensitivity_positive_box`. The extremal-vertex truth and one-sided acceptance rule are asserted in `joint_sensitivity_calibration.rs`; the endpoint bootstrap's containment behavior is asserted in `joint_mechanism_sensitivity.rs`. The promoted Rust/Python interval producers and version-3 range consumer now exist. These historical measurements do not attest the current branch, so current source-surface attestation is still required before release.

## Other measured or adjacent closed routes

| Exact coordinate or boundary | Disposition |
| --- | --- |
| Frequentist `DirectionalDerivative`, fixed DAG, explicit or accepted structure, additive-GAM plug-in gradient, complete-row IID design, 90% analytic coefficient-sandwich interval, two direction coordinates (`j_0`, `j_1`) | Measured in `cov.directional_derivative.dag.frequentist.analytic_se.l90.*`, but retain point-only public output and `response.derivative_interval_withheld`. The band is internal to `calibration-internal`; a public producer and artifact consumer for its exact identity-scale claim have not passed. Owner: `antecedent-estimate`, response facade/IO. |
| Frequentist `ResponseJacobian`, the same fixed-DAG provider/design, 90% analytic coefficient-sandwich interval, four Jacobian coordinates (`j_0`–`j_3`) | Measured in `cov.response_jacobian.dag.frequentist.analytic_se.l90.*`, but retain point-only public output and `response.derivative_interval_withheld` for the same public-path gap. No transformed-scale or simultaneous band inherits these pointwise records. Owner: `antecedent-estimate`, response facade/IO. |
| Learned continuous trial transport percentile bootstrap, declared IID designs | The grid failed; retain `estimator_inference_mismatch` and the withheld percentile route. The separately calibrated analytic influence interval remains licensed. Any revised percentile method needs a fresh full grid. Owner: `antecedent-estimate`, transport facade/IO. |
| Clustered DML AIPW: one-way 95% t interval versus dyadic/multiway and row bootstrap | One-way interval is already licensed with `cov.average_effect.dag.frequentist.analytic_se.l95.cluster_dml_t_wald_interval`. Dyadic/multiway and row-bootstrap alternatives lack matching whole-method coverage; retain `cluster_interval_not_licensed` / `dyadic_dependence_not_licensed`. Owner: `antecedent-estimate`, clustered facade. |
| Joint-cell, matched/descriptive/tier and partial-family interval extensions | No matching public whole-method calibration and artifact claim is registered for these closed variants; retain their individual reason codes from `promotion_baseline.md` (`penalized_interval_not_licensed`, `ml_nuisance_not_licensed`, `cell_not_licensed`, `score_table_unavailable` or design-specific nonidentification). A measured neighboring point or one-way interval does not open them. Owners: the respective estimator and facade modules. |

The baseline table classifies all 66 closed promotion routes of v2.2.0 individually. No support or stage reason code changes in this decision. A new cell must cite the exact candidate coordinate and pass its own public producer, consumer and calibration gates before a row changes status.

## Conditional ADMG gID scope decision

Do not scope a gID or surrogate-experiment extension of `transport.admg_conditional` as a licensed row. That theorem assumes one source with the complete experimental family and explicitly excludes heterogeneous surrogate experiments. There is no gID theorem-scoped derivation, source-regime factor map, positive and obstruction witnesses, or independently checked provider for a new row. The conditional point route remains licensed only for its stated complete-source family; requests for a surrogate/gID interpretation remain outside that contract. A future gID cell needs a separate promotion record and route rather than widening the existing row.

## Retained failed constructions and distinct corrections

The original run at `623f16dd9d33205b0d0a86b6bb15c4f0be4773a5` measured these failures. The 2,000-replicate rechecks below had zero skipped datasets and 500 successful inner bootstrap replicates. Nominal coverage is 0.95: MCSE is 0.004873397172404482, the three-MCSE lower bound is 0.9353798084827865, and the two-MCSE precision floor is 0.940253205655191. Both unchanged checks apply; an observation inside the wider band can still fail the precision floor.

| Original construction / exact grid coordinate | Initial coverage | Recheck coverage | Recorded failure |
| --- | ---: | ---: | --- |
| `temporal_checked_response_percentile_l95`, p0 / 64 units | 370/400 | 1873/2000 = 0.9365 | Below precision floor |
| `temporal_checked_response_basic_l95`, p0 / 64 units | 366/400 | 1867/2000 = 0.9335 | Below three-MCSE band |
| `temporal_two_step_units_percentile_l95`, p2 / 200 units | 370/400 | 1861/2000 = 0.9305 | Below three-MCSE band |
| `temporal_two_step_units_basic_l95`, p2 / 200 units | 370/400 | 1857/2000 = 0.9285 | Below three-MCSE band |
| `binary_missingness_whole_row_recovery_l95`, p1 / 2,000 rows | 370/400 | 1874/2000 = 0.9370 | Below precision floor |

Source receipts are the original `target/calibration-records/<harness>__<test>.p<point>.recheck.tallies.jsonl` and corresponding logs. These historical failed tallies are not passing coverage records and are not relabeled by the corrected method selections. Their estimator paths and failed harness cases are retired; the registry resolves their original emitters only from immutable Git source at the original measurement commit. They are no longer active release methods or future work obligations. Passing checked paired-effect percentile/basic methods remain selected in the release gate.

The distinct temporal correction is equal-tailed whole-unit bootstrap-t with 500 replicates, exact balanced complete-history unit contributions and positive original/per-resample unit-mean standard errors. Its three new coverage IDs use `bootstrap_studentized` and the suffixes `temporal_two_step_units_studentized_l95`, `temporal_checked_response_studentized_l95`, and `temporal_checked_effect_studentized_l95`. The unit grid stays 50/100/200 and the checked response/effect grid stays 64/128/256; original DGPs, truths, seeds and denominator/precision rules are retained. The checked native `BalancedTemporalEstimator` adapter and independent replay certify only their bounded complete binary-history construction. An overridable public Rust score hook remains unmeasured and does not independently certify arbitrary callbacks. Unit/time identity, aligned standard errors and pivots must replay. The historical source artifact retains unmeasured standing and no finite-sample guarantee. The exact three studentized methods passed their5c76724f grid; final measured-envelope registration depends on normal source replay acceptance and current changed-surface attestation.

The distinct recovery correction is whole-row BCa with 2,000 bootstrap replicates, exact multiplicity-weighted delete-one-row acceleration and midrank bias correction. Its new ID is `cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95`. The original n1000/2000/4000 grid, effect truth 0.36, seeds and coverage/precision denominator rules remain. Any failed bootstrap replicate, invalid delete-one support, degenerate acceleration or unresolved adjusted tail refuses. The version3 `sampled_observation_recovery_bca_v3` artifact binds jackknife summaries, adjustments and all bootstrap work; the failed percentile estimator and version2 runtime are retired. The original BCa source candidate remains unmeasured. Its5c76724f method grid passed, but the corrected native scientific-key basis requires actual final remeasurement before normal measured-envelope activation; neither the old provenance-dependent key nor component variance grants that license.
