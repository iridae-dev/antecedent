# 2.3 C0 interval dispositions

The values below are the checked-in 2.2 calibration records, not new 2.3 measurements. All three public interval families remain closed at kickoff. Their point estimates and, for X3, the assumption range remain independently available. An internal calibrated method does not license a public producer or consumer.

At the 2.3 kickoff, `scripts/gate_calibration_attestation.sh` passed: the X1 and X3 facets matched their measurement commits; the smoothed-dose changes used the reviewed bit-identical replay waiver already recorded by that gate. This historical result attests those internal measurements, not the current branch or a public interval route. The release cut requires fresh attestation after implementation stabilizes.

## Versioned release evidence reports

`python3 scripts/release_evidence_report.py` reports both `parity/promotion_2_2.toml` and `parity/promotion_2_3.toml`, including every record's declared status, inference claim, fixture and route citations, and allocated coverage ids. `--json` retains the complete fixture and route metadata for downstream evidence reports; `--release-version 2.3` selects the current release. The release inventory invokes this report, and the gate self-tests exercise missing dispositions, missing failed gates, stale dispositions and incorrectly opened carried routes. An inventory success proves metadata consistency; execution, calibration and release certification are explicitly unverified by this report.

`parity/promotion_carryovers.toml` is the enforced disposition contract for every carried-forward record in either release. The report rejects omissions and stale entries, requires concrete failed gates and corrective actions, takes corrective owners directly from the promotion registry, and requires every carried route to retain its closed status, reason code and named refusal evidence. The promotion gate separately checks those routes against their owning support/stage registries and executes their refusal witnesses.

The legacy `b_exit_report.py` defaults to the retained 2.2 B stories. `--promotion` accepts only the canonical `parity/promotion_2_2.toml` or `parity/promotion_2_3.toml` registry in the selected tree; alternate paths and unknown flags refuse instead of silently reporting a different registry. Selecting 2.3 explicitly emits its inventory; it refuses 2.2 story/interval inputs and cannot satisfy `--release` or implementation/calibration requirements. With `--self-test`, both reporter self-tests execute. The old B story map cannot serve as evidence for the different 2.3 package map. `scripts/b_exit_report_cli_selftest.py` exercises these CLI boundaries.

| Current carried-forward record | Precise failed obligations | Unchanged public refusal |
| --- | --- | --- |
| `2.3A.X4.joint_bayesian_transport` | Whole conjugate-Gaussian source-target posterior calibration is unmeasured. | `cell_not_licensed` / `bayesian_transport.route_frozen` |
| `2.3A.X4.binary_nested_markov_pilot` | Separate licensed identification binding, posterior/interval implementation, and whole-posterior calibration. The existing MLE point fit does not implement the frozen posterior claim. | `cell_not_licensed` / `nested_markov.route_frozen` |
| `2.3A.X5.dependent_temporal_interval` | Whole dependence-preserving temporal interval calibration is unmeasured. | `cell_not_licensed` / `temporal_interval.route_frozen` |
| `2.3A.X8.transported_path_specific_counterfactual` | General joint cross-world transport theorem and matching provider/artifact evidence. The separate affine-additive row does not license this route. | `cell_not_licensed` / `transported_counterfactual.route_frozen` |
| `2.3A.X10.sampled_observation_recovery` | Whole recovery-path interval calibration, including overlapping-margin dependence, is unmeasured. | `cell_not_licensed` / `sampled_recovery.route_frozen`; component-only variance also refuses with `sampled_recovery.component_variance_only` |
| `2.3A.X4.learned_joint_transport` | Whole polynomial-basis conjugate-Gaussian source-target posterior calibration is unmeasured. | `cell_not_licensed` / `learned_joint_transport.route_frozen` |

The disposition contract also includes the retained `2.2B.X3.joint_sensitivity_uncertainty` public producer/consumer gap described below. No status, route license, coverage record or waiver is changed by reporting these blockers. Calibration remains the final step after implementation stabilizes.

## X1 multi-source mz transport

Decision: retain `cell_not_licensed` / `mz_transport.interval_withheld` for the joint percentile-bootstrap interval. The estimator is compiled only with `calibration-internal`; the released artifact producer records a withheld interval and its consumer rejects an interval-bearing artifact. The 2.3 promotion gate therefore lacks a released public producer and independent public consumer for either dependence design. Unknown or incompatible dependence additionally withholds the interval through `transport.unsupported_dependence`.

| Sampling design | n | Observed 95% coverage | Replicates |
| --- | ---: | ---: | ---: |
| Independent studies | 400 | 0.9470 | 2,000 |
| Independent studies | 800 | 0.9625 | 400 |
| Independent studies | 1,600 | 0.9455 | 2,000 |
| Shared units | 200 | 0.9575 | 400 |
| Shared units | 400 | 0.9500 | 400 |
| Shared units | 800 | 0.9425 | 400 |

Source: `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.multi_source_mz_independent_studies` and `...multi_source_mz_shared_units` in `coverage_records.toml`. Corrective owner: `antecedent-estimate`, `antecedent-io`, and facade/Python transport. Opening requires a normal-build estimator, a measured-design-specific public interval route, consumer replay under the same dependence declaration, and exact-coordinate remeasurement after any number-moving change. The existing point route stays licensed.

## X4 smoothed dose response

Decision: retain `cell_not_licensed` / `dose_response.interval_withheld` at the Rust and Python interval routes. The joint outer-refit percentile bootstrap is `calibration-internal`; the public v1 smoothed-dose artifact has no interval field. A public interval could not pass independent artifact consumption now. These records concern the pointwise smoothed functional `ψ_h` at the tested grid dose; they do not cover a simultaneous band, derivative or unsmoothed truth. Smoothing-bias diagnostics and quadrature differences remain separate from sampling coverage.

| Sampling design | n | Observed 95% pointwise coverage | Replicates | Recorded gate |
| --- | ---: | ---: | ---: | --- |
| Independent samples | 1,050 | 0.9475 | 400 | pass |
| Independent samples | 2,100 | 0.9375 | 400 | pass |
| Independent samples | 4,200 | 0.9410 | 2,000 | pass |
| Nested cohort | 1,050 | 0.9450 | 400 | pass |
| Nested cohort | 2,100 | 0.9475 | 400 | pass |
| Nested cohort | 4,200 | 0.9375 | 400 | pass |

Source: `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.smoothed_dose_transport_independent_samples_psi_h` and `...smoothed_dose_transport_nested_cohort_psi_h`. Corrective owner: `antecedent-estimate`, `antecedent-io`, and facade/Python transport. A new public route needs the exact method and design bound to an interval-bearing artifact with independent replay, followed by the full grid after stabilization. The 2.2 point grid and its support/quadrature checks remain licensed.

## X3 joint sensitivity uncertainty

Decision: retain `cell_not_licensed` / `joint_sensitivity.interval_withheld` for `2.2B.X3.joint_sensitivity_uncertainty`. The interval calculation remains `calibration-internal`, and no public interval artifact exists. The zero box is a two-sided percentile interval. At a positive box, the reported conservative interval's upper endpoint targets the **true upper extremal bound** `U`, the effect at the maximizing vertex of the true law; the registered record is one-sided (`side = upper`). Coverage of an arbitrary interior effect is not a two-sided claim licensed by that record. The 2.2 assumption range is unaffected.

| Sensitivity box | n | Observed coverage | Replicates | Registered role |
| --- | ---: | ---: | ---: | --- |
| Zero | 200 | 0.9325 | 400 | two-sided gated |
| Zero | 400 | 0.9375 | 400 | two-sided gated |
| Zero | 800 | 0.9525 | 400 | two-sided gated |
| Positive | 200 | 0.9725 | 400 | one-sided upper |
| Positive | 400 | 0.9700 | 400 | one-sided upper |
| Positive | 800 | 0.9775 | 400 | one-sided upper |

Source: `cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.z_joint_sensitivity_zero_box` and `...z_joint_sensitivity_positive_box`. The extremal-vertex truth and one-sided acceptance rule are asserted in `joint_sensitivity_calibration.rs`; the endpoint bootstrap's containment behavior is asserted in `joint_mechanism_sensitivity.rs`. Corrective owner: `antecedent-validate`, `antecedent-estimate`, `antecedent-io`, and facade/Python transport. Promotion requires a public interval producer, a versioned consumer that replays the assumption and sampling coordinates, and remeasurement if those numerical paths change.

## Other measured or adjacent closed 2.2 routes

| Exact coordinate or boundary | 2.3 disposition |
| --- | --- |
| Frequentist `DirectionalDerivative`, fixed DAG, explicit or accepted structure, additive-GAM plug-in gradient, complete-row IID design, 90% analytic coefficient-sandwich interval, two direction coordinates (`j_0`, `j_1`) | Measured in `cov.directional_derivative.dag.frequentist.analytic_se.l90.*`, but retain point-only public output and `response.derivative_interval_withheld`. The band is internal to `calibration-internal`; a public producer and artifact consumer for its exact identity-scale claim have not passed. Owner: `antecedent-estimate`, response facade/IO. |
| Frequentist `ResponseJacobian`, the same fixed-DAG provider/design, 90% analytic coefficient-sandwich interval, four Jacobian coordinates (`j_0`–`j_3`) | Measured in `cov.response_jacobian.dag.frequentist.analytic_se.l90.*`, but retain point-only public output and `response.derivative_interval_withheld` for the same public-path gap. No transformed-scale or simultaneous band inherits these pointwise records. Owner: `antecedent-estimate`, response facade/IO. |
| Learned continuous trial transport percentile bootstrap, declared IID designs | The 2.2 grid failed; retain `estimator_inference_mismatch` and the withheld percentile route. The separately calibrated analytic influence interval remains licensed. Any revised percentile method needs a fresh full grid. Owner: `antecedent-estimate`, transport facade/IO. |
| Clustered DML AIPW: one-way 95% t interval versus dyadic/multiway and row bootstrap | One-way interval is already licensed with `cov.average_effect.dag.frequentist.analytic_se.l95.cluster_dml_t_wald_interval`. Dyadic/multiway and row-bootstrap alternatives lack matching whole-method coverage; retain `cluster_interval_not_licensed` / `dyadic_dependence_not_licensed`. Owner: `antecedent-estimate`, clustered facade. |
| E5 joint-cell, E7 matched/descriptive/tier and E3 partial-family interval extensions | No matching public whole-method calibration and artifact claim is registered for these closed variants; retain their individual reason codes from `baseline_2_2_for_2_3.md` (`penalized_interval_not_licensed`, `ml_nuisance_not_licensed`, `cell_not_licensed`, `score_table_unavailable` or design-specific nonidentification). A measured neighboring point or one-way interval does not open them. Owners: the respective E3/E5/E7 estimator and facade modules. |

The baseline table classifies all 66 closed 2.2 promotion routes individually. No support or stage reason code changes in this C0 decision. A new 2.3 cell must cite the exact candidate coordinate and pass its own public producer, consumer and calibration gates before a row changes status.

## Conditional ADMG gID scope decision

Do not scope a gID or surrogate-experiment extension of `2.2B.X2.admg_conditional_transport` as a 2.3 A licensed row. That 2.2 theorem assumes one source with the complete experimental family and explicitly excludes heterogeneous surrogate experiments. The current plan has no gID theorem-scoped derivation, source-regime factor map, positive and obstruction witnesses, or independently checked provider for a new row. The 2.2 conditional point route remains licensed only for its stated complete-source family; requests for a surrogate/gID interpretation remain outside that contract. A future gID cell needs a separate promotion record and route rather than widening the existing row.
