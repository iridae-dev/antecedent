//! The compiled dose-grid, caller-learner smoothed-dose and external feedback fit families
//! reproduce the direct entries they replace bit for bit, and recover independent truths.
//!
//! * `Study::compile_dose_grid_bias_aware` (`DoseGridBiasAware`): the row, worst-case biases,
//!   bias-aware intervals, band and critical value equal `estimate_dose_grid_bias_aware` (and
//!   its declared-rows form) under the same context; on a noise-free cubic response the
//!   interval of every coordinate contains the analytic truth and the error is within the
//!   worst-case bias bound, and a noise-free quadratic is reproduced exactly with zero bias.
//! * `Study::compile_smoothed_dose_learners` (`SmoothedDoseLearnerTransport`): the grid equals
//!   `estimate_smoothed_dose_with_factories` with the same factories, and with the library's
//!   learners as caller factories it matches the 50-digit reference of
//!   `crates/antecedent-estimate/tests/oracle_data/smoothed_dose_declared_reference.json`.
//! * `Study::compile_external_feedback_fit` (`ExternalFeedbackFit`): the fit equals
//!   `fit_external_feedback` for analytic and finite-difference declarations, and the fitted
//!   gain of a clamped-regressor design is the closed-form least-squares slope.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits of both paths are compared exactly")]
#![allow(
    deprecated,
    reason = "the twins call the direct facade entries the compiled routes replace"
)]

use antecedent::Study;
use antecedent::analysis::compiled_dose_grid_bias_aware::DoseGridBiasAwareRequest;
use antecedent::analysis::compiled_external_feedback::{
    ExternalFeedbackFitError, ExternalFeedbackFitRequest,
};
use antecedent::analysis::compiled_smoothed_dose_learners::SmoothedDoseLearnersRequest;
use antecedent::analysis::declared_learners::LearnerFactories;
use antecedent_core::ExecutionContext;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

// ---- bias-aware dose grid -----------------------------------------------------------------

mod dose_grid {
    use super::*;
    use antecedent::analysis::dose_grid_functional::{
        BiasAwareDoseGrid, ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest,
        estimate_dose_grid_bias_aware,
    };
    use antecedent_estimate::dose_grid_functional::{
        RowDeclaration, RowDependence, estimate_dose_grid_bias_aware_declared,
    };

    const GRID: [f64; 4] = [0.6, 1.2, 1.8, 2.4];

    /// 3001 doses on `[0, 3]` with `m(d) = mean(d)` plus an alternating `noise`.
    fn rows(mean: fn(f64) -> f64, noise: f64) -> (Vec<f64>, Vec<f64>) {
        let dose: Vec<f64> = (0..3001).map(|i| 3.0 * f64::from(i) / 3000.0).collect();
        let outcome = dose
            .iter()
            .enumerate()
            .map(|(i, &d)| mean(d) + if i % 2 == 0 { noise } else { -noise })
            .collect();
        (dose, outcome)
    }

    fn request<'a>(
        dose: &'a [f64],
        outcome: &'a [f64],
        functional: DoseFunctional,
    ) -> DoseGridRequest<'a> {
        DoseGridRequest {
            dose,
            outcome,
            grid: &GRID,
            bandwidth: 0.4,
            bandwidth_range: (0.1, 1.0),
            minimum_local_ess: 10.0,
            functional,
            claims: ClaimSet { pointwise_level: true, derivative: true, simultaneous_band: true },
            design: DoseDesign::RandomizedDose,
        }
    }

    fn cubic(d: f64) -> f64 {
        d * d * d
    }

    fn quadratic(d: f64) -> f64 {
        1.0 + d + d * d
    }

    /// Estimates and standard errors of the reported coordinates.
    fn coordinates(fit: &BiasAwareDoseGrid) -> (Vec<f64>, Vec<f64>) {
        if let Some(contrast) = &fit.row.contrast {
            return (vec![contrast.estimate], vec![contrast.standard_error]);
        }
        if fit.row.derivatives.is_empty() {
            (
                fit.row.levels.iter().map(|point| point.level).collect(),
                fit.row.levels.iter().map(|point| point.standard_error).collect(),
            )
        } else {
            (
                fit.row.derivatives.iter().map(|point| point.derivative).collect(),
                fit.row.derivatives.iter().map(|point| point.standard_error).collect(),
            )
        }
    }

    fn assert_same_bits(compiled: &BiasAwareDoseGrid, direct: &BiasAwareDoseGrid) {
        let (estimates, errors) = coordinates(compiled);
        let (direct_estimates, direct_errors) = coordinates(direct);
        assert_eq!(bits(&estimates), bits(&direct_estimates), "estimates");
        assert_eq!(bits(&errors), bits(&direct_errors), "standard errors");
        let bounds = |fit: &BiasAwareDoseGrid| {
            fit.intervals
                .iter()
                .flat_map(|interval| [interval.lower.to_bits(), interval.upper.to_bits()])
                .collect::<Vec<_>>()
        };
        assert_eq!(bounds(compiled), bounds(direct), "interval bounds");
        assert_eq!(
            bits(&compiled.smoothing_bias_bound),
            bits(&direct.smoothing_bias_bound),
            "worst-case biases"
        );
        assert_eq!(
            compiled.band.as_ref().map(|band| (bits(&band.lower), bits(&band.upper))),
            direct.band.as_ref().map(|band| (bits(&band.lower), bits(&band.upper))),
            "band"
        );
        assert_eq!(
            compiled.band_critical_value.map(f64::to_bits),
            direct.band_critical_value.map(f64::to_bits),
            "band critical value"
        );
        assert_eq!(compiled, direct, "every other field");
    }

    #[test]
    fn the_compiled_bias_aware_grid_equals_the_direct_entry() {
        let (dose, outcome) = rows(cubic, 0.05);
        for functional in [
            DoseFunctional::Level,
            DoseFunctional::Derivative,
            DoseFunctional::Contrast { from: 0.6, to: 2.4 },
        ] {
            let row = request(&dose, &outcome, functional);
            let row = if matches!(functional, DoseFunctional::Contrast { .. }) {
                DoseGridRequest {
                    claims: ClaimSet { simultaneous_band: false, ..row.claims },
                    ..row
                }
            } else {
                row
            };
            let ctx = ExecutionContext::for_tests(3);
            let direct = estimate_dose_grid_bias_aware(&row, 6.0, 300, &ctx).unwrap();
            let compiled =
                Study::compile_dose_grid_bias_aware(DoseGridBiasAwareRequest::new(&row, 6.0, 300))
                    .unwrap();
            assert_eq!(compiled.plan_view().route().as_str(), "DoseGridBiasAware");
            let run = compiled.execute(&ExecutionContext::for_tests(3)).unwrap();
            assert_same_bits(&run.fit, &direct);
            // The measured coverage of this kernel is recorded under this estimator key.
            assert_eq!(run.calibration_estimator, "dose_grid_bias_aware");
            assert_eq!(run.lowered_program_identity, compiled.lowered().identity());
        }
    }

    #[test]
    fn the_compiled_bias_aware_grid_equals_the_declared_rows_entry() {
        let (dose, outcome) = rows(cubic, 0.05);
        let clusters: Vec<u32> = (0..dose.len()).map(|i| u32::try_from(i / 30).unwrap()).collect();
        let weights: Vec<f64> =
            (0..dose.len()).map(|i| 1.0 + f64::from(u8::try_from(i % 3).unwrap())).collect();
        let row = request(&dose, &outcome, DoseFunctional::Level);
        let ctx = ExecutionContext::for_tests(11);
        let direct = estimate_dose_grid_bias_aware_declared(
            &row,
            &RowDeclaration {
                sampling_weights: Some(&weights),
                dependence: RowDependence::Clustered(&clusters),
            },
            6.0,
            300,
            &ctx,
        )
        .unwrap();
        let compiled = Study::compile_dose_grid_bias_aware(
            DoseGridBiasAwareRequest::new(&row, 6.0, 300)
                .sampling_weights(weights.clone())
                .clusters(clusters.clone()),
        )
        .unwrap();
        let run = compiled.execute(&ExecutionContext::for_tests(11)).unwrap();
        assert_same_bits(&run.fit, &direct);
    }

    #[test]
    fn the_compiled_bias_aware_intervals_contain_the_analytic_cubic_truth() {
        // m(d) = d^3 has m''' = 6: the local quadratic's error at d is exactly
        // sum_i l_i (D_i - d)^3, which the worst-case bound M/6 sum_i |l_i| |D_i - d|^3 with
        // M = 6 dominates term by term, so every interval and band contains the truth.
        let (dose, outcome) = rows(cubic, 0.0);
        let truth_of = |functional: DoseFunctional| -> Vec<f64> {
            match functional {
                DoseFunctional::Level => GRID.iter().map(|&d| cubic(d)).collect(),
                DoseFunctional::Derivative => GRID.iter().map(|&d| 3.0 * d * d).collect(),
                DoseFunctional::Contrast { from, to } => vec![cubic(to) - cubic(from)],
            }
        };
        for functional in [
            DoseFunctional::Level,
            DoseFunctional::Derivative,
            DoseFunctional::Contrast { from: 0.6, to: 2.4 },
        ] {
            let row = request(&dose, &outcome, functional);
            let row = DoseGridRequest {
                claims: ClaimSet {
                    simultaneous_band: !matches!(functional, DoseFunctional::Contrast { .. }),
                    ..row.claims
                },
                ..row
            };
            let compiled =
                Study::compile_dose_grid_bias_aware(DoseGridBiasAwareRequest::new(&row, 6.0, 300))
                    .unwrap();
            let fit = compiled.execute(&ExecutionContext::for_tests(5)).unwrap().fit;
            let (estimates, _) = coordinates(&fit);
            let truth = truth_of(functional);
            assert_eq!(estimates.len(), truth.len());
            for (j, (&estimate, &truth)) in estimates.iter().zip(&truth).enumerate() {
                let bias = fit.smoothing_bias_bound[j];
                assert!(bias > 0.0, "{functional:?} {j}: a cubic has smoothing bias");
                assert!(
                    (estimate - truth).abs() <= bias * (1.0 + 1e-9) + 1e-12,
                    "{functional:?} {j}: |{estimate} - {truth}| exceeds the bound {bias}"
                );
                let interval = &fit.intervals[j];
                assert!(interval.lower <= truth && truth <= interval.upper, "{functional:?} {j}");
                if let Some(band) = &fit.band {
                    assert!(band.lower[j] <= truth && truth <= band.upper[j], "{functional:?} {j}");
                }
            }
        }
        // A quadratic is reproduced exactly: zero bias under M = 0, contrast
        // m(2.4) - m(0.6) = 1.8 + 5.76 - 0.36 = 7.2.
        let (dose, outcome) = rows(quadratic, 0.0);
        let row = request(&dose, &outcome, DoseFunctional::Contrast { from: 0.6, to: 2.4 });
        let row =
            DoseGridRequest { claims: ClaimSet { simultaneous_band: false, ..row.claims }, ..row };
        let fit = Study::compile_dose_grid_bias_aware(DoseGridBiasAwareRequest::new(&row, 0.0, 0))
            .unwrap()
            .execute(&ExecutionContext::for_tests(5))
            .unwrap()
            .fit;
        let contrast = fit.row.contrast.as_ref().unwrap();
        assert!((contrast.estimate - 7.2).abs() < 1e-9, "{}", contrast.estimate);
        assert_eq!(fit.smoothing_bias_bound, vec![0.0]);
        assert!(fit.band.is_none());
    }

    #[test]
    fn a_negative_curvature_bound_refuses_through_the_compiled_route() {
        let (dose, outcome) = rows(cubic, 0.05);
        let row = request(&dose, &outcome, DoseFunctional::Level);
        let ctx = ExecutionContext::for_tests(3);
        let direct = estimate_dose_grid_bias_aware(&row, -1.0, 300, &ctx).unwrap_err();
        let compiled =
            Study::compile_dose_grid_bias_aware(DoseGridBiasAwareRequest::new(&row, -1.0, 300))
                .unwrap()
                .execute(&ctx)
                .unwrap_err();
        assert_eq!(compiled.to_string(), direct.to_string());
        assert!(compiled.to_string().contains("curvature bound"), "{compiled}");
        let observational = DoseGridRequest { design: DoseDesign::Observational, ..row };
        assert!(
            Study::compile_dose_grid_bias_aware(DoseGridBiasAwareRequest::new(
                &observational,
                6.0,
                300
            ))
            .unwrap()
            .execute(&ctx)
            .is_err(),
            "an observational dose is refused"
        );
    }

    #[test]
    fn the_dose_grid_program_identity_binds_every_premise_and_row() {
        let (dose, outcome) = rows(cubic, 0.05);
        let row = request(&dose, &outcome, DoseFunctional::Level);
        let identity = |request: DoseGridBiasAwareRequest| {
            Study::compile_dose_grid_bias_aware(request).unwrap().lowered().identity()
        };
        let base = identity(DoseGridBiasAwareRequest::new(&row, 6.0, 300));
        assert_eq!(base, identity(DoseGridBiasAwareRequest::new(&row, 6.0, 300)), "replayable");
        assert_ne!(base, identity(DoseGridBiasAwareRequest::new(&row, 5.0, 300)), "bound");
        assert_ne!(base, identity(DoseGridBiasAwareRequest::new(&row, 6.0, 301)), "replicates");
        let mut changed = outcome.clone();
        changed[17] = f64::from_bits(changed[17].to_bits() ^ 1);
        let changed_row = DoseGridRequest { outcome: &changed, ..row };
        assert_ne!(base, identity(DoseGridBiasAwareRequest::new(&changed_row, 6.0, 300)), "rows");
        assert_ne!(
            base,
            identity(DoseGridBiasAwareRequest::new(&row, 6.0, 300).serial_lag(2)),
            "dependence"
        );
        let contrast = request(&dose, &outcome, DoseFunctional::Contrast { from: 0.6, to: 2.4 });
        assert_ne!(
            base,
            identity(DoseGridBiasAwareRequest::new(&contrast, 6.0, 300)),
            "functional"
        );
    }
}

// ---- smoothed dose with caller learners ---------------------------------------------------

mod smoothed_dose {
    use super::*;
    use std::sync::Arc;

    use antecedent::estimate::{
        SmoothedDoseInput, SmoothedDoseOptions, estimate_smoothed_dose_with_factories,
    };
    use antecedent_core::{SmoothedDoseTransportQuery, SmoothingKernel, VariableId};
    use antecedent_estimate::{DoseBasis, TrialSampling};
    use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
    use antecedent_identify::TransportIdentifier;
    use antecedent_learn::{LearnerFactory, PredictionTask, resolve_for};
    use serde_json::Value;

    fn reference() -> Vec<Value> {
        serde_json::from_str::<Value>(include_str!(
            "../../antecedent-estimate/tests/oracle_data/smoothed_dose_declared_reference.json"
        ))
        .unwrap()["cases"]
            .as_array()
            .cloned()
            .unwrap()
    }

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn input(case: &Value) -> SmoothedDoseInput {
        SmoothedDoseInput {
            features: vec![0],
            covariates: vec![floats(&case["covariate"])],
            outcome: floats(&case["outcome"]),
            dose: floats(&case["dose"]),
            dose_density: floats(&case["density"]),
            source: case["source"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect(),
            sampling: TrialSampling::NestedCohort,
            sampling_weights: Some(floats(&case["weights"])),
            cluster: case["groups"]
                .is_array()
                .then(|| floats(&case["groups"]).into_iter().map(|g| g as u32).collect()),
            serial_lag: case["lag"].as_u64().map(|l| u32::try_from(l).unwrap()),
        }
    }

    fn diagram() -> SelectionDiagram {
        let mut graph = Admg::with_variables(3);
        for (from, to) in [(0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        SelectionDiagram::try_new(graph, vec![VariableId::from_raw(0)]).unwrap()
    }

    fn query(case: &Value) -> SmoothedDoseTransportQuery {
        let grid = floats(&case["grid"]);
        SmoothedDoseTransportQuery {
            outcome: VariableId::from_raw(2),
            dose: VariableId::from_raw(1),
            source_population: Arc::from("trial"),
            target_population: Arc::from("target"),
            grid: Arc::from(grid.as_slice()),
            bandwidth: case["h"].as_f64().unwrap(),
            kernel: SmoothingKernel::Epanechnikov,
            dose_support: (0.0, 4.0),
            density_provenance: Arc::from("known"),
        }
    }

    fn options(case: &Value) -> SmoothedDoseOptions {
        SmoothedDoseOptions {
            folds: 3,
            basis: DoseBasis {
                degree: u8::try_from(case["degree"].as_u64().unwrap()).unwrap(),
                knots: floats(&case["knots"]),
                interactions: case["interactions"].as_bool().unwrap(),
            },
            min_membership_probability: 0.02,
            min_local_ess: 2.0,
            min_distinct_doses: 2,
            ..SmoothedDoseOptions::default()
        }
    }

    /// The library's own outcome and membership learners, handed over as caller factories.
    fn library_learners(
        options: &SmoothedDoseOptions,
        identity: &str,
    ) -> (Arc<dyn LearnerFactory>, Arc<dyn LearnerFactory>, LearnerFactories) {
        let outcome: Arc<dyn LearnerFactory> =
            Arc::from(resolve_for(options.outcome, PredictionTask::Regression).unwrap());
        let membership: Arc<dyn LearnerFactory> =
            Arc::from(resolve_for(options.membership, PredictionTask::BinaryProbability).unwrap());
        let learners =
            LearnerFactories::new(Arc::clone(&outcome), Arc::clone(&membership), identity).unwrap();
        (outcome, membership, learners)
    }

    /// Bits of each grid point's estimate, plug-in, augmentation and influence standard error.
    fn grid_bits(estimate: &antecedent_estimate::SmoothedDoseEstimate) -> Vec<[u64; 4]> {
        estimate
            .grid
            .iter()
            .map(|point| {
                [
                    point.estimate.to_bits(),
                    point.plug_in.to_bits(),
                    point.augmentation.to_bits(),
                    point.influence_se_diagnostic.to_bits(),
                ]
            })
            .collect()
    }

    fn compiled_request(case: &Value, identity: &str) -> SmoothedDoseLearnersRequest {
        let options = options(case);
        let (_, _, learners) = library_learners(&options, identity);
        SmoothedDoseLearnersRequest::new(diagram(), query(case), input(case), options, learners)
    }

    #[test]
    fn the_compiled_smoothed_dose_with_caller_learners_equals_the_factory_kernel() {
        for case in reference().into_iter().filter(|case| case["lag"].is_null()) {
            let name = case["name"].as_str().unwrap();
            let (query, rows, options) = (query(&case), input(&case), options(&case));
            let (outcome, membership, learners) = library_learners(&options, "library learners");
            let id =
                TransportIdentifier::new().identify(&diagram(), &query.transport_query()).unwrap();
            let direct = estimate_smoothed_dose_with_factories(
                &id,
                &query,
                &rows,
                &options,
                &ExecutionContext::for_tests(7),
                &*outcome,
                &*membership,
            )
            .expect(name);
            let ctx = ExecutionContext::for_tests(7);
            let compiled = Study::compile_smoothed_dose_learners(
                SmoothedDoseLearnersRequest::new(diagram(), query, rows, options, learners),
                &ctx,
            )
            .expect(name);
            assert_eq!(compiled.plan_view().route().as_str(), "SmoothedDoseLearnerTransport");
            assert_eq!(compiled.identification(), &id, "{name}: the same certificate");
            let run = compiled.execute(&ctx).expect(name);
            assert_eq!(grid_bits(&run.estimate), grid_bits(&direct), "{name}");
            assert!(run.estimate.external_nuisances_unverified, "{name}");
            assert_eq!(run.estimate, direct, "{name}: every other field");
            assert_eq!((run.claim, run.calibration), ("point_only", "not_measured"));
        }
    }

    #[test]
    fn the_compiled_smoothed_dose_with_caller_learners_matches_the_multiprecision_reference() {
        let close = |actual: f64, expected: f64, tol: f64, what: &str| {
            assert!(
                (actual - expected).abs() <= tol * expected.abs().max(1.0),
                "{what}: {actual} vs {expected}"
            );
        };
        for case in reference().into_iter().filter(|case| case["lag"].is_null()) {
            let name = case["name"].as_str().unwrap();
            let ctx = ExecutionContext::for_tests(7);
            let run = Study::compile_smoothed_dose_learners(
                compiled_request(&case, "library learners"),
                &ctx,
            )
            .expect(name)
            .execute(&ctx)
            .expect(name);
            let folds: Vec<u64> =
                run.estimate.folds.assignment.iter().map(|f| u64::from(*f)).collect();
            let want: Vec<u64> =
                case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(folds, want, "{name}: fold assignment");
            for (point, expected) in
                run.estimate.grid.iter().zip(case["points"].as_array().unwrap())
            {
                let at = format!("{name} dose {}", point.dose);
                close(point.plug_in, expected["plug_in"].as_f64().unwrap(), 1e-8, &at);
                close(point.augmentation, expected["augmentation"].as_f64().unwrap(), 1e-7, &at);
                close(point.estimate, expected["estimate"].as_f64().unwrap(), 1e-8, &at);
                close(point.influence_se_diagnostic, expected["se"].as_f64().unwrap(), 1e-7, &at);
            }
        }
    }

    #[test]
    fn caller_learners_on_time_ordered_rows_refuse_through_the_compiled_route() {
        let case = reference().into_iter().find(|case| !case["lag"].is_null()).unwrap();
        let ctx = ExecutionContext::for_tests(7);
        let error = Study::compile_smoothed_dose_learners(
            compiled_request(&case, "library learners"),
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap_err();
        assert!(error.to_string().contains("route_not_supported"), "{error}");
    }

    #[test]
    fn the_smoothed_dose_program_identity_binds_the_rows_and_the_learners() {
        let case = reference().into_iter().find(|case| case["lag"].is_null()).unwrap();
        let ctx = ExecutionContext::for_tests(7);
        let identity = |request: SmoothedDoseLearnersRequest| {
            Study::compile_smoothed_dose_learners(request, &ctx).unwrap().lowered().identity()
        };
        let base = identity(compiled_request(&case, "library learners"));
        assert_eq!(base, identity(compiled_request(&case, "library learners")), "replayable");
        assert_ne!(base, identity(compiled_request(&case, "other learners")), "learners");
        let options = options(&case);
        let (_, _, learners) = library_learners(&options, "library learners");
        let mut rows = input(&case);
        rows.outcome[3] = f64::from_bits(rows.outcome[3].to_bits() ^ 1);
        assert_ne!(
            base,
            identity(SmoothedDoseLearnersRequest::new(
                diagram(),
                query(&case),
                rows,
                options,
                learners
            )),
            "rows"
        );
    }
}

// ---- external feedback fit ----------------------------------------------------------------

mod external_feedback {
    use super::*;
    use std::sync::Arc;

    use antecedent::estimate::external_cyclic_fit::{ExternalFitPolicy, fit_external_feedback};
    use antecedent_model::external_mechanism::{
        DerivativeDeclaration, DifferenceScheme, ExternalFeedbackSystem, ExternalFitParameter,
        ExternalMechanismImplementation, ExternalMechanismSpec, MechanismRegistry, SystemEquation,
    };
    use antecedent_model::nonlinear_equilibrium::{
        Activation, Clamp, Equation, EquilibriumError, RegimeRow,
    };

    /// `p0 * tanh(x)`: a saturating response with a global slope bound of 0.9.
    struct Saturating {
        spec: ExternalMechanismSpec,
    }
    const CONTENT: &[u8] = b"lab/saturating v1: p0 * tanh(x)";
    impl Saturating {
        fn new(derivative: DerivativeDeclaration) -> Arc<Self> {
            let mut spec = ExternalMechanismSpec::for_content(
                "lab/saturating",
                "1",
                CONTENT,
                vec![1],
                vec!["gain".into()],
                derivative,
            );
            spec.input_lipschitz = Some(vec![0.9]);
            Arc::new(Self { spec })
        }
    }
    impl ExternalMechanismImplementation for Saturating {
        fn spec(&self) -> ExternalMechanismSpec {
            self.spec.clone()
        }
        fn content(&self) -> &[u8] {
            CONTENT
        }
        fn value(&self, inputs: &[f64], parameters: &[f64]) -> Result<f64, EquilibriumError> {
            Ok(parameters[0] * inputs[0].tanh())
        }
        fn input_gradient(
            &self,
            inputs: &[f64],
            parameters: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![parameters[0] * (1.0 - inputs[0].tanh().powi(2))]))
        }
        fn parameter_gradient(
            &self,
            inputs: &[f64],
            _parameters: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![inputs[0].tanh()]))
        }
    }

    fn template(spec: &ExternalMechanismSpec) -> ExternalFeedbackSystem {
        ExternalFeedbackSystem {
            equations: vec![
                SystemEquation::External {
                    id: "response".into(),
                    intercept: 0.0,
                    parameters: vec![0.0],
                    mechanism: spec.clone(),
                },
                SystemEquation::Builtin {
                    equation: Equation {
                        id: "feedback".into(),
                        intercept: 0.0,
                        weights: vec![0.2, 0.0],
                        activation: Activation::Tanh,
                        expression: None,
                        parameters: vec![],
                    },
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        }
    }

    /// Clamped regressor rows: `x1` is hard-clamped, `x0 = 0.7 tanh(x1) +- 0.1`.
    fn rows() -> Vec<RegimeRow> {
        [-2.0_f64, -1.0, 0.5, 1.0, 2.0]
            .into_iter()
            .flat_map(|x| {
                [-0.1, 0.1].map(|noise| RegimeRow {
                    values: vec![0.7 * x.tanh() + noise, x],
                    clamps: vec![Clamp { variable: 1, value: x }],
                })
            })
            .collect()
    }

    const TARGETS: [ExternalFitParameter; 1] =
        [ExternalFitParameter::Parameter { equation: 0, parameter: 0 }];

    fn declarations() -> [DerivativeDeclaration; 2] {
        [
            DerivativeDeclaration::Analytic,
            DerivativeDeclaration::FiniteDifference {
                step: 1e-6,
                scheme: DifferenceScheme::Central,
            },
        ]
    }

    fn registry(implementation: &Arc<Saturating>) -> MechanismRegistry {
        let mut registry = MechanismRegistry::new();
        registry.register(implementation.clone()).unwrap();
        registry
    }

    fn request(spec: &ExternalMechanismSpec, rows: Vec<RegimeRow>) -> ExternalFeedbackFitRequest {
        ExternalFeedbackFitRequest::new(
            template(spec),
            TARGETS.to_vec(),
            rows,
            ExternalFitPolicy::default(),
        )
    }

    fn gain(system: &ExternalFeedbackSystem) -> f64 {
        let SystemEquation::External { parameters, .. } = &system.equations[0] else {
            panic!("external equation");
        };
        parameters[0]
    }

    #[test]
    fn the_compiled_external_fit_equals_the_direct_fit() {
        for derivative in declarations() {
            let implementation = Saturating::new(derivative);
            let registry = registry(&implementation);
            let ctx = ExecutionContext::for_tests(7);
            let direct = fit_external_feedback(
                &template(&implementation.spec),
                Some(&registry),
                &TARGETS,
                &rows(),
                ExternalFitPolicy::default(),
                &ctx,
            )
            .unwrap();
            let compiled =
                Study::compile_external_feedback_fit(request(&implementation.spec, rows()))
                    .unwrap();
            assert_eq!(compiled.plan_view().route().as_str(), "ExternalFeedbackFit");
            let run = compiled.execute(Some(&registry), &ctx).unwrap();
            assert_eq!(gain(&run.fit.system).to_bits(), gain(&direct.system).to_bits());
            assert_eq!(run.fit.log_likelihood.to_bits(), direct.log_likelihood.to_bits());
            assert_eq!(run.fit.score_norm.to_bits(), direct.score_norm.to_bits());
            assert_eq!(run.fit, direct, "{derivative:?}: every field");
            assert_eq!(run.claim, "model_conditional_point");
        }
    }

    #[test]
    fn the_compiled_external_fit_recovers_the_least_squares_gain() {
        // With x1 hard-clamped in every row the joint likelihood's data term is the Gaussian
        // regression of x0 on tanh(x1), so the maximum-likelihood gain is the closed-form
        // least-squares slope.
        let xs: Vec<f64> = rows().iter().map(|row| row.values[1].tanh()).collect();
        let ys: Vec<f64> = rows().iter().map(|row| row.values[0]).collect();
        let closed = xs.iter().zip(&ys).map(|(x, y)| x * y).sum::<f64>()
            / xs.iter().map(|x| x * x).sum::<f64>();
        for derivative in declarations() {
            let implementation = Saturating::new(derivative);
            let run = Study::compile_external_feedback_fit(request(&implementation.spec, rows()))
                .unwrap()
                .execute(Some(&registry(&implementation)), &ExecutionContext::for_tests(7))
                .unwrap();
            assert!(run.fit.converged, "{derivative:?}: {:?}", run.fit);
            assert!((gain(&run.fit.system) - closed).abs() < 1e-6, "{derivative:?}");
            assert_eq!(
                run.fit.trust,
                antecedent_model::external_mechanism::MechanismTrust::VerifiedExternal
            );
        }
    }

    #[test]
    fn a_compiled_external_fit_without_a_resolver_is_refused() {
        let implementation = Saturating::new(DerivativeDeclaration::Analytic);
        let error = Study::compile_external_feedback_fit(request(&implementation.spec, rows()))
            .unwrap()
            .execute(None, &ExecutionContext::for_tests(7))
            .unwrap_err();
        assert!(matches!(error, ExternalFeedbackFitError::Model(_)), "{error:?}");
        assert!(error.to_string().contains("mechanism_resolver_missing"), "{error}");
    }

    #[test]
    fn the_external_fit_program_identity_binds_the_system_and_rows_but_not_the_resolver() {
        let implementation = Saturating::new(DerivativeDeclaration::Analytic);
        let identity = |request: ExternalFeedbackFitRequest| {
            Study::compile_external_feedback_fit(request).unwrap().lowered().identity()
        };
        let base = identity(request(&implementation.spec, rows()));
        assert_eq!(base, identity(request(&implementation.spec, rows())), "replayable");
        let mut changed = rows();
        changed[0].values[0] += 0.5;
        assert_ne!(base, identity(request(&implementation.spec, changed)), "rows");
        let other = Saturating::new(DerivativeDeclaration::FiniteDifference {
            step: 1e-6,
            scheme: DifferenceScheme::Central,
        });
        assert_ne!(base, identity(request(&other.spec, rows())), "mechanism declaration");
        // Two registries holding the same verified implementation give the same fit.
        let compiled =
            Study::compile_external_feedback_fit(request(&implementation.spec, rows())).unwrap();
        let ctx = ExecutionContext::for_tests(7);
        let first = compiled.execute(Some(&registry(&implementation)), &ctx).unwrap();
        let second = compiled.execute(Some(&registry(&implementation)), &ctx).unwrap();
        assert_eq!(first.fit, second.fit);
        assert_eq!(first.lowered_program_identity, base);
    }
}
