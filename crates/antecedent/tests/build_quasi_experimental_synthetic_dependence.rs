//! Quasi-experimental designs that now honour a declaration they once refused.
//!
//! - A declared clustered or serial design on the single treated unit of a synthetic control is
//!   honoured by the placebo-variance and the conformal interval, which equal an exact
//!   rational-arithmetic reference (`oracle_data/synthetic_dependence_reference.py`), and is
//!   refused only where the route cannot carry the interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, CausalError, InferenceMode, Study, StudyResult};
use antecedent_core::{DeclaredSamplingDesign, SyntheticControlQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::rollout_resampling::DesignEffectKind;

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(100.0).n_draws(2_000))
}

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("oracle_data/synthetic_dependence_reference.json")).unwrap()
}

/// The long unit-major table of a reference case: `(data, units, periods, treated label)`.
fn long_panel(case: &serde_json::Value) -> (TabularData, Vec<Arc<str>>, Vec<i64>) {
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    let names = case["unit_names"].as_array().unwrap();
    for (name, row) in names.iter().zip(case["panel"].as_array().unwrap()) {
        for (t, value) in row.as_array().unwrap().iter().enumerate() {
            y.push(value.as_f64().unwrap());
            units.push(Arc::<str>::from(name.as_str().unwrap()));
            periods.push(t as i64 + 1);
        }
    }
    (TabularData::from_f64_columns([("y", y.as_slice())]).unwrap(), units, periods)
}

fn synthetic_study(
    case: &serde_json::Value,
    design: Option<DeclaredSamplingDesign>,
) -> Result<StudyResult, CausalError> {
    let (data, units, periods) = long_panel(case);
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        case["intervention_period"].as_i64().unwrap(),
    );
    let mut builder = Study::tabular(data).query(query);
    if let Some(design) = design {
        builder = builder.sampling_design(design);
    }
    builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(31)))
}

#[test]
fn a_single_treated_unit_synthetic_control_honours_a_declared_clustered_design_by_the_placebo_variance()
 {
    let reference = reference();
    let case = &reference["placebo"];
    let exact = case["effect"].as_f64().unwrap();
    // The undeclared design is the point estimate of the same fit.
    let point = synthetic_study(case, None).unwrap().synthetic_control().unwrap().effect;
    assert!((point - exact).abs() < 1e-6, "{point} vs {exact}");
    // The declared clustered design reports the placebo variance of the exact reference.
    let result = synthetic_study(case, Some(DeclaredSamplingDesign::ClusteredPanel)).unwrap();
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("quasi.multi_unit_synthetic_control_placebo")
    );
    let rollout = result.quasi_rollout().unwrap();
    let overall = rollout.overall().unwrap();
    assert!((overall.estimate - exact).abs() < 1e-6, "{} vs {exact}", overall.estimate);
    // Five donors give five equally likely placebo assignments; the product averages 500 seeded
    // draws of them, so its standard error matches the exact population value to Monte Carlo error.
    let exact_se = case["placebo_se"].as_f64().unwrap();
    let se = overall.standard_error.unwrap();
    assert!((se - exact_se).abs() < 0.15 * exact_se, "{se} vs exact {exact_se}");
    let interval = overall.interval.unwrap();
    let z = 1.959_963_984_540_054;
    assert!((interval[0] - (overall.estimate - z * se)).abs() < 1e-9);
    assert!((interval[1] - (overall.estimate + z * se)).abs() < 1e-9);
    assert_eq!(rollout.replicates_ok, 500);
}

#[test]
fn a_single_treated_unit_synthetic_control_honours_a_declared_serial_design_by_the_conformal_interval()
 {
    let reference = reference();
    let case = &reference["conformal"];
    let exact = case["effect"].as_f64().unwrap();
    let result = synthetic_study(case, Some(DeclaredSamplingDesign::DependentTimeSeries)).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("quasi.synthetic_control_conformal"));
    let rollout = result.quasi_rollout().unwrap();
    let overall = rollout.overall().unwrap();
    assert!((overall.estimate - exact).abs() < 1e-6, "{} vs {exact}", overall.estimate);
    // The hull of the retained constant-effect nulls, found by exact bisection on the exact p-value.
    let want = case["overall_interval"].as_array().unwrap();
    let got = overall.interval.unwrap();
    for (got, want) in got.iter().zip(want) {
        let want = want.as_f64().unwrap();
        assert!((got - want).abs() < 2e-5, "overall endpoint {got} vs exact {want}");
    }
    let periods: Vec<_> =
        rollout.effects.iter().filter(|effect| effect.kind == DesignEffectKind::Period).collect();
    let wanted = case["periods"].as_array().unwrap();
    assert_eq!(periods.len(), wanted.len());
    for (effect, want) in periods.iter().zip(wanted) {
        assert_eq!(effect.period, want["period"].as_i64());
        assert!((effect.estimate - want["estimate"].as_f64().unwrap()).abs() < 1e-6);
        let interval = effect.interval.unwrap();
        for (got, want) in interval.iter().zip(want["interval"].as_array().unwrap()) {
            let want = want.as_f64().unwrap();
            assert!((got - want).abs() < 2e-5, "period endpoint {got} vs exact {want}");
        }
    }
}

#[test]
fn a_declared_serial_design_with_too_short_a_pre_period_is_refused_with_the_next_attempt() {
    // Four pre-periods cannot support a 5% conformal test: one over pre-periods plus one is 0.2.
    let reference = reference();
    let placebo = &reference["placebo"];
    let error = synthetic_study(placebo, Some(DeclaredSamplingDesign::DependentTimeSeries))
        .err()
        .expect("a 6-period pre window cannot reach the 5% conformal level");
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
    let text = error.to_string();
    assert!(
        text.contains("pre-periods") && text.contains("Next attempt")
            || text.contains("next attempt"),
        "{text}"
    );
}

#[test]
fn a_declared_design_the_interval_cannot_be_combined_with_is_refused_not_ignored() {
    let reference = reference();
    let case = &reference["conformal"];
    let (data, units, periods) = long_panel(case);
    let base = || {
        SyntheticControlQuery::new(
            VariableId::from_raw(0),
            units.clone(),
            periods.clone(),
            Arc::<str>::from("treated"),
            25,
        )
    };
    let run = |query: SyntheticControlQuery, design, inference: Option<InferenceMode>| {
        let mut builder = Study::tabular(data.clone()).query(query).sampling_design(design);
        if let Some(inference) = inference {
            builder = builder.inference(inference);
        }
        builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(32)))
    };
    // A unit-randomization test, a ridge augmentation and (for the serial design) the synthetic
    // difference in differences have no such interval; the Bayesian donor-weight posterior reads no lag.
    let refused = [
        run(base().with_uniform_unit_randomization(), DeclaredSamplingDesign::ClusteredPanel, None),
        run(base().with_augmentation(1.0), DeclaredSamplingDesign::DependentTimeSeries, None),
        run(base().difference_in_differences(), DeclaredSamplingDesign::DependentTimeSeries, None),
        run(base(), DeclaredSamplingDesign::ClusteredPanel, Some(bayes())),
    ];
    for (index, outcome) in refused.into_iter().enumerate() {
        let error = outcome.err().unwrap_or_else(|| panic!("case {index}: the design was ignored"));
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "case {index}: {error}");
        assert!(error.to_string().contains("Next attempt"), "case {index}: {error}");
    }
    // The independent design needs no interval and runs as the point estimate.
    let point = run(base(), DeclaredSamplingDesign::IndependentSubjectTrajectories, None).unwrap();
    assert!(point.synthetic_control().is_some());
}

// ---------------------------------------------------------------------------------------------
// The Bayesian design routes refuse a table that defines no contrast with a typed reason code.
// ---------------------------------------------------------------------------------------------

fn bayesian_run(
    data: TabularData,
    query: impl Into<antecedent::CausalQuery>,
) -> Result<StudyResult, CausalError> {
    Study::tabular(data)
        .query(query)
        .inference(bayes())
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(13)))
}

#[test]
fn the_bayesian_design_routes_refuse_an_undefined_contrast_with_a_typed_reason_and_a_next_attempt()
{
    use antecedent::PanelDidQuery;
    let outcome = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data = || TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let labels =
        |names: &[&str]| -> Vec<Arc<str>> { names.iter().map(|n| Arc::from(*n)).collect() };
    // Synthetic control on one period: no pre-treatment path.
    let sc = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        labels(&["treated", "d0", "d1", "d2", "d3", "d4"]),
        vec![1_i64; 6],
        "treated",
        1,
    );
    let error = bayesian_run(data(), sc).err().unwrap();
    assert_eq!(error.reason_code(), Some("invalid_argument"), "{error}");
    assert!(error.to_string().contains("Next attempt"), "{error}");
    // Staggered event study on one period: no event-time contrast.
    let subjects = labels(&["s0", "s1", "s2", "s3", "s4", "s5"]);
    let es = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        subjects.clone(),
        subjects.clone(),
        vec![1_i64; 6],
        vec![0_i64, 0, 0, 2, 2, 2],
    );
    let error = bayesian_run(data(), es).err().unwrap();
    assert_eq!(error.reason_code(), Some("invalid_argument"), "{error}");
    assert!(error.to_string().contains("Next attempt"), "{error}");
    // The same event study on an unbalanced panel is a missing-data refusal.
    let unbalanced = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        subjects.clone(),
        subjects,
        vec![1_i64, 2, 1, 2, 1, 1],
        vec![0_i64, 0, 0, 2, 2, 2],
    );
    let error = bayesian_run(data(), unbalanced).err().unwrap();
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");
    assert!(error.to_string().contains("Next attempt"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// The sharp discontinuity pools over multiple imputations of a missing outcome.
// ---------------------------------------------------------------------------------------------

mod discontinuity_imputation {
    use super::*;
    use antecedent::{EstimatorId, IdentifierId};
    use antecedent_core::AverageEffectQuery;
    use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy};
    use antecedent_estimate::multiple_imputation::{ImputationColumn, ImputationUnits, Imputer};
    use antecedent_graph::{Dag, DenseNodeId};

    const ROWS: usize = 600;

    fn columns(gaps: &[usize]) -> [Vec<f64>; 3] {
        let r: Vec<f64> = (0..ROWS).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / ROWS as f64).collect();
        let t: Vec<f64> = r.iter().map(|&x| f64::from(u8::from(x >= 0.0))).collect();
        let mut y: Vec<f64> = (0..ROWS)
            .map(|i| 3.0 * t[i] + 0.5 * r[i] + 0.05 * ((i * 37 % 17) as f64 / 17.0 - 0.5))
            .collect();
        for &gap in gaps {
            y[gap] = f64::NAN;
        }
        [t, y, r]
    }

    fn table(columns: &[Vec<f64>; 3]) -> TabularData {
        TabularData::from_f64_columns([
            ("t", columns[0].as_slice()),
            ("y", columns[1].as_slice()),
            ("r", columns[2].as_slice()),
        ])
        .unwrap()
    }

    fn run(
        columns: &[Vec<f64>; 3],
        missing: Option<MissingDeclaration>,
    ) -> Result<StudyResult, CausalError> {
        let mut graph = Dag::with_variables(3);
        for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let mut builder = Study::tabular(table(columns))
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .identifier(antecedent::IdentifierId::RdSharp)
            .estimator(EstimatorId::RdSharp)
            .rd_config(VariableId::from_raw(2), 0.0, 1.0)
            .bootstrap_replicates(0);
        if let Some(missing) = missing {
            builder = builder.missing(missing);
        }
        builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(41)))
    }

    fn effect_of(result: &StudyResult) -> (f64, f64) {
        let effect = result.estimate().as_effect().unwrap();
        (effect.ate, effect.uncertainty.analytic.se)
    }

    #[test]
    fn a_sharp_discontinuity_with_missing_outcomes_pools_to_the_hand_rubin_reference() {
        const IMPUTATIONS: u32 = 5;
        let _ = IdentifierId::RdSharp;
        let spec =
            ImputationSpec::new().with_imputations(IMPUTATIONS).with_iterations(4).with_seed(17);
        let gaps: Vec<usize> = (0..ROWS).filter(|row| row % 11 == 4).collect();
        let gapped = columns(&gaps);
        let result =
            run(&gapped, Some(MissingDeclaration::new(MissingPolicy::MultipleImputation(spec))))
                .unwrap();
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert_eq!(report.policy().name(), "multiple_imputation");
        let report = report.imputation.as_ref().expect("an imputation report");
        assert_eq!(report.pooling.as_ref(), "rubin");
        assert_eq!(report.points.len(), IMPUTATIONS as usize);

        // The completed tables from the public imputer over every column in table order, each
        // fitted by the discontinuity estimator alone, pooled by Rubin's rules.
        let imputer = Imputer::new(
            ["t", "y", "r"]
                .into_iter()
                .zip(&gapped)
                .map(|(name, values)| ImputationColumn::new(name, values.clone()))
                .collect(),
            spec,
            ImputationUnits::Rows,
        )
        .unwrap();
        let (mut points, mut variances) = (Vec::new(), Vec::new());
        for index in 0..IMPUTATIONS {
            let completed = imputer.impute(index, &ExecutionContext::for_tests(41)).unwrap();
            let values: [Vec<f64>; 3] = [0, 1, 2].map(|c| completed.columns()[c].to_vec());
            assert!(values[1].iter().all(|value| value.is_finite()));
            let (point, se) = effect_of(&run(&values, None).unwrap());
            points.push(point);
            variances.push(se * se);
        }
        let m = f64::from(IMPUTATIONS);
        let mean = points.iter().sum::<f64>() / m;
        let within = variances.iter().sum::<f64>() / m;
        let between = points.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (m - 1.0);
        let total = within + (1.0 + 1.0 / m) * between;
        for (got, want) in report.points.iter().zip(&points) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        assert!((report.within - within).abs() < 1e-12 * (1.0 + within));
        assert!((report.between - between).abs() < 1e-12 * (1.0 + between));
        assert!((report.total - total).abs() < 1e-12 * (1.0 + total));
        let (ate, se) = effect_of(&result);
        assert!((ate - mean).abs() < 1e-12, "{ate} vs {mean}");
        assert!((se - total.sqrt()).abs() < 1e-12);
        assert!((ate - 3.0).abs() < 0.1, "the jump is 3: {ate}");
        assert!(
            between > 0.0 || within > 0.0,
            "the pooled variance must be positive: {between} {within}"
        );
    }

    #[test]
    fn a_bayesian_discontinuity_with_missing_outcomes_stacks_the_completed_tables_posteriors() {
        use antecedent::{BayesianConfig, InferenceMode};
        const IMPUTATIONS: u32 = 4;
        let spec =
            ImputationSpec::new().with_imputations(IMPUTATIONS).with_iterations(4).with_seed(5);
        let gaps: Vec<usize> = (0..ROWS).filter(|row| row % 13 == 6).collect();
        let gapped = columns(&gaps);
        let bayes = || InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512));
        let fit = |columns: &[Vec<f64>; 3], missing: Option<MissingDeclaration>| {
            let mut graph = Dag::with_variables(3);
            for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
                graph
                    .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                    .unwrap();
            }
            let mut builder = Study::tabular(table(columns))
                .graph(graph)
                .query(AverageEffectQuery::binary_ate(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                ))
                .identifier(antecedent::IdentifierId::RdSharp)
                .estimator(EstimatorId::BayesianRdLocalLinear)
                .rd_config(VariableId::from_raw(2), 0.0, 1.0)
                .inference(bayes())
                .refute(antecedent::RefuteSuite::None);
            if let Some(missing) = missing {
                builder = builder.missing(missing);
            }
            builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(43)))
        };
        let pooled =
            fit(&gapped, Some(MissingDeclaration::new(MissingPolicy::MultipleImputation(spec))))
                .unwrap();
        let posterior = pooled.posterior.as_ref().expect("a stacked posterior");
        assert_eq!(posterior.draws.n_draws, 512 * IMPUTATIONS as usize);
        let imputer = Imputer::new(
            ["t", "y", "r"]
                .into_iter()
                .zip(&gapped)
                .map(|(name, values)| ImputationColumn::new(name, values.clone()))
                .collect(),
            spec,
            ImputationUnits::Rows,
        )
        .unwrap();
        let mut means = Vec::new();
        for index in 0..IMPUTATIONS {
            let completed = imputer.impute(index, &ExecutionContext::for_tests(43)).unwrap();
            let values: [Vec<f64>; 3] = [0, 1, 2].map(|c| completed.columns()[c].to_vec());
            let table_fit = fit(&values, None).unwrap();
            means.push(table_fit.estimate().as_effect().unwrap().ate);
        }
        // Equal-weight stacking: the mixture mean is the mean of the completed tables' means.
        let want = means.iter().sum::<f64>() / f64::from(IMPUTATIONS);
        let got = pooled.estimate().as_effect().unwrap().ate;
        assert!((got - want).abs() < 1e-9 * (1.0 + want.abs()), "{got} vs {want}");
        assert!((got - 3.0).abs() < 0.1, "{got}");
    }
}

// ---------------------------------------------------------------------------------------------
// A switchback with one period for every sequence never switches back.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_one_period_switchback_is_refused_with_a_typed_reason_and_a_two_period_switchback_runs() {
    use antecedent_core::{RandomizationDesign, RandomizedEffectQuery};
    let outcome: Vec<f64> = (0..8).map(f64::from).collect();
    let data = || TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let query = |periods: [&str; 8], units: [&str; 8]| {
        RandomizedEffectQuery::with_design(
            RandomizationDesign::Switchback { periods: periods.map(Arc::<str>::from).into() },
            VariableId::from_raw(0),
            [true, false, false, true, true, false, false, true],
            [0.5; 8],
            units.map(Arc::<str>::from),
            ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"].map(Arc::<str>::from),
            ("off", "on"),
        )
    };
    let run = |query: RandomizedEffectQuery| {
        Study::tabular(data())
            .query(query)
            .build()
            .and_then(|study| study.run(&ExecutionContext::for_tests(51)))
    };
    let one_period = query(["p0"; 8], ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]);
    let error = run(one_period).err().expect("a switchback that never switches back ran");
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
    let text = error.to_string();
    assert!(text.contains("never switches back") && text.contains("Next attempt"), "{text}");
    let two_periods = query(
        ["p0", "p1", "p0", "p1", "p0", "p1", "p0", "p1"],
        ["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"],
    );
    assert!(run(two_periods).is_ok());
}
