//! The declared missing-data support of every route cell, the builder's refusals of
//! unsupported declarations, and multiple imputation with valid pooling on the average-effect
//! route (frequentist Rubin pooling and the Bayesian stacked posterior); the other routes that
//! execute it are exercised in `missing_imputation_routes`.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise replays"
)]

use antecedent::analysis::missing_support::{ALL_QUERY_KINDS, MissingRouteCell};
use antecedent::{
    BayesianConfig, CausalError, EstimatorId, InferenceMode, MissingInference, MissingModality,
    RefuteSuite, Study, StudyResult, missing_support_table, route_missing_support,
};
use antecedent_core::{
    AverageEffectQuery, CausalQueryKind, DeclaredSamplingDesign, ExecutionContext, VariableId,
};
use antecedent_data::{
    ImputationModel, ImputationSpec, MissingDeclaration, MissingPolicy, MissingnessAssumption,
    TabularData,
};

/// The policies each route cell is expected to execute, stated independently of the
/// registry: the registry must say exactly this.
fn expected(cell: MissingRouteCell) -> Option<&'static [&'static str]> {
    use CausalQueryKind as Q;
    use MissingModality as M;
    match (cell.query, cell.modality) {
        (
            Q::AverageEffect | Q::ConditionalEffect | Q::Response | Q::Distribution | Q::Mediation,
            M::Tabular,
        ) => Some(&["error", "complete_case", "multiple_imputation"]),
        (Q::RandomizedEffect, M::Tabular) => {
            Some(&["error", "complete_case", "inverse_probability", "multiple_imputation"])
        }
        (Q::PanelDid, M::Tabular) => Some(&["error", "balanced_panel", "available_case"]),
        (Q::TemporalEffect | Q::Mediation, M::Series) => {
            Some(&["error", "complete_case", "multiple_imputation"])
        }
        _ => None,
    }
}

#[test]
fn every_route_cell_matches_the_declared_support() {
    let cells = MissingRouteCell::all();
    assert_eq!(cells.len(), ALL_QUERY_KINDS.len() * MissingModality::ALL.len() * 2);
    let mut covered = 0;
    for cell in cells {
        let row = route_missing_support(cell);
        let names: Option<Vec<&str>> =
            row.map(|row| row.supported.iter().map(|p| p.name()).collect());
        assert_eq!(names.as_deref(), expected(cell), "{}", cell.label());
        if let Some(row) = row {
            covered += 1;
            assert!(row.supports(row.default), "{}", cell.label());
            assert!(row.supports(row.next_attempt), "{}", cell.label());
            let imputes = row.supports(MissingPolicy::MultipleImputation(ImputationSpec::new()));
            assert_eq!(imputes, !row.imputation_estimators.is_empty(), "{}", cell.label());
            // Any imputation specification is the same policy for support.
            let custom = MissingPolicy::MultipleImputation(
                ImputationSpec::new().with_imputations(50).with_seed(9),
            );
            assert_eq!(row.supports(custom), imputes);
        }
    }
    let rows: usize = missing_support_table().iter().map(|row| row.inference.len()).sum();
    assert_eq!(covered, rows, "every registry row covers exactly its cells");
    let bayesian = route_missing_support(MissingRouteCell {
        query: CausalQueryKind::AverageEffect,
        modality: MissingModality::Tabular,
        inference: MissingInference::Bayesian,
    })
    .unwrap();
    assert_eq!(bayesian.imputation_estimators, ["bayesian.gcomp"]);
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x4D49_0001)
}

/// `SplitMix64` uniform stream.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

const EFFECT: f64 = 1.0;

/// Confounder `z` drives treatment and outcome, an auxiliary `w` predicts the outcome and
/// its missingness: `y = 1·t + 0.7 z + 1.2 w + e`, `y` missing with probability rising in
/// `w`, `z` missing for one row in ten at random. Adjusting for `z` identifies the effect;
/// complete cases select on `w` and keep it unbiased only by luck of the design, so the
/// test compares the imputed estimate with the truth and its spread with complete cases.
fn table(n: usize, seed: u64) -> TabularData {
    let mut rng = Stream(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let zi = rng.normal();
        let wi = rng.normal();
        let ti = f64::from(u8::from(rng.next() < 1.0 / (1.0 + (-zi).exp())));
        let yi = EFFECT * ti + 0.7 * zi + 1.2 * wi + rng.normal();
        let p_missing = 1.0 / (1.0 + (-(1.5 * wi - 0.5)).exp());
        y.push(if rng.next() < p_missing { f64::NAN } else { yi });
        z.push(if i % 10 == 3 { f64::NAN } else { zi });
        t.push(ti);
        w.push(wi);
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

fn graph() -> antecedent_graph::Dag {
    let mut g = antecedent_graph::Dag::empty();
    let ids: Vec<_> = (0..4)
        .map(|i| g.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(i))).unwrap())
        .collect();
    g.insert_directed(ids[2], ids[0]).unwrap();
    g.insert_directed(ids[2], ids[1]).unwrap();
    g.insert_directed(ids[0], ids[1]).unwrap();
    g.insert_directed(ids[3], ids[1]).unwrap();
    g
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

fn imputation(spec: ImputationSpec) -> MissingDeclaration {
    MissingDeclaration::new(MissingPolicy::MultipleImputation(spec))
}

fn run(
    data: &TabularData,
    declaration: Option<MissingDeclaration>,
    inference: Option<InferenceMode>,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let mut builder =
        Study::tabular(data.clone()).query(query()).graph(graph()).refute(RefuteSuite::None);
    if let Some(inference) = inference {
        builder = builder.inference(inference);
    }
    if let Some(declaration) = declaration {
        builder = builder.missing(declaration);
    }
    builder.build()?.prepare(ctx)?.estimate(data, ctx)
}

fn refusal_text(error: &CausalError) -> String {
    assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
    error.to_string()
}

#[test]
fn unsupported_policies_are_refused_at_build_with_the_next_attempt() {
    let data = table(200, 1);
    let built = |declaration: MissingDeclaration| {
        Study::tabular(data.clone()).query(query()).graph(graph()).missing(declaration).build()
    };
    let panel = built(MissingDeclaration::new(MissingPolicy::BalancedPanel)).unwrap_err();
    let text = refusal_text(&panel);
    assert!(text.contains("use one of: error, complete_case, multiple_imputation"), "{text}");
    assert!(text.contains("next attempt: declare `multiple_imputation`"), "{text}");

    let not_at_random = built(imputation(
        ImputationSpec::new().with_assumption(MissingnessAssumption::MissingNotAtRandom),
    ))
    .unwrap_err();
    let text = refusal_text(&not_at_random);
    assert!(text.contains("multiple_imputation.not_at_random"), "{text}");
    assert!(text.contains("sensitivity analysis"), "{text}");

    let too_few = built(imputation(ImputationSpec::new().with_imputations(1))).unwrap_err();
    assert_eq!(too_few.reason_code(), Some("invalid_argument"), "{too_few}");

    let matching = Study::tabular(data.clone())
        .query(query())
        .graph(graph())
        .estimator(EstimatorId::PropensityMatching)
        .missing(imputation(ImputationSpec::new()))
        .build()
        .unwrap_err();
    let text = refusal_text(&matching);
    assert!(text.contains("propensity.matching"), "{text}");
    assert!(text.contains("next attempt: declare `complete_case`"), "{text}");

    let clustered = Study::tabular(data.clone())
        .query(query())
        .graph(graph())
        .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
        .missing(imputation(ImputationSpec::new()))
        .build()
        .unwrap_err();
    let text = refusal_text(&clustered);
    assert!(text.contains("multiple_imputation.clusters_required"), "{text}");

    // Covariates are auxiliary variables under imputation, and refused under complete case.
    let covariates = built(
        MissingDeclaration::new(MissingPolicy::CompleteCase)
            .with_covariates(vec![VariableId::from_raw(3)]),
    )
    .unwrap_err();
    refusal_text(&covariates);
    assert!(
        built(imputation(ImputationSpec::new()).with_covariates(vec![VariableId::from_raw(3)]))
            .is_ok()
    );
}

#[test]
fn multiple_imputation_pools_by_rubins_rules_and_recovers_the_effect() {
    let ctx = ctx();
    let mut points = Vec::new();
    for seed in 0..8_u64 {
        let data = table(800, 100 + seed);
        let spec = ImputationSpec::new().with_imputations(10).with_iterations(5).with_seed(seed);
        let result = run(&data, Some(imputation(spec)), None, &ctx).unwrap();
        let effect = result.estimate.as_effect().unwrap();
        let report = result.missing_data.as_ref().unwrap();
        let pooled = report.imputation.as_ref().unwrap();
        assert_eq!(report.policy().name(), "multiple_imputation");
        assert!(report.declared && report.non_default);
        assert_eq!(report.accounting.rows_kept, 800, "every row is kept");
        assert_eq!(report.accounting.rows_dropped(), 0);
        assert_eq!(pooled.pooling.as_ref(), "rubin");
        assert_eq!(pooled.points.len(), 10);
        assert!(pooled.between > 0.0 && pooled.within > 0.0);
        let mean = pooled.points.iter().sum::<f64>() / 10.0;
        assert!((effect.ate - mean).abs() < 1e-12, "the pooled point is the mean");
        let total = pooled.within + 1.1 * pooled.between;
        assert!((pooled.total - total).abs() < 1e-12);
        assert!((effect.uncertainty.analytic.se - total.sqrt()).abs() < 1e-12);
        assert!(result.interval.is_none(), "the pooled interval stays closed");
        let imputed_y = pooled.imputed_cells.iter().find(|(v, _)| v.as_ref() == "y").unwrap().1;
        assert!(imputed_y > 100, "{imputed_y}");
        points.push(effect.ate);
    }
    let mean = points.iter().sum::<f64>() / points.len() as f64;
    assert!((mean - EFFECT).abs() < 0.1, "mean imputed effect {mean}");
}

#[test]
fn the_bayesian_counterpart_stacks_the_completed_tables_posteriors() {
    let data = table(600, 7);
    let spec = ImputationSpec::new().with_imputations(5).with_iterations(5).with_seed(3);
    let bayes = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200));
    let result = run(&data, Some(imputation(spec)), Some(bayes), &ctx()).unwrap();
    let report = result.missing_data.as_ref().unwrap().imputation.as_ref().unwrap();
    assert_eq!(report.pooling.as_ref(), "stacked_posterior");
    assert_eq!(report.estimator.as_ref(), "bayesian.gcomp");
    let posterior = result.posterior.as_ref().unwrap();
    assert_eq!(posterior.draws.n_draws, 1_000, "five tables of 200 draws, equal weights");
    let column = posterior.effect_column().unwrap();
    let mean = posterior.summaries.mean[column];
    assert!((mean - EFFECT).abs() < 0.25, "posterior mean {mean}");
    assert!(report.total >= report.within, "the mixture keeps the between axis");
    assert!(report.between > 0.0);
}

#[test]
fn the_imputed_result_is_deterministic_and_the_specification_enters_the_identity() {
    let data = table(400, 21);
    let spec = ImputationSpec::new().with_imputations(4).with_iterations(3).with_seed(5);
    let serial = run(&data, Some(imputation(spec)), None, &ExecutionContext::for_tests(9)).unwrap();
    let threaded =
        run(&data, Some(imputation(spec)), None, &ExecutionContext::production(9, 4)).unwrap();
    let a = serial.estimate.as_effect().unwrap();
    let b = threaded.estimate.as_effect().unwrap();
    assert_eq!(a.ate.to_bits(), b.ate.to_bits());
    assert_eq!(a.uncertainty.analytic.se.to_bits(), b.uncertainty.analytic.se.to_bits());

    let snapshot = |result: &StudyResult| {
        result.executed_contract.as_ref().map(|contract| contract.identities.data_snapshot)
    };
    let reseeded = run(&data, Some(imputation(spec.with_seed(6))), None, &ctx()).unwrap();
    assert!(snapshot(&serial).is_some());
    assert_ne!(snapshot(&serial), snapshot(&reseeded), "the seed is part of the claim identity");
    let joint =
        run(&data, Some(imputation(spec.with_model(ImputationModel::JointNormal))), None, &ctx());
    // The treatment is binary and complete, so the joint normal model accepts the table.
    let joint = joint.unwrap();
    assert_ne!(snapshot(&serial), snapshot(&joint), "the model is part of the claim identity");
    let complete_case =
        run(&data, Some(MissingDeclaration::new(MissingPolicy::CompleteCase)), None, &ctx())
            .unwrap();
    assert_ne!(snapshot(&serial), snapshot(&complete_case));
}

#[test]
fn a_tiny_admission_pool_refuses_the_imputation() {
    let data = table(300, 4);
    let tiny = ExecutionContext::for_tests(1)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 64));
    let error = Study::tabular(data.clone())
        .query(query())
        .graph(graph())
        .refute(RefuteSuite::None)
        .missing(imputation(ImputationSpec::new().with_imputations(2)))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
        .estimate(&data, &tiny)
        .unwrap_err();
    assert!(error.to_string().contains("admission"), "{error}");
}
