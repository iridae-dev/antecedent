//! Multiple imputation on the routes whose answer has several coordinates and on a single
//! series: conditional effects, response curves, interventional distributions, static
//! mediation, randomized designs and the temporal routes.
//!
//! Each frequentist test reproduces the route's completed tables with the public imputer,
//! refits the same study on every completed table with no missing-data declaration, and pools
//! the coordinates and covariances by hand (Rubin's rules, written out here). The route's own
//! report must equal that reference, so the dispatch, the vector pooling and the write-back
//! into the result are all checked against numbers this file computes.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::type_complexity,
    reason = "integration test with deterministic fixtures and a hand-written reference"
)]

use std::sync::Arc;

use antecedent::result::ImputationReport;
use antecedent::{
    BayesianConfig, CausalError, DesignPosteriorModel, EstimatorId, IdentifierId, InferenceMode,
    RandomizedEffectQuery, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain,
    DeclaredSamplingDesign, ExecutionContext, GridSpec, Intervention,
    InterventionalDistributionQuery, Lag, MediationContrast, MediationQuery, OutcomeFunctional,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, TemporalEffectQuery,
    TemporalPolicy, Value, VariableId,
};
use antecedent_data::{
    ImputationModel, ImputationSpec, MissingDeclaration, MissingPolicy, TabularData, TimeSeriesData,
};
use antecedent_estimate::multiple_imputation::{self, ImputationColumn, ImputationUnits, Imputer};
use antecedent_estimate::randomized_posterior::{ArmNormalPrior, ArmOutcomeModel};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};

const IMPUTATIONS: u32 = 6;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x4D49_0002)
}

fn spec() -> ImputationSpec {
    ImputationSpec::new().with_imputations(IMPUTATIONS).with_iterations(4).with_seed(11)
}

fn declaration(spec: ImputationSpec) -> MissingDeclaration {
    MissingDeclaration::new(MissingPolicy::MultipleImputation(spec))
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Run `body` on a thread with the stack a prepared study result needs in a debug build.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("imputation-routes".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
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

    fn bernoulli(&mut self, p: f64) -> bool {
        self.next() < p
    }
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ------------------------------------------------------------------ the hand-written reference

/// Rubin's rules over `m` coordinate vectors and their covariance matrices (row-major).
struct Pooled {
    point: Vec<f64>,
    within: Vec<f64>,
    between: Vec<f64>,
    total: Vec<f64>,
    relative_increase: f64,
}

fn pool(points: &[Vec<f64>], covariances: &[Vec<f64>]) -> Pooled {
    let m = points.len() as f64;
    let p = points[0].len();
    let point: Vec<f64> = (0..p).map(|i| points.iter().map(|q| q[i]).sum::<f64>() / m).collect();
    let mut within = vec![0.0; p * p];
    let mut between = vec![0.0; p * p];
    for (q, u) in points.iter().zip(covariances) {
        for i in 0..p {
            for j in 0..p {
                within[i * p + j] += u[i * p + j] / m;
                between[i * p + j] += (q[i] - point[i]) * (q[j] - point[j]) / (m - 1.0);
            }
        }
    }
    let total: Vec<f64> =
        within.iter().zip(&between).map(|(u, b)| u + (1.0 + 1.0 / m) * b).collect();
    let inverse = invert(&within, p);
    let mut trace = 0.0;
    for i in 0..p {
        for k in 0..p {
            trace += between[i * p + k] * inverse[k * p + i];
        }
    }
    let relative_increase = (1.0 + 1.0 / m) * trace / p as f64;
    Pooled { point, within, between, total, relative_increase }
}

/// Gauss-Jordan inverse of a small row-major matrix.
fn invert(matrix: &[f64], p: usize) -> Vec<f64> {
    let mut a = matrix.to_vec();
    let mut inverse = vec![0.0; p * p];
    for i in 0..p {
        inverse[i * p + i] = 1.0;
    }
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&x, &y| a[x * p + col].abs().total_cmp(&a[y * p + col].abs()))
            .unwrap();
        for k in 0..p {
            a.swap(col * p + k, pivot * p + k);
            inverse.swap(col * p + k, pivot * p + k);
        }
        let scale = a[col * p + col];
        for k in 0..p {
            a[col * p + k] /= scale;
            inverse[col * p + k] /= scale;
        }
        for row in 0..p {
            if row != col {
                let factor = a[row * p + col];
                for k in 0..p {
                    a[row * p + k] -= factor * a[col * p + k];
                    inverse[row * p + k] -= factor * inverse[col * p + k];
                }
            }
        }
    }
    inverse
}

/// The `D1` statistic of the hypothesis that every coordinate is zero.
fn hand_wald(pooled: &Pooled) -> f64 {
    let p = pooled.point.len();
    let inverse = invert(&pooled.within, p);
    let mut quadratic = 0.0;
    for i in 0..p {
        for j in 0..p {
            quadratic += pooled.point[i] * inverse[i * p + j] * pooled.point[j];
        }
    }
    quadratic / (p as f64 * (1.0 + pooled.relative_increase))
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + b.abs())
}

fn assert_all_close(label: &str, got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(close(*a, *b, tolerance), "{label}[{i}]: {a} against {b}");
    }
}

/// The report's vector pooling must equal the hand pooling of the reference fits.
fn assert_vector_matches(report: &ImputationReport, points: &[Vec<f64>], covs: &[Vec<f64>]) {
    let reference = pool(points, covs);
    let vector = report.vector.as_ref().expect("a vector route reports its joint pooling");
    let p = reference.point.len();
    assert_eq!(vector.labels.len(), p);
    assert_eq!(report.pooling.as_ref(), "rubin");
    assert_eq!(vector.points.len(), points.len());
    for (got, want) in vector.points.iter().zip(points) {
        assert_all_close("per-table coordinates", got, want, 1e-8);
    }
    assert_all_close("pooled point", &vector.point, &reference.point, 1e-8);
    assert_all_close("within", &vector.within, &reference.within, 1e-8);
    assert_all_close("between", &vector.between, &reference.between, 1e-8);
    assert_all_close("total", &vector.total, &reference.total, 1e-8);
    let m = points.len() as f64;
    for i in 0..p * p {
        let expected = vector.within[i] + (1.0 + 1.0 / m) * vector.between[i];
        assert!(close(vector.total[i], expected, 1e-12), "T = U + (1 + 1/m) B at {i}");
    }
    assert!(close(vector.average_relative_increase.unwrap(), reference.relative_increase, 1e-6));
    let wald = vector.wald.as_ref().expect("the within covariance is invertible");
    assert!(close(wald.statistic, hand_wald(&reference), 1e-6), "Wald {}", wald.statistic);
    assert_eq!(wald.df_numerator, p as f64);
    assert_eq!(vector.coordinate_df.len(), p);
    assert_eq!(vector.coordinate_fraction_missing_information.len(), p);
    // The scalar fields describe the first coordinate.
    assert!(close(report.total, reference.total[0], 1e-8));
    assert!(close(report.within, reference.within[0], 1e-8));
    assert!(close(report.between, reference.between[0], 1e-8));
    assert_eq!(report.points.len(), points.len());
    assert!(!report.imputed_cells.is_empty());
    assert_eq!(report.assumptions_digest.len(), 64);
    assert!(report.assumptions_digest.chars().all(|c| c.is_ascii_hexdigit()));
}

// ------------------------------------------------------------------ completed tables

/// A table of named float columns.
struct Columns {
    names: Vec<&'static str>,
    values: Vec<Vec<f64>>,
}

impl Columns {
    fn table(&self) -> TabularData {
        Self::build(&self.names, &self.values)
    }

    fn build(names: &[&'static str], values: &[Vec<f64>]) -> TabularData {
        TabularData::from_f64_columns(
            names.iter().zip(values).map(|(name, column)| (*name, column.as_slice())),
        )
        .unwrap()
    }

    /// The completed tables the route's imputation draws: every column is in the imputation
    /// model, in table order, and rows are the resampling unit.
    fn completed(&self, spec: ImputationSpec, extra: Vec<ImputationColumn>) -> Vec<TabularData> {
        self.completed_with(spec, extra, &[])
    }

    /// [`Self::completed`] with the product of each named pair in every other conditional model.
    fn completed_with(
        &self,
        spec: ImputationSpec,
        extra: Vec<ImputationColumn>,
        interactions: &[(&str, &str)],
    ) -> Vec<TabularData> {
        let mut columns: Vec<ImputationColumn> = self
            .names
            .iter()
            .zip(&self.values)
            .map(|(name, values)| ImputationColumn::new(*name, values.clone()))
            .collect();
        columns.extend(extra);
        let pairs: Vec<(Arc<str>, Arc<str>)> =
            interactions.iter().map(|(a, b)| (Arc::from(*a), Arc::from(*b))).collect();
        let imputer = Imputer::new(columns, spec, ImputationUnits::Rows)
            .unwrap()
            .with_interactions(&pairs)
            .unwrap();
        let ctx = ctx();
        (0..spec.imputations)
            .map(|j| {
                let table = imputer.impute(j, &ctx).unwrap();
                let values: Vec<Vec<f64>> = table
                    .columns()
                    .iter()
                    .take(self.names.len())
                    .map(|column| column.to_vec())
                    .collect();
                Self::build(&self.names, &values)
            })
            .collect()
    }
}

fn report_of(result: &StudyResult) -> &ImputationReport {
    let missing = result.missing_data().as_ref().expect("a missing-data report");
    assert_eq!(missing.policy().name(), "multiple_imputation");
    assert!(missing.declared && missing.non_default);
    missing.imputation.as_ref().expect("an imputation report")
}

fn refusal(error: &CausalError) -> String {
    assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
    error.to_string()
}

// ------------------------------------------------------------------ conditional effect

const MODIFIER_EFFECT: f64 = 0.5;

/// `y = 1 + 2 t + 0.5 t w + 0.8 w + e`; `y` goes missing with probability rising in `w`, and
/// `w` is missing completely at random for one row in ten.
fn conditional_columns(n: usize, seed: u64) -> Columns {
    let mut rng = Stream(seed);
    let (mut t, mut y, mut w) = (vec![], vec![], vec![]);
    for i in 0..n {
        let wi = rng.normal();
        let ti = f64::from(u8::from(rng.bernoulli(0.5)));
        let yi = 1.0 + 2.0 * ti + MODIFIER_EFFECT * ti * wi + 0.8 * wi + rng.normal();
        y.push(if rng.bernoulli(logistic(1.2 * wi - 0.4)) { f64::NAN } else { yi });
        w.push(if i % 10 == 3 { f64::NAN } else { wi });
        t.push(ti);
    }
    Columns { names: vec!["t", "y", "w"], values: vec![t, y, w] }
}

fn conditional_graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn conditional_query(functional: OutcomeFunctional) -> CausalQuery {
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(v(0), v(1))
                .with_effect_modifiers([v(2)])
                .with_outcome_functional(functional),
        )
        .unwrap(),
    )
}

fn run_conditional(
    data: &TabularData,
    functional: OutcomeFunctional,
    inference: InferenceMode,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let mut builder = Study::tabular(data.clone())
        .graph(conditional_graph())
        .query(conditional_query(functional))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
}

#[test]
fn a_conditional_effect_pools_to_the_hand_pooled_reference() {
    on_large_stack(|| {
        let columns = conditional_columns(1_500, 21);
        let data = columns.table();
        let result = run_conditional(
            &data,
            OutcomeFunctional::Mean,
            InferenceMode::Frequentist,
            Some(declaration(spec())),
        )
        .unwrap();
        let report = report_of(&result);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in columns.completed_with(spec(), Vec::new(), &[("t", "w")]) {
            let reference = run_conditional(
                &completed,
                OutcomeFunctional::Mean,
                InferenceMode::Frequentist,
                None,
            )
            .unwrap();
            let effect = reference.estimate().as_effect().unwrap();
            points.push(vec![effect.ate]);
            covs.push(vec![effect.uncertainty.analytic.se.powi(2)]);
        }
        assert_vector_matches(report, &points, &covs);
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(vector.covariance_basis.as_ref(), "published_variance");
        // The pooled answer is the result's own estimate.
        let effect = result.estimate().as_effect().unwrap();
        assert!(close(effect.ate, vector.point[0], 1e-12));
        assert!(close(effect.uncertainty.analytic.se, vector.total[0].sqrt(), 1e-12));
        assert!(result.interval().is_none(), "the pooled interval stays closed");
        // The marginal effect is 2 + 0.5 E[w] = 2; imputation recovers it where the missing
        // outcomes depend on the modifier.
        assert!((effect.ate - 2.0).abs() < 0.3, "pooled conditional effect {}", effect.ate);
        assert!(vector.between[0] > 0.0 && vector.within[0] > 0.0);
        assert!(report.imputed_cells.iter().find(|(n, _)| n.as_ref() == "y").unwrap().1 > 100);
        assert!(report.vector.as_ref().unwrap().extras.is_empty());

        // The digest is the identity of the assumptions: a replay agrees, any change moves it.
        let again = run_conditional(
            &data,
            OutcomeFunctional::Mean,
            InferenceMode::Frequentist,
            Some(declaration(spec())),
        )
        .unwrap();
        assert_eq!(report_of(&again).assumptions_digest, report.assumptions_digest);
        let reseeded = run_conditional(
            &data,
            OutcomeFunctional::Mean,
            InferenceMode::Frequentist,
            Some(declaration(spec().with_seed(12))),
        )
        .unwrap();
        assert_ne!(report_of(&reseeded).assumptions_digest, report.assumptions_digest);
    });
}

#[test]
fn a_conditional_grid_pools_its_cells_with_the_joint_influence_covariance() {
    on_large_stack(|| {
        let columns = conditional_columns(1_500, 22);
        let data = columns.table();
        let functional = OutcomeFunctional::exceedance_grid([0.5, 1.5, 2.5]);
        let result = run_conditional(
            &data,
            functional.clone(),
            InferenceMode::Frequentist,
            Some(declaration(spec())),
        )
        .unwrap();
        let report = report_of(&result);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in columns.completed_with(spec(), Vec::new(), &[("t", "w")]) {
            let reference =
                run_conditional(&completed, functional.clone(), InferenceMode::Frequentist, None)
                    .unwrap();
            let effect = reference.estimate().as_effect().unwrap();
            let cdf = effect.joint.exceedance_cdf.as_ref().unwrap();
            let joint = effect.joint.covariance.as_ref().unwrap();
            let p = cdf.len();
            assert_eq!(joint.dim, p);
            points.push(cdf.to_vec());
            covs.push((0..p * p).map(|k| joint.get(k / p, k % p)).collect());
        }
        assert_vector_matches(report, &points, &covs);
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(vector.covariance_basis.as_ref(), "joint_covariance");
        assert_eq!(vector.labels.len(), 6, "two arms at three thresholds");
        // Cross-covariances are retained: the cells share rows, so the covariance is not diagonal.
        assert!(vector.total.iter().enumerate().any(|(k, x)| k % 7 != 0 && x.abs() > 1e-12));
        // The result's own grid carries the pooled cells and total covariance.
        let effect = result.estimate().as_effect().unwrap();
        assert_all_close(
            "grid cells",
            effect.joint.exceedance_cdf.as_ref().unwrap(),
            &vector.point,
            1e-12,
        );
        let joint = effect.joint.covariance.as_ref().unwrap();
        assert_all_close("grid covariance", &joint.values, &vector.total, 1e-12);
        assert!(effect.scores.inference.is_none(), "the first table's bands are cleared");
    });
}

#[test]
fn a_bayesian_conditional_effect_stacks_the_tables_joint_draws() {
    on_large_stack(|| {
        let columns = conditional_columns(900, 23);
        let data = columns.table();
        let bayes = || InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200));
        let result =
            run_conditional(&data, OutcomeFunctional::Mean, bayes(), Some(declaration(spec())))
                .unwrap();
        let report = report_of(&result);
        assert_eq!(report.pooling.as_ref(), "stacked_posterior");
        assert_eq!(report.estimator.as_ref(), "conditional.bayesian");
        let posterior = result.posterior.as_ref().unwrap();
        assert_eq!(posterior.draws.n_draws, 200 * IMPUTATIONS as usize, "equal weights");
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(vector.covariance_basis.as_ref(), "posterior_draws");
        assert!(vector.wald.is_none() && vector.coordinate_df.iter().all(Option::is_none));
        // The stacked mean is the mean of the completed tables' posterior means.
        let mut means = Vec::new();
        for completed in columns.completed_with(spec(), Vec::new(), &[("t", "w")]) {
            let reference =
                run_conditional(&completed, OutcomeFunctional::Mean, bayes(), None).unwrap();
            let posterior = reference.posterior.as_ref().unwrap();
            means.push(posterior.summaries.mean[posterior.effect_column().unwrap()]);
        }
        let want = means.iter().sum::<f64>() / means.len() as f64;
        let effect = result.estimate().as_effect().unwrap();
        assert!(close(effect.ate, want, 1e-9), "stacked mean {} against {want}", effect.ate);
        for (j, mean) in means.iter().enumerate() {
            assert!(close(vector.points[j][0], *mean, 1e-9), "table {j}");
        }
        assert!(vector.total[0] >= vector.within[0], "the mixture keeps the between axis");
        assert!(vector.between[0] > 0.0);
    });
}

// ------------------------------------------------------------------ response curve

/// `y = 1.7 t + 0.8 w + e` with confounder `w` driving `t`; `y` is missing with probability
/// rising in `w` and `t`, and `w` for one row in twelve.
fn response_columns(n: usize, seed: u64) -> Columns {
    let mut rng = Stream(seed);
    let (mut t, mut y, mut w) = (vec![], vec![], vec![]);
    for i in 0..n {
        let wi = rng.normal();
        let ti = 0.5 * wi + rng.normal();
        let yi = 1.7 * ti + 0.8 * wi + 0.5 * rng.normal();
        y.push(if rng.bernoulli(logistic(wi + 0.5 * ti - 0.6)) { f64::NAN } else { yi });
        w.push(if i % 12 == 5 { f64::NAN } else { wi });
        t.push(ti);
    }
    Columns { names: vec!["t", "y", "w"], values: vec![t, y, w] }
}

fn response_graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph
}

const CURVE: [f64; 3] = [-0.5, 0.0, 0.5];

fn run_response(
    data: &TabularData,
    inference: InferenceMode,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from(CURVE))),
    });
    let mut builder = Study::tabular(data.clone())
        .graph(response_graph())
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
}

fn curve(result: &StudyResult) -> Vec<f64> {
    match &result.response().unwrap().estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("expected a point-identified curve, got {other:?}"),
    }
}

#[test]
fn a_response_curve_pools_its_points_jointly() {
    on_large_stack(|| {
        let columns = response_columns(1_600, 31);
        let data = columns.table();
        let result =
            run_response(&data, InferenceMode::Frequentist, Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in columns.completed(spec(), Vec::new()) {
            let reference = run_response(&completed, InferenceMode::Frequentist, None).unwrap();
            let values = curve(&reference);
            let p = values.len();
            let joint = reference
                .estimate()
                .as_effect()
                .unwrap()
                .joint
                .covariance
                .as_ref()
                .expect("the curve retains its joint influence covariance");
            assert_eq!(joint.dim, p);
            points.push(values);
            covs.push((0..p * p).map(|k| joint.get(k / p, k % p)).collect());
        }
        assert_vector_matches(report, &points, &covs);
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(vector.covariance_basis.as_ref(), "joint_covariance");
        let pooled_curve = curve(&result);
        assert_all_close("curve", &pooled_curve, &vector.point, 1e-12);
        for (&dose, value) in CURVE.iter().zip(&pooled_curve) {
            assert!((value - 1.7 * dose).abs() < 0.3, "dose {dose}: {value}");
        }
        // The response carries a pointwise band built from the pooled total variance.
        let response = result.response().unwrap();
        let antecedent_core::ResponseUncertainty::PointwiseBand { lower, upper, .. } =
            &response.uncertainty
        else {
            panic!("a pooled curve publishes a pointwise band")
        };
        for i in 0..CURVE.len() {
            assert!(lower[i] < pooled_curve[i] && pooled_curve[i] < upper[i]);
        }
        assert!(result.interval().is_none());
    });
}

#[test]
fn a_bayesian_response_curve_stacks_the_credible_draws() {
    on_large_stack(|| {
        let columns = response_columns(900, 32);
        let data = columns.table();
        let bayes = || InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(300));
        let result = run_response(&data, bayes(), Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        assert_eq!(report.pooling.as_ref(), "stacked_posterior");
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(vector.labels.len(), CURVE.len());
        assert_eq!(vector.covariance_basis.as_ref(), "posterior_draws");
        let mut means: Vec<Vec<f64>> = Vec::new();
        for completed in columns.completed(spec(), Vec::new()) {
            means.push(curve(&run_response(&completed, bayes(), None).unwrap()));
        }
        let want: Vec<f64> = (0..CURVE.len())
            .map(|i| means.iter().map(|m| m[i]).sum::<f64>() / means.len() as f64)
            .collect();
        assert_all_close("stacked curve", &curve(&result), &want, 1e-8);
        let response = result.response().unwrap();
        let draws = response.uncertainty.credible_draws().expect("the stacked draws are retained");
        assert_eq!(draws.n_draws, 300 * IMPUTATIONS as usize);
        assert_eq!(draws.n_coordinates(), CURVE.len());
    });
}

// ------------------------------------------------------------------ interventional distribution

/// Binary `z -> t`, `z -> y`, `t -> y` with `y` and `z` missing completely at random.
fn distribution_columns(seed: u64) -> Columns {
    let mut rng = Stream(seed);
    let (mut t, mut y, mut z) = (vec![], vec![], vec![]);
    for _ in 0..1_000 {
        let zi = f64::from(u8::from(rng.bernoulli(0.5)));
        let ti = f64::from(u8::from(rng.bernoulli(0.3 + 0.4 * zi)));
        let yi = f64::from(u8::from(rng.bernoulli(0.2 + 0.4 * ti + 0.1 * zi)));
        t.push(ti);
        y.push(if rng.bernoulli(0.15) { f64::NAN } else { yi });
        z.push(if rng.bernoulli(0.1) { f64::NAN } else { zi });
    }
    Columns { names: vec!["t", "y", "z"], values: vec![t, y, z] }
}

fn run_distribution(
    data: &TabularData,
    bootstrap: u32,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let query =
        InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), Value::f64(1.0))]);
    let mut builder = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Distribution(query))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
}

#[test]
fn an_interventional_distribution_pools_its_atoms_and_mean() {
    on_large_stack(|| {
        let columns = distribution_columns(41);
        let data = columns.table();
        let result = run_distribution(&data, 60, Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in columns.completed(spec(), Vec::new()) {
            let reference = run_distribution(&completed, 60, None).unwrap();
            let dist = reference.distribution().unwrap();
            let mut point: Vec<f64> = dist.atoms.iter().map(|a| a.probability).collect();
            let mut variance: Vec<f64> =
                dist.atom_uncertainty.iter().map(|u| u.se_bootstrap.unwrap().powi(2)).collect();
            point.push(dist.mean);
            variance.push(dist.se_bootstrap.unwrap().powi(2));
            let p = point.len();
            let mut cov = vec![0.0; p * p];
            for (i, var) in variance.iter().enumerate() {
                cov[i * p + i] = *var;
            }
            points.push(point);
            covs.push(cov);
        }
        assert_vector_matches(report, &points, &covs);
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(vector.covariance_basis.as_ref(), "marginal_variances");
        assert_eq!(vector.labels.last().unwrap().as_ref(), "mean");
        // The result's distribution carries the pooled atoms with bounded intervals.
        let dist = result.distribution().unwrap();
        let atoms = dist.atoms.len();
        for (i, atom) in dist.atoms.iter().enumerate() {
            assert!(close(atom.probability, vector.point[i], 1e-12));
            let se = dist.atom_uncertainty[i].se_bootstrap.unwrap();
            assert!(close(se, vector.total[i * (atoms + 1) + i].sqrt(), 1e-12));
        }
        assert!(close(dist.mean, vector.point[atoms], 1e-12));
        let total: f64 = dist.atoms.iter().map(|a| a.probability).sum();
        assert!((total - 1.0).abs() < 1e-9, "pooled atoms stay a distribution ({total})");
        // The truth: P(y = 1 | do(t = 1)) = 0.6 + 0.1 * 0.5 = 0.65.
        assert!((dist.mean - 0.65).abs() < 0.08, "pooled interventional mean {}", dist.mean);

        // Without bootstrap replicates the plug-in publishes no variance to pool.
        let error = run_distribution(&data, 0, Some(declaration(spec()))).unwrap_err();
        let text = refusal(&error);
        assert!(text.contains("standard error"), "{text}");
    });
}

// ------------------------------------------------------------------ static mediation

/// `z -> t, m, y`; `t -> m, y`; `m -> y` with direct 0.4 and mediator slope 0.5 (mediated
/// `0.6 * 0.5 = 0.3`); `y` missing with probability rising in `z`.
fn mediation_columns(n: usize, seed: u64) -> Columns {
    let mut rng = Stream(seed);
    let (mut t, mut m, mut y, mut z) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let zi = rng.normal();
        let ti = 0.5 * zi + rng.normal();
        let mi = 0.6 * ti + 0.2 * zi + 0.3 * rng.normal();
        let yi = 0.4 * ti + 0.5 * mi + 0.3 * zi + 0.2 * rng.normal();
        y.push(if rng.bernoulli(logistic(zi - 0.8)) { f64::NAN } else { yi });
        m.push(if rng.bernoulli(0.08) { f64::NAN } else { mi });
        t.push(ti);
        z.push(zi);
    }
    Columns { names: vec!["t", "m", "y", "z"], values: vec![t, m, y, z] }
}

fn mediation_graph() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (a, b) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph
}

fn run_mediation(
    data: &TabularData,
    contrast: MediationContrast,
    bootstrap: u32,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let mut query = MediationQuery::binary(v(0), v(2), [v(1)], contrast);
    query.control = Intervention::set(v(0), Value::f64(0.0));
    query.active = Intervention::set(v(0), Value::f64(1.0));
    let mut builder = Study::tabular(data.clone())
        .graph(mediation_graph())
        .query(CausalQuery::Mediation(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
}

#[test]
fn a_static_mediation_pools_the_contrast_and_its_unvaried_components() {
    on_large_stack(|| {
        let columns = mediation_columns(1_200, 51);
        let data = columns.table();
        let result =
            run_mediation(&data, MediationContrast::Mediated, 60, Some(declaration(spec())))
                .unwrap();
        let report = report_of(&result);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        let mut components: Vec<[f64; 3]> = Vec::new();
        for completed in columns.completed(spec(), Vec::new()) {
            let reference =
                run_mediation(&completed, MediationContrast::Mediated, 60, None).unwrap();
            let effect = reference.estimate().as_effect().unwrap();
            let se = effect.uncertainty.bootstrap.se.unwrap();
            points.push(vec![effect.ate]);
            covs.push(vec![se * se]);
            let parts = reference.payload().mediation().unwrap();
            components.push([parts.total.unwrap(), parts.direct.unwrap(), parts.mediated.unwrap()]);
        }
        assert_vector_matches(report, &points, &covs);
        let vector = report.vector.as_ref().unwrap();
        // Components with no published variance are pooled by their mean with the
        // between-imputation variance beside them.
        assert_eq!(vector.extras.len(), 3);
        for (k, name) in ["total", "direct", "mediated"].iter().enumerate() {
            let extra = &vector.extras[k];
            assert_eq!(extra.label.as_ref(), *name);
            let mean = components.iter().map(|c| c[k]).sum::<f64>() / components.len() as f64;
            assert!(close(extra.point, mean, 1e-9), "{name}: {} against {mean}", extra.point);
            let spread = components.iter().map(|c| (c[k] - mean).powi(2)).sum::<f64>()
                / (components.len() as f64 - 1.0);
            assert!(close(extra.between, spread, 1e-8), "{name} between");
        }
        let parts = result.payload().mediation().unwrap();
        assert!(close(parts.total.unwrap(), vector.extras[0].point, 1e-12));
        assert!(close(parts.mediated.unwrap(), result.estimate().as_effect().unwrap().ate, 1e-6));
        assert!(
            (result.estimate().as_effect().unwrap().ate - 0.3).abs() < 0.1,
            "pooled mediated effect {}",
            result.estimate().as_effect().unwrap().ate
        );

        // A mediation fit without bootstrap replicates publishes no variance to pool.
        let error = run_mediation(&data, MediationContrast::Mediated, 0, Some(declaration(spec())))
            .unwrap_err();
        assert!(refusal(&error).contains("standard error"));
    });
}

// ------------------------------------------------------------------ randomized trial

const TRIAL_EFFECT: f64 = 0.3;

struct Trial {
    outcome: Vec<f64>,
    baseline: Vec<f64>,
    assignment: Vec<bool>,
}

/// Bernoulli(1/2) trial; the outcome is missing with probability rising in the baseline and
/// differently by arm, so responders are a selected sample.
fn trial(n: usize, seed: u64) -> Trial {
    let mut rng = Stream(seed);
    let mut t = Trial { outcome: vec![], baseline: vec![], assignment: vec![] };
    for i in 0..n {
        let arm = i % 2 == 1;
        let x = rng.normal().clamp(-2.0, 2.0);
        let y = 1.0 + TRIAL_EFFECT * f64::from(u8::from(arm)) + 2.0 * x + rng.normal();
        let b = if arm { 1.6 } else { 0.8 };
        let missing = !rng.bernoulli(logistic(0.8 - b * x));
        t.outcome.push(if missing { f64::NAN } else { y });
        t.baseline.push(x);
        t.assignment.push(arm);
    }
    t
}

impl Trial {
    fn table(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("outcome", self.outcome.as_slice()),
            ("baseline", self.baseline.as_slice()),
        ])
        .unwrap()
    }

    fn query(&self) -> RandomizedEffectQuery {
        let n = self.outcome.len();
        let units: Vec<Arc<str>> = (0..n).map(|i| Arc::from(format!("p{i}"))).collect();
        RandomizedEffectQuery::bernoulli_itt(
            v(0),
            self.assignment.clone(),
            vec![0.5; n],
            units.clone(),
            units,
            ("control", "treated"),
        )
    }

    /// The imputation model also holds the arm each row was assigned.
    fn arm(&self) -> ImputationColumn {
        ImputationColumn::new(
            "__assignment",
            self.assignment.iter().map(|&a| f64::from(u8::from(a))).collect::<Vec<f64>>(),
        )
    }

    fn run(
        &self,
        data: &TabularData,
        inference: InferenceMode,
        missing: Option<MissingDeclaration>,
    ) -> Result<StudyResult, CausalError> {
        let mut builder = Study::tabular(data.clone()).query(self.query()).inference(inference);
        if let Some(missing) = missing {
            builder = builder.missing(missing);
        }
        builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
    }
}

#[test]
fn a_randomized_trial_imputes_with_its_arm_and_pools_the_design_variance() {
    on_large_stack(|| {
        let t = trial(6_000, 61);
        let data = t.table();
        let columns = Columns {
            names: vec!["outcome", "baseline"],
            values: vec![t.outcome.clone(), t.baseline.clone()],
        };
        let result = t.run(&data, InferenceMode::Frequentist, Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in columns.completed(spec(), vec![t.arm()]) {
            let reference = t.run(&completed, InferenceMode::Frequentist, None).unwrap();
            let fit = reference.randomized_effect().unwrap();
            points.push(vec![fit.effect]);
            covs.push(vec![fit.variance_upper_bound]);
        }
        assert_vector_matches(report, &points, &covs);
        let fit = result.randomized_effect().unwrap();
        let vector = report.vector.as_ref().unwrap();
        assert!(close(fit.effect, vector.point[0], 1e-12));
        assert!(close(fit.variance_upper_bound, vector.total[0], 1e-12));
        assert_eq!(fit.interval_95, None, "no interval is published for a pooled design");
        assert!(fit.uncertainty.contains("multiple_imputation"));
        assert_eq!(fit.control_units + fit.treatment_units, 6_000, "every assignment is kept");
        // Responders are a selected sample: complete cases miss the effect, imputation with the
        // arm in the model recovers it.
        let complete = t
            .run(
                &data,
                InferenceMode::Frequentist,
                Some(MissingDeclaration::new(MissingPolicy::CompleteCase)),
            )
            .unwrap();
        let naive = complete.randomized_effect().unwrap().effect;
        assert!((naive - TRIAL_EFFECT).abs() > 0.2, "complete-case effect {naive}");
        assert!((fit.effect - TRIAL_EFFECT).abs() < 0.25, "imputed effect {}", fit.effect);
        let accounting = &report_of(&result).imputed_cells;
        assert!(accounting.iter().any(|(name, cells)| name.as_ref() == "outcome" && *cells > 500));
        // The arm is complete and is never imputed.
        assert!(
            accounting.iter().any(|(name, cells)| name.as_ref() == "__assignment" && *cells == 0)
        );
    });
}

#[test]
fn a_bayesian_randomized_trial_stacks_the_arm_contrast_posteriors() {
    on_large_stack(|| {
        let t = trial(2_000, 62);
        let data = t.table();
        let columns = Columns {
            names: vec!["outcome", "baseline"],
            values: vec![t.outcome.clone(), t.baseline.clone()],
        };
        // The outcome is real, so each arm is declared Gaussian with a weak Normal-Inverse-Gamma
        // prior; a prior alone does not identify the contrast.
        let bayes = || {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200).design_model(
                DesignPosteriorModel::ArmOutcome {
                    model: ArmOutcomeModel::Gaussian(ArmNormalPrior {
                        mean: 0.0,
                        pseudo_units: 1.0e-3,
                        variance_shape: 1.0,
                        variance_scale: 1.0,
                    }),
                },
            ))
        };
        let result = t.run(&data, bayes(), Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        assert_eq!(report.pooling.as_ref(), "stacked_posterior");
        let posterior = result.posterior.as_ref().unwrap();
        assert_eq!(posterior.draws.n_draws, 200 * IMPUTATIONS as usize);
        let mut means = Vec::new();
        for completed in columns.completed(spec(), vec![t.arm()]) {
            let reference = t.run(&completed, bayes(), None).unwrap();
            let posterior = reference.posterior.as_ref().unwrap();
            means.push(posterior.summaries.mean[posterior.effect_column().unwrap()]);
        }
        let want = means.iter().sum::<f64>() / means.len() as f64;
        assert!(close(result.estimate().as_effect().unwrap().ate, want, 1e-8));
        assert_eq!(report.vector.as_ref().unwrap().covariance_basis.as_ref(), "posterior_draws");
    });
}

// ------------------------------------------------------------------ a single series

fn series_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn pulse() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(v(0), v(1), 1.0)
            .with_horizon_steps(1)
            .with_policy(TemporalPolicy::pulse(-1)),
    )
}

/// `x` is an AR(1) process, `y_t = 2 x_{t-1} + e_t`; `x` is missing completely at random.
fn series_columns(n: usize, seed: u64, share: f64) -> Columns {
    let mut rng = Stream(seed);
    let mut x = Vec::with_capacity(n);
    let mut previous = rng.normal();
    for _ in 0..n {
        previous = 0.7 * previous + 0.7 * rng.normal();
        x.push(previous);
    }
    let y: Vec<f64> =
        (0..n).map(|t| if t == 0 { 0.0 } else { 2.0 * x[t - 1] + 0.5 * rng.normal() }).collect();
    let observed: Vec<f64> =
        x.iter().map(|value| if rng.bernoulli(share) { f64::NAN } else { *value }).collect();
    Columns { names: vec!["x", "y"], values: vec![observed, y] }
}

fn series_of(names: &[&'static str], values: &[Vec<f64>]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        names.iter().zip(values).map(|(name, column)| (*name, column.as_slice())),
        1,
    )
    .unwrap()
}

fn run_series(
    data: &TimeSeriesData,
    inference: InferenceMode,
    bootstrap: u32,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let mut builder = Study::series(data.clone())
        .query(pulse())
        .graph(series_graph())
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate_retained(&ctx())
}

/// The completed series the route draws: the lagged model with one step on each side and
/// contiguous blocks of the cube root of the length.
fn completed_series(columns: &Columns, spec: ImputationSpec, lags: usize) -> Vec<TimeSeriesData> {
    let n = columns.values[0].len();
    let block = ((n as f64).cbrt().ceil() as usize).max(2 * lags + 1);
    let units = multiple_imputation::bind_series_units(None, n, block).unwrap();
    let imputer = Imputer::new(
        columns
            .names
            .iter()
            .zip(&columns.values)
            .map(|(name, values)| ImputationColumn::new(*name, values.clone()))
            .collect(),
        spec,
        units,
    )
    .unwrap()
    .with_dynamics(lags)
    .unwrap();
    let ctx = ctx();
    (0..spec.imputations)
        .map(|j| {
            let table = imputer.impute(j, &ctx).unwrap();
            let values: Vec<Vec<f64>> = table.columns().iter().map(|c| c.to_vec()).collect();
            series_of(&columns.names, &values)
        })
        .collect()
}

#[test]
fn a_temporal_effect_imputes_by_the_lagged_model_and_pools_the_block_bootstrap() {
    on_large_stack(|| {
        let columns = series_columns(500, 71, 0.15);
        let data = series_of(&columns.names, &columns.values);
        let result =
            run_series(&data, InferenceMode::Frequentist, 80, Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        assert_eq!(report.lags, 1, "the pulse reads one step of history");
        assert_eq!(report.unit.as_ref(), "group", "a series resamples contiguous blocks");
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in completed_series(&columns, spec(), 1) {
            let reference = run_series(&completed, InferenceMode::Frequentist, 80, None).unwrap();
            let effect = reference.estimate().as_effect().unwrap();
            let se = if effect.uncertainty.analytic.se.is_finite() {
                effect.uncertainty.analytic.se
            } else {
                effect.uncertainty.bootstrap.se.unwrap()
            };
            points.push(vec![effect.ate]);
            covs.push(vec![se * se]);
        }
        assert_vector_matches(report, &points, &covs);
        let effect = result.estimate().as_effect().unwrap();
        assert!((effect.ate - 2.0).abs() < 0.25, "pooled temporal effect {}", effect.ate);
        // A different lagged neighbourhood is a different analysis.
        let other = series_columns(500, 71, 0.15);
        let wide = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
            .with_horizon_steps(1)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_max_history_lag(Some(2));
        let wide = Study::series(series_of(&other.names, &other.values))
            .query(CausalQuery::TemporalEffect(wide))
            .graph(series_graph())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(80)
            .missing(declaration(spec()))
            .build()
            .unwrap()
            .prepare(&ctx())
            .unwrap()
            .estimate_retained(&ctx())
            .unwrap();
        assert_eq!(report_of(&wide).lags, 2);
        assert_ne!(report_of(&wide).assumptions_digest, report.assumptions_digest);
        // The dependent time-series design is the only one a series accepts, and the joint
        // normal model has no lagged form.
        let clustered = Study::series(series_of(&columns.names, &columns.values))
            .query(pulse())
            .graph(series_graph())
            .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
            .missing(declaration(spec()))
            .build();
        assert!(clustered.is_err(), "a single series is not a clustered panel");
        let joint = run_series(
            &data,
            InferenceMode::Frequentist,
            0,
            Some(declaration(spec().with_model(ImputationModel::JointNormal))),
        )
        .unwrap_err();
        assert!(refusal(&joint).contains("chained-equations"), "{joint}");
    });
}

#[test]
fn a_bayesian_temporal_effect_stacks_the_completed_series_posteriors() {
    on_large_stack(|| {
        let columns = series_columns(400, 72, 0.12);
        let data = series_of(&columns.names, &columns.values);
        let bayes = || InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200));
        let result = run_series(&data, bayes(), 0, Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        assert_eq!(report.pooling.as_ref(), "stacked_posterior");
        assert_eq!(report.lags, 1);
        let mut means = Vec::new();
        for completed in completed_series(&columns, spec(), 1) {
            let reference = run_series(&completed, bayes(), 0, None).unwrap();
            let posterior = reference.posterior.as_ref().unwrap();
            means.push(posterior.summaries.mean[posterior.effect_column().unwrap()]);
        }
        let want = means.iter().sum::<f64>() / means.len() as f64;
        assert!(close(result.estimate().as_effect().unwrap().ate, want, 1e-8));
        assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 200 * IMPUTATIONS as usize);
    });
}

// ------------------------------------------------------------------ temporal mediation

fn mediation_series_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, v(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, v(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, v(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, v(3), Lag::from_raw(1)).unwrap();
    for (from, to) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// `m_t = 0.8 x_{t-1} + 0.3 z_{t-1} + e`, `y_t = 0.55 m_t + 0.4 x_{t-1} + 0.3 z_{t-1} + e`.
fn mediation_series_columns(n: usize, seed: u64) -> Columns {
    let mut rng = Stream(seed);
    let (mut z, mut x, mut m, mut y) = (vec![0.0], vec![0.0], vec![0.0], vec![0.0]);
    for t in 1..n {
        let zt = 0.5 * z[t - 1] + 0.8 * rng.normal();
        let xt = 0.5 * zt + rng.normal();
        let mt = 0.8 * x[t - 1] + 0.3 * z[t - 1] + 0.2 * rng.normal();
        let yt = 0.55 * mt + 0.4 * x[t - 1] + 0.3 * z[t - 1] + 0.2 * rng.normal();
        z.push(zt);
        x.push(xt);
        m.push(mt);
        y.push(yt);
    }
    let blank = |column: Vec<f64>, rng: &mut Stream| -> Vec<f64> {
        column.into_iter().map(|v| if rng.bernoulli(0.06) { f64::NAN } else { v }).collect()
    };
    Columns {
        names: vec!["x", "m", "y", "z"],
        values: vec![
            blank(x, &mut rng),
            blank(m, &mut rng),
            blank(y, &mut rng),
            blank(z, &mut rng),
        ],
    }
}

fn run_mediation_series(
    data: &TimeSeriesData,
    bootstrap: u32,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let query = MediationQuery::binary(v(0), v(2), [v(1)], MediationContrast::Mediated)
        .with_horizons(vec![1])
        .unwrap();
    let mut builder = Study::series(data.clone())
        .graph(mediation_series_graph())
        .query(CausalQuery::Mediation(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate_retained(&ctx())
}

#[test]
fn a_temporal_mediation_pools_total_direct_and_mediated_with_their_block_bootstrap_variances() {
    on_large_stack(|| {
        let columns = mediation_series_columns(700, 81);
        let data = series_of(&columns.names, &columns.values);
        let result = run_mediation_series(&data, 80, Some(declaration(spec()))).unwrap();
        let report = report_of(&result);
        assert_eq!(report.lags, 1);
        let mut points = Vec::new();
        let mut covs = Vec::new();
        for completed in completed_series(&columns, spec(), 1) {
            let reference = run_mediation_series(&completed, 80, None).unwrap();
            let slice = &reference.payload().mediation_grid().unwrap().slices[0];
            let antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                block,
                ..
            } = &slice.uncertainty
            else {
                panic!("frequentist temporal mediation reports its block bootstrap")
            };
            let ses = [block.total.unwrap(), block.direct.unwrap(), block.mediated.unwrap()];
            points.push(vec![
                slice.estimate.total.unwrap(),
                slice.estimate.direct.unwrap(),
                slice.estimate.mediated.unwrap(),
            ]);
            let mut cov = vec![0.0; 9];
            for (i, se) in ses.iter().enumerate() {
                cov[i * 3 + i] = se * se;
            }
            covs.push(cov);
        }
        assert_vector_matches(report, &points, &covs);
        let vector = report.vector.as_ref().unwrap();
        assert_eq!(
            vector.labels.iter().map(|l| l.as_ref()).collect::<Vec<_>>(),
            ["total@1", "direct@1", "mediated@1"]
        );
        // The result's decomposition carries the pooled components.
        let parts = result.payload().mediation().unwrap();
        assert!(close(parts.total.unwrap(), vector.point[0], 1e-12));
        assert!(close(parts.direct.unwrap(), vector.point[1], 1e-12));
        assert!(close(parts.mediated.unwrap(), vector.point[2], 1e-12));
        let requested = result.estimate().as_effect().unwrap();
        assert!(close(requested.ate, vector.point[2], 1e-9), "the requested contrast is mediated");
        assert!(close(requested.uncertainty.analytic.se, vector.total[8].sqrt(), 1e-9));
        // The truth: mediated 0.55 * 0.8 = 0.44, total 0.44 + 0.4.
        assert!((vector.point[2] - 0.44).abs() < 0.2, "pooled mediated {}", vector.point[2]);
    });
}
