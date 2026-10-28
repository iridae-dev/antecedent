//! Multiple imputation on the matching and stratification estimators of the average-effect
//! route, and on a graph-posterior mixture of effect posteriors.
//!
//! Each frequentist test reproduces the route's completed tables with the public imputer,
//! refits the same study on every completed table with no missing-data declaration (propensity
//! fit, matching and stratum formation all run again on each table), and pools the per-table
//! estimates and variances by Rubin's rules written out here. The route's report must equal that
//! reference. The graph-posterior tests compare the stacked mixture with per-table references
//! and check that a construction without a justified pooling rule is refused with its own
//! detail.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and a hand-written reference"
)]

use antecedent::result::ImputationReport;
use antecedent::{
    BayesianConfig, CausalError, EstimatorId, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, MediationContrast, MediationQuery,
    VariableId,
};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{AdjacencyMask, GraphPosterior, SamplingSummary};
use antecedent_estimate::multiple_imputation::{ImputationColumn, ImputationUnits, Imputer};
use antecedent_prob::InferenceDiagnostics;

const IMPUTATIONS: u32 = 6;
const EFFECT: f64 = 1.0;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x4D49_0003)
}

fn spec() -> ImputationSpec {
    ImputationSpec::new().with_imputations(IMPUTATIONS).with_iterations(4).with_seed(17)
}

fn declaration() -> MissingDeclaration {
    MissingDeclaration::new(MissingPolicy::MultipleImputation(spec()))
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(v(0), v(1))
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
        let (u, w) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos()
    }
}

const NAMES: [&str; 4] = ["t", "y", "z", "w"];

/// Treatment `t`, outcome `y = t + 0.7 z + 1.2 w + e`, confounder `z` and a predictor `w` of the
/// outcome and of its missingness; `y` is missing at random given `w`, `z` for one row in ten.
fn raw_columns(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Stream(seed);
    let (mut t, mut y, mut z, mut w) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let zi = rng.normal();
        let wi = rng.normal();
        let ti = f64::from(u8::from(rng.next() < 1.0 / (1.0 + (-0.8 * zi).exp())));
        let yi = EFFECT * ti + 0.7 * zi + 1.2 * wi + rng.normal();
        let p_missing = 1.0 / (1.0 + (-(1.2 * wi - 0.6)).exp());
        y.push(if rng.next() < p_missing { f64::NAN } else { yi });
        z.push(if i % 10 == 3 { f64::NAN } else { zi });
        t.push(ti);
        w.push(wi);
    }
    vec![t, y, z, w]
}

fn build_table(values: &[Vec<f64>]) -> TabularData {
    TabularData::from_f64_columns(
        NAMES.iter().zip(values).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

/// The completed tables the route's imputation draws: all four columns are in the imputation
/// model, in table order, and rows are the resampling unit.
fn completed_tables(values: &[Vec<f64>]) -> Vec<TabularData> {
    let columns: Vec<ImputationColumn> = NAMES
        .iter()
        .zip(values)
        .map(|(name, column)| ImputationColumn::new(*name, column.clone()))
        .collect();
    let imputer = Imputer::new(columns, spec(), ImputationUnits::Rows).unwrap();
    let ctx = ctx();
    (0..spec().imputations)
        .map(|j| {
            let table = imputer.impute(j, &ctx).unwrap();
            let completed: Vec<Vec<f64>> =
                table.columns().iter().take(NAMES.len()).map(|c| c.to_vec()).collect();
            build_table(&completed)
        })
        .collect()
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

fn refusal(error: &CausalError) -> String {
    assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
    error.to_string()
}

fn report_of(result: &StudyResult) -> &ImputationReport {
    let missing = result.missing_data().as_ref().expect("a missing-data report");
    assert_eq!(missing.policy().name(), "multiple_imputation");
    missing.imputation.as_ref().expect("an imputation report")
}

// ------------------------------------------------------------------ the hand-written reference

/// Rubin's rules for one scalar: point, within, between, total and the Rubin degrees of freedom
/// `(m - 1) / lambda^2` with `lambda = (1 + 1/m) B / T` (the Barnard–Rubin value for an
/// unbounded complete-data degrees of freedom).
fn rubin(points: &[f64], variances: &[f64]) -> (f64, f64, f64, f64, f64) {
    let m = points.len() as f64;
    let point = points.iter().sum::<f64>() / m;
    let within = variances.iter().sum::<f64>() / m;
    let between = points.iter().map(|q| (q - point).powi(2)).sum::<f64>() / (m - 1.0);
    let total = within + (1.0 + 1.0 / m) * between;
    let lambda = (1.0 + 1.0 / m) * between / total;
    (point, within, between, total, (m - 1.0) / (lambda * lambda))
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + b.abs())
}

// ------------------------------------------------------------------ matching and stratification

fn run_adjustment(
    data: &TabularData,
    estimator: EstimatorId,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let mut builder = Study::tabular(data.clone())
        .query(query())
        .graph(graph())
        .estimator(estimator)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.run(&ctx())
}

#[test]
fn matching_and_stratification_pool_by_rubins_rules_and_refit_on_every_table() {
    let values = raw_columns(900, 41);
    let data = build_table(&values);
    let completed = completed_tables(&values);
    for (estimator, name) in [
        (EstimatorId::PropensityMatching, "propensity.matching"),
        (EstimatorId::PropensityStratification, "propensity.stratification"),
        (EstimatorId::DistanceMatching, "distance.matching"),
    ] {
        let result = run_adjustment(&data, estimator, Some(declaration())).unwrap();
        let report = report_of(&result);
        assert_eq!(report.estimator.as_ref(), name);
        assert_eq!(report.pooling.as_ref(), "rubin");

        // The reference: the whole estimator on each completed table by a separate path.
        let mut points = Vec::new();
        let mut variances = Vec::new();
        for table in &completed {
            let fit = run_adjustment(table, estimator, None).unwrap();
            let effect = fit.estimate().as_effect().unwrap();
            let se = effect.uncertainty.analytic.se;
            assert!(se.is_finite() && se > 0.0, "{name}: per-table standard error {se}");
            points.push(effect.ate);
            variances.push(se * se);
        }
        let (point, within, between, total, df) = rubin(&points, &variances);
        assert!(between > 0.0, "{name}: the tables differ, so the estimator was refit");
        assert_eq!(report.points.len(), points.len());
        for (got, want) in report.points.iter().zip(&points) {
            assert!(close(*got, *want, 1e-8), "{name}: per-table {got} against {want}");
        }
        let pooled = result.estimate().as_effect().unwrap();
        assert!(close(pooled.ate, point, 1e-8), "{name}: pooled point");
        assert!(close(report.within, within, 1e-8), "{name}: within");
        assert!(close(report.between, between, 1e-8), "{name}: between");
        assert!(close(report.total, total, 1e-8), "{name}: total");
        assert!(close(pooled.uncertainty.analytic.se, total.sqrt(), 1e-8), "{name}: se");
        assert!(close(report.df.unwrap(), df, 1e-6), "{name}: Rubin degrees of freedom");
        assert!(close(report.total, report.within + (1.0 + 1.0 / 6.0) * report.between, 1e-12));
        assert!(result.interval().is_none(), "the pooled interval stays closed");
        assert!((pooled.ate - EFFECT).abs() < 0.35, "{name}: pooled effect {}", pooled.ate);
    }
}

#[test]
fn a_stratified_imputation_carries_the_whole_procedure_bootstrap_as_its_within_variance() {
    let values = raw_columns(700, 43);
    let data = build_table(&values);
    let result = Study::tabular(data.clone())
        .query(query())
        .graph(graph())
        .estimator(EstimatorId::PropensityStratification)
        .bootstrap_replicates(40)
        .refute(RefuteSuite::None)
        .missing(declaration())
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let report = report_of(&result);
    assert_eq!(report.estimator.as_ref(), "propensity.stratification");
    assert!(report.within > 0.0 && report.between > 0.0);
    assert!(close(report.total, report.within + (1.0 + 1.0 / 6.0) * report.between, 1e-12));
    let pooled = result.estimate().as_effect().unwrap();
    assert!(close(pooled.uncertainty.analytic.se, report.total.sqrt(), 1e-12));
}

#[test]
fn an_estimator_without_a_pooled_variance_is_refused_at_build_with_the_next_attempt() {
    let data = build_table(&raw_columns(200, 3));
    let error = Study::tabular(data)
        .query(query())
        .graph(graph())
        .estimator(EstimatorId::Dml)
        .missing(declaration())
        .build()
        .unwrap_err();
    let text = refusal(&error);
    assert!(text.contains("dml"), "{text}");
    assert!(text.contains("next attempt: declare `complete_case`"), "{text}");
}

// ------------------------------------------------------------------ graph-posterior mixtures

/// Variables `[t, y, z, w]`. Atom one adjusts for the confounder `z`; atom two leaves `z` out of
/// the treatment's parents, so its backdoor set is empty and its effect is confounded.
fn two_atom_posterior(weights: [f64; 2]) -> GraphPosterior {
    let n = 4;
    let adjusted = set_edge(
        set_edge(set_edge(set_edge(0_u64, n, 0, 1, true), n, 2, 0, true), n, 2, 1, true),
        n,
        3,
        1,
        true,
    );
    let unadjusted =
        set_edge(set_edge(set_edge(0_u64, n, 0, 1, true), n, 2, 1, true), n, 3, 1, true);
    atoms(&weights, vec![adjusted, unadjusted])
}

fn atoms(weights: &[f64], masks: Vec<AdjacencyMask>) -> GraphPosterior {
    GraphPosterior::from_atoms(
        4,
        weights.to_vec(),
        masks,
        vec![0.0; 16],
        vec![0.0; 16],
        SamplingSummary {
            ess: 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("imputed_graph_posterior_fixture"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_algorithm("fixture")
}

fn single_atom(index: usize) -> GraphPosterior {
    let n = 4;
    let mask = if index == 0 {
        set_edge(
            set_edge(set_edge(set_edge(0_u64, n, 0, 1, true), n, 2, 0, true), n, 2, 1, true),
            n,
            3,
            1,
            true,
        )
    } else {
        set_edge(set_edge(set_edge(0_u64, n, 0, 1, true), n, 2, 1, true), n, 3, 1, true)
    };
    atoms(&[1.0], vec![mask])
}

fn run_posterior(
    data: &TabularData,
    posterior: GraphPosterior,
    missing: Option<MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let inference =
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400).prior_scale(30.0));
    let mut builder = Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(query())
        .inference(inference)
        .refute(RefuteSuite::None);
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
}

/// Mean, standard deviation and draw count of a result's effect posterior.
fn effect_summary(result: &StudyResult) -> (f64, f64, usize) {
    let posterior = result.posterior.as_ref().expect("an effect posterior");
    let column = posterior.effect_column().expect("an effect column");
    (posterior.summaries.mean[column], posterior.summaries.sd[column], posterior.draws.n_draws)
}

#[test]
fn a_graph_posterior_mixture_is_pooled_by_stacking_draws_with_its_graph_weights() {
    let values = raw_columns(800, 47);
    let data = build_table(&values);
    let weights = [0.7, 0.3];
    let result = run_posterior(&data, two_atom_posterior(weights), Some(declaration())).unwrap();
    let report = report_of(&result);
    assert_eq!(report.pooling.as_ref(), "stacked_graph_posterior");
    assert_eq!(report.estimator.as_ref(), "bayesian.gcomp");
    assert_eq!(report.points.len(), IMPUTATIONS as usize);

    // The references: the mixture and each atom alone, refit on every completed table.
    let completed = completed_tables(&values);
    let mut table_draws = 0;
    let mut mixture_means = Vec::new();
    let mut atom_means = [0.0, 0.0];
    for table in &completed {
        let (mean, _, draws) =
            effect_summary(&run_posterior(table, two_atom_posterior(weights), None).unwrap());
        table_draws = draws;
        mixture_means.push(mean);
        for (index, slot) in atom_means.iter_mut().enumerate() {
            *slot += effect_summary(&run_posterior(table, single_atom(index), None).unwrap()).0
                / f64::from(IMPUTATIONS);
        }
    }
    let (pooled_mean, pooled_sd, pooled_draws) = effect_summary(&result);
    assert_eq!(pooled_draws, table_draws * IMPUTATIONS as usize, "equal-weight stack of tables");
    assert!(
        (atom_means[0] - atom_means[1]).abs() > 0.1,
        "the two atoms must disagree for the weights to be identifiable: {atom_means:?}"
    );

    // The stacked mean is the mean of the per-table means, up to the draws' Monte Carlo error.
    let reference = mixture_means.iter().sum::<f64>() / mixture_means.len() as f64;
    let mc = 8.0 * pooled_sd / (pooled_draws as f64).sqrt();
    assert!((pooled_mean - reference).abs() < mc, "stacked {pooled_mean} against {reference}");
    // The graph weights survive: the mixture mean is the weighted mean of the atoms' means.
    let weighted = weights[0] * atom_means[0] + weights[1] * atom_means[1];
    assert!((pooled_mean - weighted).abs() < mc, "weighted {weighted} against {pooled_mean}");
    let even = 0.5 * (atom_means[0] + atom_means[1]);
    assert!((pooled_mean - even).abs() > mc, "equal weights would not reproduce {pooled_mean}");
    let mean_of_points = report.points.iter().sum::<f64>() / report.points.len() as f64;
    assert!(close(mean_of_points, pooled_mean, 1e-9), "report points are the per-table means");
    assert!(report.between >= 0.0 && report.total >= report.within);
}

#[test]
fn a_graph_posterior_without_a_justified_pooling_rule_is_refused_with_its_own_detail() {
    let data = build_table(&raw_columns(200, 5));
    // A frequentist study has no effect posterior to mix.
    let frequentist = Study::tabular(data.clone())
        .graph_posterior(two_atom_posterior([0.5, 0.5]))
        .query(query())
        .missing(declaration())
        .build()
        .unwrap_err();
    let text = refusal(&frequentist);
    assert!(text.contains("multiple_imputation.graph_posterior_frequentist"), "{text}");
    assert!(text.contains("next attempt: declare `complete_case`"), "{text}");

    // A mediation contrast over the mixture is answered atom by atom.
    let mediation = MediationQuery::binary(v(0), v(1), [v(2)], MediationContrast::Mediated);
    let by_atom = Study::tabular(data)
        .graph_posterior(two_atom_posterior([0.5, 0.5]))
        .query(CausalQuery::Mediation(mediation))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100)))
        .missing(declaration())
        .build()
        .unwrap_err();
    let text = refusal(&by_atom);
    assert!(text.contains("multiple_imputation.graph_posterior_query"), "{text}");
}
