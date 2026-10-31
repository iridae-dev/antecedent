//! Multiple imputation on the augmented inverse-probability and inverse-probability-weighting
//! estimators of the average-effect route.
//!
//! The test reproduces the route's completed tables with the public imputer, refits the same
//! study on every completed table with no missing-data declaration (the propensity model and
//! the outcome model run again on each table), and pools the per-table estimates and variances
//! by Rubin's rules written out here. The route's report must equal that reference; the
//! missing data are missing at random given the observed predictors, so the pooled effect also
//! recovers the simulated effect where the complete cases are shifted.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and a hand-written reference"
)]

use antecedent::result::ImputationReport;
use antecedent::{CausalError, EstimatorId, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};
use antecedent_estimate::multiple_imputation::{ImputationColumn, ImputationUnits, Imputer};

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
fn aipw_and_weighting_pool_by_rubins_rules_and_refit_on_every_table() {
    let values = raw_columns(900, 43);
    let data = build_table(&values);
    let completed = completed_tables(&values);
    for (estimator, name) in [
        (EstimatorId::Aipw, "aipw"),
        (EstimatorId::PropensityWeighting, "propensity.weighting"),
    ] {
        let result = run_adjustment(&data, estimator, Some(declaration())).unwrap();
        let report = report_of(&result);
        assert_eq!(report.estimator.as_ref(), name);
        assert_eq!(report.pooling.as_ref(), "rubin");

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
        // Observed gaps are 0.014 and 0.033 against a pooled standard error of 0.12.
        assert!(
            (pooled.ate - EFFECT).abs() < 0.15,
            "{name}: pooled effect {} vs the simulated {EFFECT}",
            pooled.ate
        );
    }
}

