//! The compiled sensitivity ranges of a prepared study's own result against the direct kernel
//! calls, and against values computed here without the kernel.
//!
//! `PreparedStudy::sensitivity_range` reads the treatment, outcome, mediator and adjustment set
//! of the compiled result on the handle's retained table and calls the same kernel function as
//! the direct path: `linear_adjustment_fit`, `ovb_range` and `benchmark_bound`; `evalue_range`
//! on the result's own effect and standard error as a standardized mean difference;
//! `mediation_sensitivity_fit` and `mediation_range`; `mediation_components_fit` and
//! `mediation_vector_range`; `mediation_paths_fit` and `mediation_path_ovb_range`; and
//! `msm_ate_sensitivity` on the empirical strata of the adjustment set. The twin tests assert the
//! numbers are bit-identical to the direct calls on the same columns. The independent tests
//! check what the ranges mean with least squares and cell means written out here:
//!
//! * the omitted-variable bias of a dropped confounder at its sample partial R²s is exactly the
//!   gap between the short and the long regression (Cinelli and Hazlett 2020, section 4);
//! * the E-value is `RR + sqrt(RR (RR - 1))` at `RR = exp(0.91 d)`, `d` the effect over the
//!   residual standard deviation (`VanderWeele` and Ding 2017);
//! * the indirect effect at `rho = 0` is the product of the treatment-mediator and
//!   mediator-outcome coefficients;
//! * the marginal sensitivity model at `Lambda = 1` is the stratified mean contrast.
//!
//! The prior-sensitivity twin runs the compiled target-native refutation of a Bayesian study
//! (`PreparedStudy::refute_retained_target`), whose prior-sensitivity refuter refits the study
//! under each alternative prior scale and calls `prior_sensitivity_target_check`; the direct call
//! on the draws of the same refits, prepared here with the same context, gives the same report.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "bits are compared exactly; the fixtures are small and use model notation"
)]

use std::sync::Arc;

use antecedent::analysis::sensitivity_range::{SensitivityRangeOutcome, SensitivityRangeRequest};
use antecedent::analysis::target_validation::{
    TargetRefitOptions, TargetRefutationConfig, TargetRefuterKind, TargetSuite, result_quadrant,
    result_refutation_target,
};
use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, MediationContrast, MediationQuery,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_validate::target_native::{
    PriorSensitivityTarget, RefutationTarget, TargetFamily, prior_sensitivity_target_check,
};
use antecedent_validate::{
    AssumptionRange, EffectInterval, EffectScale, MediationComponent, MediationPath, MsmOutcomeLaw,
    MsmSensitivitySpec, MsmStratum, benchmark_bound, evalue_range, linear_adjustment_fit,
    mediation_components_fit, mediation_path_ovb_range, mediation_paths_fit, mediation_range,
    mediation_sensitivity_fit, mediation_vector_range, msm_ate_bounds_at, msm_ate_sensitivity,
    ovb_range,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5345_4E53)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Run `body` on a thread with the stack a prepared study result needs in a debug build.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("sensitivity-ranges".into())
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
        let (u, w) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos()
    }

    fn bernoulli(&mut self, p: f64) -> f64 {
        if self.next() < p { 1.0 } else { 0.0 }
    }
}

fn dag(variables: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(variables);
    for &(a, b) in edges {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph
}

/// Prepare `query` on `data` under `graph` and estimate it once on the same table.
fn prepared(data: &TabularData, graph: Dag, query: CausalQuery) -> (PreparedStudy, StudyResult) {
    let context = ctx();
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let result = prepared.estimate(data, &context).unwrap();
    (prepared, result)
}

fn same_range(compiled: &AssumptionRange, direct: &AssumptionRange) {
    assert_eq!(compiled.target.identity(), direct.target.identity(), "target");
    assert_eq!(compiled.assumption, direct.assumption);
    assert_eq!(compiled.parameter, direct.parameter);
    assert_eq!(compiled.tested.len(), direct.tested.len());
    for (a, b) in compiled.tested.iter().zip(direct.tested.iter()) {
        assert_eq!(a.parameter.to_bits(), b.parameter.to_bits());
        assert_eq!(a.target.to_bits(), b.target.to_bits(), "target at {}", a.parameter);
        assert_eq!(a.test.map(f64::to_bits), b.test.map(f64::to_bits), "test at {}", a.parameter);
    }
    assert_eq!(compiled.break_even.map(f64::to_bits), direct.break_even.map(f64::to_bits));
    assert_eq!(
        compiled.break_even_significance.map(f64::to_bits),
        direct.break_even_significance.map(f64::to_bits)
    );
    assert_eq!(compiled.identity, direct.identity);
}

// ------------------------------------------------------------------ hand-written least squares

/// Least squares of `y` on `columns` by the normal equations and Gaussian elimination with
/// partial pivoting: the coefficients and the residual sum of squares.
fn ols(columns: &[&[f64]], y: &[f64]) -> (Vec<f64>, f64) {
    let p = columns.len();
    let n = y.len();
    let mut a = vec![vec![0.0; p + 1]; p];
    for i in 0..p {
        for j in 0..p {
            a[i][j] = (0..n).map(|r| columns[i][r] * columns[j][r]).sum();
        }
        a[i][p] = (0..n).map(|r| columns[i][r] * y[r]).sum();
    }
    for k in 0..p {
        let pivot = (k..p).max_by(|&x, &z| a[x][k].abs().total_cmp(&a[z][k].abs())).unwrap();
        a.swap(k, pivot);
        for i in k + 1..p {
            let factor = a[i][k] / a[k][k];
            for j in k..=p {
                a[i][j] -= factor * a[k][j];
            }
        }
    }
    let mut beta = vec![0.0; p];
    for k in (0..p).rev() {
        let tail: f64 = (k + 1..p).map(|j| a[k][j] * beta[j]).sum();
        beta[k] = (a[k][p] - tail) / a[k][k];
    }
    let rss = (0..n)
        .map(|r| {
            let fitted: f64 = (0..p).map(|j| beta[j] * columns[j][r]).sum();
            (y[r] - fitted).powi(2)
        })
        .sum();
    (beta, rss)
}

fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * a.abs().max(b.abs()).max(1e-12)
}

// ------------------------------------------------------------------ fixtures

/// `t` (0/1), `y`, `z1`, `z2`: both covariates confound the treatment and the outcome.
struct Confounded {
    t: Vec<f64>,
    y: Vec<f64>,
    z1: Vec<f64>,
    z2: Vec<f64>,
}

impl Confounded {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut t, mut y, mut z1, mut z2) = (vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            let a = s.normal();
            let b = s.normal();
            let treated = s.bernoulli(1.0 / (1.0 + (-(0.6 * a + 0.5 * b)).exp()));
            t.push(treated);
            y.push(0.7 * treated + 0.8 * a + 0.6 * b + s.normal());
            z1.push(a);
            z2.push(b);
        }
        Self { t, y, z1, z2 }
    }

    fn table(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("t", self.t.as_slice()),
            ("y", self.y.as_slice()),
            ("z1", self.z1.as_slice()),
            ("z2", self.z2.as_slice()),
        ])
        .unwrap()
    }

    /// The study adjusting for both covariates.
    fn full(&self) -> (PreparedStudy, StudyResult) {
        prepared(
            &self.table(),
            dag(4, &[(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)]),
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(1))),
        )
    }

    /// The study whose graph omits `z2`'s edges, so it adjusts for `z1` alone.
    fn short(&self) -> (PreparedStudy, StudyResult) {
        prepared(
            &self.table(),
            dag(4, &[(2, 0), (2, 1), (0, 1)]),
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(1))),
        )
    }

    fn column(&self, id: VariableId) -> &[f64] {
        match id.raw() {
            0 => &self.t,
            1 => &self.y,
            2 => &self.z1,
            _ => &self.z2,
        }
    }
}

/// `t` (0/1), `m`, `y`: `t -> m -> y` and `t -> y`.
struct Mediated {
    t: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
}

impl Mediated {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut t, mut m, mut y) = (vec![], vec![], vec![]);
        for _ in 0..n {
            let treated = s.bernoulli(0.5);
            let mediator = 0.7 * treated + s.normal();
            t.push(treated);
            m.push(mediator);
            y.push(0.2 * treated + 0.9 * mediator + s.normal());
        }
        Self { t, m, y }
    }

    fn study(&self) -> (PreparedStudy, StudyResult) {
        let data = TabularData::from_f64_columns([
            ("t", self.t.as_slice()),
            ("m", self.m.as_slice()),
            ("y", self.y.as_slice()),
        ])
        .unwrap();
        prepared(
            &data,
            dag(3, &[(0, 1), (0, 2), (1, 2)]),
            CausalQuery::Mediation(MediationQuery::binary(
                v(0),
                v(2),
                [v(1)],
                MediationContrast::NaturalIndirect,
            )),
        )
    }
}

/// `t` (0/1), `y` in `{0, 1, 2, 3}`, `z` in `{0, 1, 2}`.
struct Stratified {
    t: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl Stratified {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut t, mut y, mut z) = (vec![], vec![], vec![]);
        for _ in 0..n {
            let stratum = (s.next() * 3.0).floor().min(2.0);
            let treated = s.bernoulli(0.3 + 0.2 * stratum);
            let outcome = (0.8 * treated + 0.6 * stratum + 1.2 * s.next()).floor().min(3.0);
            t.push(treated);
            y.push(outcome);
            z.push(stratum);
        }
        Self { t, y, z }
    }

    fn study(&self) -> (PreparedStudy, StudyResult) {
        let data = TabularData::from_f64_columns([
            ("t", self.t.as_slice()),
            ("y", self.y.as_slice()),
            ("z", self.z.as_slice()),
        ])
        .unwrap();
        prepared(
            &data,
            dag(3, &[(2, 0), (2, 1), (0, 1)]),
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(1))),
        )
    }

    /// The empirical strata, written out independently of the operation: strata and outcome
    /// values in order of first appearance, laws sorted ascending.
    fn strata(&self) -> Vec<MsmStratum> {
        let mut keys: Vec<f64> = Vec::new();
        for &z in &self.z {
            if !keys.contains(&z) {
                keys.push(z);
            }
        }
        let law = |rows: &[usize]| {
            let mut values: Vec<f64> = rows.iter().map(|&r| self.y[r]).collect();
            values.sort_by(f64::total_cmp);
            values.dedup();
            let probabilities = values
                .iter()
                .map(|&value| {
                    rows.iter().filter(|&&r| self.y[r] == value).count() as f64 / rows.len() as f64
                })
                .collect();
            MsmOutcomeLaw { values, probabilities }
        };
        keys.iter()
            .map(|&key| {
                let rows: Vec<usize> = (0..self.z.len()).filter(|&r| self.z[r] == key).collect();
                let treated: Vec<usize> =
                    rows.iter().copied().filter(|&r| self.t[r] == 1.0).collect();
                let control: Vec<usize> =
                    rows.iter().copied().filter(|&r| self.t[r] == 0.0).collect();
                MsmStratum {
                    mass: rows.len() as f64 / self.z.len() as f64,
                    propensity: treated.len() as f64 / rows.len() as f64,
                    treated: law(&treated),
                    control: law(&control),
                }
            })
            .collect()
    }
}

// ------------------------------------------------------------------ twins

#[test]
fn compiled_omitted_variable_bias_equals_the_direct_fit_range_and_benchmarks() {
    on_large_stack(|| {
        let fixture = Confounded::new(400, 11);
        let (study, result) = fixture.full();
        let adjustment: Vec<VariableId> = result.estimand.adjustment_set.to_vec();
        let covariates: Vec<Vec<f64>> =
            adjustment.iter().map(|&id| fixture.column(id).to_vec()).collect();
        let request = SensitivityRangeRequest::omitted_variable_bias();
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::OmittedVariableBias {
            fit,
            covariates: read,
            benchmarks,
            range,
        } = outcome
        else {
            panic!("an omitted-variable-bias outcome");
        };

        let direct = linear_adjustment_fit(&fixture.y, &fixture.t, &covariates).unwrap();
        assert_eq!(read, adjustment);
        assert_eq!(fit.summary.estimate.to_bits(), direct.summary.estimate.to_bits());
        assert_eq!(fit.summary.standard_error.to_bits(), direct.summary.standard_error.to_bits());
        assert_eq!(fit.summary.dof.to_bits(), direct.summary.dof.to_bits());
        assert_eq!(fit, direct);
        let target = result_refutation_target(&result).unwrap();
        same_range(&range, &ovb_range(target, &direct.summary, 1.0, 0.05, 1.0).unwrap());
        assert_eq!(benchmarks.len(), covariates.len());
        for (j, bound) in benchmarks.iter().enumerate() {
            let expected =
                benchmark_bound(direct.r2_treatment[j], direct.r2_outcome[j], 1.0, 1.0).unwrap();
            assert_eq!(bound.r2dz.to_bits(), expected.r2dz.to_bits());
            assert_eq!(bound.r2yz.to_bits(), expected.r2yz.to_bits());
        }
    });
}

#[test]
fn compiled_evalue_equals_the_direct_range_of_the_standardized_effect() {
    on_large_stack(|| {
        let fixture = Confounded::new(400, 12);
        let (study, result) = fixture.full();
        let effect = result.estimate().as_effect().unwrap().clone();
        let outcome =
            study.sensitivity_range(&result, &SensitivityRangeRequest::EValue, &ctx()).unwrap();
        let SensitivityRangeOutcome::EValue {
            standardized_effect,
            standardized_standard_error,
            residual_sd,
            bounds,
            range,
        } = outcome
        else {
            panic!("an E-value outcome");
        };
        assert_eq!(standardized_effect.to_bits(), (effect.ate / residual_sd).to_bits());
        let se = effect.se_bootstrap().unwrap_or(effect.se_analytic());
        assert_eq!(
            standardized_standard_error.map(f64::to_bits),
            Some((se / residual_sd).to_bits())
        );
        let (direct_bounds, direct_range) = evalue_range(
            result_refutation_target(&result).unwrap(),
            EffectInterval::Standardized {
                estimate: standardized_effect,
                standard_error: standardized_standard_error,
            },
            EffectScale::StandardizedMeanDifference,
        )
        .unwrap();
        assert_eq!(bounds.point.to_bits(), direct_bounds.point.to_bits());
        assert_eq!(bounds.limit.map(f64::to_bits), direct_bounds.limit.map(f64::to_bits));
        assert_eq!(bounds.risk_ratio.to_bits(), direct_bounds.risk_ratio.to_bits());
        same_range(&range, &direct_range);
    });
}

#[test]
fn compiled_mediation_range_equals_the_direct_fit_and_curve() {
    on_large_stack(|| {
        let fixture = Mediated::new(500, 13);
        let (study, result) = fixture.study();
        // The table holds only the treatment, the mediator and the outcome, so the working
        // model has no covariate whatever the identified adjustment set names.
        let outcome =
            study.sensitivity_range(&result, &SensitivityRangeRequest::Mediation, &ctx()).unwrap();
        let SensitivityRangeOutcome::Mediation { summary, range } = outcome else {
            panic!("a mediation outcome");
        };
        let direct = mediation_sensitivity_fit(&fixture.t, &fixture.m, &fixture.y, &[]).unwrap();
        assert_eq!(summary, direct);
        assert_eq!(summary.rho_tilde.to_bits(), direct.rho_tilde.to_bits());
        let target = result_refutation_target(&result).unwrap();
        assert_eq!(target.family(), TargetFamily::Mediation);
        same_range(&range, &mediation_range(target, &direct).unwrap());
    });
}

fn coordinate(result: &StudyResult, label: &str) -> RefutationTarget {
    let program = result.lowered_program_identity().unwrap();
    RefutationTarget::new(
        TargetFamily::Mediation,
        result_quadrant(result),
        [label],
        *program.as_bytes(),
    )
    .unwrap()
}

#[test]
fn compiled_mediation_components_and_paths_equal_the_direct_vector_ranges() {
    on_large_stack(|| {
        let fixture = Mediated::new(500, 14);
        let (study, result) = fixture.study();
        let components = mediation_components_fit(&fixture.t, &fixture.m, &fixture.y, &[]).unwrap();
        for (component, label) in
            [(MediationComponent::Indirect, "indirect"), (MediationComponent::Direct, "direct")]
        {
            let outcome = study
                .sensitivity_range(
                    &result,
                    &SensitivityRangeRequest::MediationComponent(component),
                    &ctx(),
                )
                .unwrap();
            let SensitivityRangeOutcome::MediationComponent { components: read, range } = outcome
            else {
                panic!("a mediation component outcome");
            };
            assert_eq!(read, components);
            let direct =
                mediation_vector_range(coordinate(&result, label), &[components], component)
                    .unwrap();
            same_range(&range, &direct);
        }
        let paths = mediation_paths_fit(&fixture.t, &fixture.m, &fixture.y, &[]).unwrap();
        for (path, label) in [
            (MediationPath::TreatmentMediator, "treatment_mediator"),
            (MediationPath::MediatorOutcome, "mediator_outcome"),
            (MediationPath::TreatmentOutcome, "treatment_outcome"),
            (MediationPath::Indirect, "indirect"),
        ] {
            let outcome = study
                .sensitivity_range(&result, &SensitivityRangeRequest::mediation_path(path), &ctx())
                .unwrap();
            let SensitivityRangeOutcome::MediationPath { paths: read, range } = outcome else {
                panic!("a mediation path outcome");
            };
            assert_eq!(read, paths);
            let direct = mediation_path_ovb_range(
                coordinate(&result, label),
                &[paths],
                path,
                1.0,
                0.05,
                1.0,
            )
            .unwrap();
            same_range(&range, &direct);
        }
    });
}

#[test]
fn compiled_marginal_sensitivity_equals_the_direct_bounds_on_the_empirical_strata() {
    on_large_stack(|| {
        let fixture = Stratified::new(600, 15);
        let (study, result) = fixture.study();
        assert_eq!(result.estimand.adjustment_set.to_vec(), vec![v(2)]);
        let spec = MsmSensitivitySpec::new(2.0).with_threshold(0.0);
        let outcome = study
            .sensitivity_range(
                &result,
                &SensitivityRangeRequest::MarginalSensitivity(spec.clone()),
                &ctx(),
            )
            .unwrap();
        let SensitivityRangeOutcome::MarginalSensitivity { strata, result: bounds } = outcome
        else {
            panic!("a marginal sensitivity outcome");
        };
        let expected = fixture.strata();
        assert_eq!(strata, expected);
        let direct = msm_ate_sensitivity(&expected, &spec).unwrap();
        assert_eq!(bounds.identified.to_bits(), direct.identified.to_bits());
        assert_eq!(bounds.grid.len(), direct.grid.len());
        for (a, b) in bounds.grid.iter().zip(&direct.grid) {
            assert_eq!(a.lambda.to_bits(), b.lambda.to_bits());
            assert_eq!(a.lower.to_bits(), b.lower.to_bits());
            assert_eq!(a.upper.to_bits(), b.upper.to_bits());
            let at = msm_ate_bounds_at(&expected, a.lambda).unwrap();
            assert_eq!(at.lower.to_bits(), a.lower.to_bits());
            assert_eq!(at.upper.to_bits(), a.upper.to_bits());
        }
        assert_eq!(bounds, direct);
    });
}

#[test]
fn compiled_prior_sensitivity_equals_the_direct_check_on_the_refit_posteriors() {
    on_large_stack(|| {
        let fixture = Confounded::new(300, 16);
        let data = fixture.table();
        let context = ctx();
        let base = BayesianConfig::conjugate().n_draws(200);
        let build = |config: BayesianConfig| {
            Study::tabular(data.clone())
                .graph(dag(4, &[(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)]))
                .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(1))))
                .inference(InferenceMode::Bayesian(config))
                .refute(RefuteSuite::None)
                .build()
                .unwrap()
                .prepare(&context)
                .unwrap()
        };
        let draws = |result: &StudyResult| {
            let posterior = result.posterior.as_ref().unwrap();
            posterior.draws.column(posterior.effect_column().unwrap()).unwrap().to_vec()
        };
        let study = build(base.clone());
        let result = study.estimate(&data, &context).unwrap();
        let options = TargetRefitOptions::default();
        let outcome = study
            .refute_retained_target(
                &result,
                TargetSuite::Cheap,
                &TargetRefutationConfig { replicates: 2, ..TargetRefutationConfig::default() },
                &options,
                &context,
            )
            .unwrap();
        let compiled = outcome
            .run
            .reports
            .iter()
            .find(|report| report.kind == TargetRefuterKind::PriorSensitivity)
            .expect("the compiled prior-sensitivity report");

        // The refuter perturbs the isotropic prior scale of a conjugate study by each factor.
        let alternatives: Vec<(Arc<str>, Vec<f64>)> = options
            .prior_scale_factors
            .iter()
            .map(|&factor| {
                let mut config = base.clone();
                config.prior_scale *= factor;
                let refit = build(config).estimate(&data, &context).unwrap();
                (Arc::from(format!("prior_scale x{factor}")), draws(&refit))
            })
            .collect();
        let target = result_refutation_target(&result).unwrap();
        let baseline = draws(&result);
        let direct = prior_sensitivity_target_check(&PriorSensitivityTarget {
            target: &target,
            baseline: &baseline,
            alternatives: &alternatives,
            tolerance: options.prior_shift_tolerance,
        })
        .unwrap();
        assert_eq!(compiled.statistic.value.to_bits(), direct.statistic.value.to_bits());
        assert_eq!(compiled.refuted.len(), direct.refuted.len());
        for (a, b) in compiled.refuted.iter().zip(direct.refuted.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(compiled.passed, direct.passed);
        assert_eq!(compiled.target.identity(), direct.target.identity());
    });
}

// ------------------------------------------------------------------ independent checks

#[test]
fn omitted_variable_bias_of_the_dropped_confounder_is_the_short_long_gap() {
    on_large_stack(|| {
        let fixture = Confounded::new(400, 21);
        let (study, result) = fixture.short();
        assert_eq!(result.estimand.adjustment_set.to_vec(), vec![v(2)], "adjusts for z1 only");
        let outcome = study
            .sensitivity_range(&result, &SensitivityRangeRequest::omitted_variable_bias(), &ctx())
            .unwrap();
        let SensitivityRangeOutcome::OmittedVariableBias { fit, .. } = outcome else {
            panic!("an omitted-variable-bias outcome");
        };
        let ones = vec![1.0; fixture.t.len()];
        let one = ones.as_slice();
        let (t, y, z1, z2) = (
            fixture.t.as_slice(),
            fixture.y.as_slice(),
            fixture.z1.as_slice(),
            fixture.z2.as_slice(),
        );
        let (short, _) = ols(&[one, t, z1], y);
        let (long, _) = ols(&[one, t, z1, z2], y);
        // Sample partial R²s of the omitted z2: with the treatment given z1, and with the
        // outcome given the treatment and z1.
        let (_, d_reduced) = ols(&[one, z1], t);
        let (_, d_full) = ols(&[one, z1, z2], t);
        let (_, y_reduced) = ols(&[one, t, z1], y);
        let (_, y_full) = ols(&[one, t, z1, z2], y);
        let r2dz = 1.0 - d_full / d_reduced;
        let r2yz = 1.0 - y_full / y_reduced;
        assert!(close(fit.summary.estimate, short[1], 1e-9), "short coefficient");
        let gap = (short[1] - long[1]).abs();
        assert!(gap > 0.05, "the omitted confounder moves the estimate: {gap}");
        let bound = fit.summary.bias_bound(r2dz, r2yz);
        assert!(close(bound, gap, 1e-7), "bias bound {bound} against the gap {gap}");
    });
}

#[test]
fn evalue_is_the_closed_form_of_the_standardized_effect() {
    on_large_stack(|| {
        let fixture = Confounded::new(400, 22);
        let (study, result) = fixture.full();
        let effect = result.estimate().as_effect().unwrap().clone();
        let outcome =
            study.sensitivity_range(&result, &SensitivityRangeRequest::EValue, &ctx()).unwrap();
        let SensitivityRangeOutcome::EValue { residual_sd, bounds, .. } = outcome else {
            panic!("an E-value outcome");
        };
        let n = fixture.t.len();
        let ones = vec![1.0; n];
        let mut columns: Vec<&[f64]> = vec![ones.as_slice(), fixture.t.as_slice()];
        for id in result.estimand.adjustment_set.iter() {
            columns.push(fixture.column(*id));
        }
        let p = columns.len();
        let (_, rss) = ols(&columns, &fixture.y);
        let sd = (rss / (n - p) as f64).sqrt();
        assert!(close(residual_sd, sd, 1e-9), "residual sd {residual_sd} against {sd}");
        let evalue = |rr: f64| {
            let rr = if rr >= 1.0 { rr } else { 1.0 / rr };
            rr + (rr * (rr - 1.0)).sqrt()
        };
        let d = effect.ate / sd;
        assert!(close(bounds.point, evalue((0.91 * d).exp()), 1e-9));
        let s = effect.se_bootstrap().unwrap_or(effect.se_analytic()) / sd;
        let (low, high) = ((0.91 * d - 1.78 * s).exp(), (0.91 * d + 1.78 * s).exp());
        let nearest = if d >= 0.0 { low.max(1.0) } else { high.min(1.0) };
        let limit = if nearest == 1.0 { 1.0 } else { evalue(nearest) };
        assert!(close(bounds.limit.unwrap(), limit, 1e-9));
    });
}

#[test]
fn indirect_effect_at_zero_correlation_is_the_product_of_coefficients() {
    on_large_stack(|| {
        let fixture = Mediated::new(500, 23);
        let (study, result) = fixture.study();
        let outcome =
            study.sensitivity_range(&result, &SensitivityRangeRequest::Mediation, &ctx()).unwrap();
        let range = outcome.range().unwrap();
        let at_zero = range.tested.iter().find(|p| p.parameter == 0.0).unwrap().target;
        let ones = vec![1.0; fixture.t.len()];
        let (one, t, m) = (ones.as_slice(), fixture.t.as_slice(), fixture.m.as_slice());
        let (a, _) = ols(&[one, t], m);
        let (b, _) = ols(&[one, t, m], &fixture.y);
        assert!(close(at_zero, a[1] * b[2], 1e-9), "{at_zero} against {}", a[1] * b[2]);
    });
}

#[test]
fn marginal_sensitivity_at_lambda_one_is_the_stratified_contrast() {
    on_large_stack(|| {
        let fixture = Stratified::new(600, 24);
        let (study, result) = fixture.study();
        let outcome = study
            .sensitivity_range(
                &result,
                &SensitivityRangeRequest::MarginalSensitivity(MsmSensitivitySpec::new(2.0)),
                &ctx(),
            )
            .unwrap();
        let SensitivityRangeOutcome::MarginalSensitivity { result: bounds, .. } = outcome else {
            panic!("a marginal sensitivity outcome");
        };
        let n = fixture.z.len() as f64;
        let mut contrast = 0.0;
        for stratum in [0.0, 1.0, 2.0] {
            let mean = |arm: f64| {
                let values: Vec<f64> = (0..fixture.z.len())
                    .filter(|&r| fixture.z[r] == stratum && fixture.t[r] == arm)
                    .map(|r| fixture.y[r])
                    .collect();
                values.iter().sum::<f64>() / values.len() as f64
            };
            let mass = fixture.z.iter().filter(|&&z| z == stratum).count() as f64 / n;
            contrast += mass * (mean(1.0) - mean(0.0));
        }
        assert!(close(bounds.identified, contrast, 1e-12));
        let first = bounds.grid[0];
        assert!(close(first.lower, contrast, 1e-12) && close(first.upper, contrast, 1e-12));
        assert!(
            bounds.grid.windows(2).all(|w| w[1].lower <= w[0].lower && w[1].upper >= w[0].upper)
        );
    });
}

#[test]
fn a_result_from_another_handle_is_refused() {
    on_large_stack(|| {
        let fixture = Confounded::new(200, 31);
        let (study, _) = fixture.full();
        let (_, other) = fixture.short();
        assert!(study.sensitivity_range(&other, &SensitivityRangeRequest::EValue, &ctx()).is_err());
        let mediated = Mediated::new(200, 32);
        let (_, effect_result) = fixture.full();
        let (mediation_study, _) = mediated.study();
        assert!(
            mediation_study
                .sensitivity_range(&effect_result, &SensitivityRangeRequest::Mediation, &ctx())
                .is_err()
        );
    });
}
