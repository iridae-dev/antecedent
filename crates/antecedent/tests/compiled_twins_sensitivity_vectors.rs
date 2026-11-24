//! The compiled vector, posterior and marginal-structural-model sensitivity ranges against the
//! direct kernel calls, and against values computed here without the kernel.
//!
//! * Vector targets: `sensitivity_vector_range` reads one compiled result per coordinate (one
//!   average-effect or mediation study per outcome) and calls `evalue_vector_range`,
//!   `ovb_vector_range`, `mediation_vector_range`, `mediation_path_ovb_range`,
//!   `component_evalue_range` and `component_ovb_range` on the per-coordinate summaries the
//!   scalar analysis computes; the twins rebuild the target and the summaries here and assert
//!   bit equality.
//! * Posteriors: a Bayesian average effect under a Gaussian outcome model retains its effect and
//!   residual-variance draws, which `evalue_posterior_range` and `ovb_posterior_range` read; the
//!   mediation posteriors read the reference posterior of the linear working model. The twins
//!   call the kernels on the same draws; the independent checks verify the bias scale against
//!   least squares written here and the indirect effect at `rho = 0` against `a gamma` per draw.
//! * Marginal structural model: the period parameters of a longitudinal regime study fitted with
//!   the marginal structural model method, read with `msm_parameter_fit`, `component_ovb_range`,
//!   `component_evalue_range` and `msm_parameter_fit_refitting_weights` on the route's own
//!   stabilized weights; the independent check is the route's own published period effects.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    reason = "bits are compared exactly; the fixtures are small and use model notation"
)]

use std::sync::Arc;

use antecedent::analysis::sensitivity_range::{
    MsmRefitRequest, SensitivityRangeOutcome, SensitivityRangeRequest, SensitivityVectorOutcome,
    SensitivityVectorRequest, sensitivity_vector_range,
};
use antecedent::analysis::target_validation::{result_quadrant, result_refutation_target};
use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CanonicalWriter, CausalQuery, ExecutionContext, LongitudinalRegimeMethod,
    LongitudinalRegimeQuery, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_validate::target_native::{RefutationTarget, TargetFamily};
use antecedent_validate::{
    AssumptionRange, ComponentModel, EffectInterval, EffectScale, MediationComponent,
    MediationPath, MediationPosterior, MsmWeightPeriod, OvbPosterior,
    component_evalue_posterior_range, component_evalue_range, component_ovb_posterior_range,
    component_ovb_range, evalue_posterior_range, evalue_vector_range, linear_adjustment_fit,
    mediation_components_fit, mediation_path_ovb_range, mediation_paths_fit,
    mediation_posterior_range, mediation_vector_range, msm_parameter_fit,
    msm_parameter_fit_refitting_weights, ovb_posterior_range, ovb_vector_range,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5645_4354)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("sensitivity-vectors".into())
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

fn prepared(
    data: &TabularData,
    graph: Dag,
    query: CausalQuery,
    inference: InferenceMode,
) -> (PreparedStudy, StudyResult) {
    let context = ctx();
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .inference(inference)
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

/// Least squares by the normal equations: the coefficients and the residual sum of squares.
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
    let rss =
        (0..n).map(|r| (y[r] - (0..p).map(|j| beta[j] * columns[j][r]).sum::<f64>()).powi(2)).sum();
    (beta, rss)
}

fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * a.abs().max(b.abs()).max(1e-12)
}

/// The vector target the operation defines: the coordinates' outcome names, their programs.
fn vector_target(
    family: TargetFamily,
    results: &[&StudyResult],
    labels: &[&str],
) -> RefutationTarget {
    let mut writer = CanonicalWriter::new("antecedent.sensitivity_vector_target");
    for result in results {
        writer.bytes(result.lowered_program_identity().unwrap().as_bytes());
    }
    RefutationTarget::new(
        family,
        result_quadrant(results[0]),
        labels.iter().copied(),
        writer.finish(),
    )
    .unwrap()
}

// ------------------------------------------------------------------ fixtures

/// `t` (0/1), `y1`, `y2`, `z`: `z` confounds the treatment and both outcomes.
struct TwoOutcomes {
    t: Vec<f64>,
    y1: Vec<f64>,
    y2: Vec<f64>,
    z: Vec<f64>,
}

impl TwoOutcomes {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut t, mut y1, mut y2, mut z) = (vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            let c = s.normal();
            let treated = s.bernoulli(1.0 / (1.0 + (-0.7 * c).exp()));
            t.push(treated);
            y1.push(0.8 * treated + 0.6 * c + s.normal());
            y2.push(0.4 * treated - 0.5 * c + s.normal());
            z.push(c);
        }
        Self { t, y1, y2, z }
    }

    fn table(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("t", self.t.as_slice()),
            ("y1", self.y1.as_slice()),
            ("y2", self.y2.as_slice()),
            ("z", self.z.as_slice()),
        ])
        .unwrap()
    }

    fn study(&self, outcome: u32, inference: InferenceMode) -> (PreparedStudy, StudyResult) {
        prepared(
            &self.table(),
            dag(4, &[(3, 0), (3, 1), (3, 2), (0, 1), (0, 2)]),
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(outcome))),
            inference,
        )
    }
}

/// `t` (0/1), `m`, `y1`, `y2`: one mediator, two outcomes.
struct TwoMediatedOutcomes {
    t: Vec<f64>,
    m: Vec<f64>,
    y1: Vec<f64>,
    y2: Vec<f64>,
}

impl TwoMediatedOutcomes {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut t, mut m, mut y1, mut y2) = (vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            let treated = s.bernoulli(0.5);
            let mediator = 0.7 * treated + s.normal();
            t.push(treated);
            m.push(mediator);
            y1.push(0.2 * treated + 0.9 * mediator + s.normal());
            y2.push(0.5 * treated + 0.3 * mediator + s.normal());
        }
        Self { t, m, y1, y2 }
    }

    fn study(&self, outcome: u32) -> (PreparedStudy, StudyResult) {
        let data = TabularData::from_f64_columns([
            ("t", self.t.as_slice()),
            ("m", self.m.as_slice()),
            ("y1", self.y1.as_slice()),
            ("y2", self.y2.as_slice()),
        ])
        .unwrap();
        prepared(
            &data,
            dag(4, &[(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]),
            CausalQuery::Mediation(MediationQuery::binary(
                v(0),
                v(outcome),
                [v(1)],
                MediationContrast::NaturalIndirect,
            )),
            InferenceMode::Frequentist,
        )
    }
}

// ------------------------------------------------------------------ vector targets

#[test]
fn compiled_vector_evalue_and_ovb_equal_the_direct_vector_ranges() {
    on_large_stack(|| {
        let fixture = TwoOutcomes::new(400, 41);
        let (s1, r1) = fixture.study(1, InferenceMode::Frequentist);
        let (s2, r2) = fixture.study(2, InferenceMode::Frequentist);
        let coordinates = [(&s1, &r1), (&s2, &r2)];
        let target = vector_target(TargetFamily::JointContrast, &[&r1, &r2], &["y1", "y2"]);

        let outcome =
            sensitivity_vector_range(&coordinates, &SensitivityVectorRequest::EValue, &ctx())
                .unwrap();
        let SensitivityVectorOutcome::EValue { standardized, bounds, range } = outcome else {
            panic!("a vector E-value outcome");
        };
        // Each coordinate is the scalar analysis of its own study.
        for ((study, result), coordinate) in coordinates.iter().zip(&standardized) {
            let scalar =
                study.sensitivity_range(result, &SensitivityRangeRequest::EValue, &ctx()).unwrap();
            let SensitivityRangeOutcome::EValue {
                standardized_effect,
                standardized_standard_error,
                residual_sd,
                ..
            } = scalar
            else {
                panic!("a scalar E-value outcome");
            };
            assert_eq!(coordinate.0.to_bits(), standardized_effect.to_bits());
            assert_eq!(
                coordinate.1.map(f64::to_bits),
                standardized_standard_error.map(f64::to_bits)
            );
            assert_eq!(coordinate.2.to_bits(), residual_sd.to_bits());
        }
        let intervals: Vec<EffectInterval> = standardized
            .iter()
            .map(|c| EffectInterval::Standardized { estimate: c.0, standard_error: c.1 })
            .collect();
        let (direct_bounds, direct_range) = evalue_vector_range(
            target.clone(),
            &intervals,
            EffectScale::StandardizedMeanDifference,
        )
        .unwrap();
        assert_eq!(bounds, direct_bounds);
        same_range(&range, &direct_range);

        let request = SensitivityVectorRequest::OmittedVariableBias {
            q: 1.0,
            alpha: 0.05,
            r2yz_extreme: 1.0,
        };
        let outcome = sensitivity_vector_range(&coordinates, &request, &ctx()).unwrap();
        let SensitivityVectorOutcome::OmittedVariableBias { summaries, range } = outcome else {
            panic!("a vector omitted-variable-bias outcome");
        };
        let direct: Vec<_> = [&fixture.y1, &fixture.y2]
            .iter()
            .map(|y| {
                linear_adjustment_fit(y, &fixture.t, std::slice::from_ref(&fixture.z))
                    .unwrap()
                    .summary
            })
            .collect();
        assert_eq!(summaries, direct);
        same_range(&range, &ovb_vector_range(target, &direct, 1.0, 0.05, 1.0).unwrap());
    });
}

#[test]
fn compiled_vector_mediation_ranges_equal_the_direct_component_ranges() {
    on_large_stack(|| {
        let fixture = TwoMediatedOutcomes::new(500, 42);
        let (s1, r1) = fixture.study(2);
        let (s2, r2) = fixture.study(3);
        let coordinates = [(&s1, &r1), (&s2, &r2)];
        let target = || vector_target(TargetFamily::Mediation, &[&r1, &r2], &["y1", "y2"]);
        let outcomes = [&fixture.y1, &fixture.y2];

        let components: Vec<_> = outcomes
            .iter()
            .map(|y| mediation_components_fit(&fixture.t, &fixture.m, y, &[]).unwrap())
            .collect();
        for component in [MediationComponent::Indirect, MediationComponent::Direct] {
            let outcome = sensitivity_vector_range(
                &coordinates,
                &SensitivityVectorRequest::MediationComponent(component),
                &ctx(),
            )
            .unwrap();
            let SensitivityVectorOutcome::MediationComponent { components: read, range } = outcome
            else {
                panic!("a mediation component outcome");
            };
            assert_eq!(read, components);
            same_range(&range, &mediation_vector_range(target(), &components, component).unwrap());
        }

        let paths: Vec<_> = outcomes
            .iter()
            .map(|y| mediation_paths_fit(&fixture.t, &fixture.m, y, &[]).unwrap())
            .collect();
        for path in [MediationPath::MediatorOutcome, MediationPath::Indirect] {
            let request = SensitivityVectorRequest::MediationPath {
                path,
                q: 1.0,
                alpha: 0.05,
                r2yz_extreme: 1.0,
            };
            let outcome = sensitivity_vector_range(&coordinates, &request, &ctx()).unwrap();
            let SensitivityVectorOutcome::MediationPath { paths: read, range } = outcome else {
                panic!("a mediation path outcome");
            };
            assert_eq!(read, paths);
            same_range(
                &range,
                &mediation_path_ovb_range(target(), &paths, path, 1.0, 0.05, 1.0).unwrap(),
            );
        }

        let request = SensitivityVectorRequest::MediationComponentOvb {
            q: 1.0,
            alpha: 0.05,
            r2yz_extreme: 1.0,
        };
        let outcome = sensitivity_vector_range(&coordinates, &request, &ctx()).unwrap();
        let SensitivityVectorOutcome::OmittedVariableBias { summaries, range } = outcome else {
            panic!("a component omitted-variable-bias outcome");
        };
        let mediator_outcome: Vec<_> = paths.iter().map(|p| p.mediator_outcome).collect();
        assert_eq!(summaries, mediator_outcome);
        same_range(
            &range,
            &component_ovb_range(
                target(),
                &mediator_outcome,
                1.0,
                0.05,
                1.0,
                ComponentModel::Mediation,
            )
            .unwrap(),
        );

        let outcome = sensitivity_vector_range(
            &coordinates,
            &SensitivityVectorRequest::MediationComponentEValue,
            &ctx(),
        )
        .unwrap();
        let SensitivityVectorOutcome::EValue { standardized, bounds, range } = outcome else {
            panic!("a component E-value outcome");
        };
        for (result, coordinate) in [&r1, &r2].iter().zip(&standardized) {
            let effect = &result.mediation().unwrap().effect;
            assert_eq!(coordinate.0.to_bits(), (effect.ate / coordinate.2).to_bits());
        }
        let intervals: Vec<EffectInterval> = standardized
            .iter()
            .map(|c| EffectInterval::Standardized { estimate: c.0, standard_error: c.1 })
            .collect();
        let (direct_bounds, direct_range) = component_evalue_range(
            target(),
            &intervals,
            EffectScale::StandardizedMeanDifference,
            ComponentModel::Mediation,
        )
        .unwrap();
        assert_eq!(bounds, direct_bounds);
        same_range(&range, &direct_range);
    });
}

// ------------------------------------------------------------------ posteriors

#[test]
fn compiled_posterior_evalue_and_ovb_read_the_studys_own_draws() {
    on_large_stack(|| {
        let fixture = TwoOutcomes::new(400, 43);
        let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400));
        let (study, result) = fixture.study(1, inference);
        let posterior = result.posterior.as_ref().unwrap();
        let effect = posterior.draws.column(posterior.effect_column().unwrap()).unwrap().to_vec();
        let variance_column = posterior
            .draws
            .schema
            .quantities
            .iter()
            .position(|q| matches!(q, antecedent_prob::PosteriorQuantityKind::ResidualVariance))
            .expect("a Gaussian outcome model retains its residual variance");
        let variance = posterior.draws.column(variance_column).unwrap().to_vec();
        let target = result_refutation_target(&result).unwrap();

        let outcome = study
            .sensitivity_range(
                &result,
                &SensitivityRangeRequest::EValuePosterior { level: 0.95 },
                &ctx(),
            )
            .unwrap();
        let SensitivityRangeOutcome::EValuePosterior { draws, summary, range } = outcome else {
            panic!("a posterior E-value outcome");
        };
        let expected: Vec<f64> = effect.iter().zip(&variance).map(|(e, v)| e / v.sqrt()).collect();
        assert_eq!(draws, expected);
        let (direct_summary, direct_range) = evalue_posterior_range(
            target.clone(),
            std::slice::from_ref(&expected),
            EffectScale::StandardizedMeanDifference,
            0.95,
        )
        .unwrap();
        assert_eq!(summary, direct_summary);
        same_range(&range, &direct_range);

        let request = SensitivityRangeRequest::OmittedVariableBiasPosterior {
            level: 0.95,
            r2yz_extreme: 1.0,
        };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::OmittedVariableBiasPosterior {
            coefficients,
            bias_scale,
            range,
        } = outcome
        else {
            panic!("a posterior omitted-variable-bias outcome");
        };
        assert_eq!(coefficients, effect);
        // Independent: sd(T | z) on the linear-adjustment degrees of freedom, by hand.
        let n = fixture.t.len();
        let ones = vec![1.0; n];
        let (_, rss) = ols(&[ones.as_slice(), fixture.z.as_slice()], &fixture.t);
        let treatment_sd = (rss / (n - 3) as f64).sqrt();
        for (scale, v) in bias_scale.iter().zip(&variance) {
            assert!(close(*scale, v.sqrt() / treatment_sd, 1e-9), "{scale}");
        }
        let direct = ovb_posterior_range(
            target,
            &OvbPosterior {
                coefficients: std::slice::from_ref(&coefficients),
                bias_scale: std::slice::from_ref(&bias_scale),
            },
            0.95,
            1.0,
        )
        .unwrap();
        same_range(&range, &direct);
    });
}

#[test]
fn compiled_mediation_posteriors_equal_the_direct_ranges_on_the_reference_draws() {
    on_large_stack(|| {
        let fixture = TwoMediatedOutcomes::new(500, 44);
        let (study, result) = fixture.study(2);
        let program = *result.lowered_program_identity().unwrap().as_bytes();
        let coordinate = |label: &str| {
            RefutationTarget::new(
                TargetFamily::Mediation,
                result_quadrant(&result),
                [label],
                program,
            )
            .unwrap()
        };
        let request = SensitivityRangeRequest::MediationPosterior {
            component: MediationComponent::Indirect,
            level: 0.95,
            draws: 600,
        };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MediationPosterior { posterior, range, .. } = outcome else {
            panic!("a mediation posterior outcome");
        };
        // The same context draws the same reference posterior.
        let again = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MediationPosterior { posterior: repeat, .. } = again else {
            panic!("a mediation posterior outcome");
        };
        assert_eq!(posterior, repeat);
        let direct = mediation_posterior_range(
            coordinate("indirect"),
            &MediationPosterior {
                beta_mediator: std::slice::from_ref(&posterior.beta_mediator),
                sigma_total: std::slice::from_ref(&posterior.sigma_total),
                sigma_mediator: std::slice::from_ref(&posterior.sigma_mediator),
                rho_tilde: std::slice::from_ref(&posterior.rho_tilde),
                total_effect: std::slice::from_ref(&posterior.total_effect),
            },
            MediationComponent::Indirect,
            0.95,
        )
        .unwrap();
        same_range(&range, &direct);
        // Independent: per draw the indirect effect at rho = 0 is a gamma.
        for d in 0..posterior.beta_mediator.len() {
            let acme = posterior.beta_mediator[d] * posterior.sigma_total[d]
                / posterior.sigma_mediator[d]
                * posterior.rho_tilde[d];
            let product = posterior.beta_mediator[d] * posterior.mediator_outcome[d];
            assert!(close(acme, product, 1e-9), "draw {d}: {acme} against {product}");
        }
        // Independent: the mediator's residual scale on the outcome regression's degrees of
        // freedom, by hand.
        let n = fixture.t.len();
        let ones = vec![1.0; n];
        let (_, rss) = ols(&[ones.as_slice(), fixture.t.as_slice()], &fixture.m);
        assert!(close(posterior.mediator_scale, (rss / (n - 3) as f64).sqrt(), 1e-9));

        let request = SensitivityRangeRequest::MediationComponentEValuePosterior {
            component: MediationComponent::Indirect,
            level: 0.95,
            draws: 600,
        };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MediationPosterior { posterior, evalue, range } = outcome
        else {
            panic!("a mediation posterior outcome");
        };
        let standardized: Vec<f64> = (0..posterior.beta_mediator.len())
            .map(|d| {
                posterior.beta_mediator[d] * posterior.mediator_outcome[d]
                    / posterior.sigma_total[d]
            })
            .collect();
        let (summary, direct) = component_evalue_posterior_range(
            coordinate("indirect"),
            std::slice::from_ref(&standardized),
            EffectScale::StandardizedMeanDifference,
            0.95,
            ComponentModel::Mediation,
        )
        .unwrap();
        assert_eq!(evalue, Some(summary));
        same_range(&range, &direct);

        let request = SensitivityRangeRequest::MediationComponentOvbPosterior {
            level: 0.95,
            r2yz_extreme: 1.0,
            draws: 600,
        };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MediationPosterior { posterior, range, .. } = outcome else {
            panic!("a mediation posterior outcome");
        };
        let bias_scale: Vec<f64> =
            posterior.sigma_outcome.iter().map(|s| s / posterior.mediator_scale).collect();
        let direct = component_ovb_posterior_range(
            coordinate("mediator_outcome"),
            &OvbPosterior {
                coefficients: std::slice::from_ref(&posterior.mediator_outcome),
                bias_scale: std::slice::from_ref(&bias_scale),
            },
            0.95,
            1.0,
            ComponentModel::Mediation,
        )
        .unwrap();
        same_range(&range, &direct);
    });
}

// ------------------------------------------------------------------ marginal structural model

/// A two-period sequentially randomized trial with known, history-dependent probabilities, a
/// numeric history covariate per period, and the regime query fitted with the MSM method.
struct Regime {
    y: Vec<f64>,
    l1: Vec<f64>,
    l2: Vec<f64>,
    query: LongitudinalRegimeQuery,
}

impl Regime {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = Stream(seed);
        let (mut y, mut l1, mut l2) = (vec![], vec![], vec![]);
        let (mut history, mut probabilities) = (vec![], vec![]);
        for _ in 0..n {
            let first = s.normal();
            let p1 = if first > 0.0 { 0.7 } else { 0.3 };
            let a1 = s.bernoulli(p1) == 1.0;
            let second = 0.5 * f64::from(a1) + s.normal();
            let p2 = if second > 0.0 { 0.65 } else { 0.35 };
            let a2 = s.bernoulli(p2) == 1.0;
            y.push(1.0 + 2.0 * f64::from(a1) + 1.5 * f64::from(a2) + 0.8 * first + s.normal());
            l1.push(first);
            l2.push(second);
            history.extend([a1, a2]);
            probabilities.extend([p1, p2]);
        }
        let query = LongitudinalRegimeQuery {
            outcome: v(0),
            method: LongitudinalRegimeMethod::MarginalStructuralModel,
            period_outcome_predictions: None,
            known_fixed_outcome_predictions: false,
            stabilizing_numerator_probabilities: Some(Arc::from([0.5, 0.5])),
            q_predictions: None,
            observation_history: None,
            prediction_fold_ids: None,
            periods: 2,
            treatment_history: history.into(),
            regime_actions: vec![false; 2 * n].into(),
            treatment_probabilities: probabilities.into(),
            censoring_probabilities: vec![1.0; 2 * n].into(),
            outcome_observed: vec![true; n].into(),
            subject_ids: (0..n)
                .map(|i| Arc::<str>::from(format!("s{i}")))
                .collect::<Vec<_>>()
                .into(),
            fold_ids: (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into(),
            excluded_fold_predictions: false,
            probabilities_known_by_design: true,
            minimum_probability: 0.01,
            rule_id: None,
            rule_version: None,
            rule_provenance: None,
        };
        Self { y, l1, l2, query }
    }

    fn study(&self) -> (PreparedStudy, StudyResult) {
        let data = TabularData::from_f64_columns([
            ("outcome", self.y.as_slice()),
            ("l1", self.l1.as_slice()),
            ("l2", self.l2.as_slice()),
        ])
        .unwrap();
        let context = ctx();
        let prepared = Study::tabular(data.clone())
            .query(CausalQuery::LongitudinalRegime(self.query.clone()))
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        let result = prepared.estimate(&data, &context).unwrap();
        (prepared, result)
    }

    /// Exposures and stabilized weights, written out here.
    fn design(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.y.len();
        let mut exposures = vec![Vec::with_capacity(n), Vec::with_capacity(n)];
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let mut weight = 1.0;
            for t in 0..2 {
                let j = 2 * i + t;
                let a = self.query.treatment_history[j];
                let p = self.query.treatment_probabilities[j];
                weight *= if a { 0.5 / p } else { 0.5 / (1.0 - p) };
                exposures[t].push(f64::from(a));
            }
            weights.push(weight);
        }
        (exposures, weights)
    }
}

#[test]
fn compiled_msm_parameters_equal_the_direct_fit_on_the_routes_own_weights() {
    on_large_stack(|| {
        let fixture = Regime::new(600, 45);
        let (study, result) = fixture.study();
        let (exposures, weights) = fixture.design();
        let program = *result.lowered_program_identity().unwrap().as_bytes();
        let target = || {
            RefutationTarget::new(
                TargetFamily::JointContrast,
                result_quadrant(&result),
                ["period[0]", "period[1]"],
                program,
            )
            .unwrap()
        };
        let request =
            SensitivityRangeRequest::MsmParameters { q: 1.0, alpha: 0.05, r2yz_extreme: 1.0 };
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MsmParameters { summaries, weights: read, range } = outcome
        else {
            panic!("an MSM parameter outcome");
        };
        for (a, b) in read.iter().zip(&weights) {
            assert!(close(*a, *b, 1e-12));
        }
        let direct = msm_parameter_fit(&fixture.y, &exposures, &read).unwrap();
        assert_eq!(summaries, direct);
        same_range(
            &range,
            &component_ovb_range(
                target(),
                &direct,
                1.0,
                0.05,
                1.0,
                ComponentModel::MarginalStructural,
            )
            .unwrap(),
        );
        // Independent: the route's own published period effects.
        let published = result.longitudinal_regime().unwrap().period_effects.clone().unwrap();
        for (summary, effect) in summaries.iter().zip(published.iter()) {
            assert!(
                close(summary.estimate, *effect, 1e-9),
                "{} against {effect}",
                summary.estimate
            );
        }

        let outcome = study
            .sensitivity_range(&result, &SensitivityRangeRequest::MsmParameterEValue, &ctx())
            .unwrap();
        let SensitivityRangeOutcome::MsmParameterEValue { summaries, residual_sd, bounds, range } =
            outcome
        else {
            panic!("an MSM E-value outcome");
        };
        let intervals: Vec<EffectInterval> = summaries
            .iter()
            .map(|s| EffectInterval::Standardized {
                estimate: s.estimate / residual_sd,
                standard_error: Some(s.standard_error / residual_sd),
            })
            .collect();
        let (direct_bounds, direct_range) = component_evalue_range(
            target(),
            &intervals,
            EffectScale::StandardizedMeanDifference,
            ComponentModel::MarginalStructural,
        )
        .unwrap();
        assert_eq!(bounds, direct_bounds);
        same_range(&range, &direct_range);

        let mut refit = MsmRefitRequest::new(vec![vec![], vec![]], vec![vec![v(1)], vec![v(2)]], 7);
        refit.replicates = 99;
        let request = SensitivityRangeRequest::MsmParametersRefittingWeights(refit);
        let outcome = study.sensitivity_range(&result, &request, &ctx()).unwrap();
        let SensitivityRangeOutcome::MsmParametersRefittingWeights { fit, range } = outcome else {
            panic!("an MSM refit outcome");
        };
        let periods = [
            MsmWeightPeriod {
                treatment: exposures[0].clone(),
                numerator: vec![],
                denominator: vec![fixture.l1.clone()],
            },
            MsmWeightPeriod {
                treatment: exposures[1].clone(),
                numerator: vec![],
                denominator: vec![fixture.l2.clone()],
            },
        ];
        let direct =
            msm_parameter_fit_refitting_weights(&fixture.y, &exposures, &periods, 99, 7).unwrap();
        assert_eq!(fit, direct);
        same_range(
            &range,
            &component_ovb_range(
                target(),
                &direct.refit,
                1.0,
                0.05,
                1.0,
                ComponentModel::MarginalStructural,
            )
            .unwrap(),
        );
    });
}
