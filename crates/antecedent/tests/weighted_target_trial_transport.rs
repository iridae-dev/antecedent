//! Known-truth tests of trial-to-target transport toward a weighted target population.
//!
//! Trial rows have `z ~ N(0, 1)` and non-participant (target) rows have `z ~ N(1, 1)`, with a
//! randomized binary treatment in the trial and the planted effect modification
//! `tau(z) = 1 + z`. The selection probability is then exactly `sigmoid(0.5 - z)` for equal
//! trial and target sizes. The weighted target is the Gaussian kernel neighbourhood
//! `exp(-(z - c)^2 / (2 h^2))` of the target population; the product of `N(1, 1)` and the
//! kernel is a normal with mean `(1 + c / h^2) / (1 + 1 / h^2)`, so the transported effect has
//! the closed form `1 + (1 + c / h^2) / (1 + 1 / h^2)`, which the tests compare against, never
//! against a number the code under test produced.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::inference::{KernelFamily, TransportKernel};
use antecedent::{WeightedTargetDeclaration, transport_trial_to_weighted_target};
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::trial_to_target_effect;
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use antecedent_kernels::standard_normal;

struct Fixture {
    data: TabularData,
    outcome: Vec<f64>,
    treatment: Vec<bool>,
    trial: Vec<bool>,
    selection: Vec<f64>,
    treatment_probability: Vec<f64>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
}

fn identification() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Direct(PopulationFactor {
            regime: None,
            population: Arc::from("source"),
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        }),
        certificate: TransportCertificate {
            rule: Arc::from("transport.direct"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x41);
    let (mut z, mut y) = (vec![0.0; n], vec![0.0; n]);
    let (mut treatment, mut trial) = (vec![false; n], vec![false; n]);
    let (mut selection, mut mu0, mut mu1) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let in_trial = i % 2 == 0;
        trial[i] = in_trial;
        let zi = standard_normal(&mut rng) + if in_trial { 0.0 } else { 1.0 };
        z[i] = zi;
        selection[i] = 1.0 / (1.0 + (zi - 0.5).exp());
        let a = rng.next_f64() < 0.5;
        treatment[i] = a;
        y[i] = (1.0 + zi) * f64::from(a) + zi + 0.3 * standard_normal(&mut rng);
        mu0[i] = zi;
        mu1[i] = zi + 1.0 + zi;
    }
    let pairs = [("z", z)];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    Fixture {
        data: TabularData::from_f64_columns(borrowed).unwrap(),
        outcome: y,
        treatment,
        trial,
        selection,
        treatment_probability: vec![0.5; n],
        mu0,
        mu1,
    }
}

fn kernel_target(bandwidth: f64, centre: f64) -> WeightedTargetDeclaration {
    let kernel = TransportKernel::new(KernelFamily::Gaussian, bandwidth).unwrap();
    WeightedTargetDeclaration::kernel(kernel, vec![VariableId::from_raw(0)], vec![centre], None)
        .unwrap()
}

fn closed_form(centre: f64, bandwidth: f64) -> f64 {
    let h2 = bandwidth * bandwidth;
    1.0 + (1.0 + centre / h2) / (1.0 + 1.0 / h2)
}

fn transport(
    fx: &Fixture,
    declaration: &WeightedTargetDeclaration,
    with_regressions: bool,
) -> Result<antecedent::WeightedTrialTransport, antecedent::CausalError> {
    transport_trial_to_weighted_target(
        declaration,
        &fx.data,
        &identification(),
        &fx.outcome,
        &fx.treatment,
        &fx.trial,
        &fx.selection,
        &fx.treatment_probability,
        with_regressions.then_some((fx.mu0.as_slice(), fx.mu1.as_slice())),
    )
}

#[test]
fn a_kernel_target_effect_equals_the_closed_form_weighted_effect() {
    let fx = fixture(12_000, 31);
    for (centre, bandwidth) in [(1.0, 0.8), (1.5, 1.2), (0.5, 1.0)] {
        let result = transport(&fx, &kernel_target(bandwidth, centre), true).unwrap();
        let truth = closed_form(centre, bandwidth);
        let estimate = result.estimate;
        assert!(
            (estimate.effect.ipw - truth).abs() < 4.0 * estimate.ipw_se,
            "ipw {} se {} truth {truth}",
            estimate.effect.ipw,
            estimate.ipw_se
        );
        let aipw = estimate.effect.aipw.unwrap();
        assert!(
            (aipw - truth).abs() < 4.0 * estimate.aipw_se.unwrap().max(1e-3),
            "aipw {aipw} truth {truth}"
        );
    }
    // The compiled study executes from its retained plan once its builder is gone.
    let fx = fixture(12_000, 36);
    let z = compiled::z_of(&fx);
    let ctx = ExecutionContext::for_tests(37);
    let builder = compiled::study(&fx, &z, Some(compiled::declaration(0.8, 1.0))).unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    assert!(prepared.plan().lowered().is_some(), "a lowered plan");
    let result = prepared.estimate(&compiled::table(&fx, &z), &ctx).unwrap();
    let effect = result.estimate().as_effect().unwrap();
    let truth = closed_form(1.0, 0.8);
    assert!(
        (effect.ate - truth).abs() < 4.0 * effect.se_analytic(),
        "ate {} truth {truth}",
        effect.ate
    );
}

#[test]
fn the_weighted_target_differs_from_the_unweighted_target_effect() {
    let fx = fixture(12_000, 32);
    let weighted = transport(&fx, &kernel_target(0.5, 2.0), false).unwrap();
    let unweighted = trial_to_target_effect(
        &identification(),
        &fx.outcome,
        &fx.treatment,
        &fx.trial,
        &fx.selection,
        &fx.treatment_probability,
        None,
    )
    .unwrap();
    // The target population has mean effect 1 + 1 = 2; the neighbourhood of z = 2 has more.
    assert!(weighted.estimate.effect.ipw - unweighted.ipw > 0.3);
}

#[test]
fn constant_weights_equal_the_unweighted_contrast() {
    let fx = fixture(8_000, 33);
    let n = fx.outcome.len();
    let uniform =
        WeightedTargetDeclaration::row_weights(vec![1.0; n], vec![VariableId::from_raw(0)])
            .unwrap();
    let weighted = transport(&fx, &uniform, true).unwrap();
    let unweighted = trial_to_target_effect(
        &identification(),
        &fx.outcome,
        &fx.treatment,
        &fx.trial,
        &fx.selection,
        &fx.treatment_probability,
        Some((&fx.mu0, &fx.mu1)),
    )
    .unwrap();
    assert!((weighted.estimate.effect.ipw - unweighted.ipw).abs() < 1e-9);
    assert!((weighted.estimate.effect.aipw.unwrap() - unweighted.aipw.unwrap()).abs() < 1e-9);
}

#[test]
fn the_report_carries_the_declaration_and_the_target_overlap() {
    let fx = fixture(6_000, 34);
    let result = transport(&fx, &kernel_target(0.8, 1.0), false).unwrap();
    let report = &result.report;
    assert_eq!(report.family(), Some(KernelFamily::Gaussian));
    assert_eq!(report.center(), Some(&[1.0][..]));
    let overlap = result.estimate.target_overlap;
    assert!(overlap.kish_ess > 20.0 && overlap.n_support > 0);
    assert!(result.estimate.trial_overlap.kish_ess > 20.0);
    // A declaration with another bandwidth is another claim.
    let other = transport(&fx, &kernel_target(1.0, 1.0), false).unwrap();
    assert_ne!(report.identity(), other.report.identity());
}

#[test]
fn a_target_without_overlap_refuses() {
    let fx = fixture(6_000, 35);
    let error = transport(&fx, &kernel_target(0.2, 7.0), false).unwrap_err();
    assert!(error.to_string().contains("transport_target_overlap_failed"), "{error}");
}

// ---------------------------------------------------------------------------
// The compiled transport query: a study with a selection diagram and trial columns.
// ---------------------------------------------------------------------------

mod compiled {
    use antecedent::{RefuteSuite, Study, TransportTrialSpec};
    use antecedent_core::{
        CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional,
        ResponseQuery, TransportQuery, VariableId,
    };
    use antecedent_data::{TableView, TabularData};
    use antecedent_graph::{Dag, DenseNodeId};

    use super::{Fixture, KernelFamily, TransportKernel, WeightedTargetDeclaration, closed_form};

    const A: u32 = 0;
    const Y: u32 = 1;
    const TRIAL: u32 = 2;
    const SEL: u32 = 3;
    const PROP: u32 = 4;
    const Z: u32 = 5;

    pub(super) fn table(fx: &Fixture, z: &[f64]) -> TabularData {
        let flag = |values: &[bool]| -> Vec<f64> { values.iter().map(|v| f64::from(*v)).collect() };
        let columns: Vec<(&str, Vec<f64>)> = vec![
            ("a", flag(&fx.treatment)),
            ("y", fx.outcome.clone()),
            ("trial", flag(&fx.trial)),
            ("s", fx.selection.clone()),
            ("e", fx.treatment_probability.clone()),
            ("z", z.to_vec()),
        ];
        let borrowed: Vec<(&str, &[f64])> =
            columns.iter().map(|(name, values)| (*name, values.as_slice())).collect();
        TabularData::from_f64_columns(borrowed).unwrap()
    }

    fn query() -> TransportQuery {
        let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(Y),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(A),
                GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
            ),
        });
        TransportQuery::new(response, "trial", "target", [VariableId::from_raw(A)])
    }

    pub(super) fn study(
        fx: &Fixture,
        z: &[f64],
        target: Option<WeightedTargetDeclaration>,
    ) -> Result<Study, antecedent::CausalError> {
        let mut dag = Dag::with_variables(6);
        dag.insert_directed(DenseNodeId::from_raw(A), DenseNodeId::from_raw(Y)).unwrap();
        let mut builder = Study::tabular(table(fx, z))
            .graph(dag)
            .query(CausalQuery::Transport(query()))
            .selection_targets(std::sync::Arc::<[VariableId]>::from(Vec::new()))
            .transport_trial(TransportTrialSpec {
                trial: VariableId::from_raw(TRIAL),
                selection_probability: VariableId::from_raw(SEL),
                treatment_probability: VariableId::from_raw(PROP),
            })
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        if let Some(target) = target {
            builder = builder.weighted_target(target);
        }
        builder.build()
    }

    pub(super) fn z_of(fx: &Fixture) -> Vec<f64> {
        fx.data.float64_values(VariableId::from_raw(0)).unwrap()
    }

    pub(super) fn declaration(bandwidth: f64, centre: f64) -> WeightedTargetDeclaration {
        WeightedTargetDeclaration::kernel(
            TransportKernel::new(KernelFamily::Gaussian, bandwidth).unwrap(),
            vec![VariableId::from_raw(Z)],
            vec![centre],
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_compiled_query_reports_the_closed_form_weighted_target_effect() {
        let fx = super::fixture(12_000, 36);
        let z = z_of(&fx);
        for (centre, bandwidth) in [(1.0, 0.8), (1.5, 1.2)] {
            let study = study(&fx, &z, Some(declaration(bandwidth, centre))).unwrap();
            let result = study.run(&ExecutionContext::for_tests(36)).unwrap();
            let effect = result.estimate().as_effect().unwrap();
            let truth = closed_form(centre, bandwidth);
            assert!(
                (effect.ate - truth).abs() < 4.0 * effect.se_analytic(),
                "ate {} se {} truth {truth}",
                effect.ate,
                effect.se_analytic()
            );
            assert!(format!("{result:?}").contains("weighted_target_population"));
        }
    }

    #[test]
    fn the_compiled_query_carries_a_weighted_target_under_bayesian_inference() {
        let fx = super::fixture(12_000, 39);
        let z = z_of(&fx);
        let truth = closed_form(1.0, 0.8);
        let mut dag = Dag::with_variables(6);
        dag.insert_directed(DenseNodeId::from_raw(A), DenseNodeId::from_raw(Y)).unwrap();
        let study = Study::tabular(table(&fx, &z))
            .graph(dag)
            .query(CausalQuery::Transport(query()))
            .selection_targets(std::sync::Arc::<[VariableId]>::from(Vec::new()))
            .transport_trial(TransportTrialSpec {
                trial: VariableId::from_raw(TRIAL),
                selection_probability: VariableId::from_raw(SEL),
                treatment_probability: VariableId::from_raw(PROP),
            })
            .weighted_target(declaration(0.8, 1.0))
            .inference(antecedent::InferenceMode::Bayesian(
                antecedent::BayesianConfig::laplace().n_draws(2_000),
            ))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(39)).unwrap();
        assert!(result.posterior.is_some(), "a posterior over the weighted contrast");
        let effect = result.estimate().as_effect().unwrap();
        assert!((effect.ate - truth).abs() < 0.25, "posterior mean {} truth {truth}", effect.ate);
    }

    #[test]
    fn a_weighted_target_without_overlap_refuses_before_estimation() {
        let fx = super::fixture(6_000, 37);
        let z = z_of(&fx);
        let error = study(&fx, &z, Some(declaration(0.2, 7.0))).expect_err("refused at build");
        assert!(error.to_string().contains("transport_target_overlap_failed"), "{error}");
    }

    #[test]
    fn label_population_transport_is_unchanged() {
        let fx = super::fixture(8_000, 38);
        let z = z_of(&fx);
        let labelled = study(&fx, &z, None).unwrap();
        let result = labelled.run(&ExecutionContext::for_tests(38)).unwrap();
        let unweighted = antecedent_estimate::trial_to_target_effect(
            &super::identification(),
            &fx.outcome,
            &fx.treatment,
            &fx.trial,
            &fx.selection,
            &fx.treatment_probability,
            None,
        )
        .unwrap();
        assert!((result.estimate().as_effect().unwrap().ate - unweighted.ipw).abs() < 1e-12);
    }
}
