//! The compiled trial-to-target transport route reads caller-supplied nuisances and computes
//! what the direct facade calls compute, and both recover known truths.
//!
//! Twins (bit equality, same kernel on the same rows):
//!
//! * a [`TransportQuery`] study with [`StudyBuilder::transport_outcome_regressions`] carries the
//!   augmented (AIPW) contrast `trial_to_target_effect_with_context` returns with `mu0`/`mu1`,
//!   and still publishes the IPW point, standard error and Wald interval of the cell;
//! * the same study toward a weighted target carries the IPW point and standard error and the
//!   AIPW point and standard error of `transport_trial_to_weighted_target`;
//! * `Study::compile_transport_response_grid` evaluates the grid
//!   `transport_augmented_response_grid_with_context` returns.
//!
//! Neither path draws from the seed, so the context seed cannot separate them.
//!
//! Independent checks (never a number the code under test produced):
//!
//! * trial rows have `z ~ N(0, 1)`, target rows `z ~ N(1, 1)`, a randomized binary treatment
//!   and the planted effect `tau(z) = 1 + z`, so with equal arm sizes the selection
//!   probability is exactly `1 / (1 + exp(z - 1/2))`, the transported effect is
//!   `E[1 + z | target] = 2`, and the true regressions are `mu0 = z`, `mu1 = 1 + 2 z`; the
//!   compiled augmented contrast on those regressions recovers 2, and toward the Gaussian
//!   kernel target of centre `c` and bandwidth `h` the closed form
//!   `1 + (1 + c / h^2) / (1 + 1 / h^2)`;
//! * a four-row grid whose augmented means are worked by hand from the formula.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits of both paths are compared exactly")]
#![allow(
    deprecated,
    reason = "the twins call the direct facade entries the compiled route replaces"
)]
use std::sync::Arc;

use antecedent::analysis::transport_response_grid::TransportResponseGridRequest;
use antecedent::analysis::transport_supplied_nuisance::{
    TransportOutcomeRegressions, TransportResponseGridColumns,
};
use antecedent::inference::{KernelFamily, TransportKernel};
use antecedent::result::PublishedScalarUncertainty;
use antecedent::{
    CausalQuery, InferenceMode, RefuteSuite, Study, StudyBuilder, TransportTrialSpec,
    WeightedTargetDeclaration,
};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery, StreamDomain,
    TransportQuery, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{EffectEstimate, OverlapPolicy, trial_to_target_ipw_se};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use antecedent_kernels::standard_normal;

const A: u32 = 0;
const Y: u32 = 1;
const TRIAL: u32 = 2;
const SEL: u32 = 3;
const PROP: u32 = 4;
const Z: u32 = 5;
const MU0: u32 = 6;
const MU1: u32 = 7;
const AUGMENTATION: &str = "estimate.transport.trial_aipw_supplied";

struct Fixture {
    z: Vec<f64>,
    outcome: Vec<f64>,
    treatment: Vec<bool>,
    trial: Vec<bool>,
    selection: Vec<f64>,
    propensity: Vec<f64>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
}

/// Alternating trial and target rows under the planted effect `1 + z`, with the true
/// regressions.
fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x53);
    let mut fx = Fixture {
        z: Vec::with_capacity(n),
        outcome: Vec::with_capacity(n),
        treatment: Vec::with_capacity(n),
        trial: Vec::with_capacity(n),
        selection: Vec::with_capacity(n),
        propensity: vec![0.5; n],
        mu0: Vec::with_capacity(n),
        mu1: Vec::with_capacity(n),
    };
    for i in 0..n {
        let in_trial = i % 2 == 0;
        let z = standard_normal(&mut rng) + if in_trial { 0.0 } else { 1.0 };
        let a = rng.next_f64() < 0.5;
        fx.z.push(z);
        fx.trial.push(in_trial);
        fx.treatment.push(a);
        fx.selection.push(1.0 / (1.0 + (z - 0.5).exp()));
        fx.outcome.push((1.0 + z) * f64::from(u8::from(a)) + z + 0.3 * standard_normal(&mut rng));
        fx.mu0.push(z);
        fx.mu1.push(1.0 + 2.0 * z);
    }
    fx
}

fn flags(values: &[bool]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(u8::from(value))).collect()
}

fn table(fx: &Fixture) -> TabularData {
    let (a, trial) = (flags(&fx.treatment), flags(&fx.trial));
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", fx.outcome.as_slice()),
        ("trial", trial.as_slice()),
        ("s", fx.selection.as_slice()),
        ("e", fx.propensity.as_slice()),
        ("z", fx.z.as_slice()),
        ("mu0", fx.mu0.as_slice()),
        ("mu1", fx.mu1.as_slice()),
    ])
    .unwrap()
}

fn binary_query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(Y),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(A),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(A)])
}

fn trial_spec() -> TransportTrialSpec {
    TransportTrialSpec {
        trial: VariableId::from_raw(TRIAL),
        selection_probability: VariableId::from_raw(SEL),
        treatment_probability: VariableId::from_raw(PROP),
    }
}

fn regressions() -> TransportOutcomeRegressions {
    TransportOutcomeRegressions::new(VariableId::from_raw(MU0), VariableId::from_raw(MU1))
}

fn builder(fx: &Fixture) -> StudyBuilder {
    let mut dag = Dag::with_variables(8);
    dag.insert_directed(DenseNodeId::from_raw(A), DenseNodeId::from_raw(Y)).unwrap();
    Study::tabular(table(fx))
        .graph(dag)
        .query(CausalQuery::Transport(binary_query()))
        .selection_targets(Arc::<[VariableId]>::from(Vec::new()))
        .transport_trial(trial_spec())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

fn kernel_target(bandwidth: f64, centre: f64) -> WeightedTargetDeclaration {
    WeightedTargetDeclaration::kernel(
        TransportKernel::new(KernelFamily::Gaussian, bandwidth).unwrap(),
        vec![VariableId::from_raw(Z)],
        vec![centre],
        None,
    )
    .unwrap()
}

fn closed_form(centre: f64, bandwidth: f64) -> f64 {
    let h2 = bandwidth * bandwidth;
    1.0 + (1.0 + centre / h2) / (1.0 + 1.0 / h2)
}

fn direct_identification() -> TransportIdentification {
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

/// The bits a structured field of the augmentation diagnostic states.
fn field_bits(result: &antecedent::StudyResult, key: &str) -> Option<u64> {
    let diagnostic = result.diagnostics.iter().find(|d| d.code.as_ref() == AUGMENTATION)?;
    let (_, value) = diagnostic.fields.iter().find(|(k, _)| k.as_ref() == key)?;
    Some(u64::from_str_radix(value, 16).expect("16 hex digits"))
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("compiled-twin".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("spawn twin thread")
        .join()
        .expect("twin body panicked");
}

#[test]
fn supplied_regressions_carry_the_direct_aipw_beside_the_unchanged_ipw_publication() {
    run_on_large_stack(|| {
        for (n, seed) in [(600_usize, 11_u64), (2_000, 12)] {
            let fx = fixture(n, seed);
            let ctx = ExecutionContext::for_tests(seed);
            let direct = antecedent::estimate::trial_to_target_effect_with_context(
                &direct_identification(),
                &fx.outcome,
                &fx.treatment,
                &fx.trial,
                &fx.selection,
                &fx.propensity,
                Some((fx.mu0.as_slice(), fx.mu1.as_slice())),
                &ctx,
            )
            .unwrap();
            let direct_se = trial_to_target_ipw_se(
                &fx.outcome,
                &fx.treatment,
                &fx.trial,
                &fx.selection,
                &fx.propensity,
                direct.ipw,
            )
            .unwrap();
            let study = builder(&fx).transport_outcome_regressions(regressions()).build().unwrap();
            let result = study.run(&ctx).unwrap();
            let transported = *result.transport().expect("transport payload");
            let aipw = direct.aipw.expect("direct AIPW with regressions");
            assert_eq!(transported.ipw.to_bits(), direct.ipw.to_bits(), "n={n}: ipw");
            assert_eq!(transported.aipw.map(f64::to_bits), Some(aipw.to_bits()), "n={n}: aipw");
            assert_eq!(transported.overlap, direct.overlap, "n={n}: overlap");
            assert_eq!(field_bits(&result, "aipw_bits"), Some(aipw.to_bits()), "n={n}");
            assert_eq!(field_bits(&result, "aipw_se_bits"), None, "n={n}: no AIPW error");

            // The publication is the cell's IPW one, as without regressions.
            let effect = result.estimate().as_effect().unwrap();
            assert_eq!(effect.ate.to_bits(), direct.ipw.to_bits(), "n={n}: point");
            assert_eq!(effect.se_analytic().to_bits(), direct_se.to_bits(), "n={n}: se");
            let published = PublishedScalarUncertainty::select(effect).interval(effect).unwrap();
            let reference = EffectEstimate::new(
                direct.ipw,
                direct_se,
                effect.provenance.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let wald = PublishedScalarUncertainty::select(&reference).interval(&reference).unwrap();
            assert_eq!(published.0.to_bits(), wald.0.to_bits(), "n={n}: lower");
            assert_eq!(published.1.to_bits(), wald.1.to_bits(), "n={n}: upper");

            let without = builder(&fx).build().unwrap().run(&ctx).unwrap();
            let plain = without.estimate().as_effect().unwrap();
            assert_eq!(plain.ate.to_bits(), effect.ate.to_bits(), "n={n}: unchanged point");
            assert!(without.transport().unwrap().aipw.is_none());
        }
    });
}

#[test]
fn supplied_regressions_toward_a_weighted_target_equal_the_direct_weighted_contrasts() {
    run_on_large_stack(|| {
        let fx = fixture(6_000, 21);
        let ctx = ExecutionContext::for_tests(21);
        for (centre, bandwidth) in [(1.0, 0.8), (1.5, 1.2)] {
            let declaration = kernel_target(bandwidth, centre);
            let direct = antecedent::transport_trial_to_weighted_target(
                &declaration,
                &table(&fx),
                &direct_identification(),
                &fx.outcome,
                &fx.treatment,
                &fx.trial,
                &fx.selection,
                &fx.propensity,
                Some((fx.mu0.as_slice(), fx.mu1.as_slice())),
            )
            .unwrap();
            let study = builder(&fx)
                .weighted_target(declaration)
                .transport_outcome_regressions(regressions())
                .build()
                .unwrap();
            let result = study.run(&ctx).unwrap();
            let effect = result.estimate().as_effect().unwrap();
            let estimate = &direct.estimate;
            assert_eq!(effect.ate.to_bits(), estimate.effect.ipw.to_bits(), "point");
            assert_eq!(effect.se_analytic().to_bits(), estimate.ipw_se.to_bits(), "se");
            let transported = *result.transport().expect("transport payload");
            assert_eq!(transported.aipw.map(f64::to_bits), estimate.effect.aipw.map(f64::to_bits));
            assert_eq!(field_bits(&result, "aipw_bits"), estimate.effect.aipw.map(f64::to_bits));
            assert_eq!(field_bits(&result, "aipw_se_bits"), estimate.aipw_se.map(f64::to_bits));
        }
    });
}

#[test]
fn the_compiled_augmented_contrast_recovers_the_planted_transported_effect() {
    run_on_large_stack(|| {
        let fx = fixture(20_000, 31);
        let ctx = ExecutionContext::for_tests(31);
        let result = builder(&fx)
            .transport_outcome_regressions(regressions())
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let aipw = result.transport().unwrap().aipw.unwrap();
        assert!((aipw - 2.0).abs() < 0.15, "augmented contrast {aipw}, truth 2");

        let (centre, bandwidth) = (1.0, 0.8);
        let weighted = builder(&fx)
            .weighted_target(kernel_target(bandwidth, centre))
            .transport_outcome_regressions(regressions())
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let aipw = weighted.transport().unwrap().aipw.unwrap();
        let truth = closed_form(centre, bandwidth);
        assert!((aipw - truth).abs() < 0.2, "weighted augmented contrast {aipw}, truth {truth}");
    });
}

#[test]
fn supplied_regressions_refuse_where_the_route_would_ignore_them() {
    run_on_large_stack(|| {
        let fx = fixture(400, 41);
        let ctx = ExecutionContext::for_tests(41);
        let bayesian = builder(&fx)
            .inference(InferenceMode::Bayesian(antecedent::BayesianConfig::laplace().n_draws(200)))
            .transport_outcome_regressions(regressions())
            .build()
            .unwrap()
            .run(&ctx)
            .expect_err("a Bayesian study does not read supplied regressions");
        assert!(bayesian.to_string().contains("option_not_applicable"), "{bayesian}");
    });
}

/// Four rows, two grid points; the augmented means are worked by hand below.
fn grid_table() -> TabularData {
    TabularData::from_f64_columns([
        ("a", [0.0, 1.0, 0.0, 1.0].as_slice()),
        ("y", [2.0, 4.0, 0.0, 0.0].as_slice()),
        ("trial", [1.0, 1.0, 0.0, 0.0].as_slice()),
        ("s", [0.5, 0.25, 0.5, 0.5].as_slice()),
        ("m", [1.5, 3.0, 0.0, 0.0].as_slice()),
        ("m0", [0.0, 0.0, 1.0, 2.0].as_slice()),
        ("m1", [0.0, 0.0, 3.0, 5.0].as_slice()),
        ("w0", [1.0, 0.5, 0.0, 0.0].as_slice()),
        ("w1", [0.25, 2.0, 0.0, 0.0].as_slice()),
    ])
    .unwrap()
}

fn grid_request() -> TransportResponseGridRequest {
    let mut admg = Admg::with_variables(9);
    admg.insert_directed(DenseNodeId::from_raw(A), DenseNodeId::from_raw(Y)).unwrap();
    let diagram = SelectionDiagram::try_new(admg, Vec::new()).unwrap();
    let id = VariableId::from_raw;
    TransportResponseGridRequest::new(
        diagram,
        binary_query(),
        grid_table(),
        TransportTrialSpec {
            trial: id(2),
            selection_probability: id(3),
            treatment_probability: id(3),
        },
        TransportResponseGridColumns::new(id(4), vec![id(5), id(6)], vec![id(7), id(8)]),
    )
}

#[test]
fn the_compiled_response_grid_equals_the_hand_worked_augmented_means() {
    // Selection odds (1 - s) / s are 1 and 3 on the trial rows; residuals y - m are 0.5 and 1.
    // Point 0: (1 + 2 + 1 * 1 * 0.5 + 3 * 0.5 * 1) / 2 = 2.5.
    // Point 1: (3 + 5 + 1 * 0.25 * 0.5 + 3 * 2 * 1) / 2 = 7.0625.
    let ctx = ExecutionContext::for_tests(7);
    let compiled = Study::compile_transport_response_grid(grid_request(), &ctx).unwrap();
    let run = compiled.execute(&ctx).unwrap();
    assert_eq!(run.estimate.grid.as_ref(), &[0.0, 1.0]);
    assert_eq!(run.estimate.mean[0].to_bits(), 2.5_f64.to_bits());
    assert_eq!(run.estimate.mean[1].to_bits(), 7.0625_f64.to_bits());
    assert_eq!(run.claim, "point_only");
    assert_eq!(run.calibration, "not_measured");
    assert_eq!(run.lowered_program_identity, compiled.lowered().identity());
    assert_eq!(compiled.plan_view().route().as_str(), "TransportResponseGrid");
}

#[test]
fn the_compiled_response_grid_equals_the_direct_supplied_equation() {
    let ctx = ExecutionContext::for_tests(8);
    let compiled = Study::compile_transport_response_grid(grid_request(), &ctx).unwrap();
    let run = compiled.execute(&ctx).unwrap();
    let column = |values: [f64; 4]| values.to_vec();
    let direct = antecedent::estimate::transport_augmented_response_grid_with_context(
        compiled.identification(),
        &column([2.0, 4.0, 0.0, 0.0]),
        &[true, true, false, false],
        &column([0.5, 0.25, 0.5, 0.5]),
        &antecedent::estimate::ResponseGridInputs {
            grid: &[0.0, 1.0],
            observed_regression: &[1.5, 3.0, 0.0, 0.0],
            target_regression: &[0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 5.0],
            response_weight: &[1.0, 0.5, 0.0, 0.0, 0.25, 2.0, 0.0, 0.0],
        },
        &ctx,
    )
    .unwrap();
    assert_eq!(run.estimate, direct, "grid, means, effective sample sizes and overlap");
    let undeclared = antecedent::estimate::transport_augmented_response_grid(
        &direct_identification(),
        &column([2.0, 4.0, 0.0, 0.0]),
        &[true, true, false, false],
        &column([0.5, 0.25, 0.5, 0.5]),
        &antecedent::estimate::ResponseGridInputs {
            grid: &[0.0, 1.0],
            observed_regression: &[1.5, 3.0, 0.0, 0.0],
            target_regression: &[0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 5.0],
            response_weight: &[1.0, 0.5, 0.0, 0.0, 0.25, 2.0, 0.0, 0.0],
        },
    )
    .unwrap();
    assert_eq!(run.estimate, undeclared, "the policy-free entry is the same evaluation");
}

#[test]
fn a_response_grid_with_a_column_count_off_the_grid_refuses_at_compile() {
    let ctx = ExecutionContext::for_tests(9);
    let mut request = grid_request();
    request.columns = TransportResponseGridColumns::new(
        VariableId::from_raw(4),
        vec![VariableId::from_raw(5)],
        vec![VariableId::from_raw(7)],
    );
    let error = Study::compile_transport_response_grid(request, &ctx).expect_err("refused");
    assert!(error.to_string().contains("grid points"), "{error}");
}

#[test]
fn the_grid_program_identity_binds_every_input_the_kernel_reads() {
    // No wire artifact is emitted: the program identity is the re-derivation. The same request
    // lowers to the same identity, and a changed weight value or a swapped column changes it.
    let ctx = ExecutionContext::for_tests(10);
    let identity = |request| {
        Study::compile_transport_response_grid(request, &ctx).unwrap().lowered().identity()
    };
    let base = identity(grid_request());
    assert_eq!(base, identity(grid_request()));
    let mut changed = grid_request();
    let mut columns: Vec<(String, Vec<f64>)> = ["a", "y", "trial", "s", "m", "m0", "m1", "w0"]
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let id = VariableId::from_raw(u32::try_from(index).unwrap());
            ((*name).to_owned(), grid_table().float64_values(id).unwrap())
        })
        .collect();
    columns.push(("w1".to_owned(), vec![0.25, 2.5, 0.0, 0.0]));
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    changed.data = TabularData::from_f64_columns(borrowed).unwrap();
    assert_ne!(base, identity(changed), "a changed response weight");
    let mut swapped = grid_request();
    swapped.columns = TransportResponseGridColumns::new(
        VariableId::from_raw(4),
        vec![VariableId::from_raw(6), VariableId::from_raw(5)],
        vec![VariableId::from_raw(7), VariableId::from_raw(8)],
    );
    assert_ne!(base, identity(swapped), "swapped grid regressions");
}
