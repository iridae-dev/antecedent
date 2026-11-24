//! The compiled artifact-producer families and the direct producers they replace give the same
//! numbers.
//!
//! Each family compiles the input its direct producer takes and, after the support gate of its
//! unit, calls that producer with the same arguments: the nested-Markov Fisher interval and
//! posterior, the learned and the Gaussian joint transport fits, the shared-row scenario
//! covariance, the external feedback equilibrium, and the sampled counterfactual transport
//! estimate and its interval. Every comparison here is on the bytes of the sealed artifacts
//! (which carry every stored number) and on the bits of each measured point, interval bound and
//! level, with the calibration slots compared whole.
//!
//! Seeds: both paths take the caller's `ExecutionContext`, and each comparison hands them the
//! same `ExecutionContext::for_tests(seed)`; the posterior, joint transport and bootstrap draws
//! come from the seeds the inputs declare, which the compiled program keeps unchanged.
//!
//! The fixtures are the ones the producers' own artifact tests use; the counterfactual transport
//! fixtures are included by path from the identifier's tests.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    dead_code,
    reason = "test fixtures: exact bit comparisons are the assertion"
)]

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent::Study;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::counterfactual_transport_sampled::{
    OverlapDeclaration, SampledAxis, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTransportLimits, SampledTransportRequest, SamplingDesign, UnseenCellRule,
};
use antecedent_estimate::external_cyclic_fit::ExternalFitPolicy;
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_estimate::learned_joint_transport::LearnedJointModel;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_expr::ExactDiscreteLaw;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, PopulationFactor, TransportCertificate, TransportFormula,
    TransportIdentification, identify_counterfactual_transport,
};
use antecedent_io::external_cyclic_artifact::{ExternalCyclicRequest, ExternalFitRequest};
use antecedent_io::joint_bayesian_transport_artifact::JointBayesianArtifactWire;
use antecedent_io::learned_joint_transport_artifact::LearnedJointArtifactWire;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::nested_markov_artifact::NestedFisherArtifact;
use antecedent_io::nested_markov_bayesian_artifact::Artifact as NestedPosteriorArtifact;
use antecedent_io::sampled_counterfactual_transport_artifact::{
    SampledCounterfactualTransportArtifactWire as SampledWire,
    SampledCounterfactualTransportConsumeLimits, sampled_counterfactual_transport_artifact,
};
use antecedent_io::scenario_covariance_artifact::{
    CovarianceMethodWire, CovarianceScenarioWire, CovarianceSpec, FunctionalWire, RowTableWire,
    ScenarioCovarianceConsumeLimits, ScoreTermWire,
};
use antecedent_learn::nested_markov_bayesian::{Options, Prior};
use antecedent_model::external_mechanism::{
    DerivativeDeclaration, ExternalFeedbackSystem, ExternalFitParameter,
    ExternalMechanismImplementation, ExternalMechanismSpec, MechanismRegistry, SystemEquation,
    TrustedMechanismResolver,
};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumError, RegimeRow, SolvePolicy,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, all_cases};

/// The support unit of a compiled program's route, with its route name.
fn assert_bound(view: &antecedent::planner::PlanView, route: &str, unit: &str) {
    assert_eq!(view.route().as_str(), route);
    let decision = view.unit_support().expect("the route is bound to a unit");
    assert_eq!(decision.unit(), unit);
    assert!(decision.supported(), "{unit} is supported");
}

/// Two measured executions report the same scalars bit for bit under the same records.
fn assert_same_measured(direct: &MeasuredInference, compiled: &MeasuredInference) {
    assert_eq!(direct.export(), compiled.export());
    assert_eq!(direct.source_artifact(), compiled.source_artifact());
    let (direct, compiled) = (direct.report(), compiled.report());
    assert_eq!(direct.route, compiled.route);
    assert_eq!(direct.calibration, compiled.calibration);
    assert_eq!(direct.scalars.len(), compiled.scalars.len());
    for (d, c) in direct.scalars.iter().zip(&compiled.scalars) {
        assert_eq!(d.name, c.name);
        assert_eq!(d.point.to_bits(), c.point.to_bits(), "{}", d.name);
        assert_eq!(d.lower.to_bits(), c.lower.to_bits(), "{}", d.name);
        assert_eq!(d.upper.to_bits(), c.upper.to_bits(), "{}", d.name);
        assert_eq!(d.level.to_bits(), c.level.to_bits(), "{}", d.name);
        assert_eq!(d.calibration, c.calibration, "{}", d.name);
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// Binary nested Markov (fixtures of the nested-Markov artifact tests).
// ---------------------------------------------------------------------------

fn b(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

const PU: f64 = 0.3;

/// Observed law of the latent-variable SCM, summed over the latent variable by hand.
fn scm_law(px1: f64, eps: f64) -> [f64; 16] {
    let p2 = |x1: usize, u: usize| 0.2 + 0.3 * x1 as f64 + 0.4 * u as f64;
    let p3 = |x2: usize| 0.25 + 0.5 * x2 as f64;
    let p4 =
        |x3: usize, u: usize, x1: usize| 0.15 + 0.35 * x3 as f64 + 0.3 * u as f64 + eps * x1 as f64;
    let mut law = [0.0; 16];
    for x1 in 0..2 {
        for x2 in 0..2 {
            for x3 in 0..2 {
                for x4 in 0..2 {
                    let mut mass = 0.0;
                    for u in 0..2 {
                        mass += b(px1, x1)
                            * b(PU, u)
                            * b(p2(x1, u), x2)
                            * b(p3(x2), x3)
                            * b(p4(x3, u, x1), x4);
                    }
                    law[x1 * 8 + x2 * 4 + x3 * 2 + x4] = mass;
                }
            }
        }
    }
    law
}

fn counts(n: f64) -> NestedMarkovInput {
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts {
            regime: Regime::Observational,
            levels: vec![2, 2, 2, 2],
            cells: scm_law(0.4, 0.0).iter().map(|p| (p * n).round()).collect(),
        }],
    }
}

#[test]
fn nested_fisher_candidate_and_measured_scalars_equal_the_compiled_family() {
    let input = counts(1_000_000.0);
    let ctx = ExecutionContext::for_tests(21);
    let compiled =
        Study::compile_nested_markov_fisher(input.clone(), FitOptions::default(), 0.95, &ctx)
            .unwrap();
    assert_bound(
        &compiled.plan_view(),
        "NestedMarkovFisherInterval",
        "identify.binary_nested_markov_fisher",
    );

    let direct = NestedFisherArtifact::build(&input, &FitOptions::default(), 0.95, &ctx).unwrap();
    let twin = compiled.fit(&ctx).unwrap();
    assert_eq!(direct.export().unwrap(), twin.export().unwrap());
    assert_eq!(bits(&direct.parameter_covariance), bits(&twin.parameter_covariance));
    for (d, c) in direct.interval_candidates.iter().zip(&twin.interval_candidates) {
        assert_eq!(bits(d), bits(c));
    }

    let (direct_artifact, direct) =
        NestedFisherArtifact::build_measured(&input, &FitOptions::default(), 0.95, &ctx).unwrap();
    let (twin_artifact, twin) = compiled.measure(&ctx).unwrap();
    assert_eq!(direct_artifact.export().unwrap(), twin_artifact.export().unwrap());
    assert_same_measured(&direct, &twin);
}

#[test]
fn nested_posterior_candidate_equals_the_compiled_family() {
    let input = NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts {
            regime: Regime::Observational,
            levels: vec![2; 4],
            cells: vec![1.0; 16],
        }],
    };
    let sampler = Options { seed: 817, ..Options::default() };
    let ctx = ExecutionContext::for_tests(0);
    let compiled = Study::compile_nested_markov_posterior(
        input.clone(),
        FitOptions::default(),
        Prior::default(),
        sampler.clone(),
        &ctx,
    )
    .unwrap();
    assert_bound(&compiled.plan_view(), "NestedMarkovPosterior", "identify.binary_nested_markov");
    let direct = NestedPosteriorArtifact::build(
        &input,
        &FitOptions::default(),
        Prior::default(),
        sampler,
        &ctx,
    )
    .unwrap();
    let twin = compiled.fit(&ctx).unwrap();
    assert_eq!(direct.export().unwrap(), twin.export().unwrap());
    assert_eq!(direct.posterior, twin.posterior);
    assert_eq!(bits(&direct.posterior.mean), bits(&twin.posterior.mean));
}

#[test]
fn nested_posterior_measured_scalars_equal_the_compiled_family() {
    // The frozen sampler and a Beta(2, 2) prior: the measured protocol the Python producer runs.
    let input = counts(2_000.0);
    let prior = Prior { alpha: [2.0; 11], beta: [2.0; 11] };
    let sampler = Options {
        chains: 4,
        warmup: 2048,
        draws: 4096,
        max_proposals: 5_000_000,
        seed: 817,
        credible_mass: 0.95,
    };
    let ctx = ExecutionContext::for_tests(3);
    let compiled = Study::compile_nested_markov_posterior(
        input.clone(),
        FitOptions::default(),
        prior.clone(),
        sampler.clone(),
        &ctx,
    )
    .unwrap();
    let direct = NestedPosteriorArtifact::build_measured(
        &input,
        &FitOptions::default(),
        prior,
        sampler,
        &ctx,
    );
    let twin = compiled.measure(&ctx);
    match (direct, twin) {
        (Ok((direct_artifact, direct)), Ok((twin_artifact, twin))) => {
            assert_eq!(direct_artifact.export().unwrap(), twin_artifact.export().unwrap());
            assert_same_measured(&direct, &twin);
        }
        (Err(direct), Err(twin)) => assert_eq!(direct.to_string(), twin.to_string()),
        (direct, twin) => {
            panic!("the paths disagree: direct {:?}, compiled {:?}", direct.err(), twin.err())
        }
    }
}

// ---------------------------------------------------------------------------
// Joint source-target transport (fixtures of the joint transport artifact tests).
// ---------------------------------------------------------------------------

const FEATURE: u32 = 7;

fn source(id: &str, offset: usize, effect: f64) -> SourceData {
    let n = 40;
    let (mut x, mut a, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let k = (i + offset) as f64;
        let xi = 1.5 * (0.37 * k).sin() + 0.2;
        let treated = i % 2 == 0;
        let t = if treated { 1.0 } else { 0.0 };
        x.push(xi);
        a.push(treated);
        y.push(
            1.0 + 0.5 * xi
                + 0.4 * xi * xi
                + t * (effect + 1.5 * xi + 1.2 * xi * xi)
                + 0.3 * (1.3 * k).cos(),
        );
    }
    SourceData {
        id: id.into(),
        identity: DataIdentity {
            snapshot_digest: format!("snap-{id}"),
            datum_ids: (0..n).map(|i| format!("{id}-{i}")).collect(),
        },
        treatment: a,
        outcome: y,
        covariates: vec![x],
        noise_variance: 0.09,
    }
}

fn identified() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Standardize {
            over: Arc::from([VariableId::from_raw(FEATURE)]),
            source_response: PopulationFactor {
                population: Arc::from("source"),
                regime: None,
                variables: Arc::from([VariableId::from_raw(100)]),
                conditioned_on: Arc::from([VariableId::from_raw(FEATURE)]),
                interventions: Arc::from([VariableId::from_raw(101)]),
            },
            target_law: PopulationFactor {
                population: Arc::from("target"),
                regime: None,
                variables: Arc::from([VariableId::from_raw(FEATURE)]),
                conditioned_on: Arc::from([]),
                interventions: Arc::from([]),
            },
        },
        certificate: TransportCertificate {
            rule: Arc::from("transport.sid.standardize"),
            selection_targets: Arc::from([VariableId::from_raw(FEATURE)]),
            premises: Arc::from([Arc::from("standardizers are pre-treatment")]),
        },
    }
}

fn learned_model() -> LearnedJointModel {
    LearnedJointModel {
        graph: TransportGraphClass::FixedDag,
        features: vec![FEATURE],
        basis_degree: 2,
        varying: VaryingBlock::Intercept,
        sharing: SourceSharing::IndependentVaryingBlocks,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(5, 0.0, 4.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 4.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    }
}

fn gaussian_model() -> JointTransportModel {
    JointTransportModel {
        graph: TransportGraphClass::FixedDag,
        features: vec![FEATURE],
        varying: VaryingBlock::Intercept,
        sharing: SourceSharing::IndependentVaryingBlocks,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(3, 0.0, 4.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 4.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    }
}

fn target() -> TargetData {
    TargetData {
        identity: DataIdentity {
            snapshot_digest: "snap-target".into(),
            datum_ids: (0..30).map(|j| format!("t-{j}")).collect(),
        },
        rows: 30,
        covariates: vec![(0..30_i32).map(|j| 0.4 * (0.21 * f64::from(j)).cos() + 0.3).collect()],
    }
}

fn sources() -> Vec<SourceData> {
    vec![source("s1", 0, 2.0), source("s2", 400, 2.0)]
}

const JOINT_OPTIONS: JointTransportOptions = JointTransportOptions { draws: 3000, seed: 77 };

#[test]
fn learned_joint_transport_fit_and_measured_effect_equal_the_compiled_family() {
    let ctx = ExecutionContext::for_tests(1);
    let compiled = Study::compile_learned_joint_transport(
        identified(),
        learned_model(),
        sources(),
        target(),
        JOINT_OPTIONS,
        &ctx,
    )
    .unwrap();
    assert_bound(&compiled.plan_view(), "LearnedJointTransport", "transport.learned_joint");

    let (direct, direct_fit) = LearnedJointArtifactWire::build(
        &identified(),
        &learned_model(),
        &sources(),
        &target(),
        JOINT_OPTIONS,
        &ctx,
    )
    .unwrap();
    let (twin, twin_fit) = compiled.fit(&ctx).unwrap();
    assert_eq!(direct.export().unwrap(), twin.export().unwrap());
    assert_eq!(direct_fit, twin_fit);

    let (direct, direct_fit, direct_measured) = LearnedJointArtifactWire::build_measured(
        &identified(),
        &learned_model(),
        &sources(),
        &target(),
        JOINT_OPTIONS,
        0.95,
        &ctx,
    )
    .unwrap();
    let (twin, twin_fit, twin_measured) = compiled.measure(0.95, &ctx).unwrap();
    assert_eq!(direct.export().unwrap(), twin.export().unwrap());
    assert_eq!(direct_fit, twin_fit);
    assert_same_measured(&direct_measured, &twin_measured);
}

#[test]
fn joint_bayesian_measured_effect_equals_the_compiled_family() {
    let ctx = ExecutionContext::for_tests(1);
    let compiled = Study::compile_joint_bayesian_transport(
        identified(),
        gaussian_model(),
        sources(),
        target(),
        JOINT_OPTIONS,
        &ctx,
    )
    .unwrap();
    assert_bound(&compiled.plan_view(), "JointBayesianTransport", "transport.joint_bayesian");
    let (direct, direct_fit, direct_measured) = JointBayesianArtifactWire::build_measured(
        &identified(),
        &gaussian_model(),
        &sources(),
        &target(),
        JOINT_OPTIONS,
        0.95,
        &ctx,
    )
    .unwrap();
    let (twin, twin_fit, twin_measured) = compiled.measure(0.95, &ctx).unwrap();
    assert_eq!(direct.export().unwrap(), twin.export().unwrap());
    assert_eq!(direct_fit, twin_fit);
    assert_same_measured(&direct_measured, &twin_measured);
}

// ---------------------------------------------------------------------------
// Shared-row scenario covariance (fixtures of the scenario covariance artifact tests).
// ---------------------------------------------------------------------------

const ROWS: [[i64; 3]; 6] = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]];

fn term(coefficient: f64, pattern: &[(&str, i64)]) -> ScoreTermWire {
    ScoreTermWire {
        coefficient,
        pattern: pattern.iter().map(|(c, v)| ((*c).to_string(), *v)).collect(),
    }
}

fn functional(name: &str) -> FunctionalWire {
    match name {
        "a" => FunctionalWire::LinearScore {
            terms: vec![term(1.0, &[("x", 1), ("y", 1)]), term(-1.0, &[("x", 0), ("y", 1)])],
        },
        "c" => FunctionalWire::LinearScore { terms: vec![term(1.0, &[("z", 1), ("y", 1)])] },
        "adjusted" => FunctionalWire::AdjustedContrast {
            treatment: "x".into(),
            outcome: "y".into(),
            adjustment: vec!["z".into()],
            treated: 1,
            control: 0,
            outcome_value: 1,
        },
        _ => FunctionalWire::AdjustedContrast {
            treatment: "x".into(),
            outcome: "y".into(),
            adjustment: Vec::new(),
            treated: 1,
            control: 0,
            outcome_value: 1,
        },
    }
}

fn covariance_spec(method: CovarianceMethodWire) -> CovarianceSpec {
    CovarianceSpec {
        table: RowTableWire {
            columns: vec!["z".into(), "x".into(), "y".into()],
            unit_ids: (0..ROWS.len()).map(|i| format!("u{i}")).collect(),
            rows: ROWS.iter().map(|r| r.to_vec()).collect(),
        },
        scenarios: ["a", "c", "adjusted", "crude"]
            .iter()
            .map(|name| CovarianceScenarioWire {
                id: (*name).into(),
                snapshot: "snapshot:rows".into(),
                dependence: "shared_rows".into(),
                unit_ids: None,
                functional: functional(name),
            })
            .collect(),
        method,
    }
}

fn assert_same_covariance(
    direct: &antecedent_estimate::scenario_covariance::ScenarioCovariance,
    twin: &antecedent_estimate::scenario_covariance::ScenarioCovariance,
) {
    assert_eq!(bits(&direct.means), bits(&twin.means));
    assert_eq!(bits(&direct.covariance), bits(&twin.covariance));
    assert_eq!(bits(&direct.entry_sampling_variance), bits(&twin.entry_sampling_variance));
    assert_eq!(direct.failed_mass.to_bits(), twin.failed_mass.to_bits());
    assert_eq!(direct.replicate_ids, twin.replicate_ids);
    assert_eq!(direct.replicate_digest, twin.replicate_digest);
    assert_eq!(direct.row_identity_digest, twin.row_identity_digest);
    assert_eq!(direct, twin);
}

#[test]
fn scenario_covariance_equals_the_compiled_family_under_both_methods() {
    let ctx = ExecutionContext::for_tests(1);
    for method in [
        CovarianceMethodWire::SharedRowBootstrap {
            replicates: 300,
            seed: 7,
            max_failure_fraction: 0.0,
        },
        CovarianceMethodWire::ExactEnumeration { max_compositions: 10_000, max_failure_mass: 0.0 },
    ] {
        let spec = covariance_spec(method);
        let compiled = Study::compile_scenario_covariance(spec.clone(), &ctx).unwrap();
        assert_bound(
            &compiled.plan_view(),
            "SharedScenarioCovariance",
            "transport.shared_scenario_covariance",
        );
        assert_same_covariance(&spec.compute(&ctx).unwrap(), &compiled.covariance(&ctx).unwrap());
        let limits = ScenarioCovarianceConsumeLimits::default();
        assert_same_covariance(
            &spec.compute_within(&ctx, &limits).unwrap(),
            &compiled.covariance_within(&ctx, &limits).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// External feedback equilibrium (fixtures of the external cyclic artifact tests).
// ---------------------------------------------------------------------------

const GAIN_SOURCE: &str = "SECRET_IMPLEMENTATION_MARKER gain * tanh(x)";

struct Gain {
    spec: ExternalMechanismSpec,
    content: Vec<u8>,
}

impl Gain {
    fn new() -> Arc<Self> {
        let content = GAIN_SOURCE.as_bytes().to_vec();
        let mut spec = ExternalMechanismSpec::for_content(
            "lab/gain",
            "1.0",
            &content,
            vec![1],
            vec!["gain".into()],
            DerivativeDeclaration::Analytic,
        );
        spec.input_lipschitz = Some(vec![0.9]);
        Arc::new(Self { spec, content })
    }
}

impl ExternalMechanismImplementation for Gain {
    fn spec(&self) -> ExternalMechanismSpec {
        self.spec.clone()
    }
    fn content(&self) -> &[u8] {
        &self.content
    }
    fn value(&self, inputs: &[f64], p: &[f64]) -> Result<f64, EquilibriumError> {
        Ok(p[0] * inputs[0].tanh())
    }
    fn input_gradient(
        &self,
        inputs: &[f64],
        p: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(Some(vec![p[0] * (1.0 - inputs[0].tanh().powi(2))]))
    }
    fn parameter_gradient(
        &self,
        inputs: &[f64],
        _p: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(Some(vec![inputs[0].tanh()]))
    }
}

fn external_request() -> ExternalCyclicRequest {
    let spec = Gain::new().spec.clone();
    let system = ExternalFeedbackSystem {
        equations: vec![
            SystemEquation::External {
                id: "response".into(),
                intercept: 0.0,
                parameters: vec![0.0],
                mechanism: spec,
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
    };
    let rows = [-2.0_f64, -1.0, 0.5, 1.0, 2.0]
        .into_iter()
        .flat_map(|x| {
            [-0.1, 0.1].map(|noise| RegimeRow {
                values: vec![0.7 * x.tanh() + noise, x],
                clamps: vec![Clamp { variable: 1, value: x }],
            })
        })
        .collect();
    ExternalCyclicRequest {
        system,
        fit: Some(ExternalFitRequest {
            parameters: vec![ExternalFitParameter::Parameter { equation: 0, parameter: 0 }],
            rows,
            policy: ExternalFitPolicy::default(),
        }),
        worlds: vec![vec![], vec![Clamp { variable: 1, value: 1.0 }]],
        noise: vec![vec![0.1, 0.2], vec![-0.1, -0.2]],
        initial: vec![0.0; 2],
        policy: SolvePolicy::default(),
        factual: Some(vec![0.3, 0.7]),
    }
}

#[test]
fn external_feedback_report_equals_the_compiled_family() {
    let ctx = ExecutionContext::for_tests(11);
    let mut registry = MechanismRegistry::new();
    registry.register(Gain::new()).unwrap();
    let resolver: Option<&dyn TrustedMechanismResolver> = Some(&registry);
    let request = external_request();
    let compiled = Study::compile_external_feedback(request.clone(), &ctx).unwrap();
    assert_bound(
        &compiled.plan_view(),
        "ExternalFeedbackEquilibrium",
        "cyclic.external_mechanisms",
    );
    let direct = request.execute(resolver, &ctx).unwrap();
    let twin = compiled.evaluate(resolver, &ctx).unwrap();
    assert_eq!(antecedent_io::to_cbor(&direct).unwrap(), antecedent_io::to_cbor(&twin).unwrap());
    assert_eq!(direct, twin);
    // Without a resolver both refuse before anything runs, with the same typed refusal.
    let direct = request.execute(None, &ctx).unwrap_err();
    let twin = compiled.evaluate(None, &ctx).unwrap_err();
    assert_eq!(direct.to_string(), twin.to_string());
}

// ---------------------------------------------------------------------------
// Sampled counterfactual transport (fixtures of the sampled artifact tests).
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn multinomial(probabilities: &[f64], n: u64, rng: &mut Rng) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    for p in probabilities {
        running += p;
        cumulative.push(running);
    }
    let last = probabilities.iter().rposition(|p| *p > 0.0).unwrap();
    let mut counts = vec![0u64; probabilities.len()];
    for _ in 0..n {
        let u = rng.unit();
        counts[cumulative.partition_point(|c| *c <= u).min(last)] += 1;
    }
    counts
}

fn table_of(law: &ExactDiscreteLaw, counts: Vec<u64>) -> SampledRegimeCounts {
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().unwrap()))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|a| SampledAxis {
                variable: a.variable,
                levels: a.values.iter().map(|v| v.as_f64().unwrap()).collect(),
            })
            .collect(),
        counts,
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: law
            .probabilities()
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == 0.0)
            .map(|(i, _)| i)
            .collect(),
    })
    .unwrap()
}

fn case_named(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).unwrap()
}

fn derive(case: &Case, label: &str) -> CounterfactualTransportDerivation {
    let target = case.targets.iter().find(|t| t.label == label).unwrap();
    let problem = adapter::problem(case);
    let catalog = adapter::catalog(case);
    let sources = adapter::sources(case, &adapter::evidence_controllable(case));
    let query = adapter::query(case, &target.query, sources);
    match identify_counterfactual_transport(
        &problem,
        &query,
        &catalog.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
    {
        CounterfactualTransportDecision::Identified(d) => *d,
        other => panic!("expected a derivation, got {other:?}"),
    }
}

fn sampled_request(
    case: &Case,
    derivation: &CounterfactualTransportDerivation,
    units: u64,
) -> SampledTransportRequest {
    let adapted = adapter::catalog(case);
    let data = adapter::laws(case, &adapted);
    let mut rng = Rng(11);
    let tables: Vec<SampledRegimeCounts> = data
        .laws()
        .iter()
        .filter(|law| {
            derivation
                .cited
                .iter()
                .any(|c| c.regime == law.regime() && c.population.as_ref() == law.population())
        })
        .map(|law| table_of(law, multinomial(law.probabilities(), units, &mut rng)))
        .collect();
    SampledTransportRequest {
        tables,
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: UnseenCellRule::Refuse,
        limits: SampledTransportLimits::default(),
    }
}

#[test]
fn sampled_counterfactual_transport_estimate_and_interval_equal_the_compiled_families() {
    let case = case_named("transportable_latent_district");
    let derivation = derive(&case, "Y_{x1}=1");
    let request = sampled_request(&case, &derivation, 1_000);
    let ctx = ExecutionContext::for_tests(5);

    let compiled = Study::compile_sampled_counterfactual_transport(
        derivation.clone(),
        request.clone(),
        Vec::new(),
        &ctx,
    )
    .unwrap();
    assert_bound(
        &compiled.plan_view(),
        "SampledCounterfactualTransport",
        "counterfactual.transport_sampled",
    );
    let (direct, direct_estimate) =
        sampled_counterfactual_transport_artifact(&derivation, &request, &[], &ctx).unwrap();
    let (twin, twin_estimate) = compiled.estimate(&ctx).unwrap();
    let bytes = direct.export().unwrap();
    assert_eq!(bytes, twin.export().unwrap());
    assert_eq!(format!("{direct_estimate:?}"), format!("{twin_estimate:?}"));

    let limits = SampledCounterfactualTransportConsumeLimits::default();
    let interval = Study::compile_sampled_counterfactual_transport_interval(
        &bytes,
        direct.identity(),
        limits,
        &ctx,
    )
    .unwrap();
    assert_bound(
        &interval.plan_view(),
        "SampledCounterfactualTransportInterval",
        "counterfactual.transport_sampled_interval",
    );
    let direct = SampledWire::measure(&bytes, &direct.identity(), limits, &ctx).unwrap();
    let twin = interval.measure(&ctx).unwrap();
    assert_same_measured(&direct, &twin);
    assert_eq!(direct.report().scalars[0].name, "target_probability");
}
