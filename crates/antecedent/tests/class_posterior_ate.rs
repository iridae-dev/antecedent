//! CPDAG/PAG graph-posterior AverageEffect: policy, mass, and weight basis.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use antecedent::{
    BayesianConfig, CellStatus, InferenceMode, RefuteSuite, SemanticApplicability,
    StructuralAggregationPolicy, StructuralWeightBasis, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec,
    IdentificationStatus, Intervention, ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    set_edge,
};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;
use antecedent_validate::{
    CustomEffectValidator, RefutationProblem, RefutationReport, ValidationError,
};

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn ate() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

fn known_truth_mixture_pin() -> serde_json::Value {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    expected["static_average_effect"].clone()
}

fn gaussian(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            sum += (state >> 11) as f64 / (1u64 << 53) as f64;
        }
        sum - 6.0
    }
}

fn class_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + z[i] + g();
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Balanced `(z, t)` design from `known_truth_mixtures`: `Y = 2T + 2Z ± 0.2`.
fn mixture_data(n: usize) -> TabularData {
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut confounder = Vec::with_capacity(n);
    for _ in 0..(n / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

fn two_completion_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

fn reverse_causal_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(d(1), d(0)).unwrap();
    g
}

fn oriented(from_z: bool) -> Cpdag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(0), d(1)).unwrap();
    if from_z {
        g.insert_directed(d(2), d(0)).unwrap();
        g.insert_directed(d(2), d(1)).unwrap();
    }
    Cpdag::from_dag(&g)
}

/// `t → y`, `z → y`: different mask from `oriented(false)`, same empty adjustment.
fn outcome_parent_cpdag() -> Cpdag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    Cpdag::from_dag(&g)
}

fn cpdag_posterior(weights: &[f64], graphs: &[Cpdag]) -> GraphPosterior {
    let n = graphs[0].node_count();
    let masks: Vec<u64> =
        graphs.iter().map(|graph| adjacency_mask_from_cpdag(graph).unwrap()).collect();
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("class_posterior_ate"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

fn pag_from_dag_edges(edges: &[(u32, u32)]) -> Pag {
    let mut g = Pag::with_variables(3);
    for &(from, to) in edges {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

fn pag_posterior(weights: &[f64], graphs: &[Pag]) -> GraphPosterior {
    let n = graphs[0].node_count();
    let mut masks = Vec::new();
    let mut marks = Vec::new();
    for graph in graphs {
        let (adj, mark) = adjacency_masks_from_pag(graph).unwrap();
        masks.push(adj);
        marks.push(mark);
    }
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("class_posterior_ate"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_masks(marks)
    .unwrap()
}

fn run(data: TabularData, gp: GraphPosterior, inference: InferenceMode) -> antecedent::StudyResult {
    run_with(data, gp, inference, RefuteSuite::None, &[])
}

fn run_with(
    data: TabularData,
    gp: GraphPosterior,
    inference: InferenceMode,
    refute: RefuteSuite,
    validators: &[Arc<dyn CustomEffectValidator>],
) -> antecedent::StudyResult {
    let ctx = ExecutionContext::for_tests(1);
    Study::tabular(data)
        .graph_posterior(gp)
        .query(ate())
        .refute(refute)
        .inference(inference)
        .custom_validators(validators.to_vec())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
}

struct FixedValidator {
    name: &'static str,
    pass: bool,
}

impl CustomEffectValidator for FixedValidator {
    fn name(&self) -> &str {
        self.name
    }

    fn validate(
        &self,
        problem: &RefutationProblem<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        Ok(RefutationReport::new(
            self.name,
            problem.original.ate,
            problem.original.ate,
            if self.pass { 1.0 } else { 0.0 },
            true,
            self.pass,
            (!self.pass).then(|| Arc::from("forced fail")),
            0,
        ))
    }
}

struct FailNth {
    seen: AtomicU32,
    n: u32,
}

impl CustomEffectValidator for FailNth {
    fn name(&self) -> &str {
        "fail.nth"
    }

    fn validate(
        &self,
        problem: &RefutationProblem<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        let pass = self.seen.fetch_add(1, Ordering::Relaxed) + 1 != self.n;
        Ok(RefutationReport::new(
            "fail.nth",
            problem.original.ate,
            problem.original.ate,
            if pass { 1.0 } else { 0.0 },
            true,
            pass,
            (!pass).then(|| Arc::from("nth completion forced fail")),
            0,
        ))
    }
}

fn has_policy(result: &antecedent::StudyResult, policy: StructuralAggregationPolicy) -> bool {
    result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
            && d.message.contains(policy.as_str())
    })
}

#[test]
fn class_posterior_cpdag_and_pag_average_effect_none_runs() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let weights = [0.8, 0.2];
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        let cpdag = run(
            data.clone(),
            cpdag_posterior(&weights, &[two_completion_cpdag(), reverse_causal_cpdag()]),
            inference.clone(),
        );
        let mixture = cpdag.structural_response.as_ref().expect("class posterior mixture");
        assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
        assert!(
            (mixture.unidentified_mass - 0.2).abs() < 1e-12,
            "unidentified posterior mass must be retained: {}",
            mixture.unidentified_mass
        );
        assert_eq!(mixture.atoms.len(), 2);
        let identified_w =
            mixture.atoms.iter().find(|atom| atom.value.is_some()).map(|atom| atom.weight).unwrap();
        let unidentified_w =
            mixture.atoms.iter().find(|atom| atom.value.is_none()).map(|atom| atom.weight).unwrap();
        assert!(
            (identified_w - 0.8).abs() < 1e-12,
            "outer atoms keep posterior weights, not completion thirds"
        );
        assert!((unidentified_w - 0.2).abs() < 1e-12);
        assert_eq!(cpdag.identification.status, IdentificationStatus::GraphDependent);
        assert!(cpdag.diagnostics.iter().any(|d| {
            d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                && d.message.contains("completion enumeration is not posterior probability")
        }));

        let pag = run(
            data.clone(),
            pag_posterior(
                &weights,
                &[pag_from_dag_edges(&[(2, 0), (0, 1)]), pag_from_dag_edges(&[(1, 0)])],
            ),
            inference,
        );
        let pag_mix = pag.structural_response.as_ref().expect("pag class posterior mixture");
        assert_eq!(pag_mix.weight_basis, StructuralWeightBasis::PosteriorProbability);
        assert!((pag_mix.unidentified_mass - 0.2).abs() < 1e-12);
        assert_eq!(pag.identification.status, IdentificationStatus::GraphDependent);
    }
}

#[test]
fn class_posterior_disagreeing_estimands_are_not_a_scalar_mixture() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let fake_mixture = pin["expected_effect_given_identified"].as_f64().unwrap();
    let data = class_data(n, 3);
    let gp = cpdag_posterior(&[0.5, 0.5], &[oriented(false), oriented(true)]);
    let result = run(data, gp, InferenceMode::Frequentist);
    assert!(
        has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms),
        "disagreeing adjustment sets must not be treated as one estimand"
    );
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    assert!(
        !result.estimate.ate.is_finite() || (result.estimate.ate - fake_mixture).abs() > 0.2,
        "disagreeing estimands must not publish the DAG-mixture scalar {fake_mixture}, got {}",
        result.estimate.ate
    );
    let mixture = result.structural_response.as_ref().unwrap();
    assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
    assert!(mixture.conditional_on_identified.is_none());
    assert!(
        mixture.identified_set.is_some(),
        "GraphDependent atoms still publish the identified set"
    );
}

#[test]
fn class_graph_posterior_inspect_and_execute_agree_on_support_status() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let weights = [0.8, 0.2];
    let ctx = ExecutionContext::for_tests(1);
    let posteriors = [
        ("Cpdag", cpdag_posterior(&weights, &[two_completion_cpdag(), reverse_causal_cpdag()])),
        (
            "Pag",
            pag_posterior(
                &weights,
                &[pag_from_dag_edges(&[(2, 0), (0, 1)]), pag_from_dag_edges(&[(1, 0)])],
            ),
        ),
    ];

    for (label, gp) in posteriors {
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
        ] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let builder = Study::tabular(data.clone())
                    .graph_posterior(gp.clone())
                    .query(ate())
                    .refute(suite)
                    .inference(inference.clone())
                    .bootstrap_replicates(0);
                let inspected = builder.clone().inspect().unwrap();
                assert_eq!(
                    inspected.support_status,
                    Some(CellStatus::Licensed),
                    "{label} {inference:?} {suite:?} inspect must publish licensed"
                );
                let preflight = builder.clone().capability().unwrap();
                assert_eq!(
                    preflight.applicability,
                    SemanticApplicability::Licensed,
                    "{label} {inference:?} {suite:?} capability must agree with inspect"
                );
                let built = builder.build().unwrap();
                assert_eq!(
                    built.inspect().unwrap().support_status,
                    inspected.support_status,
                    "{label} built Study::inspect must reuse the same support_status"
                );
                let result = built.run(&ctx).unwrap();
                assert_eq!(
                    result.support_status, inspected.support_status,
                    "{label} {inference:?} {suite:?} execute must match cheap inspect"
                );
                assert_eq!(
                    result.identification.status,
                    IdentificationStatus::GraphDependent,
                    "{label} {inference:?} {suite:?} must retain unidentified posterior mass"
                );
            }
        }
    }
}

#[test]
fn class_posterior_cpdag_and_pag_average_effect_cheap_and_full_run() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let weights = [0.8, 0.2];
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let cpdag = run_with(
                data.clone(),
                cpdag_posterior(&weights, &[two_completion_cpdag(), reverse_causal_cpdag()]),
                inference.clone(),
                suite,
                &[],
            );
            assert!(
                !cpdag.refutations.is_empty(),
                "{inference:?} {suite:?} CPDAG must run per-atom refuters"
            );
            assert!(
                cpdag.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "refute.envelope.class_posterior"
                        && d.message.contains("completion")
                        && d.message.contains("posterior")
                }),
                "{inference:?} {suite:?} CPDAG must name inner completion vs outer posterior mixing"
            );
            let mixture = cpdag.structural_response.as_ref().expect("class posterior mixture");
            assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12);

            let pag = run_with(
                data.clone(),
                pag_posterior(
                    &weights,
                    &[pag_from_dag_edges(&[(2, 0), (0, 1)]), pag_from_dag_edges(&[(1, 0)])],
                ),
                inference.clone(),
                suite,
                &[],
            );
            assert!(
                !pag.refutations.is_empty(),
                "{inference:?} {suite:?} PAG must run per-atom refuters"
            );
            assert!(
                pag.diagnostics
                    .iter()
                    .any(|d| { d.code.as_ref() == "refute.envelope.class_posterior" })
            );
            let pag_mix = pag.structural_response.as_ref().expect("pag class posterior mixture");
            assert!((pag_mix.unidentified_mass - 0.2).abs() < 1e-12);
        }
    }
}

#[test]
fn class_posterior_mixed_refutation_pass_and_fail() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let gp = cpdag_posterior(&[0.6, 0.4], &[oriented(false), outcome_parent_cpdag()]);

    let passed = run_with(
        data.clone(),
        gp.clone(),
        InferenceMode::Frequentist,
        RefuteSuite::None,
        &[Arc::new(FixedValidator { name: "forced.pass", pass: true })],
    );
    assert!(
        has_policy(&passed, StructuralAggregationPolicy::SameEstimandWeightedMean),
        "same empty adjustment must mix by posterior weight"
    );
    let pass_report = passed
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == "forced.pass")
        .expect("mixed pass report");
    assert!(pass_report.passed, "every contributing atom passed");
    assert!(passed.diagnostics.iter().any(|d| {
        d.code.as_ref() == "refute.envelope.class_posterior"
            && d.message.contains("pass only if every contributing atom passes")
    }));

    let failed = run_with(
        data,
        gp,
        InferenceMode::Frequentist,
        RefuteSuite::None,
        &[Arc::new(FailNth { seen: AtomicU32::new(0), n: 2 })],
    );
    let fail_report = failed
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == "fail.nth")
        .expect("mixed fail report");
    assert!(!fail_report.passed, "one failing completion must fail-close the mixed check");
}

#[test]
fn class_posterior_disagreeing_estimands_skip_scalar_mixture_refuter() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let fake_mixture = pin["expected_effect_given_identified"].as_f64().unwrap();
    let data = class_data(n, 3);
    let gp = cpdag_posterior(&[0.5, 0.5], &[oriented(false), oriented(true)]);
    let result = run_with(data, gp, InferenceMode::Frequentist, RefuteSuite::Cheap, &[]);
    assert!(has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms));
    assert!(
        result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "refute.envelope.class_posterior"
                && d.message.contains("outer scalar mix skipped")
                && d.message.contains("pooled mixture ATE")
        }),
        "disagreeing estimands must not invent a scalar mixture refuter"
    );
    assert!(
        !result.estimate.ate.is_finite() || (result.estimate.ate - fake_mixture).abs() > 0.2,
        "disagreeing estimands must not publish the DAG-mixture scalar {fake_mixture}"
    );
    for report in &result.refutations {
        assert!(
            !report.original_ate.is_finite() || (report.original_ate - fake_mixture).abs() > 0.2,
            "refuter {} must not compare against the pooled mixture ATE {fake_mixture}, got {}",
            report.refuter,
            report.original_ate
        );
    }
}

#[test]
fn class_posterior_known_truth_mixture_pin() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let atom_effects: Vec<f64> = pin["identified_atom_effects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    let data = mixture_data(n);
    let cpdag =
        cpdag_posterior(&weights, &[oriented(false), oriented(true), reverse_causal_cpdag()]);
    let result = run(data.clone(), cpdag, InferenceMode::Frequentist);
    assert!(
        has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms),
        "Cpdag: disagreeing class atoms must not scalar-mix the fixture"
    );
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    assert!(
        !result.estimate.ate.is_finite(),
        "Cpdag: scalar ate must stay withheld when estimands disagree"
    );
    let mixture = result.structural_response.as_ref().expect("Cpdag: mixture");
    assert!((mixture.unidentified_mass - unidentified_truth).abs() < 1e-12);
    assert!(mixture.conditional_on_identified.is_none());
    let identified_set = mixture.identified_set.as_ref().expect("Cpdag: identified set");
    assert!(
        (identified_set.lower[0] - atom_effects[1]).abs() < tolerance
            && (identified_set.upper[0] - atom_effects[0]).abs() < tolerance,
        "Cpdag: identified set [{}, {}] truth [{}, {}]",
        identified_set.lower[0],
        identified_set.upper[0],
        atom_effects[1],
        atom_effects[0]
    );
    let identified_values: Vec<f64> = mixture
        .atoms
        .iter()
        .filter_map(|atom| match atom.value.as_ref()? {
            antecedent_core::ResponseValue::Scalar(v) => Some(*v),
            _ => None,
        })
        .collect();
    for truth in &atom_effects {
        assert!(
            identified_values.iter().any(|v| (v - truth).abs() < tolerance),
            "Cpdag: missing atom effect {truth}; got {identified_values:?}"
        );
    }
}

#[test]
fn class_posterior_same_estimand_two_atom_mixture_pin() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    let direct_effect = pin["identified_atom_effects"][0].as_f64().unwrap();
    let data = mixture_data(n);
    let weights = [0.6, 0.4];
    let cpdag = cpdag_posterior(&weights, &[oriented(false), outcome_parent_cpdag()]);
    let result = run(data, cpdag, InferenceMode::Frequentist);
    assert!(
        has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean),
        "Cpdag: same empty adjustment must scalar-mix"
    );
    assert!(
        (result.estimate.ate - direct_effect).abs() < tolerance,
        "Cpdag: mixture mean={} truth={direct_effect}",
        result.estimate.ate
    );
    let mixture = result.structural_response.as_ref().expect("Cpdag: mixture");
    let conditional = match mixture.conditional_on_identified.as_ref() {
        Some(antecedent_core::ResponseValue::Scalar(value)) => *value,
        _ => panic!("Cpdag: conditional_on_identified"),
    };
    assert!((conditional - direct_effect).abs() < tolerance);
    assert!(
        result.estimate.se_analytic.is_finite(),
        "Cpdag: two-atom same-estimand mix must publish a joint IF SE"
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.graph_posterior.joint_if_se"),
        "Cpdag: joint IF diagnostic"
    );
}

#[test]
fn class_posterior_frequentist_numeric_pins() {
    // Licensed evidence entry covering both the disagreeing-estimand fixture
    // (identified set + retained unidentified mass) and the same-estimand
    // two-atom mixture scalar + joint IF SE pin.
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let atom_effects: Vec<f64> = pin["identified_atom_effects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    let direct_effect = atom_effects[0];
    let data = mixture_data(n);
    let disagree =
        cpdag_posterior(&weights, &[oriented(false), oriented(true), reverse_causal_cpdag()]);
    let disagree_result = run(data.clone(), disagree, InferenceMode::Frequentist);
    assert!(has_policy(&disagree_result, StructuralAggregationPolicy::GraphDependentAtoms));
    assert!(!disagree_result.estimate.ate.is_finite());
    let disagree_mix = disagree_result.structural_response.as_ref().unwrap();
    assert!((disagree_mix.unidentified_mass - unidentified_truth).abs() < 1e-12);
    let same = cpdag_posterior(&[0.6, 0.4], &[oriented(false), outcome_parent_cpdag()]);
    let same_result = run(data, same, InferenceMode::Frequentist);
    assert!(has_policy(&same_result, StructuralAggregationPolicy::SameEstimandWeightedMean));
    assert!((same_result.estimate.ate - direct_effect).abs() < tolerance);
    assert!(same_result.estimate.se_analytic.is_finite());
}

#[test]
fn two_completion_cpdag_mask_identifies() {
    let cpdag = two_completion_cpdag();
    let mask = adjacency_mask_from_cpdag(&cpdag).unwrap();
    let back = antecedent_discovery::cpdag_from_adjacency_mask(mask, 3).unwrap();
    assert_eq!(back.undirected_edge_count(), 1);
    let envelope = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_cpdag_envelope(&back, &ate());
    assert!(envelope.is_ok(), "{envelope:?}");
    let envelope = envelope.unwrap();
    assert!(
        envelope.identified_weight.0 > 0.0,
        "status={:?} identified={} cases={}",
        envelope.status,
        envelope.identified_weight.0,
        envelope.cases.len()
    );
    assert!(
        envelope.cases.iter().any(|case| !case.result.estimands.is_empty()),
        "identified completions must carry an estimand"
    );
}

#[test]
fn identified_pag_mask_roundtrips() {
    // T→Y is visible only when a parent of T is nonadjacent to Y (Z→T, T→Y).
    let pag = pag_from_dag_edges(&[(2, 0), (0, 1)]);
    let original = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(&pag, &ate());
    assert!(original.as_ref().is_ok_and(|e| e.identified_weight.0 > 0.0), "original {original:?}");
    let (mask, mark) = adjacency_masks_from_pag(&pag).unwrap();
    let back = antecedent_discovery::pag_from_adjacency_mask(mask, mark, 3).unwrap();
    assert!(back.has_edge(d(2), d(0)));
    assert!(back.has_edge(d(0), d(1)));
    let envelope = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(&back, &ate());
    assert!(envelope.is_ok(), "{envelope:?}");
    let envelope = envelope.unwrap();
    assert!(
        envelope.identified_weight.0 > 0.0,
        "status={:?} identified={}",
        envelope.status,
        envelope.identified_weight.0
    );
}

#[test]
fn class_posterior_graph_posterior_conditional_effect_runs() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let weights = [0.8, 0.2];
    let query = antecedent_core::ConditionalEffectQuery::try_new(
        ate().with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    for (label, gp) in [
        ("Cpdag", cpdag_posterior(&weights, &[two_completion_cpdag(), reverse_causal_cpdag()])),
        (
            "Pag",
            pag_posterior(
                &weights,
                &[pag_from_dag_edges(&[(2, 0), (0, 1)]), pag_from_dag_edges(&[(1, 0)])],
            ),
        ),
    ] {
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
        ] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let result = Study::tabular(data.clone())
                    .graph_posterior(gp.clone())
                    .query(CausalQuery::ConditionalEffect(query.clone()))
                    .refute(suite)
                    .inference(inference.clone())
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(1))
                    .unwrap();
                assert!(
                    has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms)
                        || result.structural_response.is_some(),
                    "{label} {inference:?} {suite:?}: class conditional graph-posterior must publish structural aggregation"
                );
                assert!(
                    result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                    }),
                    "{label} {inference:?} {suite:?}: structural_aggregation diagnostic"
                );
                match suite {
                    RefuteSuite::None => assert!(
                        result.refutations.is_empty(),
                        "{label} {inference:?}: none must not run atom refuters"
                    ),
                    RefuteSuite::Cheap | RefuteSuite::Full => {
                        assert!(
                            result.diagnostics.iter().any(|d| {
                                d.code.as_ref() == "refute.envelope.class_posterior"
                            }),
                            "{label} {inference:?} {suite:?}: per-atom class-posterior refuters"
                        );
                        assert!(
                            !result.refutations.is_empty(),
                            "{label} {inference:?} {suite:?}: per-atom refutations must run"
                        );
                        let mixed = result.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "refute.envelope.class_posterior"
                                && d.message.contains("pass only if every contributing atom passes")
                        });
                        if has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean)
                        {
                            assert!(
                                mixed,
                                "{label} {inference:?} {suite:?}: SameEstimandWeightedMean mixes atom reports"
                            );
                        }
                    }
                    RefuteSuite::PlaceboAndRcc => {}
                }
            }
        }
    }
}

#[test]
fn class_posterior_graph_posterior_response_runs() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let weights = [0.8, 0.2];
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(t, GridSpec::Linspace { start: 0.0, end: 1.0, points: 5 }),
    });
    let level = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(t, Value::f64(0.5))]),
    });
    for (label, gp) in [
        ("Cpdag", cpdag_posterior(&weights, &[two_completion_cpdag(), reverse_causal_cpdag()])),
        (
            "Pag",
            pag_posterior(
                &weights,
                &[pag_from_dag_edges(&[(2, 0), (0, 1)]), pag_from_dag_edges(&[(1, 0)])],
            ),
        ),
    ] {
        for (query_label, query) in
            [("ResponseCurve", curve.clone()), ("InterventionResponse", level.clone())]
        {
            for inference in [
                InferenceMode::Frequentist,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
            ] {
                let result = Study::tabular(data.clone())
                    .graph_posterior(gp.clone())
                    .query(CausalQuery::Response(query.clone()))
                    .refute(RefuteSuite::None)
                    .inference(inference.clone())
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(1))
                    .unwrap();
                let structural = result.structural_response.as_ref().expect("structural response");
                assert_eq!(structural.weight_basis, StructuralWeightBasis::PosteriorProbability);
                assert!(
                    (structural.unidentified_mass - 0.2).abs() < 1e-12,
                    "{label} {query_label} {inference:?}: unidentified mass retained"
                );
                assert!(
                    result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                    }),
                    "{label} {query_label} {inference:?}: structural_aggregation diagnostic"
                );
                assert!(
                    result.response.is_some(),
                    "{label} {query_label} {inference:?}: response payload"
                );
            }
        }
    }
}

#[test]
fn class_posterior_graph_posterior_intervention_response_cheap_and_full_run() {
    let pin = known_truth_mixture_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = class_data(n, 17);
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let level = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(t, Value::f64(0.5))]),
    });
    for (label, gp) in [
        ("Cpdag", cpdag_posterior(&[1.0], &[two_completion_cpdag()])),
        ("Pag", pag_posterior(&[1.0], &[pag_from_dag_edges(&[(2, 0), (0, 1)])])),
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = Study::tabular(data.clone())
                .graph_posterior(gp.clone())
                .query(CausalQuery::Response(level.clone()))
                .refute(suite)
                .inference(InferenceMode::Frequentist)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(1))
                .unwrap();
            assert!(
                result.structural_response.is_some(),
                "{label} {suite:?}: class IR mixture"
            );
            assert!(
                !result.refutations.is_empty(),
                "{label} {suite:?} must run plugin-level refuters"
            );
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.evalue.not_a_contrast"),
                "{label} {suite:?}: not a contrast-shaped ATE suite"
            );
            assert!(
                result.refutations.iter().any(|r| r.refuter.contains("overlap")),
                "{label} {suite:?}: overlap on the intervention level"
            );
            if suite == RefuteSuite::Full {
                let stability_report = result.refutations.iter().any(|r| {
                    r.refuter.contains("bootstrap")
                        || r.refuter.contains("data_subset")
                        || r.refuter.contains("graph")
                });
                let stability_requested = result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "refute.validator.not_applicable"
                        && d.fields.iter().any(|(k, v)| {
                            k.as_ref() == "validator"
                                && matches!(v.as_ref(), "bootstrap" | "data_subset" | "graph")
                        })
                });
                assert!(
                    stability_report || stability_requested,
                    "{label} {suite:?}: full must request sampling-stability"
                );
            }
        }
    }
}

#[test]
fn class_posterior_atom_kind_defaults_to_dag_masks() {
    // A DAG-shaped posterior must not silently take the class path.
    let direct = set_edge(0, 3, 0, 1, true);
    let gp = GraphPosterior::new(
        3,
        vec![1.0],
        vec![direct],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0,
        InferenceDiagnostics::analytic("dag_default"),
        0,
    )
    .unwrap();
    assert_eq!(gp.atom_kind, GraphPosteriorAtomKind::Dag);
}

#[test]
fn class_posterior_does_not_publish_largest_atom_posterior_as_aggregate() {
    let gp = cpdag_posterior(&[0.6, 0.4], &[oriented(false), outcome_parent_cpdag()]);
    let result = run(
        mixture_data(320),
        gp,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    );
    assert!(has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean));
    assert!(result.estimate.ate.is_finite());
    assert!(result.posterior.is_none());
    let mixture = result.structural_response.as_ref().unwrap();
    assert_eq!(mixture.atoms.iter().filter(|a| a.posterior.is_some()).count(), 2);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.graph_posterior.posterior_withheld")
    );
}
