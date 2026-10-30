//! Intrinsic causal influence against closed forms.
//!
//! Linear structural models with independent noise have `Var(E[Y | U_S]) = Σ_{i∈S} c_i² σ_i²`,
//! where `c_i` is the total path coefficient from source `i`'s noise to `Y` and `σ_i` its noise
//! scale, so the Shapley value of source `i` is `c_i² σ_i²` and the contributions sum to
//! `Var(Y)`. A product mechanism `y = a·b + e` gives a game with interaction, whose Shapley
//! values are `Var(a·b)/2` for each factor and `σ_e²` for the noise.
//!
//! # Tolerances
//!
//! The nested estimator shares its random numbers across coalitions, so `v̂(S)` is close to
//! `Σ_{j∈S} ŝ_j`, where `ŝ_j` is the sample variance of `m·B` independent terms of variance
//! `c_j² σ_j²`. The standard error of the contribution of source `j` is therefore close to
//! `sqrt(2 / (B (m − 1))) · c_j² σ_j²`; the tests compare with six times the larger of this
//! derived value and the standard error the estimator reports, which keeps the check tied to
//! the estimator's own uncertainty without letting a noisy standard error from few replicate
//! batches loosen or tighten it by chance.

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "degrees of freedom and counts are exact small integers; column indices are below 2"
)]

use std::sync::Arc;

use antecedent_attribution::{
    AttributionError, ComponentStatus, FitResampling, IntrinsicAllocation, IntrinsicInfluenceQuery,
    IntrinsicInfluenceResult, NoiseIndependence, UncertaintyComponent, intrinsic_influence,
};
use antecedent_core::{
    CachePolicy, CausalSchemaBuilder, ComponentId, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_model::{
    BasisTerm, CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, ParentBasis, SelectionPolicy, sample_observational,
};

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn var(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

fn model(edges: &[(u32, u32)], slots: Vec<MechanismSlot>) -> CompiledCausalModel {
    let mut graph = Dag::with_variables(u32::try_from(slots.len()).unwrap());
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    CompiledCausalModel::compile(graph)
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

/// `a = 1 + e_a`, `b = 0.5 + 2a + e_b`, `c = -1 + 3b + e_c`, noise scales 1, 0.5, 2.
/// Total coefficients of `(e_a, e_b, e_c)` on `c` are `(6, 3, 1)`.
fn chain() -> (CompiledCausalModel, [f64; 3]) {
    let m = model(
        &[(0, 1), (1, 2)],
        vec![linear(1.0, &[], 1.0), linear(0.5, &[2.0], 0.5), linear(-1.0, &[3.0], 2.0)],
    );
    (m, [36.0, 2.25, 4.0])
}

/// `x1, x2 -> y` with equal coefficients and noise scales: `x1` and `x2` are exchangeable.
fn collider() -> (CompiledCausalModel, [f64; 3]) {
    let m = model(
        &[(0, 2), (1, 2)],
        vec![linear(0.0, &[], 1.0), linear(0.0, &[], 1.0), linear(0.0, &[1.5, 1.5], 0.7)],
    );
    (m, [2.25, 2.25, 0.49])
}

/// `y = a·b + e` with standard normal roots and `σ_e = 0.5`: the product has variance 1.
fn product() -> CompiledCausalModel {
    let basis = ParentBasis::new(
        2,
        Arc::from(vec![0.0, 0.0]),
        Arc::from(vec![1.0, 1.0]),
        Arc::from(vec![
            Arc::<[f64]>::from(Vec::<f64>::new()),
            Arc::<[f64]>::from(Vec::<f64>::new()),
        ]),
        Arc::from(vec![
            BasisTerm::Power { parent: 0, degree: 1 },
            BasisTerm::Power { parent: 1, degree: 1 },
            BasisTerm::Product { left: 0, right: 1 },
        ]),
    )
    .unwrap();
    model(
        &[(0, 2), (1, 2)],
        vec![
            linear(0.0, &[], 1.0),
            linear(0.0, &[], 1.0),
            MechanismSlot::LinearBasis {
                intercept: 0.0,
                basis,
                coeffs: Arc::from(vec![0.0, 0.0, 1.0]),
                sigma: 0.5,
            },
        ],
    )
}

/// `y = a^2 + b + e` with standard normal roots and `σ_e = 0.5`: additive in the noise terms, so
/// the contributions are `Var(a^2) = 2`, `Var(b) = 1` and `σ_e^2 = 0.25`.
fn quadratic() -> CompiledCausalModel {
    let edges = [(0, 2), (1, 2)];
    let basis = ParentBasis::new(
        2,
        Arc::from(vec![0.0, 0.0]),
        Arc::from(vec![1.0, 1.0]),
        Arc::from(vec![
            Arc::<[f64]>::from(Vec::<f64>::new()),
            Arc::<[f64]>::from(Vec::<f64>::new()),
        ]),
        Arc::from(vec![
            BasisTerm::Power { parent: 0, degree: 2 },
            BasisTerm::Power { parent: 1, degree: 1 },
        ]),
    )
    .unwrap();
    model(
        &edges,
        vec![
            linear(0.0, &[], 1.0),
            linear(0.0, &[], 1.0),
            MechanismSlot::LinearBasis {
                intercept: 0.0,
                basis,
                coeffs: Arc::from(vec![1.0, 1.0]),
                sigma: 0.5,
            },
        ],
    )
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn run(
    m: &CompiledCausalModel,
    target: u32,
    query: &IntrinsicInfluenceQuery,
) -> Result<IntrinsicInfluenceResult, AttributionError> {
    let query = IntrinsicInfluenceQuery { target: var(target), ..query.clone() };
    intrinsic_influence(m, NoiseIndependence::Declared, None, &query, &ctx())
}

fn query() -> IntrinsicInfluenceQuery {
    IntrinsicInfluenceQuery::new(var(0)).with_simulation(400, 20, 4).with_seed(5)
}

/// Six times the larger of the derived and the reported standard error of a contribution.
fn tolerance(truth: f64, outer: usize, replicates: usize, reported: f64) -> f64 {
    let derived = (2.0 / (replicates as f64 * (outer as f64 - 1.0))).sqrt() * truth;
    6.0 * derived.max(reported)
}

fn reported_se(result: &IntrinsicInfluenceResult, i: usize) -> f64 {
    result.sources[i].quantified_stderr().stderr
}

fn assert_matches_truth(result: &IntrinsicInfluenceResult, truth: &[f64], outer: usize, b: usize) {
    assert_eq!(result.sources.len(), truth.len());
    for (i, &t) in truth.iter().enumerate() {
        let got = result.sources[i].contribution;
        let tol = tolerance(t, outer, b, reported_se(result, i));
        assert!((got - t).abs() <= tol, "source {i}: got {got}, truth {t}, tol {tol}");
    }
}

#[test]
fn a_linear_chain_recovers_squared_path_coefficients_times_noise_variance() {
    let (m, truth) = chain();
    let result = run(&m, 2, &query()).unwrap();
    assert_eq!(
        result.sources.iter().map(|s| s.source.variable()).collect::<Vec<_>>(),
        vec![var(0), var(1), var(2)]
    );
    assert_matches_truth(&result, &truth, 400, 4);
    let total: f64 = truth.iter().sum();
    let tol = tolerance(total, 400, 4, result.total_variance_stderr);
    assert!((result.total_variance - total).abs() <= tol);
}

#[test]
fn a_collider_recovers_each_parent_noise_share() {
    let (m, truth) = collider();
    let result = run(&m, 2, &query()).unwrap();
    assert_matches_truth(&result, &truth, 400, 4);
}

#[test]
fn only_ancestors_of_the_target_and_the_target_are_sources() {
    let (m, _) = chain();
    let result = run(&m, 1, &query()).unwrap();
    let sources: Vec<_> = result.sources.iter().map(|s| s.source.variable()).collect();
    assert_eq!(sources, vec![var(0), var(1)]);
    // Coefficients of (e_a, e_b) on b are (2, 1): 4·1 and 1·0.25.
    assert_matches_truth(&result, &[4.0, 0.25], 400, 4);
}

/// The efficiency gap `Var(Y) − Σφ` is estimation error: `v(∅) = 0` and `v(all) = Var(Y)`.
///
/// Its standard deviation is bounded by that of the direct variance estimate,
/// `V·sqrt(2 / (B (m k − 1)))`, plus that of `v̂(∅)`, about `(V / k)·sqrt(2 / (B (m − 1)))`.
#[test]
fn shapley_efficiency_holds_within_estimation_error() {
    let (m, truth) = chain();
    let q = query();
    let result = run(&m, 2, &q).unwrap();
    let v: f64 = truth.iter().sum();
    let (outer, inner, b) = (q.outer_draws as f64, q.inner_draws as f64, q.replicates as f64);
    let theory = v * (2.0 / (b * (outer * inner - 1.0))).sqrt()
        + v / inner * (2.0 / (b * (outer - 1.0))).sqrt();
    let tol = 6.0 * theory.max(result.allocation_residual_stderr);
    assert!(
        (result.contribution_sum() - result.total_variance).abs() <= tol,
        "sum {} vs total {}",
        result.contribution_sum(),
        result.total_variance
    );
    assert!(
        (result.total_variance - result.contribution_sum() - result.allocation_residual).abs()
            < 1e-9
    );
    assert!(result.allocation_residual.abs() <= tol);
}

#[test]
fn exchangeable_sources_receive_equal_contributions() {
    let (m, _) = collider();
    let result = run(&m, 2, &query()).unwrap();
    let (a, b) = (&result.sources[0], &result.sources[1]);
    let se = a.quantified_stderr().stderr.hypot(b.quantified_stderr().stderr);
    let derived = (2.0 / (4.0 * 399.0_f64)).sqrt() * 2.25 * 2.0_f64.sqrt();
    assert!(
        (a.contribution - b.contribution).abs() <= 6.0 * se.max(derived),
        "{} vs {}",
        a.contribution,
        b.contribution
    );
}

/// The naive variance of the inner means overstates `v(S)` by `(Var(Y) − v(S)) / k`, which
/// shrinks every contribution by `1 − 1/k` and opens an efficiency gap of `Var(Y)/k`. With two
/// inner draws that is a factor of two, which the corrected estimator must not show.
#[test]
fn the_inner_variance_correction_removes_the_nested_bias() {
    let m = product();
    let q = IntrinsicInfluenceQuery::new(var(2)).with_simulation(1500, 2, 4).with_seed(2);
    let result = run(&m, 2, &q).unwrap();
    // phi_a = phi_b = Var(a b) / 2 = 0.5 and phi_e = 0.25.
    let truth = [0.5, 0.5, 0.25];
    for (i, &t) in truth.iter().enumerate() {
        let got = result.sources[i].contribution;
        // v(S) combines terms of variance up to 1.25; a loose six-sigma band that excludes the
        // uncorrected value t/2.
        assert!((got - t).abs() <= 0.15, "source {i}: got {got}, truth {t}");
    }
    assert!(result.allocation_residual.abs() <= 0.15);
}

#[test]
fn exact_and_sampled_shapley_agree_on_a_game_with_interaction() {
    let m = product();
    let exact = run(
        &m,
        2,
        &IntrinsicInfluenceQuery::new(var(2))
            .with_simulation(600, 20, 4)
            .with_allocation(IntrinsicAllocation::Exact)
            .with_seed(3),
    )
    .unwrap();
    let sampled = run(
        &m,
        2,
        &IntrinsicInfluenceQuery::new(var(2))
            .with_simulation(600, 20, 4)
            .with_allocation(IntrinsicAllocation::Permutation { n_permutations: 600 })
            .with_seed(3),
    )
    .unwrap();
    for i in 0..3 {
        let se = exact.sources[i]
            .quantified_stderr()
            .stderr
            .hypot(sampled.sources[i].quantified_stderr().stderr);
        assert!(
            (exact.sources[i].contribution - sampled.sources[i].contribution).abs()
                <= 6.0 * se.max(0.02),
            "source {i}: exact {} sampled {}",
            exact.sources[i].contribution,
            sampled.sources[i].contribution
        );
    }
    assert_eq!(exact.budget.exact_coalitions, 8);
    assert_eq!(exact.budget.permutations, 0);
    assert_eq!(sampled.budget.permutations, 600);
}

#[test]
fn every_interval_names_its_component_and_unmeasured_ones_say_so() {
    let (m, _) = chain();
    let exact = run(&m, 2, &query()).unwrap();
    for s in exact.sources.iter() {
        assert_eq!(s.uncertainty.len(), 3);
        assert!(matches!(
            s.status(UncertaintyComponent::PermutationSampling),
            Some(ComponentStatus::NotApplicable { .. })
        ));
        assert!(matches!(
            s.status(UncertaintyComponent::Simulation),
            Some(ComponentStatus::Quantified { stderr, .. }) if stderr.is_finite()
        ));
        assert_eq!(
            s.status(UncertaintyComponent::MechanismFit),
            Some(&ComponentStatus::Unquantified { reason: "bootstrap_not_requested" })
        );
        let partial = s.quantified_stderr();
        assert_eq!(partial.excluded, vec![UncertaintyComponent::MechanismFit]);
        assert!(!partial.is_complete());
    }

    // A bootstrap requested without data is still unquantified, with its own reason.
    let sampled = run(
        &m,
        2,
        &query()
            .with_allocation(IntrinsicAllocation::Permutation { n_permutations: 50 })
            .with_fit_bootstrap(20),
    )
    .unwrap();
    for s in sampled.sources.iter() {
        assert!(matches!(
            s.status(UncertaintyComponent::PermutationSampling),
            Some(ComponentStatus::Quantified { stderr, df, .. }) if stderr.is_finite() && *df == 49.0
        ));
        assert_eq!(
            s.status(UncertaintyComponent::MechanismFit),
            Some(&ComponentStatus::Unquantified { reason: "no_data_supplied" })
        );
    }
    assert_eq!(sampled.budget.refits, 0);
}

fn chain_data(n: usize) -> (CompiledCausalModel, TabularData) {
    let truth = model(&[(0, 1)], vec![linear(0.0, &[], 1.0), linear(0.0, &[2.0], 1.0)]);
    let c = ctx();
    let mut rng = c.rng.stream_for(StreamDomain::Attribution, 99);
    let batch =
        sample_observational(&truth, n, &mut rng, &mut MechanismWorkspace::default(), &c).unwrap();
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(n);
    let cols = (0..2)
        .map(|j| {
            OwnedColumn::Float64(
                Float64Column::new(
                    var(j as u32),
                    Arc::from(batch.values[j * n..(j + 1) * n].to_vec()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let compiled = CompiledCausalModel::compile({
        let mut g = Dag::with_variables(2);
        g.insert_directed(node(0), node(1)).unwrap();
        g
    })
    .unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    (compiled.with_mechanisms(store), data)
}

/// The posterior over the mechanisms: Dirichlet-weighted refits with the families held at the
/// ones selected on the data. It centres on the same contributions as the row bootstrap, has a
/// spread of the same order, and returns its refits as draws.
#[test]
fn the_posterior_over_the_mechanisms_returns_its_refits_beside_the_bootstrap() {
    let (m, data) = chain_data(300);
    let base = IntrinsicInfluenceQuery::new(var(1))
        .with_simulation(300, 20, 4)
        .with_fit_bootstrap(20)
        .with_seed(7);
    let bootstrap =
        intrinsic_influence(&m, NoiseIndependence::Declared, Some(&data), &base, &ctx()).unwrap();
    assert_eq!(bootstrap.method.fit_resampling, FitResampling::RowBootstrap);
    assert_eq!(bootstrap.fit_draws.len(), 20, "the refits of the bootstrap are returned too");
    let posterior_query = base.clone().with_fit_resampling(FitResampling::DirichletRowWeights);
    let posterior =
        intrinsic_influence(&m, NoiseIndependence::Declared, Some(&data), &posterior_query, &ctx())
            .unwrap();
    assert_eq!(posterior.method.fit_resampling, FitResampling::DirichletRowWeights);
    assert_eq!(posterior.budget.refits, 20);
    assert_eq!(posterior.fit_draws.len(), 20);
    assert!(posterior.fit_draws.iter().all(|draw| draw.len() == 2));
    for i in 0..2 {
        // The point answer does not depend on the resampling of the refits.
        assert_eq!(
            posterior.sources[i].contribution.to_bits(),
            bootstrap.sources[i].contribution.to_bits()
        );
        let mean = posterior.fit_draws.iter().map(|draw| draw[i]).sum::<f64>() / 20.0;
        let ComponentStatus::Quantified { stderr, df, .. } = &posterior.sources[i].whole_procedure
        else {
            panic!("whole procedure not quantified: {:?}", posterior.sources[i].whole_procedure);
        };
        assert!(stderr.is_finite() && *stderr > 0.0);
        assert_eq!(*df, 19.0);
        assert!(
            (mean - posterior.sources[i].contribution).abs() <= 6.0 * stderr.max(0.1),
            "source {i}: posterior mean {mean} against {}",
            posterior.sources[i].contribution
        );
        let ComponentStatus::Quantified { stderr: boot, .. } =
            &bootstrap.sources[i].whole_procedure
        else {
            panic!("bootstrap not quantified");
        };
        let ratio = stderr / boot;
        assert!((0.5..2.0).contains(&ratio), "source {i}: posterior over bootstrap spread {ratio}");
    }
}

#[test]
fn a_bootstrap_with_data_quantifies_mechanism_fit_separately() {
    let (m, data) = chain_data(300);
    let q = IntrinsicInfluenceQuery::new(var(1))
        .with_simulation(300, 20, 4)
        .with_fit_bootstrap(20)
        .with_seed(7);
    let result =
        intrinsic_influence(&m, NoiseIndependence::Declared, Some(&data), &q, &ctx()).unwrap();
    assert_eq!(result.budget.refits, 20);
    // y = 2x + e with unit noise: contributions 4 and 1 up to the fit's own error.
    for (i, truth) in [4.0, 1.0].into_iter().enumerate() {
        let s = &result.sources[i];
        let Some(ComponentStatus::Quantified { stderr, df, .. }) =
            s.status(UncertaintyComponent::MechanismFit)
        else {
            panic!("mechanism fit not quantified: {:?}", s.uncertainty);
        };
        assert!(stderr.is_finite() && *stderr > 0.0 && *stderr < 1.0);
        assert_eq!(*df, 19.0);
        assert!((s.contribution - truth).abs() <= 6.0 * stderr.max(0.2), "source {i}");
        assert!(s.quantified_stderr().is_complete());
        // The whole procedure keeps the Monte Carlo noise the mechanism-fit component removes,
        // so its spread is never smaller.
        let ComponentStatus::Quantified { stderr: whole, df: whole_df, ci_low, ci_high, .. } =
            &s.whole_procedure
        else {
            panic!("whole procedure not quantified: {:?}", s.whole_procedure);
        };
        assert!(whole.is_finite() && *whole >= *stderr);
        assert_eq!(*whole_df, 19.0);
        assert!(*ci_low <= s.contribution && s.contribution <= *ci_high);
    }
}

#[test]
fn without_a_bootstrap_the_whole_procedure_is_unquantified() {
    let (m, data) = chain_data(200);
    let q = IntrinsicInfluenceQuery::new(var(1)).with_simulation(100, 10, 3).with_seed(3);
    let result =
        intrinsic_influence(&m, NoiseIndependence::Declared, Some(&data), &q, &ctx()).unwrap();
    for source in result.sources.iter() {
        assert!(matches!(source.whole_procedure, ComponentStatus::Unquantified { .. }));
    }
}

#[test]
fn a_bidirected_edge_is_refused() {
    let (m, _) = chain();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(node(0), node(1)).unwrap();
    admg.insert_directed(node(1), node(2)).unwrap();
    admg.insert_bidirected(node(0), node(2)).unwrap();
    let err = intrinsic_influence(
        &m,
        NoiseIndependence::Structure(&admg),
        None,
        &IntrinsicInfluenceQuery::new(var(2)),
        &ctx(),
    )
    .unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("reason=route_not_supported: intrinsic_influence.correlated_noise"),
        "{text}"
    );
}

#[test]
fn undeclared_noise_a_mismatched_structure_and_unsupported_mechanisms_are_refused() {
    let (m, _) = chain();
    let q = IntrinsicInfluenceQuery::new(var(2));
    let text = intrinsic_influence(&m, NoiseIndependence::Undeclared, None, &q, &ctx())
        .unwrap_err()
        .to_string();
    assert!(text.contains("intrinsic_influence.noise_independence_undeclared"), "{text}");

    let mut other = Admg::with_variables(3);
    other.insert_directed(node(0), node(2)).unwrap();
    let text = intrinsic_influence(&m, NoiseIndependence::Structure(&other), None, &q, &ctx())
        .unwrap_err()
        .to_string();
    assert!(text.contains("reason=invalid_argument: intrinsic_influence.structure_mismatch"));

    let unfitted = CompiledCausalModel::compile({
        let mut g = Dag::with_variables(2);
        g.insert_directed(node(0), node(1)).unwrap();
        g
    })
    .unwrap();
    let text = intrinsic_influence(
        &unfitted,
        NoiseIndependence::Declared,
        None,
        &IntrinsicInfluenceQuery::new(var(1)),
        &ctx(),
    )
    .unwrap_err()
    .to_string();
    assert!(text.contains("intrinsic_influence.mechanism_unfitted"), "{text}");

    let state_space = model(
        &[],
        vec![MechanismSlot::LinearGaussianStateSpace {
            a: 0.5,
            process_std: 1.0,
            obs_std: 0.1,
            initial_mean: 0.0,
        }],
    );
    let text = intrinsic_influence(
        &state_space,
        NoiseIndependence::Declared,
        None,
        &IntrinsicInfluenceQuery::new(var(0)),
        &ctx(),
    )
    .unwrap_err()
    .to_string();
    assert!(text.contains("intrinsic_influence.noise_not_row_independent"), "{text}");

    let categorical = model(
        &[],
        vec![MechanismSlot::Discrete {
            support: Arc::from(vec![0.0, 1.0]),
            probs: Arc::from(vec![0.5, 0.5]),
            logit_coeffs: None,
        }],
    );
    let text = intrinsic_influence(
        &categorical,
        NoiseIndependence::Declared,
        None,
        &IntrinsicInfluenceQuery::new(var(0)),
        &ctx(),
    )
    .unwrap_err()
    .to_string();
    assert!(text.contains("intrinsic_influence.target_not_continuous"), "{text}");
}

#[test]
fn exact_allocation_beyond_the_simulation_budget_and_invalid_settings_are_refused() {
    // A chain of 26 nodes has 26 sources for its last node: 2^26 coalition values, each a
    // nested simulation, is far beyond the shared draw budget.
    let n = 26u32;
    let edges: Vec<(u32, u32)> = (1..n).map(|i| (i - 1, i)).collect();
    let slots = (0..n)
        .map(|i| if i == 0 { linear(0.0, &[], 1.0) } else { linear(0.0, &[0.5], 1.0) })
        .collect();
    let long = model(&edges, slots);
    let q = IntrinsicInfluenceQuery::new(var(n - 1)).with_allocation(IntrinsicAllocation::Exact);
    let text = intrinsic_influence(&long, NoiseIndependence::Declared, None, &q, &ctx())
        .unwrap_err()
        .to_string();
    assert!(text.contains("intrinsic_influence.draw_budget_exceeded"), "{text}");

    let (m, _) = chain();
    for bad in [
        query().with_simulation(1, 20, 4),
        query().with_simulation(400, 1, 4),
        query().with_simulation(400, 20, 1),
        query().with_level(1.0),
        query().with_fit_bootstrap(3),
        query().with_allocation(IntrinsicAllocation::Permutation { n_permutations: 1 }),
    ] {
        let text = run(&m, 2, &bad).unwrap_err().to_string();
        assert!(text.contains("intrinsic_influence.invalid_settings"), "{text}");
    }

    let huge = query().with_simulation(100_000, 100_000, 4);
    let text = run(&m, 2, &huge).unwrap_err().to_string();
    assert!(text.contains("intrinsic_influence.draw_budget_exceeded"), "{text}");
}

#[test]
fn the_same_seed_reproduces_the_result_and_a_different_seed_does_not() {
    let (m, _) = chain();
    let q = query().with_allocation(IntrinsicAllocation::Permutation { n_permutations: 40 });
    let a = run(&m, 2, &q).unwrap();
    let b = run(&m, 2, &q).unwrap();
    let bits = |r: &IntrinsicInfluenceResult| {
        r.sources.iter().map(|s| s.contribution.to_bits()).collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(a.total_variance.to_bits(), b.total_variance.to_bits());
    let c = run(&m, 2, &q.clone().with_seed(6)).unwrap();
    assert_ne!(bits(&a), bits(&c));
}

#[test]
fn a_cancelled_context_stops_the_attribution() {
    let (m, _) = chain();
    let c = ctx();
    c.cancellation.cancel();
    let q = IntrinsicInfluenceQuery { target: var(2), ..query() };
    let err = intrinsic_influence(&m, NoiseIndependence::Declared, None, &q, &c).unwrap_err();
    assert!(matches!(err, AttributionError::Cancelled | AttributionError::Model(_)), "{err:?}");
}

#[test]
fn the_result_records_the_method_and_budget() {
    let (m, _) = chain();
    let result = run(&m, 2, &query()).unwrap();
    assert_eq!(result.method.outer_draws, 400);
    assert_eq!(result.method.inner_draws, 20);
    assert_eq!(result.method.replicates, 4);
    assert!(result.method.estimator.contains("Janzing"));
    assert_eq!(result.budget.coalition_evaluations, 8);
    assert_eq!(result.budget.rows_replayed, 8 * 400 * 20 * 4);
    assert!((result.level - 0.95).abs() < f64::EPSILON);
}

/// `Var(a^2) = 2` for standard normal `a` (`E a^4 - (E a^2)^2 = 3 - 1`), computed from the
/// moments rather than by the estimator. The sample variance of `N = m B` draws of `a^2` has
/// variance `(μ4 - σ^4) / N = 56 / N` with `μ4 = E (a^2 - 1)^4 = 60`, which sets the
/// tolerance.
#[test]
fn a_nonlinear_mechanism_matches_the_moment_closed_form() {
    let m = quadratic();
    let q = IntrinsicInfluenceQuery::new(var(2)).with_simulation(1000, 10, 4).with_seed(8);
    let result = run(&m, 2, &q).unwrap();
    let derived = (56.0 / 4000.0_f64).sqrt();
    // Sources are reported in a topological order, not by variable id.
    for (variable, truth) in [(0, 2.0), (1, 1.0), (2, 0.25)] {
        let i = result
            .sources
            .iter()
            .position(|s| s.source == ComponentId::from(var(variable)))
            .unwrap();
        let got = result.sources[i].contribution;
        let se = reported_se(&result, i).max(derived * truth / 2.0);
        assert!((got - truth).abs() <= 6.0 * se, "source {variable}: got {got}, truth {truth}");
    }
    assert!((result.total_variance - 3.25).abs() <= 6.0 * (derived + result.total_variance_stderr));
}

#[test]
fn a_latent_common_cause_between_two_parents_is_refused_naming_the_pair() {
    let (m, _) = collider();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(node(0), node(2)).unwrap();
    admg.insert_directed(node(1), node(2)).unwrap();
    admg.insert_bidirected(node(0), node(1)).unwrap();
    let err = intrinsic_influence(
        &m,
        NoiseIndependence::Structure(&admg),
        None,
        &IntrinsicInfluenceQuery::new(var(2)),
        &ctx(),
    )
    .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("intrinsic_influence.correlated_noise"), "{text}");
    assert!(text.contains("variables 0 and 1"), "{text}");
}

#[test]
fn each_coalition_is_simulated_once_and_the_cache_is_reused() {
    // A chain of six nodes: the last node has six noise sources, so 64 distinct coalitions.
    let chain_edges: Vec<(u32, u32)> = (1..6).map(|i| (i - 1, i)).collect();
    let chain_slots = (0..6)
        .map(|i| if i == 0 { linear(0.0, &[], 1.0) } else { linear(0.0, &[0.9], 1.0) })
        .collect();
    let m = model(&chain_edges, chain_slots);
    let mut c = ctx();
    c.cache_policy = CachePolicy::enabled(None);
    let q = IntrinsicInfluenceQuery::new(var(5))
        .with_simulation(50, 4, 2)
        .with_allocation(IntrinsicAllocation::Permutation { n_permutations: 200 })
        .with_seed(1);
    let result = intrinsic_influence(&m, NoiseIndependence::Declared, None, &q, &c).unwrap();
    assert_eq!(result.sources.len(), 6);
    assert!(result.budget.coalition_evaluations <= 64, "{:?}", result.budget);
    assert!(result.cache_stats.hits > 0 && !result.cache_stats.saturated);
    assert_eq!(result.budget.permutations, 200);
}

/// Each component's standard error shrinks with its own budget while the other is held fixed.
///
/// Both errors scale as the inverse square root of their budget: the simulation error with
/// the outer draws `m` and the permutation error with the permutation count. A sixteen-fold
/// increase should cut either by about four; the check asks for a factor of 1.5, which twelve
/// replicate batches (a relative error of about 21% on each simulation standard error) and
/// fifty permutations keep far from chance.
#[test]
fn permutation_and_simulation_errors_shrink_with_their_own_budgets() {
    let m = product();
    let component = |result: &IntrinsicInfluenceResult, which: UncertaintyComponent| {
        let Some(ComponentStatus::Quantified { stderr, .. }) = result.sources[0].status(which)
        else {
            panic!("component {which:?} not quantified");
        };
        *stderr
    };
    let run_with = |outer: usize, permutations: usize| {
        let q = IntrinsicInfluenceQuery::new(var(2))
            .with_simulation(outer, 10, 12)
            .with_allocation(IntrinsicAllocation::Permutation { n_permutations: permutations })
            .with_seed(21);
        run(&m, 2, &q).unwrap()
    };
    let small_sim = run_with(100, 50);
    let large_sim = run_with(1600, 50);
    let sim_ratio = component(&small_sim, UncertaintyComponent::Simulation)
        / component(&large_sim, UncertaintyComponent::Simulation);
    assert!(sim_ratio >= 1.8, "simulation error ratio {sim_ratio}");

    let large_perm = run_with(100, 800);
    let perm_ratio = component(&small_sim, UncertaintyComponent::PermutationSampling)
        / component(&large_perm, UncertaintyComponent::PermutationSampling);
    assert!(perm_ratio >= 2.5, "permutation error ratio {perm_ratio}");
}

/// `a -> b -> c -> y`: `b = 2a + e_b`, `c = 1.5b + e_c`, `y = 0.5c + e_y`, noise scales
/// `(1, 0.5, 1, 0.8)`. The total coefficients of the four noise terms on `y` are
/// `(1.5, 0.75, 0.5, 1)`, so the exact contributions are `c_i² σ_i²`.
fn four_player_chain() -> (CompiledCausalModel, [f64; 4]) {
    let m = model(
        &[(0, 1), (1, 2), (2, 3)],
        vec![
            linear(0.0, &[], 1.0),
            linear(0.0, &[2.0], 0.5),
            linear(0.0, &[1.5], 1.0),
            linear(0.0, &[0.5], 0.8),
        ],
    );
    (m, [2.25, 0.140_625, 0.25, 0.64])
}

#[test]
fn exact_enumeration_reports_zero_permutation_error_and_still_reports_simulation_error() {
    let (m, truth) = four_player_chain();
    // 2500 outer draws x 10 inner draws x 4 replicate batches = 100000 rows per coalition.
    let (outer, inner, batches) = (2500, 10, 4);
    let exact = IntrinsicInfluenceQuery::new(var(3))
        .with_simulation(outer, inner, batches)
        .with_allocation(IntrinsicAllocation::Exact)
        .with_seed(17);
    let result = run(&m, 3, &exact).unwrap();
    assert_eq!(result.sources.len(), 4);
    assert_eq!(result.budget.exact_coalitions, 16);
    assert_eq!(result.budget.permutations, 0);
    assert_eq!(result.budget.rows_replayed, 16 * u64::try_from(outer * inner * batches).unwrap());
    for source in result.sources.iter() {
        // No permutation is sampled, so that component is exactly zero, not merely small.
        assert_eq!(source.component_stderr(UncertaintyComponent::PermutationSampling), Some(0.0));
        assert!(matches!(
            source.status(UncertaintyComponent::PermutationSampling),
            Some(ComponentStatus::NotApplicable { .. })
        ));
        // The inner simulation error is still measured, and is strictly positive.
        let simulation = source.component_stderr(UncertaintyComponent::Simulation).unwrap();
        assert!(simulation.is_finite() && simulation > 0.0, "{simulation}");
        // The unmeasured component is not reported as zero.
        assert_eq!(source.component_stderr(UncertaintyComponent::MechanismFit), None);
    }
    assert_matches_truth(&result, &truth, outer, batches);

    // A game with interaction allocated by sampled permutations does carry a permutation error.
    let sampled = IntrinsicInfluenceQuery::new(var(2))
        .with_simulation(400, 10, 4)
        .with_allocation(IntrinsicAllocation::Permutation { n_permutations: 50 })
        .with_seed(17);
    let result = run(&product(), 2, &sampled).unwrap();
    // The factors of the product share the interaction, so their marginal contributions
    // differ across permutations; no source reports a negative or non-finite error.
    for source in result.sources.iter() {
        let permutation =
            source.component_stderr(UncertaintyComponent::PermutationSampling).unwrap();
        assert!(permutation.is_finite() && permutation >= 0.0, "{permutation}");
    }
    let factor =
        result.sources[0].component_stderr(UncertaintyComponent::PermutationSampling).unwrap();
    assert!(factor > 0.0, "{factor}");
}

#[test]
fn intrinsic_fit_uses_active_rows_and_refuses_stored_weights() {
    use antecedent_data::TableView;
    let (model, data) = chain_data(80);
    let mut bytes = vec![0xffu8; 10];
    bytes[7] = 0x0f;
    bytes[8] = 0;
    bytes[9] = 0;
    let masked = data
        .with_analysis_mask(ValidityBitmap::from_bytes(Arc::<[u8]>::from(bytes), 80).unwrap())
        .unwrap();
    let x = data.float64_values(var(0)).unwrap();
    let y = data.float64_values(var(1)).unwrap();
    let active =
        TabularData::try_from_schema_f64(data.schema().clone(), [("x", &x[..60]), ("y", &y[..60])])
            .unwrap();
    let query = IntrinsicInfluenceQuery::new(var(1))
        .with_simulation(50, 5, 2)
        .with_fit_bootstrap(20)
        .with_seed(7);
    let masked_fit =
        intrinsic_influence(&model, NoiseIndependence::Declared, Some(&masked), &query, &ctx())
            .unwrap();
    let active_fit =
        intrinsic_influence(&model, NoiseIndependence::Declared, Some(&active), &query, &ctx())
            .unwrap();
    assert_eq!(masked_fit.fit_draws, active_fit.fit_draws);
    let ordered = query.clone().with_fit_resampling(FitResampling::BlockBootstrap { block: 4 });
    assert!(matches!(
        intrinsic_influence(&model, NoiseIndependence::Declared, Some(&masked), &ordered, &ctx()),
        Err(AttributionError::Unsupported { .. })
    ));
    let weighted = data.with_weights(Arc::from(vec![1.0; 80])).unwrap();
    assert!(matches!(
        intrinsic_influence(&model, NoiseIndependence::Declared, Some(&weighted), &query, &ctx()),
        Err(AttributionError::Unsupported { .. })
    ));
}
