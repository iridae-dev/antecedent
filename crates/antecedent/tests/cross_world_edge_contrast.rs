//! Cross-world edge contrast on a fixed Markovian DAG (2.2 A5, X8): exact SCM
//! truth for every edge set, the natural direct effect regression, per-unit
//! shared abduction (mediator and outcome noise) under linear and non-separable
//! mechanisms, nonidentification, out-of-contract and refusal boundaries,
//! cancellation, and the deterministic-replay artifact consumer.
//!
//! `consistency` is a named premise that holds by construction (abduction defines
//! each exogenous term as the residual that regenerates the observed value), so
//! no test claims to detect its violation; what is tested is well-posed
//! abduction (invertibility) and everything else the operation enforces.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::cross_world::{
    CROSS_WORLD_MAX_ROWS, CrossWorldOptions, check, consume_cross_world_artifact,
    consume_cross_world_artifact_for_data, evaluate_cross_world_effect,
};
use antecedent::gcm::{NestedOutcomeMechanism, counterfactual_ite, fit_gcm};
use antecedent::{AcceptedGraph, CausalQuery, RefuteSuite, Study};
use antecedent_core::{
    CrossWorldQuery, ExecutionContext, ExogenousCoupling, NestedCounterfactualQuery, VariableId,
    WorldId, WorldObservation, WorldSpec,
};
use antecedent_data::TabularData;
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalCpdag, TemporalDag, TemporalPag,
};
use antecedent_io::cross_world_artifact::{
    CrossWorldArtifactError, CrossWorldArtifactWire, CrossWorldQueryWire, cross_world_data_digest,
    cross_world_identity,
};

const X: u32 = 0;
const M: u32 = 1;
const Y: u32 = 2;
const EDGES: [(u32, u32); 3] = [(X, M), (X, Y), (M, Y)];

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn dag(n: u32, edges: &[(u32, u32)]) -> Dag {
    let mut g = Dag::with_variables(n);
    for &(a, b) in edges {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn mediation_dag() -> Dag {
    dag(3, &EDGES)
}

/// The part of `noise` orthogonal to the constant and `x`, so an ordinary least
/// squares fit of the mediator recovers its slope exactly.
fn orthogonal_to_constant_and_x(noise: &[f64], x: &[f64]) -> Vec<f64> {
    let n = noise.len() as f64;
    let (mn, mx) = (noise.iter().sum::<f64>() / n, x.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let sxn: f64 = x.iter().zip(noise).map(|(a, b)| (a - mx) * (b - mn)).sum();
    let slope = sxn / sxx;
    x.iter().zip(noise).map(|(a, b)| (b - mn) - slope * (a - mx)).collect()
}

struct Linear {
    x: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
}

impl Linear {
    /// `M = 0.8 X + U_M`, `Y = 1.7 X + 4 M` (no outcome noise): every structural
    /// coefficient is recovered exactly, so edge-set effects are exact.
    fn new(n: usize) -> Self {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let raw: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
        let u = orthogonal_to_constant_and_x(&raw, &x);
        let m: Vec<f64> = x.iter().zip(&u).map(|(x, u)| 0.8 * x + u).collect();
        let y: Vec<f64> = x.iter().zip(&m).map(|(x, m)| 1.7 * x + 4.0 * m).collect();
        Self { x, m, y }
    }

    fn table(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("x", self.x.as_slice()),
            ("m", self.m.as_slice()),
            ("y", self.y.as_slice()),
        ])
        .unwrap()
    }
}

/// The part of `raw` orthogonal to the constant and every column of `basis`
/// (Gram-Schmidt), scaled to unit standard deviation, so an ordinary least
/// squares fit on those columns recovers its coefficients exactly.
fn orthogonal_noise(raw: &[f64], basis: &[&[f64]]) -> Vec<f64> {
    let n = raw.len();
    let mut axes: Vec<Vec<f64>> = vec![vec![1.0 / (n as f64).sqrt(); n]];
    for column in basis {
        let mut axis = column.to_vec();
        for prior in &axes {
            let dot: f64 = axis.iter().zip(prior).map(|(a, b)| a * b).sum();
            axis.iter_mut().zip(prior).for_each(|(a, b)| *a -= dot * b);
        }
        let norm = axis.iter().map(|a| a * a).sum::<f64>().sqrt();
        axes.push(axis.iter().map(|a| a / norm).collect());
    }
    let mut out = raw.to_vec();
    for axis in &axes {
        let dot: f64 = out.iter().zip(axis).map(|(a, b)| a * b).sum();
        out.iter_mut().zip(axis).for_each(|(a, b)| *a -= dot * b);
    }
    let sd = (out.iter().map(|a| a * a).sum::<f64>() / n as f64).sqrt();
    out.iter().map(|a| a / sd).collect()
}

/// `M = 0.8 X + U_M`, `Y = 1.7 X + 4 M + U_Y` with both disturbances of unit
/// standard deviation and orthogonal to their regressors, so every coefficient is
/// recovered exactly while the outcome carries its own per-unit noise.
struct NoisyLinear {
    x: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
    um: Vec<f64>,
    uy: Vec<f64>,
}

impl NoisyLinear {
    fn new(n: usize) -> Self {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let raw_m: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
        let um = orthogonal_noise(&raw_m, &[&x]);
        let m: Vec<f64> = x.iter().zip(&um).map(|(x, u)| 0.8 * x + u).collect();
        let raw_y: Vec<f64> = (0..n).map(|i| (i as f64 * 1.7 + 0.3).sin()).collect();
        let uy = orthogonal_noise(&raw_y, &[&x, &m]);
        let y: Vec<f64> =
            x.iter().zip(&m).zip(&uy).map(|((x, m), u)| 1.7 * x + 4.0 * m + u).collect();
        Self { x, m, y, um, uy }
    }

    fn table(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("x", self.x.as_slice()),
            ("m", self.m.as_slice()),
            ("y", self.y.as_slice()),
        ])
        .unwrap()
    }
}

fn effect(query: &CrossWorldQuery, data: &TabularData, options: CrossWorldOptions) -> f64 {
    evaluate_cross_world_effect(
        mediation_dag(),
        data,
        query,
        options,
        &ExecutionContext::for_tests(11),
    )
    .unwrap()
    .point
}

fn edge_query(control: f64, active: f64, intervened: &[(u32, u32)]) -> CrossWorldQuery {
    let all = EDGES.map(|(a, b)| (v(a), v(b)));
    let chosen: Vec<_> = intervened.iter().map(|&(a, b)| (v(a), v(b))).collect();
    CrossWorldQuery::path_specific(v(X), v(Y), control, active, &all, &chosen).unwrap()
}

fn reason(error: &antecedent::CausalError) -> String {
    error.to_string()
}

/// Exact SCM truth for all eight edge sets of the mediation graph.
#[test]
fn exact_linear_scm_truth_for_every_edge_set() {
    let data = Linear::new(400).table();
    let (control, active) = (-1.0, 2.0);
    let delta = active - control;
    for mask in 0u32..8 {
        let set: Vec<(u32, u32)> = EDGES
            .iter()
            .enumerate()
            .filter(|(i, _)| (mask >> i) & 1 == 1)
            .map(|(_, e)| *e)
            .collect();
        let has = |e: (u32, u32)| set.contains(&e);
        // Structural truth: the direct edge contributes 1.7 Δ; the mediator edge
        // contributes 4 · 0.8 Δ only when the treatment reaches the mediator and
        // the mediator reaches the outcome through edges that both see the
        // intervened world. A mediator the outcome reads at baseline changes nothing.
        let truth = 1.7 * delta * f64::from(u8::from(has((X, Y))))
            + 4.0 * 0.8 * delta * f64::from(u8::from(has((X, M)) && has((M, Y))));
        let got = effect(&edge_query(control, active, &set), &data, CrossWorldOptions::default());
        assert!((got - truth).abs() < 1e-9, "edge set {set:?}: {got} vs exact {truth}");
    }
}

/// The natural direct effect is the edge set {X -> Y}; it reproduces the existing
/// licensed nested-counterfactual cell exactly, and the natural indirect effect
/// is the {X -> M, M -> Y} edge set.
#[test]
fn natural_direct_effect_regression_against_the_existing_cell() {
    let fixture = Linear::new(300);
    let data = fixture.table();
    let (control, active) = (-1.0, 2.0);
    let nested = NestedCounterfactualQuery::with_levels(v(X), v(M), v(Y), control, active).unwrap();
    let study = Study::tabular(data.clone())
        .graph(mediation_dag())
        .query(CausalQuery::NestedCounterfactual(nested))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let existing = study.run(&ExecutionContext::for_tests(417)).unwrap().estimate.ate;
    let direct = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), control, active).unwrap();
    let mine = effect(&direct, &data, CrossWorldOptions::default());
    assert!((mine - existing).abs() < 1e-12, "cross-world NDE {mine} vs existing {existing}");
    assert!((mine - 1.7 * 3.0).abs() < 1e-9);

    let indirect = CrossWorldQuery::natural_indirect(v(X), v(M), v(Y), control, active).unwrap();
    let nie = effect(&indirect, &data, CrossWorldOptions::default());
    assert!((nie - 4.0 * 0.8 * 3.0).abs() < 1e-9, "{nie}");
}

struct NonSeparable {
    x: Vec<f64>,
    m: Vec<f64>,
    u: Vec<f64>,
    /// Outcome disturbance (all zero for the noiseless fixture).
    uy: Vec<f64>,
    y: Vec<f64>,
}

impl NonSeparable {
    /// `M = 0.8 X + U_M`, `Y = 1.7 X + 0.5 M + 0.9 X M^2`: convex in the mediator
    /// and interacting with the treatment, so a unit's abduced mediator
    /// disturbance reaches the answer.
    fn new(n: usize) -> Self {
        Self::with_outcome_noise(n, 0.0)
    }

    /// The same SCM plus an outcome disturbance `U_Y = scale * sin(1.7 i + 0.3)`.
    fn with_outcome_noise(n: usize, scale: f64) -> Self {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
        let uy: Vec<f64> = (0..n).map(|i| scale * (i as f64 * 1.7 + 0.3).sin()).collect();
        let m: Vec<f64> = x.iter().zip(&u).map(|(x, u)| 0.8 * x + u).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&m)
            .zip(&uy)
            .map(|((x, m), uy)| 1.7 * x + 0.5 * m + 0.9 * x * m * m + uy)
            .collect();
        Self { x, m, u, uy, y }
    }

    fn table(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("x", self.x.as_slice()),
            ("m", self.m.as_slice()),
            ("y", self.y.as_slice()),
        ])
        .unwrap()
    }

    fn outcome(x: f64, m: f64) -> f64 {
        1.7 * x + 0.5 * m + 0.9 * x * m * m
    }
}

/// Under a non-separable outcome the per-unit shared abduction is observable:
/// the answer reads each unit's abduced disturbance, matching the per-unit truth
/// of the structural model and differing decisively from the disturbance-free
/// (conditional-mean) plug-in that separable mechanisms could not tell apart.
#[test]
fn per_unit_shared_abduction_is_observable_under_a_non_separable_mechanism() {
    let fixture = NonSeparable::new(600);
    let data = fixture.table();
    let (control, active) = (-1.0, 2.0);
    let options = CrossWorldOptions {
        mechanism: NestedOutcomeMechanism::NonSeparableBasis,
        interval_requested: false,
    };
    let direct = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), control, active).unwrap();
    let got = effect(&direct, &data, options);

    let n = fixture.x.len() as f64;
    // Structural per-unit truth: the unit keeps its own mediator disturbance.
    let m0: Vec<f64> = fixture.u.iter().map(|u| 0.8 * control + u).collect();
    let shared_truth = m0
        .iter()
        .map(|m| NonSeparable::outcome(active, *m) - NonSeparable::outcome(control, *m))
        .sum::<f64>()
        / n;
    // Plug-in: the mediator frozen at its mean, no per-unit disturbance.
    let mbar = m0.iter().sum::<f64>() / n;
    let plug_in = NonSeparable::outcome(active, mbar) - NonSeparable::outcome(control, mbar);
    assert!((shared_truth - plug_in).abs() > 1.0, "fixture cannot tell them apart");
    assert!((got - shared_truth).abs() < 0.25, "{got} vs per-unit truth {shared_truth}");
    assert!((got - plug_in).abs() > 1.0, "{got} coincides with the plug-in {plug_in}");
    // Both answers differ from the separable fit of the same data, which cannot
    // read the outcome's interaction at all.
    let separable = effect(&direct, &data, CrossWorldOptions::default());
    assert!((separable - shared_truth).abs() > 1.0, "{separable}");
}

/// The edge-level answer is compared with the node-level counterfactual total
/// `E[Y | do(X = active)] - E[Y | do(X = control)]`. Where the edge set is every
/// edge the two coincide; where it omits a path the gap is exactly that path's
/// effect; and under an interaction the natural effects do not add to the total.
#[test]
fn edge_answer_differs_from_the_node_level_intervention() {
    let (control, active) = (-1.0, 2.0);
    let delta = active - control;
    let default = CrossWorldOptions::default();

    // Linear SCM with noisy mediator and noisy outcome. The node-level total comes
    // from the existing counterfactual engine (`counterfactual_ite`), fitted
    // independently of the cross-world route.
    let fixture = NoisyLinear::new(300);
    let data = fixture.table();
    let node_total = counterfactual_ite(
        fit_gcm(mediation_dag(), &data).unwrap().model,
        &data,
        v(X),
        v(Y),
        active,
        control,
        &ExecutionContext::for_tests(3),
    )
    .unwrap()
    .mean_ite;
    let all_edges = effect(&edge_query(control, active, &EDGES), &data, default);
    assert!((node_total - (1.7 + 4.0 * 0.8) * delta).abs() < 1e-6, "node-level total {node_total}");
    assert!(
        (all_edges - node_total).abs() < 1e-6,
        "every edge intervened is the node-level total: {all_edges} vs {node_total}"
    );
    // An edge set that omits a path differs from the total by exactly the omitted
    // path's effect, a stated margin of 4 * 0.8 * delta = 9.6 (direct only) and
    // 1.7 * delta = 5.1 (indirect only), far above the fit's 1e-6 accuracy.
    let direct = effect(&edge_query(control, active, &[(X, Y)]), &data, default);
    let indirect = effect(&edge_query(control, active, &[(X, M), (M, Y)]), &data, default);
    assert!((node_total - direct - 4.0 * 0.8 * delta).abs() < 1e-6, "{direct} vs {node_total}");
    assert!((node_total - indirect - 1.7 * delta).abs() < 1e-6, "{indirect} vs {node_total}");
    assert!((node_total - direct).abs() > 9.0 && (node_total - indirect).abs() > 5.0);
    // Additive model: the natural effects add to the total.
    assert!((direct + indirect - node_total).abs() < 1e-6);
    // An edge set carrying only X -> M changes nothing when the outcome reads the
    // baseline mediator, so it is not the effect of intervening on the mediator.
    let half = effect(&edge_query(control, active, &[(X, M)]), &data, default);
    assert!(half.abs() < 1e-9, "{half}");

    // Non-separable SCM with a noisy outcome: the node-level total is the
    // structural per-unit truth of do(X); the natural effects are neither it nor
    // additive.
    let fixture = NonSeparable::with_outcome_noise(600, 1.0);
    let data = fixture.table();
    let options = CrossWorldOptions {
        mechanism: NestedOutcomeMechanism::NonSeparableBasis,
        interval_requested: false,
    };
    let n = fixture.u.len() as f64;
    let total_truth = fixture
        .u
        .iter()
        .map(|u| {
            NonSeparable::outcome(active, 0.8 * active + u)
                - NonSeparable::outcome(control, 0.8 * control + u)
        })
        .sum::<f64>()
        / n;
    let all_edges = effect(&edge_query(control, active, &EDGES), &data, options);
    let nde = effect(&edge_query(control, active, &[(X, Y)]), &data, options);
    let nie = effect(&edge_query(control, active, &[(X, M), (M, Y)]), &data, options);
    assert!((all_edges - total_truth).abs() < 0.3, "{all_edges} vs total {total_truth}");
    assert!((nde - total_truth).abs() > 1.0, "NDE {nde} vs total {total_truth}");
    assert!((nie - total_truth).abs() > 1.0, "NIE {nie} vs total {total_truth}");
    assert!((nde + nie - all_edges).abs() > 0.5, "NDE {nde} + NIE {nie} vs total {all_edges}");
}

/// The outcome's own disturbance is one exogenous term per unit shared by both
/// worlds. Each world's outcome for a unit contains that unit's `U_Y`, so the
/// counterfactual outcomes (not only their difference) match the structural
/// values built from the unit's own `U_M` and `U_Y`; a world that read another
/// unit's outcome disturbance would break both outcomes and the contrast.
#[test]
fn the_outcome_noise_is_shared_per_unit_across_worlds() {
    let fixture = NoisyLinear::new(200);
    assert!(fixture.uy.iter().map(|u| u * u).sum::<f64>() / 200.0 > 0.9, "outcome noise is real");
    let data = fixture.table();
    let engine = engine_with(
        &data,
        antecedent_model::SelectionPolicy::RequireFamily(
            antecedent_model::MechanismFamily::LinearGaussian,
        ),
    );
    let (control, active) = (-1.0, 2.0);
    // Natural indirect edge set: world 1 reads its own mediator, world 0 its own.
    let indirect = edge_query(control, active, &[(X, M), (M, Y)]);
    let evaluation = antecedent_counterfactual::cross_world::evaluate_cross_world(
        &engine,
        &data,
        &indirect,
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    let structural_y = |x: f64, m: f64, uy: f64| 1.7 * x + 4.0 * m + uy;
    for unit in 0..fixture.uy.len() {
        let (um, uy) = (fixture.um[unit], fixture.uy[unit]);
        // World 1: X -> Y reads the baseline treatment, X -> M and M -> Y the active one.
        let plus = structural_y(control, 0.8 * active + um, uy);
        let minus = structural_y(control, 0.8 * control + um, uy);
        assert!((evaluation.plus[unit] - plus).abs() < 1e-8, "unit {unit} plus");
        assert!((evaluation.minus[unit] - minus).abs() < 1e-8, "unit {unit} minus");
    }
    // The per-unit contrast (the outcome disturbance cancels) is constant and exact.
    let unit_effects = evaluation.unit_effects();
    assert!(unit_effects.iter().all(|e| (e - 4.0 * 0.8 * (active - control)).abs() < 1e-8));
}

/// Same claim under the non-separable basis with a noisy outcome, read through
/// the per-unit contrasts: they follow the unit's own mediator disturbance, and
/// the unit's outcome disturbance cancels because both worlds carry it. An
/// outcome disturbance not shared per unit leaves `U_Y(i) - U_Y(j)` in the
/// contrast.
#[test]
fn the_outcome_noise_is_shared_per_unit_under_the_non_separable_mechanism() {
    let fixture = NonSeparable::with_outcome_noise(2000, 1.0);
    let n = fixture.u.len();
    assert!(
        fixture.uy.iter().map(|u| u * u).sum::<f64>() / n as f64 > 0.4,
        "outcome noise is real"
    );
    let (control, active) = (-1.0, 2.0);
    let result = evaluate_cross_world_effect(
        mediation_dag(),
        &fixture.table(),
        &edge_query(control, active, &[(X, M), (M, Y)]),
        CrossWorldOptions {
            mechanism: NestedOutcomeMechanism::NonSeparableBasis,
            interval_requested: false,
        },
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    let mean_error = (0..n)
        .map(|i| {
            let u = fixture.u[i];
            let truth = NonSeparable::outcome(control, 0.8 * active + u)
                - NonSeparable::outcome(control, 0.8 * control + u);
            (result.unit_effects[i] - truth).abs()
        })
        .sum::<f64>()
        / n as f64;
    // The basis fit is approximate (the structural outcome is not in its span);
    // an unshared outcome disturbance is off by about |U_Y(i) - U_Y(j)|.
    let rotated =
        (0..n).map(|i| (fixture.uy[(i + 1) % n] - fixture.uy[i]).abs()).sum::<f64>() / n as f64;
    assert!(mean_error < 0.2, "mean per-unit contrast error {mean_error}");
    assert!(rotated > 4.0 * mean_error, "fixture cannot tell shared from unshared: {rotated}");
}

/// The natural direct effect regresses against the existing licensed Study cell
/// on a noisy linear SCM and on a non-separable SCM with a noisy outcome. The
/// Study route fits the separable linear-Gaussian family, so it is compared with
/// the same family (the non-separable basis is compared with the existing nested
/// operation in the crate's unit tests). Both routes fit with the same routine
/// and abduce with the same engine, so the answers differ only in summation
/// order (1e-9 absolute on a value of order 10).
#[test]
fn natural_direct_effect_regression_on_noisy_and_non_separable_fixtures() {
    let (control, active) = (-1.0, 2.0);
    let nested = NestedCounterfactualQuery::with_levels(v(X), v(M), v(Y), control, active).unwrap();
    let direct = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), control, active).unwrap();
    let study_ate = |data: &TabularData| {
        Study::tabular(data.clone())
            .graph(mediation_dag())
            .query(CausalQuery::NestedCounterfactual(nested))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(417))
            .unwrap()
            .estimate
            .ate
    };
    let noisy = NoisyLinear::new(300).table();
    let (existing, mine) =
        (study_ate(&noisy), effect(&direct, &noisy, CrossWorldOptions::default()));
    assert!((mine - existing).abs() < 1e-9, "noisy linear: {mine} vs {existing}");
    assert!((mine - 1.7 * 3.0).abs() < 1e-9, "{mine}");

    let non_separable = NonSeparable::with_outcome_noise(500, 1.0).table();
    let (existing, mine) =
        (study_ate(&non_separable), effect(&direct, &non_separable, CrossWorldOptions::default()));
    assert!((mine - existing).abs() < 1e-9, "non-separable, linear family: {mine} vs {existing}");
}

/// The check is its own stage: it names the assumptions, the rerouted edges and
/// the counterfactual nodes the estimand needs, and a consumer can recompute it.
#[test]
fn the_check_returns_a_witness_naming_assumptions_and_counterfactual_nodes() {
    let direct = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), 0.0, 1.0).unwrap();
    let witness = check(&mediation_dag(), &direct).unwrap();
    for assumption in [
        "consistency",
        "markovian_no_latent_confounding",
        "mechanism_invariance_across_worlds",
        "no_recanting_witness",
    ] {
        assert!(witness.assumptions.iter().any(|a| a == assumption), "{assumption}");
    }
    assert_eq!(witness.intervened_edges, vec![(X, Y)]);
    assert_eq!(witness.rerouted_edges, vec![(X, M, 0), (M, Y, 0)]);
    assert_eq!(witness.coupled_worlds, vec![0, 1]);
    // Y in world 1 reads X from world 1 and the mediator from the baseline world.
    let y1 = witness
        .counterfactual_nodes
        .iter()
        .find(|n| (n.variable, n.world) == (Y, 1))
        .expect("Y in world 1 is a counterfactual node");
    assert_eq!(y1.reads, vec![(X, 1), (M, 0)]);
    witness.verify(3, &direct).unwrap();
    // A different query does not verify against this witness, and a witness with
    // an edited edge list recomputes to something else.
    let indirect = CrossWorldQuery::natural_indirect(v(X), v(M), v(Y), 0.0, 1.0).unwrap();
    assert!(witness.verify(3, &indirect).is_err());
    let mut edited = witness.clone();
    edited.graph_edges.pop();
    assert!(edited.verify(3, &direct).is_err());
}

/// A recanting witness is the nonidentification finding
/// (`cross_world_not_identified`); query shapes outside the two-world contract are
/// `route_not_supported`, never a claim that the query is unidentified.
#[test]
fn nonidentified_cross_world_queries_refuse_with_cross_world_not_identified() {
    // X -> W -> Y, W -> M -> Y: intervening on X -> W and W -> Y while M reads the
    // baseline W needs W in both worlds.
    let graph = dag(4, &[(0, 1), (1, 2), (1, 3), (2, 3)]);
    let edges = [(0, 1), (1, 2), (1, 3), (2, 3)].map(|(a, b)| (v(a), v(b)));
    let recanting =
        CrossWorldQuery::path_specific(v(0), v(3), 0.0, 1.0, &edges, &[(v(0), v(1)), (v(1), v(3))])
            .unwrap();
    let error = check(&graph, &recanting).unwrap_err();
    let text = reason(&error);
    assert!(text.contains("reason=cross_world_not_identified"), "{text}");
    assert!(text.contains("cross_world.recanting_witness"), "{text}");
    // The same graph with every edge intervened has no witness.
    let total = CrossWorldQuery::path_specific(v(0), v(3), 0.0, 1.0, &edges, &edges).unwrap();
    check(&graph, &total).unwrap();

    // Three worlds are outside the two-world contract: not an identification claim.
    let three = CrossWorldQuery::new(
        vec![
            WorldSpec::new([(v(X), 0.0)], []).unwrap(),
            WorldSpec::new([(v(X), 1.0)], []).unwrap(),
            WorldSpec::new([(v(X), 2.0)], []).unwrap(),
        ],
        ExogenousCoupling::SharedAbducedExogenous,
        WorldObservation { world: WorldId::new(1), variable: v(Y) },
        WorldObservation { world: WorldId::new(0), variable: v(Y) },
    )
    .unwrap();
    let text = reason(&check(&mediation_dag(), &three).unwrap_err());
    assert!(text.contains("reason=route_not_supported"), "{text}");
    assert!(text.contains("cross_world.query_outside_contract"), "{text}");
    assert!(!text.contains("cross_world_not_identified"), "{text}");

    // A different outcome per world is not this cell's contrast.
    let mixed = CrossWorldQuery::new(
        vec![
            WorldSpec::new([(v(X), 0.0)], []).unwrap(),
            WorldSpec::new([(v(X), 1.0)], []).unwrap(),
        ],
        ExogenousCoupling::SharedAbducedExogenous,
        WorldObservation { world: WorldId::new(1), variable: v(Y) },
        WorldObservation { world: WorldId::new(0), variable: v(M) },
    )
    .unwrap();
    let text = reason(&check(&mediation_dag(), &mixed).unwrap_err());
    assert!(text.contains("reason=route_not_supported"), "{text}");
    assert!(text.contains("cross_world.query_outside_contract"), "{text}");
    assert!(!text.contains("cross_world_not_identified"), "{text}");

    // A baseline world that reads another world is another shape as well.
    let baseline_reads = CrossWorldQuery::new(
        vec![
            WorldSpec::new(
                [(v(X), 0.0)],
                [antecedent_core::EdgeRoute { parent: v(X), child: v(M), source: WorldId::new(1) }],
            )
            .unwrap(),
            WorldSpec::new([(v(X), 1.0)], []).unwrap(),
        ],
        ExogenousCoupling::SharedAbducedExogenous,
        WorldObservation { world: WorldId::new(1), variable: v(Y) },
        WorldObservation { world: WorldId::new(0), variable: v(Y) },
    )
    .unwrap();
    let text = reason(&check(&mediation_dag(), &baseline_reads).unwrap_err());
    assert!(text.contains("reason=route_not_supported"), "{text}");
    assert!(text.contains("cross_world.query_outside_contract"), "{text}");
}

/// Accepted, latent-confounded, equivalence-class and oversized structures, and
/// interval requests, are refused with their own codes; the point survives none
/// of them silently.
#[test]
fn structures_and_inference_outside_the_cell_are_refused() {
    let data = Linear::new(60).table();
    let ctx = ExecutionContext::for_tests(5);
    let query = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), 0.0, 1.0).unwrap();
    let run = |graph: &dyn Fn() -> Result<(), antecedent::CausalError>| graph().unwrap_err();
    let admg = || {
        evaluate_cross_world_effect(
            Admg::with_variables(3),
            &data,
            &query,
            CrossWorldOptions::default(),
            &ctx,
        )
        .map(|_| ())
    };
    let cpdag = || {
        evaluate_cross_world_effect(
            Cpdag::with_variables(3),
            &data,
            &query,
            CrossWorldOptions::default(),
            &ctx,
        )
        .map(|_| ())
    };
    let pag = || {
        evaluate_cross_world_effect(
            Pag::with_variables(3),
            &data,
            &query,
            CrossWorldOptions::default(),
            &ctx,
        )
        .map(|_| ())
    };
    let accepted = || {
        evaluate_cross_world_effect(
            AcceptedGraph::from(mediation_dag()),
            &data,
            &query,
            CrossWorldOptions::default(),
            &ctx,
        )
        .map(|_| ())
    };
    for refused in [&admg as &dyn Fn() -> _, &cpdag, &pag, &accepted] {
        let text = reason(&run(refused));
        assert!(text.contains("reason=cell_not_licensed"), "{text}");
        assert!(text.contains("cross_world.graph_outside_contract"), "{text}");
    }
    let interval = evaluate_cross_world_effect(
        mediation_dag(),
        &data,
        &query,
        CrossWorldOptions { interval_requested: true, ..CrossWorldOptions::default() },
        &ctx,
    )
    .unwrap_err();
    let text = reason(&interval);
    assert!(text.contains("reason=estimator_inference_mismatch"), "{text}");
    assert!(text.contains("cross_world.interval_requested"), "{text}");
    // Nine variables exceed the licensed graph size.
    let nine: Vec<Vec<f64>> =
        (0..9).map(|k| (0..30).map(|i| f64::from(i * (k + 1)).sin()).collect()).collect();
    let names: Vec<String> = (0..9).map(|k| format!("v{k}")).collect();
    let big = TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(nine.iter().map(Vec::as_slice)),
    )
    .unwrap();
    let big_query = CrossWorldQuery::natural_direct(v(0), v(1), v(2), 0.0, 1.0).unwrap();
    let text = reason(
        &evaluate_cross_world_effect(
            dag(9, &[(0, 1), (1, 2)]),
            &big,
            &big_query,
            CrossWorldOptions::default(),
            &ctx,
        )
        .unwrap_err(),
    );
    assert!(text.contains("cross_world.graph_outside_contract"), "{text}");
    // Malformed: table width disagrees with the graph.
    let two =
        TabularData::from_f64_columns([("x", &[0.0, 1.0, 2.0][..]), ("y", &[1.0, 2.0, 3.0][..])])
            .unwrap();
    let text = reason(
        &evaluate_cross_world_effect(
            mediation_dag(),
            &two,
            &query,
            CrossWorldOptions::default(),
            &ctx,
        )
        .unwrap_err(),
    );
    assert!(
        text.contains("reason=invalid_argument") && text.contains("cross_world.invalid_query"),
        "{text}"
    );
}

fn artifact_bytes() -> (Vec<u8>, f64) {
    let data = Linear::new(120).table();
    let query = edge_query(-1.0, 2.0, &[(X, M), (M, Y)]);
    let effect = evaluate_cross_world_effect(
        mediation_dag(),
        &data,
        &query,
        CrossWorldOptions::default(),
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    (effect.export_artifact().unwrap(), effect.point)
}

/// The artifact round-trips and the consumer replays the same point, witness and
/// query: same evaluator, plus a separate closed-form OLS recomputation for the
/// linear-Gaussian family (not applicable to the non-separable basis).
#[test]
fn artifact_round_trip_is_replayed_by_the_consumer() {
    let (bytes, point) = artifact_bytes();
    let ctx = ExecutionContext::for_tests(1234);
    let consumed = consume_cross_world_artifact(&bytes, &ctx).unwrap();
    assert_eq!(consumed.point.to_bits(), point.to_bits());
    assert_eq!(consumed.query, edge_query(-1.0, 2.0, &[(X, M), (M, Y)]));
    assert_eq!(consumed.witness.intervened_edges, vec![(X, M), (M, Y)]);
    assert_eq!(consumed.witness.rerouted_edges, vec![(X, Y, 0)]);
    assert!(consumed.witness.assumptions.iter().any(|a| a == "consistency"));
    assert!((consumed.point - 4.0 * 0.8 * 3.0).abs() < 1e-9);
    let wire = CrossWorldArtifactWire::decode(&bytes).unwrap();
    assert_eq!(cross_world_identity(&wire).unwrap(), wire.premises_digest);
    assert_eq!(
        cross_world_data_digest(&wire.variable_names, &wire.columns).unwrap(),
        wire.data_digest
    );
    assert_eq!(consumed.data_digest, wire.data_digest);
    // The closed-form OLS recomputation agrees on the linear-Gaussian family.
    assert!(consumed.independently_verified);
    // ... including with mediator and outcome noise.
    let noisy = evaluate_cross_world_effect(
        mediation_dag(),
        &NoisyLinear::new(200).table(),
        &edge_query(-1.0, 2.0, &[(X, M), (M, Y)]),
        CrossWorldOptions::default(),
        &ctx,
    )
    .unwrap();
    let replayed = consume_cross_world_artifact(&noisy.export_artifact().unwrap(), &ctx).unwrap();
    assert!(replayed.independently_verified);
    assert!((replayed.point - 4.0 * 0.8 * 3.0).abs() < 1e-9);
    // The non-separable basis has no closed form here: replayed, not cross-checked.
    let table = NonSeparable::with_outcome_noise(300, 1.0).table();
    let options = CrossWorldOptions {
        mechanism: NestedOutcomeMechanism::NonSeparableBasis,
        interval_requested: false,
    };
    let produced = evaluate_cross_world_effect(
        mediation_dag(),
        &table,
        &edge_query(-1.0, 2.0, &[(X, Y)]),
        options,
        &ctx,
    )
    .unwrap();
    assert!(!produced.independently_verified);
    let replayed =
        consume_cross_world_artifact(&produced.export_artifact().unwrap(), &ctx).unwrap();
    assert_eq!(replayed.point.to_bits(), produced.point.to_bits());
    assert!(!replayed.independently_verified);
}

/// The artifact is bound to its data: a caller's own digest of the table accepts
/// the artifact computed on it and refuses any other table.
#[test]
fn the_artifact_is_bound_to_the_data_it_was_computed_on() {
    let (bytes, _) = artifact_bytes();
    let ctx = ExecutionContext::for_tests(1);
    let fixture = Linear::new(120);
    let names = ["x", "m", "y"].map(String::from).to_vec();
    let columns = vec![fixture.x.clone(), fixture.m.clone(), fixture.y.clone()];
    let digest = cross_world_data_digest(&names, &columns).unwrap();
    consume_cross_world_artifact_for_data(&bytes, &digest, &ctx).unwrap();
    let mut other = columns.clone();
    other[2][0] += 1e-9;
    let other_digest = cross_world_data_digest(&names, &other).unwrap();
    assert_ne!(other_digest, digest);
    assert_eq!(
        consume_cross_world_artifact_for_data(&bytes, &other_digest, &ctx).unwrap_err(),
        CrossWorldArtifactError::DataIdentityMismatch
    );
}

/// Typed mutation tests: a stale premises digest is a premises mismatch, a stale
/// data digest a data identity mismatch, and witness, point, version and feature
/// edits their own errors.
#[test]
fn a_mutated_artifact_fails_consumption_with_a_typed_error() {
    let (bytes, _) = artifact_bytes();
    let ctx = ExecutionContext::for_tests(1);
    let base = CrossWorldArtifactWire::decode(&bytes).unwrap();
    let encode = |wire: &CrossWorldArtifactWire| wire.export().unwrap();
    let consume = |wire: &CrossWorldArtifactWire| {
        consume_cross_world_artifact(&encode(wire), &ctx).unwrap_err()
    };

    // Premises edited but not re-sealed: the digest binds each of them.
    let mut cases: Vec<(&str, CrossWorldArtifactWire)> = Vec::new();
    let mut w = base.clone();
    w.variable_names[1] = "renamed".into();
    cases.push(("variable name", w));
    let mut w = base.clone();
    w.graph_edges.pop();
    cases.push(("graph edge", w));
    let mut w = base.clone();
    w.query.worlds[1].routes.pop();
    cases.push(("edge route", w));
    let mut w = base.clone();
    w.query.worlds[1].interventions[0].1 = 3.0f64.to_bits();
    cases.push(("treatment level", w));
    let mut w = base.clone();
    w.query.plus.1 = M;
    cases.push(("observed variable", w));
    let mut w = base.clone();
    w.mechanism = "non_separable_basis".into();
    cases.push(("mechanism family", w));
    let mut w = base.clone();
    w.estimand = "another estimand".into();
    cases.push(("estimand", w));
    for (label, wire) in &cases {
        assert_eq!(consume(wire), CrossWorldArtifactError::PremisesMismatch, "{label}");
    }
    // The table is bound by its own digest: an edited value (or a stale digest)
    // is a data identity mismatch, not a premises mismatch.
    let mut w = base.clone();
    w.columns[2][5] += 1.0;
    assert_eq!(consume(&w), CrossWorldArtifactError::DataIdentityMismatch, "table value");
    let mut w = base.clone();
    w.data_digest = "0".repeat(w.data_digest.len());
    assert_eq!(consume(&w), CrossWorldArtifactError::PremisesMismatch, "data digest is a premise");

    // Re-sealed semantic edits are rejected by replay.
    let mut w = base.clone();
    w.query.worlds[1].routes.pop();
    let w = w.sealed().unwrap();
    assert!(
        matches!(consume(&w), CrossWorldArtifactError::WitnessMismatch(_)),
        "{:?}",
        consume(&w)
    );
    let mut w = base.clone();
    w.witness.rerouted_edges.clear();
    assert!(matches!(consume(&w), CrossWorldArtifactError::WitnessMismatch(_)));
    let mut w = base.clone();
    w.witness.assumptions.retain(|a| a != "consistency");
    assert!(matches!(consume(&w), CrossWorldArtifactError::WitnessMismatch(_)));
    let mut w = base.clone();
    w.witness.counterfactual_nodes.pop();
    assert!(matches!(consume(&w), CrossWorldArtifactError::WitnessMismatch(_)));
    let mut w = base.clone();
    w.point_bits = (f64::from_bits(w.point_bits) + 1.0).to_bits();
    assert_eq!(consume(&w), CrossWorldArtifactError::PointMismatch);
    let mut w = base.clone();
    w.premises_digest = "0".repeat(w.premises_digest.len());
    assert_eq!(consume(&w), CrossWorldArtifactError::PremisesMismatch);
    let mut w = base.clone();
    w.version += 1;
    assert!(matches!(consume(&w), CrossWorldArtifactError::UnsupportedSemantics(_)));
    let mut w = base.clone();
    w.required_features = vec!["other".into()];
    assert!(matches!(consume(&w), CrossWorldArtifactError::UnsupportedSemantics(_)));
    // Bytes that are not this format.
    assert!(matches!(
        consume_cross_world_artifact(b"not an artifact", &ctx),
        Err(CrossWorldArtifactError::Decode(_))
    ));
    // An unsupported mechanism family, re-sealed.
    let mut w = base.clone();
    w.mechanism = "gaussian_process".into();
    let w = w.sealed().unwrap();
    assert!(matches!(consume(&w), CrossWorldArtifactError::UnsupportedSemantics(_)));
}

fn engine_with(
    data: &TabularData,
    policy: antecedent_model::SelectionPolicy,
) -> antecedent_counterfactual::CounterfactualEngine {
    let compiled = antecedent_model::CompiledCausalModel::compile(mediation_dag()).unwrap();
    let (store, _) = antecedent_model::MechanismRegistry::standard()
        .assign_and_fit(&compiled, data, policy)
        .unwrap();
    antecedent_counterfactual::CounterfactualEngine::new(compiled.with_mechanisms(store))
}

fn evaluate_on_engine(
    engine: &antecedent_counterfactual::CounterfactualEngine,
    data: &TabularData,
) -> Result<
    antecedent_counterfactual::cross_world::CrossWorldEvaluation,
    antecedent_counterfactual::CounterfactualError,
> {
    antecedent_counterfactual::cross_world::evaluate_cross_world(
        engine,
        data,
        &edge_query(-1.0, 2.0, &[(X, Y)]),
        &ExecutionContext::for_tests(3),
    )
}

/// Well-posed abduction on both mechanism families: every exogenous term is an
/// exact inversion. This is what the operation enforces; `consistency` itself is
/// a premise that holds by construction and is not tested from data.
#[test]
fn abduction_is_exact_inversion_on_both_mechanism_families() {
    let data = Linear::new(200).table();
    let engine = engine_with(
        &data,
        antecedent_model::SelectionPolicy::RequireFamily(
            antecedent_model::MechanismFamily::LinearGaussian,
        ),
    );
    let evaluation = evaluate_on_engine(&engine, &data).unwrap();
    assert_eq!(evaluation.noise, antecedent_counterfactual::NoiseInferenceKind::Invertible);
    for mechanism in
        [NestedOutcomeMechanism::LinearGaussian, NestedOutcomeMechanism::NonSeparableBasis]
    {
        let table = NonSeparable::with_outcome_noise(300, 1.0).table();
        let effect = evaluate_cross_world_effect(
            mediation_dag(),
            &table,
            &edge_query(-1.0, 2.0, &[(X, Y)]),
            CrossWorldOptions { mechanism, interval_requested: false },
            &ExecutionContext::for_tests(5),
        )
        .unwrap();
        assert_eq!(effect.noise, antecedent_counterfactual::NoiseInferenceKind::Invertible);
    }
}

/// Ill-posed abduction is refused, not silently abduced: a deterministic
/// (constant) mechanism given varying observations has no exogenous term that
/// regenerates them, and a binary treatment fitted as a discrete node yields
/// posterior draws rather than exact inversion. Both are the typed
/// `AbductionNotExact`, and each fixture reaches a different one of the two
/// enforcing branches.
#[test]
fn ill_posed_or_non_invertible_abduction_refuses() {
    use antecedent_counterfactual::CounterfactualError;
    let data = Linear::new(200).table();
    let flat = vec![1.0; 50];
    let reference = TabularData::from_f64_columns([
        ("x", flat.as_slice()),
        ("m", flat.as_slice()),
        ("y", flat.as_slice()),
    ])
    .unwrap();
    let constant = engine_with(
        &reference,
        antecedent_model::SelectionPolicy::RequireFamily(
            antecedent_model::MechanismFamily::Constant,
        ),
    );
    let error = evaluate_on_engine(&constant, &data).unwrap_err();
    assert!(matches!(error, CounterfactualError::AbductionNotExact { .. }), "{error:?}");

    let n = 200usize;
    let x: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let m: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
    let y: Vec<f64> = x.iter().zip(&m).map(|(x, m)| 1.7 * x + 4.0 * m).collect();
    let binary = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let discrete = engine_with(&binary, antecedent_model::SelectionPolicy::BestScore);
    let error = evaluate_on_engine(&discrete, &binary).unwrap_err();
    assert!(
        matches!(&error, CounterfactualError::AbductionNotExact { message } if message.contains("exogenous terms")),
        "{error:?}"
    );
}

/// What is not claimed: model adequacy is not tested from data. Mechanisms fitted
/// on one SCM evaluate data from another without refusal, because abduction
/// defines each exogenous term as the residual that regenerates the data. This
/// pins the documented limit (the `consistency` and correct-specification
/// premises are named, never checked).
#[test]
fn model_adequacy_is_not_tested_from_data() {
    let fitted_on = Linear::new(200).table();
    let engine = engine_with(
        &fitted_on,
        antecedent_model::SelectionPolicy::RequireFamily(
            antecedent_model::MechanismFamily::LinearGaussian,
        ),
    );
    // Data from a different SCM: the mediator coefficient and outcome differ.
    let other = NonSeparable::with_outcome_noise(200, 1.0).table();
    let evaluation = evaluate_on_engine(&engine, &other).unwrap();
    assert_eq!(evaluation.noise, antecedent_counterfactual::NoiseInferenceKind::Invertible);
}

/// The cancellation token stops the evaluation: a pre-cancelled context is
/// refused as the facade's typed cancellation before any fit, and a context that
/// is not cancelled answers.
#[test]
fn a_cancelled_context_is_refused_with_the_typed_cancellation() {
    let data = Linear::new(60).table();
    let query = edge_query(-1.0, 2.0, &[(X, Y)]);
    let ctx = ExecutionContext::for_tests(4);
    ctx.cancellation.cancel();
    let error = evaluate_cross_world_effect(
        mediation_dag(),
        &data,
        &query,
        CrossWorldOptions::default(),
        &ctx,
    )
    .unwrap_err();
    assert!(
        matches!(error, antecedent::CausalError::Cancelled { stage: "cross_world" }),
        "{error:?}"
    );
    // The cancellation is observed before the fit: a table whose non-separable
    // basis cannot be fitted (twelve rows) is refused as cancelled when the
    // context is cancelled, and as a fit failure when it is not.
    let tiny = Linear::new(12).table();
    let options = CrossWorldOptions {
        mechanism: NestedOutcomeMechanism::NonSeparableBasis,
        interval_requested: false,
    };
    let fit_error = evaluate_cross_world_effect(
        mediation_dag(),
        &tiny,
        &query,
        options,
        &ExecutionContext::for_tests(4),
    )
    .unwrap_err();
    assert!(!matches!(fit_error, antecedent::CausalError::Cancelled { .. }), "{fit_error:?}");
    let error =
        evaluate_cross_world_effect(mediation_dag(), &tiny, &query, options, &ctx).unwrap_err();
    assert!(
        matches!(error, antecedent::CausalError::Cancelled { stage: "cross_world" }),
        "{error:?}"
    );
}

/// Temporal structure (a dynamic graph over lags) is outside the fixed-DAG cell:
/// it is refused with the graph-contract detail, not fitted as if it were a DAG.
#[test]
fn temporal_structures_are_refused_with_the_graph_contract_detail() {
    let data = Linear::new(60).table();
    let ctx = ExecutionContext::for_tests(5);
    let query = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), 0.0, 1.0).unwrap();
    let options = CrossWorldOptions::default();
    let errors = [
        evaluate_cross_world_effect(TemporalDag::empty(), &data, &query, options, &ctx),
        evaluate_cross_world_effect(TemporalCpdag::empty(), &data, &query, options, &ctx),
        evaluate_cross_world_effect(TemporalPag::empty(), &data, &query, options, &ctx),
    ];
    for result in errors {
        let text = reason(&result.unwrap_err());
        assert!(text.contains("reason=cell_not_licensed"), "{text}");
        assert!(text.contains("cross_world.graph_outside_contract"), "{text}");
    }
}

/// The point is a mean of per-unit contrasts, and a mean is invariant to any
/// permutation of units inside a term, so the point alone cannot tell one shared
/// exogenous term per unit from independent draws per world. The per-unit
/// contrasts can: unit `i`'s effect must be the structural contrast computed from
/// unit `i`'s own abduced mediator disturbance in both worlds. Under the
/// natural indirect edge set the intervened world reads its own mediator and the
/// baseline reads its own, so mismatched (unshared) disturbances would show here.
#[test]
fn per_unit_effects_use_one_shared_exogenous_term_across_worlds() {
    let fixture = NonSeparable::new(600);
    let data = fixture.table();
    let (control, active) = (-1.0, 2.0);
    let options = CrossWorldOptions {
        mechanism: NestedOutcomeMechanism::NonSeparableBasis,
        interval_requested: false,
    };
    let indirect = CrossWorldQuery::natural_indirect(v(X), v(M), v(Y), control, active).unwrap();
    let result = evaluate_cross_world_effect(
        mediation_dag(),
        &data,
        &indirect,
        options,
        &ExecutionContext::for_tests(21),
    )
    .unwrap();
    assert_eq!(result.unit_effects.len(), fixture.u.len());
    let mut worst = 0.0_f64;
    let mut total = 0.0_f64;
    for (unit, u) in fixture.u.iter().enumerate() {
        // Y(control, M(active)) - Y(control, M(control)) with the unit's own U_M.
        let truth = NonSeparable::outcome(control, 0.8 * active + u)
            - NonSeparable::outcome(control, 0.8 * control + u);
        let error = (result.unit_effects[unit] - truth).abs();
        worst = worst.max(error);
        total += error;
    }
    let mean_error = total / fixture.u.len() as f64;
    assert!(mean_error < 0.3, "mean per-unit error {mean_error} (worst {worst})");
    // Unshared disturbances would put a different unit's term in one world:
    // that per-unit answer is far from this one.
    let shifted: f64 = fixture
        .u
        .iter()
        .enumerate()
        .map(|(unit, _)| {
            let other = fixture.u[(unit + 1) % fixture.u.len()];
            let unshared = NonSeparable::outcome(control, 0.8 * active + other)
                - NonSeparable::outcome(control, 0.8 * control + fixture.u[unit]);
            (result.unit_effects[unit] - unshared).abs()
        })
        .sum::<f64>()
        / fixture.u.len() as f64;
    assert!(shifted > 5.0 * mean_error.max(1e-3), "fixture cannot tell shared from unshared");
}

/// Semantic edits that are re-sealed (both digests recomputed, so every digest is
/// valid) must still fail replay, and for the right reason: an edit that changes
/// the derivation is a witness mismatch, one that changes only the computed value
/// is a point mismatch, and an unsupported semantic is refused as such.
#[test]
fn resealed_semantic_mutations_fail_replay_for_the_right_reason() {
    let (bytes, point) = artifact_bytes();
    let ctx = ExecutionContext::for_tests(1);
    let base = CrossWorldArtifactWire::decode(&bytes).unwrap();
    let replay = |wire: CrossWorldArtifactWire| {
        let sealed = wire.sealed().unwrap();
        consume_cross_world_artifact(&sealed.export().unwrap(), &ctx)
    };
    assert_eq!(replay(base.clone()).unwrap().point.to_bits(), point.to_bits(), "control");

    // Table: another factual value changes the fitted model and the point.
    let mut w = base.clone();
    w.columns[2][5] += 1.0;
    assert_eq!(replay(w).unwrap_err(), CrossWorldArtifactError::PointMismatch, "table");

    // Query level: the stored witness names the original query.
    let mut w = base.clone();
    w.query.worlds[1].interventions[0].1 = 3.0f64.to_bits();
    assert!(
        matches!(replay(w).unwrap_err(), CrossWorldArtifactError::WitnessMismatch(_)),
        "treatment level"
    );

    // Query routes with a witness recomputed to match: only the value differs.
    let mut w = base.clone();
    let natural_direct = edge_query(-1.0, 2.0, &[(X, Y)]);
    w.query = CrossWorldQueryWire::from_query(&natural_direct);
    w.witness = check(&mediation_dag(), &natural_direct).unwrap();
    assert_eq!(replay(w).unwrap_err(), CrossWorldArtifactError::PointMismatch, "edge routes");

    // Routes edited without the witness.
    let mut w = base.clone();
    w.query.worlds[1].routes.pop();
    assert!(
        matches!(replay(w).unwrap_err(), CrossWorldArtifactError::WitnessMismatch(_)),
        "edge routes, stale witness"
    );

    // Graph: an edge removed while the witness keeps the original graph.
    let mut w = base.clone();
    w.graph_edges.pop();
    assert!(
        matches!(replay(w).unwrap_err(), CrossWorldArtifactError::WitnessMismatch(_)),
        "graph edge"
    );

    // Mechanism family (on a table large enough for the basis): the other family
    // does not reproduce the stored point bit for bit.
    let large = evaluate_cross_world_effect(
        mediation_dag(),
        &NoisyLinear::new(400).table(),
        &edge_query(-1.0, 2.0, &[(X, M), (M, Y)]),
        CrossWorldOptions::default(),
        &ctx,
    )
    .unwrap();
    let mut w = CrossWorldArtifactWire::decode(&large.export_artifact().unwrap()).unwrap();
    w.mechanism = "non_separable_basis".into();
    assert_eq!(replay(w).unwrap_err(), CrossWorldArtifactError::PointMismatch, "mechanism family");

    // Estimand text and an unknown mechanism family are unsupported semantics.
    let mut w = base.clone();
    w.estimand = "another estimand".into();
    assert_eq!(replay(w).unwrap_err(), CrossWorldArtifactError::UnsupportedSemantics("estimand"));
    let mut w = base.clone();
    w.mechanism = "gaussian_process".into();
    assert_eq!(
        replay(w).unwrap_err(),
        CrossWorldArtifactError::UnsupportedSemantics("mechanism family")
    );

    // What replay does not protect against: labels. Renaming a variable while
    // re-sealing changes no computation, so it replays; the artifact's data digest
    // is what tells the two apart.
    let mut w = base.clone();
    w.variable_names[1] = "renamed".into();
    let renamed = replay(w).unwrap();
    assert_eq!(renamed.point.to_bits(), point.to_bits());
    assert_ne!(renamed.data_digest, base.data_digest);
    assert_eq!(
        consume_cross_world_artifact_for_data(
            &CrossWorldArtifactWire { ..base.clone() }.export().unwrap(),
            &renamed.data_digest,
            &ctx
        )
        .unwrap_err(),
        CrossWorldArtifactError::DataIdentityMismatch
    );
}

/// The producer enforces the format's row bound, so it never writes an artifact
/// its consumer refuses; the consumer enforces it too, independently of the digests.
#[test]
fn the_row_bound_is_enforced_at_export_and_at_consumption() {
    let n = CROSS_WORLD_MAX_ROWS + 1;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let m: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
    let y: Vec<f64> = x.iter().zip(&m).map(|(x, m)| 1.7 * x + 4.0 * m).collect();
    let big = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let ctx = ExecutionContext::for_tests(2);
    // Evaluation works at any size; export refuses a table the consumer would.
    let effect = evaluate_cross_world_effect(
        mediation_dag(),
        &big,
        &edge_query(-1.0, 2.0, &[(X, Y)]),
        CrossWorldOptions::default(),
        &ctx,
    )
    .unwrap();
    let error = effect.export_artifact().unwrap_err();
    assert!(matches!(error, antecedent::CausalError::Resource { .. }), "{error:?}");
    assert!(error.to_string().contains("100000 rows"), "{error}");

    // A wire at the bound exports; the same wire one row over cannot be sealed or
    // exported, and bytes written around the writer with valid digests are refused
    // by the consumer for the bound itself.
    let (bytes, _) = artifact_bytes();
    let base = CrossWorldArtifactWire::decode(&bytes).unwrap();
    let mut over = base.clone();
    over.columns = vec![x.clone(), m.clone(), y.clone()];
    assert_eq!(over.clone().sealed().unwrap_err(), CrossWorldArtifactError::LimitsExceeded("rows"));
    assert_eq!(over.export().unwrap_err(), CrossWorldArtifactError::LimitsExceeded("rows"));
    over.data_digest = cross_world_data_digest(&over.variable_names, &over.columns).unwrap();
    over.premises_digest = cross_world_identity(&over).unwrap();
    let smuggled = antecedent_io::to_cbor(&over).unwrap();
    assert_eq!(
        consume_cross_world_artifact(&smuggled, &ctx).unwrap_err(),
        CrossWorldArtifactError::LimitsExceeded("rows")
    );
}
