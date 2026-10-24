//! Graph posteriors over lagged temporal-ADMG templates, run through the compiled route and
//! checked against the effects computed from the structural equations that generated the
//! series.
//!
//! An atom is a stationary template (a contemporaneous ADMG, lagged directed edges and lagged
//! latent-confounder edges). Each atom is identified on the unrolled window of its own template
//! and evaluated by the discrete plug-in on the lag-aligned window table; the atoms are mixed by
//! posterior weight, and an atom whose window leaves the target unidentified keeps its mass as
//! unidentified mass. The expected values come from the generating equations, not the engine.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, IdentificationStatus, MediationContrast, MediationQuery,
    ResponseValue, TemporalEffectQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, TemporalAdmgAtom,
    TemporalAdmgPosterior, with_lagged_edge,
};
use antecedent_prob::InferenceDiagnostics;

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn bern(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

/// Deterministic uniform draws (splitmix64).
struct Draws(u64);

impl Draws {
    fn flip(&mut self, p: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let unit = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
        f64::from(unit < p)
    }
}

fn atom_posterior(
    n_vars: usize,
    max_lag: u32,
    weights: &[f64],
    atoms: &[TemporalAdmgAtom],
) -> GraphPosterior {
    GraphPosterior::from_atoms(
        n_vars,
        weights.to_vec(),
        atoms.iter().map(|atom| atom.contemporaneous),
        vec![0.0; n_vars * n_vars],
        vec![0.0; n_vars * n_vars],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_admg_posterior"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg)
    .with_lagged_marginals(max_lag, vec![0.0; max_lag as usize * n_vars * n_vars])
    .unwrap()
    .with_lag_masks(atoms.iter().map(|atom| atom.lagged_directed).collect::<Vec<_>>())
    .unwrap()
    .with_lag_bidirected_masks(atoms.iter().map(|atom| atom.lagged_bidirected).collect::<Vec<_>>())
    .unwrap()
}

fn lagged(entries: &[(u32, usize, usize)], n_vars: usize) -> u64 {
    entries.iter().fold(0, |word, &(lag, from, to)| {
        with_lagged_edge(word, n_vars, lag, from, to, true).unwrap()
    })
}

fn contemporaneous(edges: &[(usize, usize)], n_vars: usize) -> AdjacencyMask {
    edges
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, n_vars, from, to, true))
}

// ---------------------------------------------------------------------------
// A pulse contrast through a latent-confounded lag structure.
// ---------------------------------------------------------------------------

const Z: usize = 0;
const X: usize = 1;
const Y: usize = 2;

/// `z_t ~ Bern(0.2 + 0.6 z_{t-1})`, `x_t ~ Bern(0.2 + 0.5 z_t)`,
/// `y_t ~ Bern(0.1 + 0.4 x_{t-1} + 0.3 z_{t-1})`.
fn effect_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut z, mut x, mut y) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        let previous_z = if t == 0 { 0.0 } else { z[t - 1] };
        z[t] = draws.flip(0.2 + 0.6 * previous_z);
        x[t] = draws.flip(0.2 + 0.5 * z[t]);
        if t > 0 {
            y[t] = draws.flip(0.1 + 0.4 * x[t - 1] + 0.3 * previous_z);
        }
    }
    TimeSeriesData::from_f64_columns(
        [("z", z.as_slice()), ("x", x.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

/// The structural effect of `x(0)` on `y(1)`.
const STRUCTURAL_EFFECT: f64 = 0.4;

/// The observational contrast `E[y(1) | x(0) = 1] - E[y(1) | x(0) = 0]` of the same law: the
/// effect a template that omits `z -> y` identifies.
fn observational_contrast() -> f64 {
    // The stationary `P(z = 1)` solves `p = 0.2 + 0.6 p`.
    let pi = 0.2 / (1.0 - 0.6);
    let p_x = |x: usize, z: usize| bern(0.2 + 0.5 * z as f64, x);
    let p_z_given_x = |x: usize| {
        let joint = |z: usize| p_x(x, z) * bern(pi, z);
        joint(1) / (joint(0) + joint(1))
    };
    let mean_y = |x: usize| 0.1 + 0.4 * x as f64 + 0.3 * p_z_given_x(x);
    mean_y(1) - mean_y(0)
}

/// The truth: `z -> x`, `z -> z`, `x -> y`, `z -> y`.
fn true_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(Z, X)], 3),
        lagged_directed: lagged(&[(1, Z, Z), (1, X, Y), (1, Z, Y)], 3),
        lagged_bidirected: 0,
    }
}

/// The truth without `z -> y` and `z -> x`: the confounding path through `z` is invisible to
/// it, so the treatment has no parent to adjust for and the contrast is the observational one.
fn omitting_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: AdjacencyMask::EMPTY,
        lagged_directed: lagged(&[(1, Z, Z), (1, X, Y)], 3),
        lagged_bidirected: 0,
    }
}

/// `x(t-1) -> y(t)` with `x(t-1) <-> y(t)`: a bow arc between the treatment and the outcome,
/// which the instrument `z -> x` does not identify.
fn bow_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(Z, X)], 3),
        lagged_directed: lagged(&[(1, Z, Z), (1, X, Y)], 3),
        lagged_bidirected: lagged(&[(1, X, Y)], 3),
    }
}

fn effect_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(var(X as u32), var(Y as u32), 1.0).with_horizon_steps(2)
}

fn scalar(value: Option<&ResponseValue>) -> Option<f64> {
    match value? {
        ResponseValue::Scalar(value) => Some(*value),
        _ => None,
    }
}

#[test]
fn the_reference_values_differ_across_templates() {
    let gap = observational_contrast() - STRUCTURAL_EFFECT;
    assert!(gap > 0.1, "the confounded contrast must differ from the effect: {gap}");
}

#[test]
fn the_atoms_are_identified_per_window_and_mixed_by_posterior_weight() {
    let ctx = ExecutionContext::for_tests(81);
    let series = effect_series(80_000, 5);
    let weights = [0.5, 0.2, 0.3];
    let gp = atom_posterior(3, 1, &weights, &[true_atom(), omitting_atom(), bow_atom()]);
    assert!(gp.is_lagged_temporal_admg());
    let expected = (0.5 * STRUCTURAL_EFFECT + 0.2 * observational_contrast()) / 0.7;
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)),
    ] {
        let built = Study::series(series.clone())
            .graph_posterior(gp.clone())
            .temporal_query(effect_query())
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap_or_else(|error| panic!("{inference:?}: {error}"));
        assert_eq!(result.logical_plan.identifier.as_deref(), Some("general.id"));
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert_eq!(mixture.atoms.len(), 3);
        assert!(
            (mixture.unidentified_mass - 0.3).abs() < 1e-12,
            "the bow atom's mass is unidentified mass: {}",
            mixture.unidentified_mass
        );
        // Atoms keep their posterior weights and their own values.
        let by_key = |key: u64| mixture.atoms.iter().find(|atom| atom.graph_key == key).unwrap();
        assert!((by_key(0).weight - 0.5).abs() < 1e-12);
        let truth = scalar(by_key(0).value.as_ref()).expect("the true template is identified");
        assert!((truth - STRUCTURAL_EFFECT).abs() < 0.02, "{truth}");
        let omitting =
            scalar(by_key(1).value.as_ref()).expect("the omitting template is identified");
        assert!((omitting - observational_contrast()).abs() < 0.02, "{omitting}");
        assert!(scalar(by_key(2).value.as_ref()).is_none(), "the bow atom has no value");
        assert_eq!(by_key(2).status, IdentificationStatus::NotIdentified);
        // The mixture is posterior-weighted over the identified mass, and graph-dependent.
        let effect = result.estimate.as_effect().unwrap().ate;
        assert!((effect - expected).abs() < 0.02, "{inference:?}: {effect} vs {expected}");
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert!(result.posterior.is_none(), "an atom posterior is conditional on its graph");
        let window = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "estimate.graph_posterior.temporal_admg_window")
            .expect("window disclosure");
        assert!(window.message.contains("3 periods"), "{}", window.message);
    }
}

#[test]
fn the_mixture_standard_error_resamples_blocks_of_windows_at_one_table() {
    let ctx = ExecutionContext::for_tests(82);
    let series = effect_series(8_000, 11);
    let gp = atom_posterior(3, 1, &[0.6, 0.4], &[true_atom(), omitting_atom()]);
    let result = Study::series(series)
        .graph_posterior(gp)
        .temporal_query(effect_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(60)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let estimate = result.estimate.as_effect().unwrap();
    let se = estimate.se_bootstrap().expect("a mixture standard error");
    assert!(se > 0.0 && se < 0.1, "{se}");
    assert_eq!(estimate.uncertainty.bootstrap.replicates_ok, Some(60));
    let expected = 0.6 * STRUCTURAL_EFFECT + 0.4 * observational_contrast();
    assert!((estimate.ate - expected).abs() < 0.06, "{} vs {expected}", estimate.ate);
}

#[test]
fn a_single_identified_atom_reproduces_the_window_route_of_its_template() {
    let ctx = ExecutionContext::for_tests(83);
    let series = effect_series(40_000, 17);
    let gp = atom_posterior(3, 1, &[1.0], &[true_atom()]);
    let result = Study::series(series)
        .graph_posterior(gp)
        .temporal_query(effect_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = result.estimate.as_effect().unwrap().ate;
    assert!((effect - STRUCTURAL_EFFECT).abs() < 0.02, "{effect}");
    let mixture = result.structural_response.as_ref().unwrap();
    assert_eq!(mixture.unidentified_mass, 0.0);
    assert_ne!(result.identification.status, IdentificationStatus::GraphDependent);
}

#[test]
fn a_prepared_handle_freezes_the_per_atom_identification_and_re_estimates() {
    let ctx = ExecutionContext::for_tests(84);
    let series = effect_series(20_000, 23);
    let gp = atom_posterior(3, 1, &[0.5, 0.2, 0.3], &[true_atom(), omitting_atom(), bow_atom()]);
    let study = Study::series(series)
        .graph_posterior(gp)
        .temporal_query(effect_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = study.run(&ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let again = prepared.estimate_retained(&ctx).unwrap();
    assert!(
        (again.estimate.as_effect().unwrap().ate - fresh.estimate.as_effect().unwrap().ate).abs()
            < 1e-12,
        "the click evaluates the frozen identification at the same data"
    );
    let mixture = again.structural_response.as_ref().unwrap();
    assert!((mixture.unidentified_mass - 0.3).abs() < 1e-12);
}

#[test]
fn contemporaneous_atoms_and_validation_suites_are_refused_with_their_next_attempt() {
    let ctx = ExecutionContext::for_tests(85);
    let series = effect_series(2_000, 29);
    // No lag masks: the atoms are contemporaneous and read tabular rows.
    let mut plain = atom_posterior(3, 1, &[1.0], &[true_atom()]);
    plain.lag_masks = None;
    plain.max_lag = None;
    let error = Study::series(series.clone())
        .graph_posterior(plain)
        .temporal_query(effect_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|built| built.run(&ctx))
        .unwrap_err();
    assert!(error.to_string().contains("lag masks"), "{error}");
    let lagged_posterior = atom_posterior(3, 1, &[1.0], &[true_atom()]);
    let refused = Study::series(series)
        .graph_posterior(lagged_posterior)
        .temporal_query(effect_query())
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .and_then(|built| built.run(&ctx));
    assert!(refused.is_err(), "cheap validation over lagged ADMG atoms is refused");
}

#[test]
fn a_posterior_scored_from_the_series_concentrates_on_the_supported_template() {
    let ctx = ExecutionContext::for_tests(87);
    let series = effect_series(20_000, 41);
    let variables = [var(Z as u32), var(X as u32), var(Y as u32)];
    let family = [true_atom(), omitting_atom()];
    let posterior =
        TemporalAdmgPosterior::new(1).run(&series, &variables, &family, None, &ctx).unwrap();
    assert!(
        posterior.weights[0] > 0.99,
        "the lagged z -> y coefficient is 0.3: {:?}",
        posterior.weights
    );
    let result = Study::series(series)
        .graph_posterior(posterior)
        .temporal_query(effect_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = result.estimate.as_effect().unwrap().ate;
    assert!((effect - STRUCTURAL_EFFECT).abs() < 0.03, "{effect}");
}

// ---------------------------------------------------------------------------
// A mediation through a lagged mediator.
// ---------------------------------------------------------------------------

const C: usize = 0;
const MX: usize = 1;
const W: usize = 2;
const M: usize = 3;
const MY: usize = 4;

fn p_w(w: usize) -> f64 {
    bern(0.35, w)
}
fn p_m(m: usize, x: usize, w: usize) -> f64 {
    bern(0.15 + 0.4 * x as f64 + 0.3 * w as f64, m)
}
fn p_y(y: usize, x: usize, m: usize) -> f64 {
    bern(0.1 + 0.2 * x as f64 + 0.3 * m as f64 + 0.2 * (x * m) as f64, y)
}
fn p_m_given(m: usize, x: usize) -> f64 {
    (0..2).map(|w| p_w(w) * p_m(m, x, w)).sum()
}

fn structural_indirect() -> f64 {
    (0..2).map(|m| p_y(1, 0, m) * (p_m_given(m, 1) - p_m_given(m, 0))).sum()
}

fn structural_direct() -> f64 {
    (0..2).map(|m| p_m_given(m, 0) * (p_y(1, 1, m) - p_y(1, 0, m))).sum()
}

fn structural_total() -> f64 {
    (0..2).map(|m| p_y(1, 1, m) * p_m_given(m, 1) - p_y(1, 0, m) * p_m_given(m, 0)).sum()
}

/// `c_t` and `w_t` i.i.d.; `x_t ~ Bern(0.25 + 0.5 c_{t-1})`; `m_t` follows `x_t` and `w_t`;
/// `y_t` follows `x_{t-1}` and `m_{t-1}` (the effect of the treatment reaches the outcome one
/// period later, directly and through the mediator).
fn mediation_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut c, mut x, mut w, mut m, mut y) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        c[t] = draws.flip(0.45);
        w[t] = draws.flip(0.35);
        let previous_c = if t == 0 { 0.0 } else { c[t - 1] };
        x[t] = draws.flip(0.25 + 0.5 * previous_c);
        m[t] = draws.flip(0.15 + 0.4 * x[t] + 0.3 * w[t]);
        if t > 0 {
            y[t] = draws.flip(0.1 + 0.2 * x[t - 1] + 0.3 * m[t - 1] + 0.2 * x[t - 1] * m[t - 1]);
        }
    }
    TimeSeriesData::from_f64_columns(
        [
            ("c", c.as_slice()),
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ],
        1,
    )
    .unwrap()
}

/// `c(t-1) -> x`, `x -> m`, `w -> m` within a period, `x(t-1) -> y`, `m(t-1) -> y`.
fn mediation_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(MX, M), (W, M)], 5),
        lagged_directed: lagged(&[(1, C, MX), (1, MX, MY), (1, M, MY)], 5),
        lagged_bidirected: 0,
    }
}

/// The same template with a latent confounder of the mediator and the outcome one period
/// later: the natural indirect effect has a recanting district and is not identified.
fn recanting_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom { lagged_bidirected: lagged(&[(1, M, MY)], 5), ..mediation_atom() }
}

fn mediation_query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(var(MX as u32), var(MY as u32), [var(M as u32)], contrast)
        .with_horizons(vec![2])
        .unwrap()
}

#[test]
fn a_mediation_posterior_mixes_each_component_over_the_atoms_that_identify_it() {
    let ctx = ExecutionContext::for_tests(86);
    let series = mediation_series(80_000, 31);
    let gp = atom_posterior(5, 1, &[0.6, 0.4], &[mediation_atom(), recanting_atom()]);
    let result = Study::series(series)
        .graph_posterior(gp)
        .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let mixture = result.structural_response.as_ref().expect("the atom mixture");
    // The recanting atom leaves the requested contrast unidentified and keeps its mass.
    assert!((mixture.unidentified_mass - 0.4).abs() < 1e-12, "{}", mixture.unidentified_mass);
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    let indirect = result.estimate.as_effect().unwrap().ate;
    assert!((indirect - structural_indirect()).abs() < 0.02, "{indirect}");
    let components = result.payload.mediation().expect("the decomposition");
    assert!((components.mediated.unwrap() - structural_indirect()).abs() < 0.02);
    // The total effect of the treatment is identified by both templates, whatever the
    // confounder of the mediator.
    assert!((components.total.unwrap() - structural_total()).abs() < 0.02);
    if let Some(direct) = components.direct {
        assert!((direct - structural_direct()).abs() < 0.02, "{direct}");
    }
}
