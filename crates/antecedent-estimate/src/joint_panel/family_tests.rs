// SPDX-License-Identifier: MIT OR Apache-2.0
//! The broader mechanism families against independent references: dense Gaussian
//! covariances, hand-written response formulas, analytic derivatives, change-of-variables
//! integrals and parameter recovery from simulated panels.
use super::decomposition::{
    ChangeComponent, DecompositionEndpoint, PairedPanelPath, PanelDecompositionRequest,
    decompose_panel_law,
};
use super::family::SPATIAL_JITTER;
use super::inference::numerical_score;
use super::likelihood::{self, CompiledPanel, Detail};
use super::*;

const ORDER: [ChangeComponent; 4] = [
    ChangeComponent::Exposure,
    ChangeComponent::Saturation,
    ChangeComponent::Population,
    ChangeComponent::Mechanism,
];

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}
fn normal(state: &mut u64) -> f64 {
    let (a, b) = (1.0 - uniform(state), uniform(state));
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

fn geo(theta: f64) -> [f64; 3] {
    let sum = 1.0 + theta + theta * theta;
    [1.0 / sum, theta / sum, theta * theta / sum]
}
fn hand_exposure(values: &[f64], history: [f64; 2], w: [f64; 3], t: usize) -> f64 {
    let at = |lag: usize| if lag <= t { values[t - lag] } else { history[2 + t - lag] };
    w[0] * at(0) + w[1] * at(1) + w[2] * at(2)
}
fn hand_saturation(e: f64, k: f64, shape: Option<f64>) -> f64 {
    match shape {
        None => e / (e + k),
        Some(s) => e.powf(s) / (e.powf(s) + k.powf(s)),
    }
}
pub(super) fn hand_mean(
    p: &PanelParameters,
    family: &PanelFamily,
    index: usize,
    group: &PanelGroup,
    t: usize,
) -> f64 {
    let k = p.half_saturation_of(family.class_of(index));
    let mut mean =
        p.intercept + p.confounder * group.confounder[t] + p.season_effects[group.season[t]];
    for (c, channel) in p.channels.iter().enumerate() {
        let e = hand_exposure(&group.channels[c], group.history[c], channel.lag, t);
        mean += channel.coefficient * hand_saturation(e, k, p.family.hill_shape.get(c).copied());
    }
    mean
}
fn hand_residuals(data: &PanelData, p: &PanelParameters) -> Vec<Vec<f64>> {
    data.groups
        .iter()
        .enumerate()
        .map(|(g, group)| {
            (0..group.outcome.len())
                .map(|t| group.outcome[t] - hand_mean(p, &data.family, g, group, t))
                .collect()
        })
        .collect()
}

/// Mechanism truth for a family: two channels, with every family coordinate populated.
pub(super) fn truth(family: &PanelFamily) -> PanelParameters {
    let lag =
        |simplex: [f64; 3], theta: f64| if family.is_geometric() { geo(theta) } else { simplex };
    PanelParameters {
        intercept: 1.0,
        channels: vec![
            PanelChannelParameters {
                id: "0".into(),
                coefficient: 4.0,
                lag: lag([0.6, 0.3, 0.1], 0.6),
            },
            PanelChannelParameters {
                id: "1".into(),
                coefficient: 3.0,
                lag: lag([0.5, 0.35, 0.15], 0.3),
            },
        ],
        interactions: vec![],
        confounder: 0.5,
        season_effects: vec![0.0],
        half_saturation: 2.0,
        rho: 0.3,
        innovation_sd: 0.15,
        group_sd: 0.4,
        family: PanelFamilyParameters {
            hill_shape: if family.is_hill() { vec![2.0, 1.0] } else { vec![] },
            class_half_saturation: vec![4.5; family.classes() - 1],
            spatial_range: if family.spatial.is_some() { 4.0 } else { 0.0 },
        },
    }
}

/// Distances between groups placed at the given positions on a line.
fn line(groups: usize, spacing: f64) -> PanelSpatial {
    let positions: Vec<f64> = (0..groups).map(|g| spacing * g as f64).collect();
    PanelSpatial {
        kernel: PanelSpatialKernel::Exponential,
        groups: (0..groups).map(|g| g.to_string()).collect(),
        distances: positions
            .iter()
            .flat_map(|a| positions.iter().map(move |b| (a - b).abs()))
            .collect(),
    }
}

/// A panel simulated from the truth with a hand-written mean and the declared residual law.
pub(super) fn simulate(
    family: PanelFamily,
    truth: &PanelParameters,
    groups: usize,
    periods: usize,
    seed: u64,
) -> PanelData {
    let mut state = seed;
    let mut data = PanelData {
        channels: vec![PanelChannel { id: "0".into() }, PanelChannel { id: "1".into() }],
        interactions: vec![],
        groups: vec![],
        seasons: 1,
        lag_order: 2,
        family,
    };
    let exposure = |state: &mut u64| {
        if uniform(state) < 0.15 { 0.0 } else { 5.0 * uniform(state) }
    };
    for g in 0..groups {
        let channels: Vec<Vec<f64>> =
            (0..2).map(|_| (0..periods).map(|_| exposure(&mut state)).collect()).collect();
        let history: Vec<[f64; 2]> =
            (0..2).map(|_| [exposure(&mut state), exposure(&mut state)]).collect();
        let confounder: Vec<f64> = (0..periods).map(|_| normal(&mut state)).collect();
        data.groups.push(PanelGroup {
            id: g.to_string(),
            history,
            channels,
            outcome: vec![0.0; periods],
            confounder,
            season: vec![0; periods],
        });
    }
    let intercepts: Vec<f64> = if let Some(spatial) = &data.family.spatial {
        // Spatial intercepts from an independently built dense Cholesky factor.
        let n = groups;
        let mut kernel: Vec<f64> = spatial
            .distances
            .iter()
            .map(|d| match spatial.kernel {
                PanelSpatialKernel::Exponential => (-d / truth.family.spatial_range).exp(),
                PanelSpatialKernel::Gaussian => (-(d / truth.family.spatial_range).powi(2)).exp(),
            })
            .collect();
        for g in 0..n {
            kernel[g * n + g] += SPATIAL_JITTER;
        }
        let lower = antecedent_stats::cholesky_spd(&kernel, n).unwrap();
        let z: Vec<f64> = (0..n).map(|_| normal(&mut state)).collect();
        (0..n)
            .map(|i| truth.group_sd * (0..=i).map(|j| lower[i * n + j] * z[j]).sum::<f64>())
            .collect()
    } else {
        (0..groups).map(|_| truth.group_sd * normal(&mut state)).collect()
    };
    for g in 0..groups {
        let mut error = truth.innovation_sd / (1.0 - truth.rho.powi(2)).sqrt() * normal(&mut state);
        for t in 0..periods {
            error = truth.rho * error + truth.innovation_sd * normal(&mut state);
            let mean = hand_mean(truth, &data.family, g, &data.groups[g], t);
            data.groups[g].outcome[t] = mean + intercepts[g] + error;
        }
    }
    data
}

/// Dense Gaussian log density of stacked residuals, and `C^-1 r`, with `C` built entry by entry.
fn dense(
    residuals: &[Vec<f64>],
    p: &PanelParameters,
    correlation: Option<&[f64]>,
) -> (f64, Vec<f64>, Vec<f64>) {
    let groups = residuals.len();
    let offsets: Vec<usize> = residuals
        .iter()
        .scan(0, |sum, r| {
            let start = *sum;
            *sum += r.len();
            Some(start)
        })
        .collect();
    let n: usize = residuals.iter().map(Vec::len).sum();
    let mut covariance = vec![0.0; n * n];
    for g in 0..groups {
        for h in 0..groups {
            let kappa = match correlation {
                Some(k) => k[g * groups + h],
                None => f64::from(u8::from(g == h)),
            };
            for t in 0..residuals[g].len() {
                for s in 0..residuals[h].len() {
                    let mut value = p.group_sd * p.group_sd * kappa;
                    if g == h {
                        value += p.innovation_sd.powi(2) / (1.0 - p.rho.powi(2))
                            * p.rho.powi(i32::try_from(t.abs_diff(s)).unwrap());
                    }
                    covariance[(offsets[g] + t) * n + offsets[h] + s] = value;
                }
            }
        }
    }
    let flat: Vec<f64> = residuals.concat();
    let lower = antecedent_stats::cholesky_spd(&covariance, n).unwrap();
    let solved = antecedent_stats::chol_solve(&lower, n, &flat).unwrap();
    let quadratic: f64 = flat.iter().zip(&solved).map(|(a, b)| a * b).sum();
    let log_likelihood = -0.5
        * (n as f64 * std::f64::consts::TAU.ln()
            + antecedent_stats::chol_log_det(&lower, n)
            + quadratic);
    (log_likelihood, solved, covariance)
}
fn exponential_correlation(spatial: &PanelSpatial, range: f64) -> Vec<f64> {
    let n = spatial.groups.len();
    let mut kernel: Vec<f64> = spatial.distances.iter().map(|d| (-d / range).exp()).collect();
    for g in 0..n {
        kernel[g * n + g] += SPATIAL_JITTER;
    }
    kernel
}

fn context(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}
fn options(initial: Option<PanelParameters>, draws: usize) -> PanelFitOptions {
    PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 800, gradient_tolerance: 1e-2 },
        inference: PanelInference::Frequentist { draws },
        initial,
    }
}
fn curve(p: &PanelParameters, channel: usize, class: usize, x: f64) -> f64 {
    p.channels[channel].coefficient
        * hand_saturation(x, p.half_saturation_of(class), p.family.hill_shape.get(channel).copied())
}

#[test]
fn reviewed_family_is_absent_from_the_schema_and_keeps_the_original_coordinates() {
    let (data, draw, _) = super::tests::fixture();
    let data_json = serde_json::to_value(&data).unwrap();
    let parameter_json = serde_json::to_value(&draw.parameters).unwrap();
    assert!(data_json.get("family").is_none() && parameter_json.get("family").is_none());
    let prior = serde_json::to_value(PanelPrior::default()).unwrap();
    assert!(prior.get("family").is_none());
    // Records written before the family existed still deserialize to the reviewed family.
    let reread: PanelData = serde_json::from_value(data_json).unwrap();
    assert!(reread.family.is_reviewed());
    assert_eq!(reread, data);
    // Original coordinate order: 2 channels, 1 product, season block, 4 lag logits, k, rho, sigma, tau.
    let layout = likelihood::layout(&data);
    assert_eq!(layout.base, 8);
    assert_eq!(layout.total, 16);
    assert_eq!(
        (layout.k0, layout.rho, layout.sigma, layout.tau),
        (layout.base + 4, Some(layout.base + 5), layout.base + 6, layout.base + 7)
    );
    assert_eq!(layout.range, None);
}

#[test]
fn broader_family_round_trips_through_the_frozen_schema_with_a_distinct_identity() {
    let family = PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 1, 1, 1] },
        spatial: Some(line(5, 1.5)),
    };
    let p = truth(&family);
    let data = simulate(family.clone(), &p, 5, 8, 11);
    data.validate().unwrap();
    let json = serde_json::to_string(&data).unwrap();
    let reread: PanelData = serde_json::from_str(&json).unwrap();
    assert_eq!(reread, data);
    assert_eq!(reread.identity().unwrap(), data.identity().unwrap());
    let draw = PanelDraw { parameters: p.clone(), group_intercepts: vec![0.1; 5] };
    let reread_draw: PanelDraw =
        serde_json::from_str(&serde_json::to_string(&draw).unwrap()).unwrap();
    assert_eq!(reread_draw, draw);
    likelihood::validate_parameters(&reread, &reread_draw.parameters).unwrap();
    // Each family choice changes the source identity; none collapses to the reviewed one.
    let reviewed = PanelData { family: PanelFamily::default(), ..data.clone() };
    assert_ne!(reviewed.identity().unwrap(), data.identity().unwrap());
    let mut other = data.clone();
    other.family.adstock = PanelAdstock::Simplex;
    assert_ne!(other.identity().unwrap(), data.identity().unwrap());
    let mut other = data.clone();
    other.family.scope = PanelSaturationScope::Classes { assignment: vec![0, 1, 1, 1, 1] };
    assert_ne!(other.identity().unwrap(), data.identity().unwrap());
    let mut other = data.clone();
    other.family.spatial.as_mut().unwrap().distances[1] += 0.5;
    other.family.spatial.as_mut().unwrap().distances[5] += 0.5;
    assert_ne!(other.identity().unwrap(), data.identity().unwrap());
    // Prior coordinates are part of the fit-policy identity only when they differ.
    let base = PanelFitOptions {
        inference: PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: PanelSamplingOptions::default(),
        },
        ..options(None, 0)
    };
    let mut changed = base.clone();
    if let PanelInference::Bayesian { prior, .. } = &mut changed.inference {
        prior.family.log_hill_shape_sd = 0.9;
    }
    assert_ne!(base.identity(&data).unwrap(), changed.identity(&data).unwrap());
    let mut unknown = serde_json::to_value(&data).unwrap();
    unknown["family"]["extra"] = serde_json::json!(1);
    assert!(serde_json::from_value::<PanelData>(unknown).is_err());
}

#[test]
fn family_declarations_that_do_not_match_the_panel_are_refused() {
    let base_family = PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 1, 1, 0] },
        spatial: Some(line(4, 1.0)),
    };
    let p = truth(&base_family);
    let good = simulate(base_family.clone(), &p, 4, 6, 3);
    good.validate().unwrap();
    let mut data = good.clone();
    data.lag_order = 0;
    assert!(data.validate().is_err(), "geometric adstock needs carryover");
    let mut data = good.clone();
    data.family.scope = PanelSaturationScope::Classes { assignment: vec![0, 2, 2, 0] };
    assert!(data.validate().is_err(), "classes must be contiguous");
    let mut data = good.clone();
    data.family.scope = PanelSaturationScope::Classes { assignment: vec![0, 1, 1] };
    assert!(data.validate().is_err(), "every group needs a class");
    let mut data = good.clone();
    data.family.spatial.as_mut().unwrap().groups.swap(0, 1);
    assert!(data.validate().is_err(), "spatial alignment is by identity");
    let mut data = good.clone();
    data.family.spatial.as_mut().unwrap().distances[1] = 9.0;
    assert!(data.validate().is_err(), "distances must be symmetric");
    let mut data = good.clone();
    data.family.spatial.as_mut().unwrap().distances[0] = 1.0;
    assert!(data.validate().is_err(), "the diagonal is zero");
    // Parameter vectors must follow the declared family.
    let compiled = CompiledPanel::new(&good).unwrap();
    likelihood::validate_parameters(&compiled, &p).unwrap();
    let mut bad = p.clone();
    bad.family.hill_shape.pop();
    assert!(likelihood::validate_parameters(&compiled, &bad).is_err());
    let mut bad = p.clone();
    bad.family.class_half_saturation.push(1.0);
    assert!(likelihood::validate_parameters(&compiled, &bad).is_err());
    let mut bad = p.clone();
    bad.family.spatial_range = 0.0;
    assert!(likelihood::validate_parameters(&compiled, &bad).is_err());
    let mut bad = p.clone();
    bad.channels[0].lag = [0.6, 0.3, 0.1];
    assert!(likelihood::validate_parameters(&compiled, &bad).is_err(), "not geometric");
    let mut bad = p;
    bad.channels[0].lag = [0.2, 0.3, 0.5];
    assert!(likelihood::validate_parameters(&compiled, &bad).is_err(), "must decay");
}

#[test]
fn geometric_weights_and_their_logit_coordinate_match_the_closed_form() {
    for logit in [-3.0_f64, -0.4, 0.0, 0.3, 2.5] {
        let theta = crate::util::sigmoid(logit);
        let w = likelihood::geometric(logit);
        for (actual, expected) in w.iter().zip(geo(theta)) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!((w[1] / w[0] - theta).abs() < 1e-13 && (w[2] / w[1] - theta).abs() < 1e-13);
    }
    let family = PanelFamily { adstock: PanelAdstock::Geometric, ..PanelFamily::default() };
    let p = truth(&family);
    let data = simulate(family, &p, 3, 6, 5);
    let encoded = likelihood::encode(&data, &p).unwrap();
    let layout = likelihood::layout(&data);
    assert_eq!(layout.k0 - layout.lag, 2, "one decay per channel, not two simplex logits");
    let theta = [0.6_f64, 0.3];
    for (c, theta) in theta.iter().enumerate() {
        assert!((encoded[layout.lag + c] - (theta / (1.0 - theta)).ln()).abs() < 1e-12);
    }
    let decoded = likelihood::decode(&data, &encoded).unwrap();
    for (a, b) in decoded.channels.iter().zip(&p.channels) {
        for (x, y) in a.lag.iter().zip(b.lag) {
            assert!((x - y).abs() < 1e-13);
        }
    }
}

#[test]
fn transformed_priors_carry_the_exact_change_of_variables_jacobian() {
    let family = PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 1, 1] },
        spatial: Some(line(4, 1.0)),
    };
    let p = truth(&family);
    let data = simulate(family, &p, 4, 6, 7);
    let layout = likelihood::layout(&data);
    let mut prior = PanelPrior::default();
    prior.family.geometric_decay = [2.0, 3.0];
    let x0 = likelihood::encode(&data, &p).unwrap();
    let at = |index: usize, value: f64| {
        let mut x = x0.clone();
        x[index] = value;
        super::inference::log_prior(&data, &x, &prior)
    };
    // Geometric decay: Beta(2,3) in theta = sigmoid(z) has log-density
    // ln(12 theta (1-theta)^2) + ln|d theta / d z|, the derivative taken numerically.
    let density = |z: f64| {
        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
        let theta = sigmoid(z);
        let h = 1e-6;
        let slope = (sigmoid(z + h) - sigmoid(z - h)) / (2.0 * h);
        12.0 * theta * (1.0 - theta).powi(2) * slope
    };
    let index = layout.lag;
    let reference = x0[index];
    let mut integral = 0.0;
    let step = 0.01;
    let mut z = -40.0;
    while z <= 40.0 {
        let ratio = (at(index, z) - at(index, reference)).exp();
        let expected = density(z) / density(reference);
        assert!(
            (ratio - expected).abs() < 1e-5 * (1.0 + expected),
            "geometric prior ratio at z={z}: {ratio} vs {expected}"
        );
        integral += ratio * density(reference) * step;
        z += step;
    }
    assert!((integral - 1.0).abs() < 1e-4, "prior integrates to {integral}");
    // Hill shape: log-normal in s means Normal in the coordinate log s; no Jacobian term.
    let index = layout.shape;
    let (mean, sd) = (prior.family.log_hill_shape_mean, prior.family.log_hill_shape_sd);
    for z in [-1.0, -0.2, 0.3, 0.9] {
        let delta = at(index, z) - at(index, 0.0);
        assert!(
            (delta - (-0.5 * ((z - mean) / sd).powi(2) + 0.5 * (mean / sd).powi(2))).abs() < 1e-12
        );
    }
    // Class half-saturation is pooled toward the first class on the log scale.
    let index = layout.class;
    let pooled = prior.family.class_log_saturation_sd;
    for z in [0.2, 1.1, 2.4] {
        let delta = at(index, z) - at(index, x0[layout.k0]);
        let expected = -0.5 * ((z - x0[layout.k0]) / pooled).powi(2);
        assert!((delta - expected).abs() < 1e-12);
    }
    // Spatial range is log-normal about the median positive distance by default.
    let index = layout.range.unwrap();
    let center = data.family.median_distance().ln();
    let sd = prior.family.log_spatial_range_sd;
    for z in [-0.5, 0.0, 1.2] {
        let delta = at(index, z) - at(index, center);
        assert!((delta + 0.5 * ((z - center) / sd).powi(2)).abs() < 1e-12);
    }
}

#[test]
fn hill_response_matches_the_hand_formula_and_shape_one_is_michaelis_menten() {
    let (mut data, mut draw, _) = super::tests::fixture();
    let programs = super::tests::programs(&data);
    let ctx = context(3);
    let reviewed = evaluate_panel_programs(&data, &draw, &programs, &ctx).unwrap();
    data.family = PanelFamily {
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 1, 1] },
        ..PanelFamily::default()
    };
    draw.parameters.family = PanelFamilyParameters {
        hill_shape: vec![1.0, 1.0],
        class_half_saturation: vec![draw.parameters.half_saturation],
        spatial_range: 0.0,
    };
    let unit = evaluate_panel_programs(&data, &draw, &programs, &ctx).unwrap();
    for (a, b) in unit.iter().zip(&reviewed) {
        for (x, y) in a.means.iter().zip(&b.means) {
            assert!((x - y).abs() < 1e-12, "shape one with equal classes is Michaelis-Menten");
        }
    }
    draw.parameters.family.hill_shape = vec![2.0, 1.5];
    draw.parameters.family.class_half_saturation = vec![3.5];
    let output = evaluate_panel_programs(&data, &draw, &programs, &ctx).unwrap();
    let p = &draw.parameters;
    for (response, program) in output.iter().zip(&programs) {
        for t in 0..24 {
            let mut total = 0.0;
            for g in 0..data.groups.len() {
                let scenario = &program.scenarios[g];
                let class = data.family.class_of(g);
                let k = p.half_saturation_of(class);
                let sat: Vec<f64> = (0..2)
                    .map(|c| {
                        let history = scenario.history[c].values;
                        let e = hand_exposure(
                            &program.channels[c].values,
                            history,
                            p.channels[c].lag,
                            t,
                        );
                        hand_saturation(e, k, Some(p.family.hill_shape[c]))
                    })
                    .collect();
                total += p.intercept
                    + p.channels[0].coefficient * sat[0]
                    + p.channels[1].coefficient * sat[1]
                    + p.interactions[0].coefficient * sat[0] * sat[1]
                    + p.confounder * scenario.confounder[t]
                    + p.season_effects[scenario.season[t]]
                    + draw.group_intercepts[g];
            }
            assert!((response.means[t] - total / 4.0).abs() < 1e-11, "{} at {t}", response.program);
        }
    }
    // Group-specific saturation is not a relabeling: moving only class one changes only its groups.
    let mut moved = draw.clone();
    moved.parameters.family.class_half_saturation = vec![7.0];
    let shifted = evaluate_panel_programs(&data, &moved, &programs, &ctx).unwrap();
    assert!(shifted.iter().zip(&output).any(|(a, b)| a.means != b.means));
}

#[test]
fn spatial_marginal_matches_the_dense_joint_covariance_and_its_conditional_moments() {
    let family = PanelFamily { spatial: Some(line(5, 1.3)), ..PanelFamily::default() };
    let p = truth(&family);
    let data = simulate(family.clone(), &p, 5, 7, 21);
    let compiled = CompiledPanel::new(&data).unwrap();
    let residuals = likelihood::residuals(&compiled, &p);
    let hand = hand_residuals(&data, &p);
    for (a, b) in residuals.iter().flatten().zip(hand.iter().flatten()) {
        assert!((a - b).abs() < 1e-12);
    }
    let correlation = exponential_correlation(family.spatial.as_ref().unwrap(), 4.0);
    let (expected, solved, covariance) = dense(&residuals, &p, Some(&correlation));
    let marginal = likelihood::marginal(&compiled, &p, &residuals, Detail::Draw).unwrap();
    assert!(
        (marginal.log_likelihood - expected).abs() < 1e-9,
        "{} vs {expected}",
        marginal.log_likelihood
    );
    // E[u_g | r] = Cov(u_g, r) C^-1 r and Var(u_g | r) = Var(u_g) - Cov C^-1 Cov'.
    let n: usize = residuals.iter().map(Vec::len).sum();
    let groups = residuals.len();
    let tau2 = p.group_sd * p.group_sd;
    let lower = antecedent_stats::cholesky_spd(&covariance, n).unwrap();
    for g in 0..groups {
        let cross: Vec<f64> = (0..groups)
            .flat_map(|h| (0..residuals[h].len()).map(move |_| h))
            .map(|h| tau2 * correlation[g * groups + h])
            .collect();
        let mean: f64 = cross.iter().zip(&solved).map(|(a, b)| a * b).sum();
        let inverse_cross = antecedent_stats::chol_solve(&lower, n, &cross).unwrap();
        let variance = tau2 * correlation[g * groups + g]
            - cross.iter().zip(&inverse_cross).map(|(a, b)| a * b).sum::<f64>();
        assert!((marginal.intercept_mean[g] - mean).abs() < 1e-8, "mean {g}");
        assert!((marginal.intercept_variance[g] - variance).abs() < 1e-8, "variance {g}");
    }
    // The score of the global intercept is 1' C^-1 r under the full spatial covariance.
    let x = likelihood::encode(&data, &p).unwrap();
    let score =
        numerical_score(&x, &context(1), &|x| likelihood::log_likelihood(&compiled, x)).unwrap();
    let analytic: f64 = solved.iter().sum();
    assert!(
        (score[0] - analytic).abs() < 1e-5 * (1.0 + analytic.abs()),
        "{} vs {analytic}",
        score[0]
    );
    // A vanishing range removes the correlation and gives independent random intercepts.
    let mut isolated = p.clone();
    isolated.family.spatial_range = 1e-4;
    let r = likelihood::residuals(&compiled, &isolated);
    let spatial = likelihood::marginal(&compiled, &isolated, &r, Detail::Value).unwrap();
    let independent: f64 = r
        .iter()
        .map(|r| {
            likelihood::ar1_random_intercept(
                r,
                isolated.rho,
                isolated.innovation_sd,
                isolated.group_sd * (1.0 + SPATIAL_JITTER).sqrt(),
            )
            .unwrap()
            .log_likelihood
        })
        .sum();
    assert!((spatial.log_likelihood - independent).abs() < 1e-7);
    // The joint draw reproduces the conditional covariance (empirically, with a fixed stream).
    let mut rng = context(9).rng.stream(5);
    let mut sums = vec![0.0; groups];
    let mut squares = vec![0.0; groups];
    let draws = 4000;
    for _ in 0..draws {
        let draw = marginal.intercepts(Some(&mut rng)).unwrap();
        for g in 0..groups {
            sums[g] += draw[g];
            squares[g] += draw[g] * draw[g];
        }
    }
    for g in 0..groups {
        let mean = sums[g] / f64::from(draws);
        let variance = squares[g] / f64::from(draws) - mean * mean;
        let tolerance = 6.0 * (marginal.intercept_variance[g] / f64::from(draws)).sqrt();
        assert!((mean - marginal.intercept_mean[g]).abs() < tolerance, "draw mean {g}");
        assert!(
            (variance / marginal.intercept_variance[g] - 1.0).abs() < 0.15,
            "draw variance {g}: {variance} vs {}",
            marginal.intercept_variance[g]
        );
    }
}

#[test]
fn scores_of_the_family_coordinates_match_analytic_derivatives_of_the_dense_likelihood() {
    let family = PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 0, 1, 1, 1] },
        spatial: None,
    };
    let p = truth(&family);
    let data = simulate(family.clone(), &p, 6, 10, 33);
    let compiled = CompiledPanel::new(&data).unwrap();
    let residuals = hand_residuals(&data, &p);
    let (_, solved, _) = dense(&residuals, &p, None);
    let x = likelihood::encode(&data, &p).unwrap();
    let layout = likelihood::layout(&data);
    let score =
        numerical_score(&x, &context(1), &|x| likelihood::log_likelihood(&compiled, x)).unwrap();
    // Row order is group-major, matching the stacked residuals; d mean / d coordinate per row.
    let rows: Vec<(usize, usize)> = data
        .groups
        .iter()
        .enumerate()
        .flat_map(|(g, group)| (0..group.outcome.len()).map(move |t| (g, t)))
        .collect();
    let shape = |c: usize| p.family.hill_shape[c];
    // d sat / d e, d sat / d log s and d sat / d log k for sat = z / (1 + z), z = (e / k)^s.
    let z = |e: f64, k: f64, s: f64| (e / k).powf(s);
    let exposure = |group: &PanelGroup, c: usize, t: usize| {
        hand_exposure(&group.channels[c], group.history[c], p.channels[c].lag, t)
    };
    let class_k = |g: usize| p.half_saturation_of(data.family.class_of(g));
    let derivative = |which: usize, g: usize, group: &PanelGroup, t: usize| -> f64 {
        match which {
            0 => 1.0,
            1 => group.confounder[t],
            2 => hand_saturation(exposure(group, 1, t), class_k(g), Some(shape(1))),
            3 => {
                let (e, k) = (exposure(group, 0, t), class_k(g));
                if e <= 0.0 {
                    return 0.0;
                }
                let zz = z(e, k, shape(0));
                p.channels[0].coefficient * shape(0) * zz * (e / k).ln() / (1.0 + zz).powi(2)
            }
            4 | 5 => {
                // Half-saturation of class one (4) or class zero (5): only its own groups move.
                let class = if which == 4 { 1 } else { 0 };
                if data.family.class_of(g) != class {
                    return 0.0;
                }
                let k = p.half_saturation_of(class);
                (0..2)
                    .map(|c| {
                        let e = exposure(group, c, t);
                        if e <= 0.0 {
                            return 0.0;
                        }
                        let zz = z(e, k, shape(c));
                        -p.channels[c].coefficient * shape(c) * zz / (1.0 + zz).powi(2)
                    })
                    .sum()
            }
            _ => {
                // Logit of the second channel's geometric decay.
                let theta = 0.3_f64;
                let sum = 1.0 + theta + theta * theta;
                let u = [1.0, theta, theta * theta];
                let du = [0.0, 1.0, 2.0 * theta];
                let dsum = 1.0 + 2.0 * theta;
                let history = group.history[1];
                let value = |lag: usize| {
                    if lag <= t { group.channels[1][t - lag] } else { history[2 + t - lag] }
                };
                let de: f64 = (0..3)
                    .map(|l| (du[l] * sum - u[l] * dsum) / (sum * sum) * value(l))
                    .sum::<f64>()
                    * theta
                    * (1.0 - theta);
                let (e, k) = (exposure(group, 1, t), class_k(g));
                if e <= 0.0 {
                    return 0.0;
                }
                let zz = z(e, k, shape(1));
                p.channels[1].coefficient * shape(1) * zz / (e * (1.0 + zz).powi(2)) * de
            }
        }
    };
    let coordinates = [
        ("intercept", 0, 0),
        ("confounder", 3, 1),
        ("coefficient 1", 2, 2),
        ("hill shape 0", layout.shape, 3),
        ("class one half-saturation", layout.class, 4),
        ("class zero half-saturation", layout.k0, 5),
        ("geometric decay 1", layout.lag + 1, 6),
    ];
    for (name, index, which) in coordinates {
        let analytic: f64 = rows
            .iter()
            .zip(&solved)
            .map(|(&(g, t), s)| s * derivative(which, g, &data.groups[g], t))
            .sum();
        assert!(
            (score[index] - analytic).abs() < 1e-5 * (1.0 + analytic.abs()),
            "{name}: numerical {} vs analytic {analytic}",
            score[index]
        );
    }
}

#[test]
fn geometric_hill_family_recovers_the_response_curves_of_a_simulated_panel() {
    let family = PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        ..PanelFamily::default()
    };
    let p = truth(&family);
    let data = simulate(family.clone(), &p, 30, 30, 101);
    // Start away from the truth: half the coefficients, an unsaturated shape, other decays.
    let mut start = p.clone();
    for channel in &mut start.channels {
        channel.coefficient *= 0.5;
        channel.lag = geo(0.5);
    }
    start.family.hill_shape = vec![1.0, 1.0];
    start.half_saturation *= 1.6;
    let fit = fit_panel(&data, &options(Some(start), 0), &context(1)).unwrap();
    assert!(fit.diagnostics.optimizer_converged, "score {}", fit.diagnostics.score_norm);
    let compiled = CompiledPanel::new(&data).unwrap();
    let truth_ll = likelihood::log_likelihood(&compiled, &likelihood::encode(&data, &p).unwrap());
    assert!(fit.diagnostics.log_likelihood >= truth_ll - 1e-6, "fit must not trail the truth");
    let q = &fit.point.parameters;
    for channel in 0..2 {
        for x in [0.5, 1.0, 2.0, 3.5, 5.0] {
            let (a, b) = (curve(q, channel, 0, x), curve(&p, channel, 0, x));
            assert!((a - b).abs() < 0.3, "channel {channel} at {x}: {a} vs {b}");
        }
    }
    // Carryover is recovered through the geometric decay.
    assert!((q.channels[0].lag[1] / q.channels[0].lag[0] - 0.6).abs() < 0.15);
    assert!((q.channels[1].lag[1] / q.channels[1].lag[0] - 0.3).abs() < 0.15);
    assert!((q.rho - 0.3).abs() < 0.15 && (q.innovation_sd - 0.15).abs() < 0.04);
    assert_eq!(q.family.hill_shape.len(), 2);
    // Pinning the shape at one (Michaelis-Menten) fits the Hill-shaped data strictly worse.
    let reviewed_family =
        PanelFamily { adstock: PanelAdstock::Geometric, ..PanelFamily::default() };
    let reviewed_data = PanelData { family: reviewed_family, ..data };
    let reviewed = fit_panel(&reviewed_data, &options(None, 0), &context(1)).unwrap();
    assert!(
        fit.diagnostics.log_likelihood > reviewed.diagnostics.log_likelihood + 5.0,
        "{} vs {}",
        fit.diagnostics.log_likelihood,
        reviewed.diagnostics.log_likelihood
    );
}

#[test]
fn group_specific_saturation_recovers_each_class_and_beats_one_shared_half_saturation() {
    let family = PanelFamily {
        scope: PanelSaturationScope::Classes { assignment: (0..24).map(|g| g / 12).collect() },
        ..PanelFamily::default()
    };
    let mut p = truth(&family);
    p.half_saturation = 1.0;
    p.family.class_half_saturation = vec![3.5];
    let data = simulate(family, &p, 24, 24, 202);
    let mut start = p.clone();
    start.half_saturation = 2.0;
    start.family.class_half_saturation = vec![2.0];
    for channel in &mut start.channels {
        channel.coefficient *= 0.6;
    }
    let fit = fit_panel(&data, &options(Some(start), 0), &context(2)).unwrap();
    assert!(fit.diagnostics.optimizer_converged);
    let q = &fit.point.parameters;
    let compiled = CompiledPanel::new(&data).unwrap();
    let truth_ll = likelihood::log_likelihood(&compiled, &likelihood::encode(&data, &p).unwrap());
    assert!(fit.diagnostics.log_likelihood >= truth_ll - 1e-6);
    for class in 0..2 {
        for channel in 0..2 {
            for x in [0.5, 1.5, 3.0, 5.0] {
                let (a, b) = (curve(q, channel, class, x), curve(&p, channel, class, x));
                assert!((a - b).abs() < 0.3, "class {class} channel {channel} at {x}: {a} vs {b}");
            }
        }
    }
    assert!(
        (q.family.class_half_saturation[0] / q.half_saturation).ln().abs() > 0.6,
        "classes must separate: {} vs {}",
        q.half_saturation,
        q.family.class_half_saturation[0]
    );
    let shared = PanelData { family: PanelFamily::default(), ..data };
    let pooled = fit_panel(&shared, &options(None, 0), &context(2)).unwrap();
    assert!(
        fit.diagnostics.log_likelihood > pooled.diagnostics.log_likelihood + 20.0,
        "{} vs {}",
        fit.diagnostics.log_likelihood,
        pooled.diagnostics.log_likelihood
    );
    // The reference-world saturation block moves every class parameter, not just the first.
    let mut changed = q.clone();
    changed.family.class_half_saturation[0] *= 2.0;
    assert!(!q.saturation_equal(&changed));
    let mut copied = q.clone();
    copied.copy_saturation_from(&changed);
    assert!(copied.saturation_equal(&changed));
}

#[test]
fn spatially_correlated_intercepts_are_recovered_and_improve_on_independent_groups() {
    let family = PanelFamily { spatial: Some(line(60, 1.0)), ..PanelFamily::default() };
    let mut p = truth(&family);
    p.group_sd = 1.0;
    p.family.spatial_range = 4.0;
    let data = simulate(family, &p, 60, 12, 303);
    let mut start = p.clone();
    start.family.spatial_range = 1.0;
    start.group_sd = 0.5;
    let fit = fit_panel(&data, &options(Some(start), 6), &context(3)).unwrap();
    assert!(fit.diagnostics.optimizer_converged, "score {}", fit.diagnostics.score_norm);
    let compiled = CompiledPanel::new(&data).unwrap();
    let truth_ll = likelihood::log_likelihood(&compiled, &likelihood::encode(&data, &p).unwrap());
    assert!(fit.diagnostics.log_likelihood >= truth_ll - 1e-6);
    let q = &fit.point.parameters;
    assert!(q.family.spatial_range > 0.0 && (q.family.spatial_range / 4.0).ln().abs() < 1.4);
    for channel in 0..2 {
        for x in [0.5, 2.0, 5.0] {
            assert!((curve(q, channel, 0, x) - curve(&p, channel, 0, x)).abs() < 0.3);
        }
    }
    let independent = PanelData { family: PanelFamily::default(), ..data.clone() };
    let flat = fit_panel(&independent, &options(None, 0), &context(3)).unwrap();
    assert!(
        fit.diagnostics.log_likelihood > flat.diagnostics.log_likelihood + 3.0,
        "{} vs {}",
        fit.diagnostics.log_likelihood,
        flat.diagnostics.log_likelihood
    );
    // Frequentist draws use the model-based covariance, not the group-cluster sandwich.
    assert_eq!(fit.diagnostics.cluster_rank_bound, None);
    assert!(fit.diagnostics.uncertainty.contains("model-based"));
    assert_eq!(fit.draws.len(), 6);
    assert!(fit.draws.iter().flatten().count() >= 1);
    for draw in fit.draws.iter().flatten() {
        assert_eq!(draw.group_intercepts.len(), 60);
        assert!(draw.parameters.family.spatial_range > 0.0);
    }
    // Neighbouring groups share their residual level: the fitted intercepts are positively
    // autocorrelated along the line, which independent intercepts need not be.
    let u = &fit.point.group_intercepts;
    let mean = u.iter().sum::<f64>() / 60.0;
    let lag_one: f64 = u.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum();
    let variance: f64 = u.iter().map(|v| (v - mean).powi(2)).sum();
    assert!(lag_one / variance > 0.2, "autocorrelation {}", lag_one / variance);
}

#[test]
fn bayesian_posterior_over_every_family_coordinate_is_sampled_with_shared_axes() {
    let family = PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 0, 1, 1, 1] },
        spatial: Some(line(6, 1.0)),
    };
    let p = truth(&family);
    let data = simulate(family, &p, 6, 12, 404);
    let options = PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 100, gradient_tolerance: 1e-2 },
        inference: PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: PanelSamplingOptions {
                chains: 2,
                warmup: 40,
                draws: 40,
                proposal_scale: 0.3,
                max_rhat: 5.0,
                min_ess: 1.0,
            },
        },
        initial: Some(p.clone()),
    };
    let fit = fit_panel(&data, &options, &context(4)).unwrap();
    assert_eq!(fit.draws.len(), 80);
    assert!(fit.diagnostics.uncertainty.contains("posterior"));
    assert_eq!(fit.diagnostics.parameter_count, likelihood::layout(&data).total);
    assert!(fit.draws.iter().flatten().count() >= 72);
    let compiled = CompiledPanel::new(&data).unwrap();
    for draw in fit.draws.iter().flatten() {
        likelihood::validate_parameters(&compiled, &draw.parameters).unwrap();
        assert_eq!(draw.parameters.family.hill_shape.len(), 2);
        assert_eq!(draw.parameters.family.class_half_saturation.len(), 1);
        assert_eq!(draw.group_intercepts.len(), 6);
    }
    let distinct: std::collections::BTreeSet<u64> = fit
        .draws
        .iter()
        .flatten()
        .map(|draw| draw.parameters.family.spatial_range.to_bits())
        .collect();
    assert!(distinct.len() > 5, "the sampler must move the spatial range");
    // Draws use jointly correlated intercepts: their pairwise sample correlation along the
    // line is positive for neighbours at the posterior's own range.
    let neighbours: f64 = fit
        .draws
        .iter()
        .flatten()
        .map(|draw| draw.group_intercepts.windows(2).map(|w| w[0] * w[1]).sum::<f64>())
        .sum();
    assert!(neighbours.is_finite());
}

#[test]
fn spatial_bootstrap_repeats_the_procedure_with_jointly_drawn_intercepts() {
    let family = PanelFamily { spatial: Some(line(5, 1.0)), ..PanelFamily::default() };
    let p = truth(&family);
    let data = simulate(family, &p, 5, 8, 505);
    let options = PanelFitOptions {
        // A permissive stopping rule fixes the global coordinates at the declared start,
        // isolating the simulation of correlated intercepts from optimizer variability.
        optimizer: PanelOptimizerOptions { max_iterations: 2, gradient_tolerance: 1e100 },
        inference: PanelInference::FrequentistBootstrap { draws: 3 },
        initial: Some(p.clone()),
    };
    let fit = fit_panel(&data, &options, &context(5)).unwrap();
    assert_eq!(fit.draws.len(), 3);
    assert!(fit.draws.iter().all(Option::is_some));
    assert!(fit.diagnostics.uncertainty.contains("bootstrap"));
    let draws: Vec<_> = fit.draws.iter().flatten().collect();
    assert_ne!(draws[0].group_intercepts, draws[1].group_intercepts);
    assert!(draws.iter().all(|draw| draw.parameters.family.spatial_range > 0.0));
}

#[test]
fn reference_world_saturation_block_carries_group_class_and_shape_parameters() {
    let (mut data, mut draw, _) = super::tests::fixture();
    data.family = PanelFamily {
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 1, 1] },
        ..PanelFamily::default()
    };
    draw.parameters.family = PanelFamilyParameters {
        hill_shape: vec![1.4, 1.1],
        class_half_saturation: vec![3.0],
        spatial_range: 0.0,
    };
    let programs = super::tests::programs(&data);
    let before =
        DecompositionEndpoint { program: programs[3].clone(), population_weights: vec![0.25; 4] };
    for component in 0..3 {
        let mut after_draw = draw.clone();
        match component {
            0 => after_draw.parameters.family.class_half_saturation[0] *= 2.0,
            1 => after_draw.parameters.family.hill_shape[0] *= 1.5,
            _ => after_draw.parameters.half_saturation *= 0.5,
        }
        let request = PanelDecompositionRequest {
            before: before.clone(),
            after: before.clone(),
            time_weights: vec![1.0 / 24.0; 24],
            order: ORDER,
        };
        let path = PairedPanelPath { before: vec![draw.clone()], after: vec![after_draw] };
        let result =
            decompose_panel_law(&data, &request, &path, &[], &ExecutionContext::for_tests(9))
                .unwrap();
        assert!(result.point.total.abs() > 1e-6, "change {component} must move the response");
        for j in 0..4 {
            let expected = if j == 1 { result.point.total } else { 0.0 };
            assert!(
                (result.point.shapley[j] - expected).abs() < 1e-10,
                "change {component} must be attributed to saturation only (block {j})"
            );
        }
    }
}

#[test]
fn compiled_expression_follows_group_classes_and_integer_hill_shapes_and_refuses_others() {
    use super::expression::{PanelExpressionControl, compile_panel_expression};
    use antecedent_core::{ValueExpr, VariableId};
    fn value(expression: &ValueExpr, x: f64) -> f64 {
        match expression {
            ValueExpr::Const(n) => n.get(),
            ValueExpr::Var(_) => x,
            ValueExpr::Neg(v) => -value(v, x),
            ValueExpr::Arith { op, lhs, rhs } => op.apply(value(lhs, x), value(rhs, x)),
            _ => panic!("unsupported test expression"),
        }
    }
    let (mut data, mut draw, _) = super::tests::fixture();
    data.family = PanelFamily {
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: vec![0, 1, 1, 0] },
        ..PanelFamily::default()
    };
    draw.parameters.family = PanelFamilyParameters {
        hill_shape: vec![2.0, 3.0],
        class_half_saturation: vec![3.5],
        spatial_range: 0.0,
    };
    let mut program = super::tests::programs(&data).remove(3);
    let ctx = ExecutionContext::for_tests(31);
    let mut weights = vec![0.0; 24];
    weights[2] = 1.0;
    let controls = [PanelExpressionControl {
        channel: "0".into(),
        period: 1,
        variable: VariableId::from_raw(99),
    }];
    let compiled = compile_panel_expression(
        &data,
        std::slice::from_ref(&draw),
        &program,
        &controls,
        &weights,
        &ctx,
    )
    .unwrap();
    for i in 0..24 {
        let x = f64::from(i) / 6.0;
        program.channels[0].values[1] = x;
        let real =
            evaluate_panel_programs(&data, &draw, std::slice::from_ref(&program), &ctx).unwrap();
        assert!((value(&compiled.expression, x) - real[0].means[2]).abs() < 1e-9);
    }
    draw.parameters.family.hill_shape[0] = 1.5;
    let refused = compile_panel_expression(
        &data,
        std::slice::from_ref(&draw),
        &program,
        &controls,
        &weights,
        &ctx,
    );
    assert!(matches!(refused, Err(PanelError::Invalid(_))));
}

#[test]
fn static_hill_and_group_classes_recover_curves_and_beat_shared_saturation() {
    let family = PanelFamily {
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: (0..24).map(|g| g / 12).collect() },
        ..PanelFamily::default()
    };
    let mut p = truth(&family);
    p.rho = 0.0;
    for channel in &mut p.channels {
        channel.lag = [1.0, 0.0, 0.0];
    }
    p.half_saturation = 1.0;
    p.family.class_half_saturation = vec![3.5];
    // A static panel reads only current exposure: weights (1,0,0) simulate it exactly.
    let mut data = simulate(family, &p, 24, 24, 707);
    data.lag_order = 0;
    data.validate().unwrap();
    let layout = likelihood::layout(&data);
    assert_eq!((layout.rho, layout.k0 - layout.lag), (None, 0));
    let compiled = CompiledPanel::new(&data).unwrap();
    let x = likelihood::encode(&data, &p).unwrap();
    assert_eq!(x.len(), layout.total);
    let decoded = likelihood::decode(&data, &x).unwrap();
    assert_eq!((decoded.rho, decoded.channels[0].lag), (0.0, [1.0, 0.0, 0.0]));
    let mut bad = p.clone();
    bad.rho = 0.2;
    assert!(likelihood::validate_parameters(&compiled, &bad).is_err());
    let truth_ll = likelihood::log_likelihood(&compiled, &x);
    let mut start = p.clone();
    for channel in &mut start.channels {
        channel.coefficient *= 0.6;
    }
    start.family.hill_shape = vec![1.0, 1.0];
    start.half_saturation *= 1.8;
    start.family.class_half_saturation = vec![2.0];
    let fit = fit_panel(&data, &options(Some(start), 0), &context(7)).unwrap();
    assert!(fit.diagnostics.optimizer_converged, "score {}", fit.diagnostics.score_norm);
    assert!(fit.diagnostics.log_likelihood >= truth_ll - 1e-6);
    let q = &fit.point.parameters;
    assert_eq!(q.rho, 0.0);
    assert_eq!(q.channels[1].lag, [1.0, 0.0, 0.0]);
    for class in 0..2 {
        for channel in 0..2 {
            for x in [0.5, 1.5, 3.0, 5.0] {
                let (a, b) = (curve(q, channel, class, x), curve(&p, channel, class, x));
                assert!((a - b).abs() < 0.3, "class {class} channel {channel} at {x}: {a} vs {b}");
            }
        }
    }
    let shared = PanelData { family: PanelFamily::default(), ..data.clone() };
    let pooled = fit_panel(&shared, &options(None, 0), &context(7)).unwrap();
    assert!(
        fit.diagnostics.log_likelihood > pooled.diagnostics.log_likelihood + 20.0,
        "{} vs {}",
        fit.diagnostics.log_likelihood,
        pooled.diagnostics.log_likelihood
    );
    // The same static family under an explicit prior: a posterior over every family
    // coordinate with shared group-effect axes and no carryover.
    let bayesian = PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 100, gradient_tolerance: 1e-2 },
        inference: PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: PanelSamplingOptions {
                chains: 2,
                warmup: 30,
                draws: 30,
                proposal_scale: 0.3,
                max_rhat: 5.0,
                min_ess: 1.0,
            },
        },
        initial: Some(q.clone()),
    };
    let posterior = fit_panel(&data, &bayesian, &context(8)).unwrap();
    assert_eq!(posterior.draws.len(), 60);
    assert!(posterior.draws.iter().flatten().count() >= 54);
    for draw in posterior.draws.iter().flatten() {
        likelihood::validate_parameters(&compiled, &draw.parameters).unwrap();
        assert_eq!(draw.parameters.rho, 0.0);
        assert_eq!(draw.parameters.family.hill_shape.len(), 2);
        assert_eq!(draw.parameters.family.class_half_saturation.len(), 1);
        assert_eq!(draw.group_intercepts.len(), 24);
    }
}
