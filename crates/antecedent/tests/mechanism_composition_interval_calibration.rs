//! Whole-method coverage of the first-order delta-method Wald interval of a composed and of a
//! transported linear-Gaussian mechanism system, against the closed-form interventional mean.
//!
//! The procedure scored is the facade end to end: fit each mechanism of a recursive chain by
//! ordinary least squares on simulated rows, hand the fitted coefficients and their
//! sampling-distribution covariance to `StudyBuilder::mechanism_composition` under a declared
//! independence of the separately fitted mechanisms, and read
//! `PreparedMechanismComposition::interval` (and, for the transported system,
//! `transport_to(..).interval`) of `E[Y | do(X = 2)]`. Their construction (gradient, variance
//! `g' Sigma g`, normal quantile) is pinned by hand in `mechanism_composition_facade.rs` and
//! `transported_mechanism_facade.rs`; this file measures whether the interval covers.
//!
//! Design and truth: the chain of those files, `X = 1 + 2 e_X`, `M = 0.5 + 2 X + e_M`,
//! `Y = -1 + 0.5 X + 3 M + 0.5 e_Y`, all errors standard normal. Equation-by-equation OLS
//! estimates of a recursive Gaussian system are asymptotically independent (the likelihood
//! factorizes), which is exactly the declared dependence, and the interventional mean is a smooth
//! function of the parameters, so the first-order interval is asymptotically exact.
//!
//! * `composed`: `E[Y | do(X = 2)] = -1 + 0.5 * 2 + 3 * (0.5 + 2 * 2) = 13.5`.
//! * `transported`: a selection node into `M` only; `M` is refitted in a target sample as
//!   `M = 1.5 + 0.5 X + 2 e_M` while `X` and `Y` are carried from the source, so
//!   `E[Y | do(X = 2)] = -1 + 0.5 * 2 + 3 * (1.5 + 0.5 * 2) = 7.5`. The interval conditions on
//!   the declared transport premises and on the two samples being independent.
//!
//! Sample sizes are on the three-point grid around 400 source rows (200, 400, 800), with
//! a target sample of the same size, `n_sim()` datasets. A dataset whose system the facade refuses
//! is a skip and counts as a miss.
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the first
//! `n_sim()` replicates, rechecked at 2000 near the edge.
//!
//! Ignored tests run via `scripts/measure_calibration_groups.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::analysis::transported_mechanism::MechanismTransportDeclaration;
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismRefit, MechanismSource,
    PreparedMechanismComposition, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{
    ContrastArm, CrossCovarianceBlock, Domain, MeanContrast, MechanismUncertainty, ParameterTerm,
    SelectionNodeDeclaration, UncertaintyMeaning,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim,
};

const BASE_ROWS: usize = 400;
const SEED_COMPOSED: u64 = 0x3C0A_0001;
const SEED_TRANSPORTED: u64 = 0x3C0A_0002;
const COMPOSED_TRUTH: f64 = 13.5;
const TRANSPORTED_TRUTH: f64 = 7.5;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// Ordinary least squares of `y` on the intercept and `columns`: coefficients (intercept first),
/// the residual standard deviation and the row-major covariance `s^2 (Z'Z)^{-1}`.
fn ols(columns: &[&[f64]], y: &[f64]) -> Option<(Vec<f64>, f64, Vec<f64>)> {
    let n = y.len();
    let k = columns.len() + 1;
    let design = |row: usize, col: usize| if col == 0 { 1.0 } else { columns[col - 1][row] };
    let mut gram = vec![0.0; k * k];
    let mut moment = vec![0.0; k];
    for (row, &target) in y.iter().enumerate() {
        for a in 0..k {
            moment[a] += design(row, a) * target;
            for b in 0..k {
                gram[a * k + b] += design(row, a) * design(row, b);
            }
        }
    }
    // Gauss-Jordan inverse of the small Gram matrix.
    let mut inverse = vec![0.0; k * k];
    for i in 0..k {
        inverse[i * k + i] = 1.0;
    }
    let mut work = gram;
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&a, &b| work[a * k + col].abs().total_cmp(&work[b * k + col].abs()))?;
        if work[pivot * k + col].abs() < 1e-12 {
            return None;
        }
        for c in 0..k {
            work.swap(col * k + c, pivot * k + c);
            inverse.swap(col * k + c, pivot * k + c);
        }
        let scale = work[col * k + col];
        for c in 0..k {
            work[col * k + c] /= scale;
            inverse[col * k + c] /= scale;
        }
        for r in 0..k {
            if r != col {
                let factor = work[r * k + col];
                for c in 0..k {
                    work[r * k + c] -= factor * work[col * k + c];
                    inverse[r * k + c] -= factor * inverse[col * k + c];
                }
            }
        }
    }
    let coefficients: Vec<f64> =
        (0..k).map(|a| (0..k).map(|b| inverse[a * k + b] * moment[b]).sum()).collect();
    let rss: f64 = (0..n)
        .map(|row| {
            let fitted: f64 = (0..k).map(|a| coefficients[a] * design(row, a)).sum();
            (y[row] - fitted).powi(2)
        })
        .sum();
    let variance = rss / (n - k) as f64;
    Some((coefficients, variance.sqrt(), inverse.iter().map(|v| v * variance).collect()))
}

fn role(variable: &str) -> QuantityRole {
    match variable {
        "X" => QuantityRole::Treatment,
        "M" => QuantityRole::Mediator,
        "Y" => QuantityRole::Outcome,
        _ => QuantityRole::Covariate,
    }
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: format!("label of {variable}"),
        role: role(variable),
        units: "u".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration_of(variable: &str, parents: &[&str]) -> MechanismDeclaration {
    MechanismDeclaration {
        output: quantity(variable),
        parents: parents.iter().map(|p| quantity(p)).collect(),
        conditioning_variable_ids: parents.iter().map(|p| (*p).to_owned()).collect(),
        origin: MechanismOriginKind::Fitted,
        artifact_id: format!("fit:{variable}"),
        evidence_id: format!("data:{variable}"),
        latent_causes: Vec::new(),
    }
}

/// A fitted linear mechanism with the sampling covariance of its OLS parameters.
fn fitted(
    variable: &str,
    parents: &[&str],
    coefficients: &[f64],
    sigma: f64,
    covariance: Vec<f64>,
) -> MechanismInput {
    let slot = MechanismSlot::LinearGaussian {
        intercept: coefficients[0],
        coeffs: Arc::from(coefficients[1..].to_vec()),
        sigma,
    };
    let mut parameters = vec![ParameterTerm::Intercept];
    parameters.extend(parents.iter().map(|p| ParameterTerm::Coefficient((*p).to_owned())));
    MechanismInput {
        declaration: declaration_of(variable, parents),
        source: MechanismSource::from_fitted_slot(&slot, parents).unwrap(),
        output_support: Domain::Real,
        parent_domains: parents.iter().map(|p| ((*p).to_owned(), Domain::Real)).collect(),
        uncertainty: Some(MechanismUncertainty {
            meaning: UncertaintyMeaning::SamplingDistribution,
            parameters,
            covariance,
        }),
        latent_loadings: Vec::new(),
    }
}

fn at_x(x: f64) -> MeanContrast {
    MeanContrast {
        outcome: "Y".into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("X".into(), x)] }],
    }
}

fn independent() -> DependenceInput {
    DependenceInput::IndependentByDesign {
        justification: "separately fitted mechanisms of a recursive Gaussian system".into(),
    }
}

/// The asymptotic independence of equation-by-equation OLS estimates as an explicit joint
/// declaration: a zero cross-covariance block for every pair of mechanisms. A bare
/// independence declaration is refused for a functional that multiplies coefficients along a
/// path, so the zero blocks are what licenses the first-order variance here.
fn zero_cross_covariance(mechanisms: &[MechanismInput]) -> DependenceInput {
    let mut ordered: Vec<&MechanismInput> = mechanisms.iter().collect();
    ordered.sort_by(|a, b| a.declaration.output.variable_id.cmp(&b.declaration.output.variable_id));
    let mut blocks = Vec::new();
    for (i, first) in ordered.iter().enumerate() {
        for second in &ordered[i + 1..] {
            let terms = |m: &MechanismInput| {
                m.uncertainty.as_ref().map(|u| u.parameters.clone()).unwrap_or_default()
            };
            let (first_parameters, second_parameters) = (terms(first), terms(second));
            blocks.push(CrossCovarianceBlock {
                first: first.declaration.output.variable_id.clone(),
                second: second.declaration.output.variable_id.clone(),
                values: vec![0.0; first_parameters.len() * second_parameters.len()],
                first_parameters,
                second_parameters,
            });
        }
    }
    DependenceInput::CrossCovariance { blocks }
}

/// Source sample of the chain and its fitted mechanisms.
fn source_mechanisms(n: usize, seed: u64) -> Option<Vec<MechanismInput>> {
    let mut noise = gaussian(seed);
    let (mut x, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let xi = 1.0 + 2.0 * noise();
        let mi = 0.5 + 2.0 * xi + noise();
        y.push(-1.0 + 0.5 * xi + 3.0 * mi + 0.5 * noise());
        x.push(xi);
        m.push(mi);
    }
    let (x_coef, x_sigma, x_cov) = ols(&[], &x)?;
    let (m_coef, m_sigma, m_cov) = ols(&[x.as_slice()], &m)?;
    let (y_coef, y_sigma, y_cov) = ols(&[x.as_slice(), m.as_slice()], &y)?;
    Some(vec![
        fitted("X", &[], &x_coef, x_sigma, x_cov),
        fitted("M", &["X"], &m_coef, m_sigma, m_cov),
        fitted("Y", &["X", "M"], &y_coef, y_sigma, y_cov),
    ])
}

fn prepare(
    mechanisms: Vec<MechanismInput>,
    dependence: DependenceInput,
) -> Option<PreparedMechanismComposition> {
    let mut edges = Vec::new();
    for mechanism in &mechanisms {
        for parent in &mechanism.declaration.parents {
            edges.push((
                parent.variable_id.clone(),
                mechanism.declaration.output.variable_id.clone(),
            ));
        }
    }
    StudyBuilder::mechanism_composition(
        MechanismCompositionRequest {
            target_population_id: "pop".into(),
            target_regime_id: "observational".into(),
            mechanisms,
            edges,
            admissions: Vec::new(),
            dependence,
            admit_declared_mechanisms: false,
        },
        &ctx(),
    )
    .ok()
}

fn composed_interval(n: usize, seed: u64) -> Option<(f64, f64)> {
    let mechanisms = source_mechanisms(n, seed)?;
    let dependence = zero_cross_covariance(&mechanisms);
    let source = prepare(mechanisms, dependence)?;
    let band = source.interval(&at_x(2.0), REPORTED_LEVEL).ok()?;
    Some((band.lower, band.upper))
}

fn transported_interval(n: usize, seed: u64) -> Option<(f64, f64)> {
    let source = prepare(source_mechanisms(n, seed)?, independent())?;
    // The target sample: X as in the source, M = 1.5 + 0.5 X + 2 e_M.
    let mut noise = gaussian(seed ^ 0x7A3C_9E01);
    let (mut x, mut m) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let xi = 1.0 + 2.0 * noise();
        m.push(1.5 + 0.5 * xi + 2.0 * noise());
        x.push(xi);
    }
    let (coef, sigma, cov) = ols(&[x.as_slice()], &m)?;
    let refit = MechanismRefit {
        variable_id: "M".into(),
        source: MechanismSource::FittedLinear {
            intercept: coef[0],
            sigma,
            coefficients: vec![("X".into(), coef[1])],
        },
        artifact_id: "fit:M:target".into(),
        evidence_id: "data:M:target".into(),
        uncertainty: Some(MechanismUncertainty {
            meaning: UncertaintyMeaning::SamplingDistribution,
            parameters: vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())],
            covariance: cov,
        }),
    };
    let declaration = MechanismTransportDeclaration {
        target_population_id: "clinic_b".into(),
        target_regime_id: "observational".into(),
        selection_nodes: vec![SelectionNodeDeclaration {
            name: "S".into(),
            points_into: vec!["M".to_owned()],
        }],
        target_refits: vec![refit],
        invariances: Vec::new(),
        dependence: independent(),
    };
    let transported = source.transport_to(&declaration, &ctx()).ok()?;
    let band = transported.interval(&at_x(2.0), REPORTED_LEVEL).ok()?;
    Some((band.lower, band.upper))
}

fn construction(query: &str, estimator: &str) -> Construction {
    Construction {
        query: query.into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: "delta_method_wald".into(),
        se_kind: "first_order_parameter_variance".into(),
        dependence: "independent_by_design".into(),
        posterior: String::new(),
        functional: "mean_contrast_do_x_2".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn measure(
    test: &'static str,
    dgp: &'static str,
    construction: &Construction,
    truth: f64,
    seed_base: u64,
    interval: fn(usize, u64) -> Option<(f64, f64)>,
) {
    let n = grid_n(BASE_ROWS);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: "delta_method_wald" },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| interval(n, grid_seed(seed_base + rep)));
    for run in runs {
        let Some(band) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            construction,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(band), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn composed_linear_gaussian_mechanisms_first_order_interval() {
    measure(
        "composed_linear_gaussian_mechanisms_first_order_interval",
        "recursive_gaussian_chain_closed_form_interventional_mean",
        &construction(
            "MechanismCompositionContrast",
            "mechanism_composition.first_order_delta_method",
        ),
        COMPOSED_TRUTH,
        SEED_COMPOSED,
        composed_interval,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn transported_linear_gaussian_mechanisms_first_order_interval() {
    measure(
        "transported_linear_gaussian_mechanisms_first_order_interval",
        "recursive_gaussian_chain_with_refitted_target_mechanism_closed_form_interventional_mean",
        &construction(
            "TransportedMechanismContrast",
            "mechanism_transport.first_order_delta_method",
        ),
        TRANSPORTED_TRUTH,
        SEED_TRANSPORTED,
        transported_interval,
    );
}

/// One replicate of each construction, quickly: the interval is ordered and finite, and its
/// centre sits within a few half-widths of the closed-form interventional mean.
#[test]
fn one_replicate_reports_a_valid_interval_near_the_closed_form_mean() {
    for (band, truth) in [
        (composed_interval(BASE_ROWS, grid_seed(SEED_COMPOSED)), COMPOSED_TRUTH),
        (transported_interval(BASE_ROWS, grid_seed(SEED_TRANSPORTED)), TRANSPORTED_TRUTH),
    ] {
        let (lower, upper) = band.expect("the composition runs");
        assert!(lower.is_finite() && upper.is_finite() && lower < upper, "[{lower}, {upper}]");
        let (centre, half) = (0.5 * (lower + upper), 0.5 * (upper - lower));
        assert!((centre - truth).abs() < 5.0 * half, "centre {centre} vs {truth}, half {half}");
    }
}
