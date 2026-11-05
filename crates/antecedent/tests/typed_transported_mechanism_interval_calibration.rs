//! Whole-method coverage of the first-order delta-method Wald interval of a typed transported
//! mechanism system, against the closed-form interventional mean of the target population.
//!
//! The procedure scored is the facade end to end: fit a Gaussian covariate mechanism and a
//! Poisson count mechanism (log link, quadratic mean) in a source sample, refit both in an
//! independent target sample, declare a selection node into the count, carry the invariant
//! covariate and use the target count fit, and read `TypedTransportedComposition::interval`
//! of `E[n | do(x = 0.5)]`. The interval is the first-order Wald band of the parameter variance
//! around a Monte Carlo evaluation of the contrast (20 000 draws; the Monte Carlo error is
//! reported apart from the parameter variance and is two orders below it at these sizes). Its
//! construction is pinned by `typed_transported_mechanism_facade.rs` and the independent delta
//! method of `antecedent-model`'s `typed_mechanism_transport.rs`; this file measures whether
//! it covers.
//!
//! Design and truth: source `x ~ N(0, 1)`, `n | x ~ Poisson(exp(0.2 + 0.3 x))`; target `x`
//! unchanged and `n | x ~ Poisson(exp(0.5 + 0.2 x + 0.25 x^2))`. The target count model is
//! correctly specified, so with `x` intervened `E[n | do(x)] = exp(0.5 + 0.2 x + 0.25 x^2)`,
//! which at `x = 0.5` is `exp(0.6625)`. Source and target samples have equal size on the
//! three-point grid around 1000 rows (500, 1000, 2000), `n_sim()` datasets. A dataset the facade
//! refuses (a failed fit or transport) is a skip and counts as a miss.
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the first
//! `n_sim()` replicates, rechecked at 2000 near the edge. The interval conditions on the
//! declared transport premises and on the two samples being independent.
//!
//! Ignored tests run via `scripts/measure_calibration_groups.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::collections::BTreeMap;

use antecedent::analysis::typed_transported_mechanism::{
    FittedMechanism, MeanSpec, MechanismSpec, ParameterDependence, ResponseFamily,
    TypedTransportRequest, fit_typed_mechanism, refit_typed_mechanism_in_target,
    transport_typed_mechanisms,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_model::compose::{
    ContrastArm, MeanContrast, MechanismContext, MechanismOrigin, MechanismProvenance,
    SelectionDiagramDeclaration, SelectionNodeDeclaration,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};

const BASE_ROWS: usize = 1000;
const SEED_BASE: u64 = 0x7E9D_0001;
const DRAWS: usize = 20_000;
const AT_X: f64 = 0.5;

type Columns = BTreeMap<String, Vec<f64>>;

fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn poisson(lambda: f64, rng: &mut CausalRng) -> f64 {
    let limit = (-lambda).exp();
    let mut p = 1.0;
    let mut k = 0.0;
    loop {
        p *= rng.next_f64();
        if p <= limit {
            return k;
        }
        k += 1.0;
    }
}

fn population(shifted: bool, n: usize, seed: u64) -> Columns {
    let mut rng = CausalRng::from_seed(seed);
    let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let counts: Vec<f64> = x
        .iter()
        .map(|v| {
            let eta = if shifted { 0.5 + 0.2 * v + 0.25 * v * v } else { 0.2 + 0.3 * v };
            poisson(eta.exp(), &mut rng)
        })
        .collect();
    let mut cols = Columns::new();
    cols.insert("x".into(), x);
    cols.insert("n".into(), counts);
    cols
}

fn context(population: &str) -> MechanismContext {
    MechanismContext { population: population.into(), regime: "observational".into() }
}

fn provenance(population: &str, variable: &str) -> MechanismProvenance {
    MechanismProvenance {
        origin: MechanismOrigin::Fitted,
        artifact_id: format!("fit:{population}:{variable}"),
        evidence_id: format!("evidence:{population}"),
    }
}

fn spec(variable: &str) -> MechanismSpec {
    if variable == "x" {
        MechanismSpec {
            variable: "x".into(),
            parents: vec![],
            mean: MeanSpec::Polynomial { degree: 1, interactions: false },
            family: ResponseFamily::Gaussian,
        }
    } else {
        MechanismSpec {
            variable: "n".into(),
            parents: vec!["x".into()],
            mean: MeanSpec::Polynomial { degree: 2, interactions: false },
            family: ResponseFamily::Poisson,
        }
    }
}

/// The fitted source and target mechanisms of one simulated pair of samples.
fn fits(
    n: usize,
    seed: u64,
) -> Option<(FittedMechanism, FittedMechanism, FittedMechanism, FittedMechanism)> {
    let ctx = ExecutionContext::for_tests(1);
    let source = population(false, n, seed);
    let target = population(true, n, seed ^ 0x51ED_2701);
    let source_x = fit_typed_mechanism(
        &spec("x"),
        &source,
        context("source"),
        provenance("source", "x"),
        None,
        &ctx,
    )
    .ok()?;
    let source_n = fit_typed_mechanism(
        &spec("n"),
        &source,
        context("source"),
        provenance("source", "n"),
        None,
        &ctx,
    )
    .ok()?;
    let target_x = refit_typed_mechanism_in_target(
        &source_x,
        &target,
        context("target"),
        provenance("target", "x"),
        &ctx,
    )
    .ok()?;
    let target_n = refit_typed_mechanism_in_target(
        &source_n,
        &target,
        context("target"),
        provenance("target", "n"),
        &ctx,
    )
    .ok()?;
    Some((source_x, source_n, target_x, target_n))
}

fn level(x: f64) -> MeanContrast {
    MeanContrast {
        outcome: "n".into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("x".into(), x)] }],
    }
}

fn interval(n: usize, seed: u64) -> Option<(f64, f64)> {
    let (source_x, source_n, target_x, target_n) = fits(n, seed)?;
    let request = TypedTransportRequest {
        target: context("target"),
        diagram: SelectionDiagramDeclaration {
            source_population: "source".into(),
            target_population: "target".into(),
            selection_nodes: vec![SelectionNodeDeclaration {
                name: "S".into(),
                points_into: vec!["n".to_owned()],
            }],
        },
        source_mechanisms: vec![source_x, source_n],
        target_mechanisms: vec![target_x, target_n],
        invariances: vec![],
        dependence: ParameterDependence::IndependentByDesign {
            justification: "disjoint independently generated samples".into(),
        },
        invariance_level: 0.01,
    };
    let ctx = ExecutionContext::for_tests(seed);
    let composition = transport_typed_mechanisms(&request, &ctx).ok()?;
    let band = composition.interval(&level(AT_X), DRAWS, &ctx, REPORTED_LEVEL).ok()?;
    Some((band.lower, band.upper))
}

fn construction() -> Construction {
    Construction {
        query: "TypedTransportedMechanismContrast".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "typed_mechanism_transport.first_order_delta_method".into(),
        interval_method: "delta_method_wald".into(),
        se_kind: "first_order_parameter_variance".into(),
        dependence: "independent_by_design".into(),
        posterior: String::new(),
        functional: "mean_contrast_do_x_0_5".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn typed_transported_poisson_mechanism_first_order_interval() {
    let n = grid_n(BASE_ROWS);
    let truth = (0.5 + 0.2 * AT_X + 0.25 * AT_X * AT_X).exp();
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "typed_transported_poisson_mechanism_first_order_interval",
            dgp: "source_and_shifted_target_poisson_count_closed_form_interventional_mean",
            interval: "delta_method_wald",
        },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| interval(n, grid_seed(SEED_BASE + rep)));
    for run in runs {
        let Some(band) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction(),
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
