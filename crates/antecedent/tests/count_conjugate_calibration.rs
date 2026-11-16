//! Coverage of the conjugate count posteriors of a Bayesian association, through the study.
//!
//! The exceedance probability `P(Y > t)` and the Poisson rate of a count outcome are drawn from
//! the stated prior (`Beta(2, 3)` and `Gamma(3, 1.5)`), the rows are simulated from that truth,
//! and the equal-tailed 95% credible interval of the single cell the association reports is
//! scored against it. Averaged over the prior the interval has exact coverage at the reported
//! level, so the two-sided band applies. The designs are `#[ignore]`d measurement designs
//! registered in `parity/calibration_groups.toml`; the same procedures called directly are
//! measured in `crates/antecedent-estimate/tests/conjugate_prior_calibration.rs`, and the
//! closed-form oracles of the route are `association_count_conjugate.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]

mod common;

use antecedent::inference::{BetaHyperparameters, GammaHyperparameters, PriorSet, PriorSpec};
use antecedent::{
    AssociationFunctional, AssociationOutcome, AssociationQuery, BayesianConfig, CausalQuery,
    InferenceMode, Study,
};
use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_kernels::sample_gamma;
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, grid_n, grid_seed, map_replicates, n_sim,
};

const SEED_BETA: u64 = 4_201_001;
const SEED_GAMMA: u64 = 4_201_101;
const BETA_TRUE: BetaHyperparameters = BetaHyperparameters { alpha: 2.0, beta: 3.0 };
const GAMMA_TRUE: GammaHyperparameters = GammaHyperparameters { shape: 3.0, rate: 1.5 };
const THRESHOLD: f64 = 0.5;

fn draw_poisson(rate: f64, rng: &mut CausalRng) -> u64 {
    let limit = (-rate).exp();
    let mut count = 0;
    let mut product = rng.next_f64();
    while product > limit {
        count += 1;
        product *= rng.next_f64();
    }
    count
}

fn run(
    data: TabularData,
    query: AssociationQuery,
    prior: PriorSet,
    seed: u64,
) -> Option<(f64, f64)> {
    let study = Study::tabular(data)
        .query(CausalQuery::Association(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().prior(prior)))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let posterior = result.association()?.estimate.terms.first()?.posterior?;
    Some((posterior.lower, posterior.upper))
}

fn exceedance_replicate(n: usize, seed: u64) -> Option<((f64, f64), f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let x = sample_gamma(BETA_TRUE.alpha, 1.0, &mut rng);
    let z = sample_gamma(BETA_TRUE.beta, 1.0, &mut rng);
    let theta = x / (x + z);
    let y: Vec<f64> =
        (0..n).map(|i| if rng.next_f64() < theta { 0.8 + 1e-3 * i as f64 } else { 0.1 }).collect();
    let data = TabularData::from_f64_columns([("y", &y[..])]).ok()?;
    let query = AssociationQuery::new(VariableId::from_raw(0), Vec::<VariableId>::new())
        .with_functional(AssociationFunctional::exceedance(THRESHOLD));
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::Beta(BETA_TRUE));
    run(data, query, prior, seed).map(|interval| (interval, theta))
}

fn rate_replicate(n: usize, seed: u64) -> Option<((f64, f64), f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let lambda = sample_gamma(GAMMA_TRUE.shape, GAMMA_TRUE.rate, &mut rng);
    let y: Vec<f64> = (0..n).map(|_| draw_poisson(lambda, &mut rng) as f64).collect();
    let data = TabularData::from_f64_columns([("y", &y[..])]).ok()?;
    let query = AssociationQuery::new(VariableId::from_raw(0), Vec::<VariableId>::new())
        .with_functional(AssociationFunctional::Conditional)
        .with_outcome_scale(AssociationOutcome::Count);
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::Gamma(GAMMA_TRUE));
    run(data, query, prior, seed).map(|interval| (interval, lambda))
}

#[test]
fn the_route_reports_the_closed_form_interval_for_a_known_cell() {
    let (interval, theta) = exceedance_replicate(40, 7).expect("an exceedance cell");
    assert!(interval.0 < interval.1);
    assert!((0.0..=1.0).contains(&theta));
    let (interval, lambda) = rate_replicate(30, 7).expect("a rate cell");
    assert!(interval.0 < interval.1 && lambda > 0.0);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn exceedance_route_covers_under_the_stated_prior() {
    let n = grid_n(40);
    let mut tally = CoverageTally::new("association_exceedance_stated_prior", REPORTED_LEVEL);
    for row in map_replicates(n_sim(), |rep| exceedance_replicate(n, SEED_BETA + rep)) {
        match row {
            Some((interval, truth)) => tally.record(Some(interval), truth),
            None => tally.skip(),
        }
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn count_rate_route_covers_under_the_stated_prior() {
    let n = grid_n(30);
    let mut tally = CoverageTally::new("association_count_rate_stated_prior", REPORTED_LEVEL);
    for row in map_replicates(n_sim(), |rep| rate_replicate(n, SEED_GAMMA + rep)) {
        match row {
            Some((interval, truth)) => tally.record(Some(interval), truth),
            None => tally.skip(),
        }
    }
    tally.assert();
}
