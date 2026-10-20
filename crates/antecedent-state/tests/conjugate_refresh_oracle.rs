//! Sequential conjugate updating equals batch updating, and double use or a prior change is
//! refused. Reference values come from the direct batch formulas
//! `Λ = Λ₀ + XᵀX`, `m = Λ⁻¹(Λ₀m₀ + Xᵀy)`, `a = a₀ + n/2`,
//! `b = b₀ + ½(|y − Xm|² + (m − m₀)ᵀΛ₀(m − m₀))` evaluated with numpy on the stacked rows.

use antecedent_state::{
    ConjugateOlsPosterior, ConjugateOlsPrior, ConjugateOlsSummary, LinearOlsSuffStats, StateError,
};

const A: ([f64; 5], [f64; 5]) = ([0.5, 1.2, -0.3, 2.0, 1.5], [1.1, 2.4, 0.2, 3.9, 3.1]);
const B: ([f64; 4], [f64; 4]) = ([-1.0, 0.8, 2.2, 0.1], [-0.9, 1.7, 4.6, 0.9]);
const C: ([f64; 3], [f64; 3]) = ([1.7, -0.6, 0.3], [3.2, -0.1, 1.0]);

fn block(x: &[f64], y: &[f64]) -> LinearOlsSuffStats {
    let mut stats = LinearOlsSuffStats::new(2);
    for (x, y) in x.iter().zip(y) {
        stats.append_row(&[1.0, *x], *y).unwrap();
    }
    stats
}

fn a() -> LinearOlsSuffStats {
    block(&A.0, &A.1)
}
fn b() -> LinearOlsSuffStats {
    block(&B.0, &B.1)
}
fn c() -> LinearOlsSuffStats {
    block(&C.0, &C.1)
}

fn prior_one() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(vec![0.0, 0.0], vec![0.5, 0.0, 0.0, 0.25], 3.0, 2.0).unwrap()
}

fn prior_two() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(vec![1.0, 1.0], vec![2.0, 0.3, 0.3, 1.5], 2.5, 1.2).unwrap()
}

fn close(actual: f64, expected: f64, label: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{label}: {actual:.17e} vs {expected:.17e}");
}

fn assert_summary(
    got: &ConjugateOlsSummary,
    mean: [f64; 2],
    precision: [f64; 4],
    shape: f64,
    rate: f64,
) {
    for (g, e) in got.mean.iter().zip(mean) {
        close(*g, e, "mean");
    }
    for (g, e) in got.precision.iter().zip(precision) {
        close(*g, e, "precision");
    }
    close(got.shape, shape, "shape");
    close(got.rate, rate, "rate");
}

const P1_ABC: ([f64; 2], [f64; 4], f64, f64) = (
    [0.614_607_066_071_943_9, 1.597_310_913_583_416],
    [12.5, 8.4, 8.4, 18.110_000_000_000_003],
    9.0,
    2.664_730_930_911_657_6,
);
const P1_AB: ([f64; 2], [f64; 4], f64, f64) = (
    [0.581_065_542_353_391_9, 1.639_982_478_234_682],
    [9.5, 7.0, 7.0, 14.770_000_000_000_003],
    7.5,
    2.613_390_735_366_588,
);
const P2_ABC: ([f64; 2], [f64; 4], f64, f64) = (
    [0.720_660_353_212_183, 1.529_971_845_405_681_7],
    [14.0, 8.700_000_000_000_001, 8.700_000_000_000_001, 19.360_000_000_000_003],
    8.5,
    1.737_929_101_612_490_3,
);

fn sequential(prior: &ConjugateOlsPrior) -> ConjugateOlsPosterior {
    let mut posterior = ConjugateOlsPosterior::new(prior);
    posterior.update(prior, "a", &a()).unwrap();
    posterior.update(prior, "b", &b()).unwrap();
    posterior.update(prior, "c", &c()).unwrap();
    posterior
}

#[test]
fn sequential_blocks_equal_the_batch_posterior() {
    for (prior, (mean, precision, shape, rate)) in [(prior_one(), P1_ABC), (prior_two(), P2_ABC)] {
        let summary = sequential(&prior).summary(&prior).unwrap();
        assert_summary(&summary, mean, precision, shape, rate);
        assert_eq!(summary.n, 12);
    }
}

#[test]
fn a_pooled_block_equals_the_sequential_blocks() {
    let prior = prior_one();
    let mut pooled = a();
    pooled.merge(&b()).unwrap();
    pooled.merge(&c()).unwrap();
    let mut single = ConjugateOlsPosterior::new(&prior);
    single.update(&prior, "all", &pooled).unwrap();
    let batch = single.summary(&prior).unwrap();
    let chained = sequential(&prior).summary(&prior).unwrap();
    for (x, y) in batch.mean.iter().zip(&chained.mean) {
        close(*x, *y, "mean");
    }
    close(batch.rate, chained.rate, "rate");
    assert_summary(&batch, P1_ABC.0, P1_ABC.1, P1_ABC.2, P1_ABC.3);
}

#[test]
fn a_posterior_used_as_a_prior_continues_to_the_same_posterior() {
    let prior = prior_one();
    let mut first = ConjugateOlsPosterior::new(&prior);
    first.update(&prior, "a", &a()).unwrap();
    first.update(&prior, "b", &b()).unwrap();
    assert_summary(&first.summary(&prior).unwrap(), P1_AB.0, P1_AB.1, P1_AB.2, P1_AB.3);
    let carried = first.to_prior().unwrap();
    let mut second = ConjugateOlsPosterior::new(&carried);
    second.update(&carried, "c", &c()).unwrap();
    assert_summary(&second.summary(&carried).unwrap(), P1_ABC.0, P1_ABC.1, P1_ABC.2, P1_ABC.3);
}

#[test]
fn the_same_observations_twice_are_refused_and_change_nothing() {
    let prior = prior_one();
    let mut posterior = sequential(&prior);
    let before = posterior.summary(&prior).unwrap();
    assert!(matches!(posterior.update(&prior, "b", &b()), Err(StateError::InvalidEvent(_))));
    // A different block under a used label is a relabelled source, also refused.
    assert!(posterior.update(&prior, "b", &c()).is_err());
    assert_eq!(posterior.summary(&prior).unwrap(), before);
}

#[test]
fn reusing_a_block_through_the_prior_and_a_later_update_is_refused() {
    let prior = prior_one();
    let mut first = ConjugateOlsPosterior::new(&prior);
    first.update(&prior, "a", &a()).unwrap();
    let carried = first.to_prior().unwrap();
    assert_eq!(carried.used_labels().collect::<Vec<_>>(), vec!["a"]);
    let mut second = ConjugateOlsPosterior::new(&carried);
    let before = second.summary(&carried).unwrap();
    assert!(matches!(second.update(&carried, "a", &a()), Err(StateError::InvalidEvent(_))));
    assert_eq!(second.summary(&carried).unwrap(), before);
    // The independent block still updates.
    second.update(&carried, "b", &b()).unwrap();
    assert_summary(&second.summary(&carried).unwrap(), P1_AB.0, P1_AB.1, P1_AB.2, P1_AB.3);
}

#[test]
fn an_invalidated_or_empty_or_wrong_width_block_is_refused() {
    let prior = prior_one();
    let mut posterior = ConjugateOlsPosterior::new(&prior);
    let mut stale = a();
    let _ = stale.apply_delta(antecedent_state::LinearOlsDelta::MaskChanged);
    assert!(posterior.update(&prior, "stale", &stale).is_err());
    assert!(posterior.update(&prior, "empty", &LinearOlsSuffStats::new(2)).is_err());
    assert!(posterior.update(&prior, "wide", &a_wide()).is_err());
    assert_eq!(posterior.summary(&prior).unwrap().n, 0);
}

fn a_wide() -> LinearOlsSuffStats {
    let mut stats = LinearOlsSuffStats::new(3);
    stats.append_row(&[1.0, 2.0, 3.0], 1.0).unwrap();
    stats
}

#[test]
fn a_prior_change_is_refused_and_reprior_rebuilds_from_every_block() {
    let one = prior_one();
    let two = prior_two();
    let mut posterior = ConjugateOlsPosterior::new(&one);
    posterior.update(&one, "a", &a()).unwrap();
    posterior.update(&one, "b", &b()).unwrap();
    // Another prior neither reads nor extends the posterior built under the first.
    assert!(matches!(posterior.summary(&two), Err(StateError::StaleCache(_))));
    assert!(matches!(posterior.update(&two, "c", &c()), Err(StateError::StaleCache(_))));
    assert_eq!(posterior.summary(&one).unwrap().n, 9);
    // The rebuild is the batch posterior under the new prior (numpy: P2 on A and B).
    let rebuilt = posterior.reprior(&two, &[("a", &a()), ("b", &b())]).unwrap();
    assert_summary(
        &rebuilt.summary(&two).unwrap(),
        [0.728_292_524_200_764_8, 1.546_408_525_176_929_5],
        [11.0, 7.3, 7.3, 16.020_000_000_000_003],
        7.0,
        1.706_260_880_175_709_5,
    );
    // Missing, extra or altered blocks are refused.
    assert!(posterior.reprior(&two, &[("a", &a())]).is_err());
    assert!(posterior.reprior(&two, &[("a", &a()), ("b", &b()), ("c", &c())]).is_err());
    assert!(posterior.reprior(&two, &[("a", &a()), ("b", &c())]).is_err());
}

#[test]
fn a_prior_that_already_used_a_block_cannot_be_the_rebuild_prior() {
    let one = prior_one();
    let mut over_a = ConjugateOlsPosterior::new(&one);
    over_a.update(&one, "a", &a()).unwrap();
    let carried = over_a.to_prior().unwrap();
    let mut posterior = ConjugateOlsPosterior::new(&one);
    posterior.update(&one, "a", &a()).unwrap();
    assert!(posterior.reprior(&carried, &[("a", &a())]).is_err());
}

#[test]
fn priors_are_validated() {
    assert!(ConjugateOlsPrior::new(vec![0.0, 0.0], vec![1.0, 0.5, 0.0, 1.0], 1.0, 1.0).is_err());
    assert!(ConjugateOlsPrior::new(vec![0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0], 1.0, 1.0).is_err());
    assert!(ConjugateOlsPrior::new(vec![0.0], vec![1.0], 0.0, 1.0).is_err());
    assert!(ConjugateOlsPrior::new(vec![0.0], vec![1.0], 1.0, f64::NAN).is_err());
    assert!(ConjugateOlsPrior::new(vec![0.0, 1.0], vec![1.0], 1.0, 1.0).is_err());
    assert!(ConjugateOlsPrior::isotropic(2, -1.0, 1.0, 1.0).is_err());
    let iso = ConjugateOlsPrior::isotropic(2, 4.0, 3.0, 2.0).unwrap();
    assert_ne!(iso.digest(), prior_one().digest());
}
