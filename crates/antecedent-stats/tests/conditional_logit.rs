//! Conditional logit: closed-form maxima, invariances, derivative checks, the cluster sandwich,
//! varying set sizes and every refusal.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CancellationToken, ExecutionContext};
use antecedent_stats::{
    ConditionalLogitDesign, ConditionalLogitError, ConditionalLogitFit, ConditionalLogitOptions,
    conditional_logit_log_likelihood, fit_conditional_logit, fit_conditional_logit_with_context,
};

/// Owned choice data: row-major features, set boundaries, chosen indices and cluster labels.
#[derive(Clone, Debug)]
struct Data {
    features: Vec<f64>,
    p: usize,
    starts: Vec<usize>,
    chosen: Vec<usize>,
    clusters: Option<Vec<u32>>,
}

impl Data {
    fn design(&self) -> ConditionalLogitDesign<'_> {
        ConditionalLogitDesign {
            features: &self.features,
            n_features: self.p,
            set_starts: &self.starts,
            chosen: &self.chosen,
            clusters: self.clusters.as_deref(),
        }
    }

    fn n_sets(&self) -> usize {
        self.chosen.len()
    }

    fn set_rows(&self, s: usize) -> Vec<&[f64]> {
        (self.starts[s]..self.starts[s + 1])
            .map(|r| &self.features[r * self.p..(r + 1) * self.p])
            .collect()
    }

    fn from_sets(p: usize, sets: &[(Vec<Vec<f64>>, usize)], clusters: Option<Vec<u32>>) -> Self {
        let mut features = Vec::new();
        let mut starts = vec![0];
        let mut chosen = Vec::new();
        for (rows, c) in sets {
            for row in rows {
                features.extend_from_slice(row);
            }
            starts.push(starts.last().copied().unwrap() + rows.len());
            chosen.push(*c);
        }
        Self { features, p, starts, chosen, clusters }
    }

    fn sets(&self) -> Vec<(Vec<Vec<f64>>, usize)> {
        (0..self.n_sets())
            .map(|s| (self.set_rows(s).into_iter().map(<[f64]>::to_vec).collect(), self.chosen[s]))
            .collect()
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 20) % n
    }
}

/// Sets of 2 to 5 options with uniform features on `[-1.5, 1.5]` and choices drawn from the
/// model at `beta`.
fn generate(seed: u64, n_sets: usize, beta: &[f64], n_clusters: Option<u64>) -> Data {
    let p = beta.len();
    let mut rng = Lcg(seed);
    let mut sets = Vec::new();
    let mut clusters = Vec::new();
    for _ in 0..n_sets {
        let k = 2 + usize::try_from(rng.below(4)).unwrap();
        let rows: Vec<Vec<f64>> =
            (0..k).map(|_| (0..p).map(|_| 3.0 * rng.uniform() - 1.5).collect()).collect();
        let eta: Vec<f64> =
            rows.iter().map(|r| r.iter().zip(beta).map(|(x, b)| x * b).sum()).collect();
        let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = eta.iter().map(|e| (e - top).exp()).collect();
        let mut u = rng.uniform() * w.iter().sum::<f64>();
        let mut pick = k - 1;
        for (j, wj) in w.iter().enumerate() {
            if u < *wj {
                pick = j;
                break;
            }
            u -= wj;
        }
        sets.push((rows, pick));
        if let Some(c) = n_clusters {
            clusters.push(u32::try_from(rng.below(c)).unwrap());
        }
    }
    Data::from_sets(p, &sets, n_clusters.map(|_| clusters))
}

fn fit(data: &Data) -> ConditionalLogitFit {
    fit_conditional_logit(&data.design(), ConditionalLogitOptions::default())
        .unwrap_or_else(|e| panic!("fit refused: {e}"))
}

fn refusal(data: &Data) -> ConditionalLogitError {
    match fit_conditional_logit(&data.design(), ConditionalLogitOptions::default()) {
        Ok(fit) => panic!("expected a refusal, got coefficients {:?}", fit.coefficients),
        Err(e) => e,
    }
}

fn normwise(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    got.iter().zip(want).fold(0.0_f64, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

/// Two options per set with feature values `base` and `base + d`; option 1 is chosen in `a`
/// sets and option 0 in `b` sets.
fn binary(a: usize, b: usize, base: f64, d: f64, clusters: Option<Vec<u32>>) -> Data {
    let mut sets = Vec::new();
    for _ in 0..a {
        sets.push((vec![vec![base], vec![base + d]], 1));
    }
    for _ in 0..b {
        sets.push((vec![vec![base], vec![base + d]], 0));
    }
    Data::from_sets(1, &sets, clusters)
}

#[test]
fn two_options_one_feature_has_the_closed_form_maximum() {
    // With two options the model is a logistic regression through the origin on the feature
    // difference d: P(option 1) = 1 / (1 + exp(-beta d)). With a choices of option 1 among n
    // the maximum is p = a / n, beta = ln(a / b) / d, and the information is n p (1 - p) d^2.
    let (a, b, d) = (6, 2, 0.5);
    let data = binary(a, b, 1.0, d, None);
    let fit = fit(&data);
    let n = (a + b) as f64;
    let p = a as f64 / n;
    let beta = (a as f64 / b as f64).ln() / d;
    assert!(close(fit.coefficients[0], beta, 1e-10), "{:?}", fit.coefficients);
    let ll = a as f64 * p.ln() + b as f64 * (1.0 - p).ln();
    assert!(close(fit.log_likelihood, ll, 1e-12));
    assert!(close(fit.null_log_likelihood, -n * 2.0_f64.ln(), 1e-15));
    let info = n * p * (1.0 - p) * d * d;
    assert!(close(fit.information[0], info, 1e-10));
    assert!(close(fit.covariance[0], 1.0 / info, 1e-10));
    assert!(fit.cluster_covariance.is_none());
    assert!(fit.iterations > 0);
    assert!(fit.gradient_norm <= 1e-12 * n * d);
}

#[test]
fn equal_counts_give_zero_coefficient_without_iterating() {
    let data = binary(3, 3, -2.0, 1.25, None);
    let fit = fit(&data);
    assert!(fit.coefficients[0].abs() <= 1e-15);
    assert_eq!(fit.iterations, 0);
    assert!(close(fit.log_likelihood, fit.null_log_likelihood, 1e-15));
}

#[test]
fn fit_is_invariant_to_within_set_shifts_and_to_option_and_set_order() {
    let data = generate(0x9E37_79B9_7F4A_7C15, 90, &[0.9, -0.7, 0.4], Some(17));
    let base = fit(&data);

    // Add a constant vector to every option of each set (different per set and per feature).
    let mut shifted = data.clone();
    for s in 0..data.n_sets() {
        for r in data.starts[s]..data.starts[s + 1] {
            for k in 0..data.p {
                shifted.features[r * data.p + k] += 37.5 * (s as f64 - 40.0) + 3.0 * k as f64;
            }
        }
    }
    // Reverse the options within every set.
    let reversed_options = Data::from_sets(
        data.p,
        &data
            .sets()
            .into_iter()
            .map(|(mut rows, c)| {
                let k = rows.len();
                rows.reverse();
                (rows, k - 1 - c)
            })
            .collect::<Vec<_>>(),
        data.clusters.clone(),
    );
    // Reverse the order of the sets, keeping each set's cluster.
    let mut reversed_sets = data.sets();
    reversed_sets.reverse();
    let reversed_sets = Data::from_sets(
        data.p,
        &reversed_sets,
        data.clusters.as_ref().map(|c| c.iter().rev().copied().collect()),
    );

    for (label, other) in
        [("shift", &shifted), ("options", &reversed_options), ("sets", &reversed_sets)]
    {
        let f = fit(other);
        assert!(normwise(&f.coefficients, &base.coefficients) <= 1e-9, "{label}");
        assert!(close(f.log_likelihood, base.log_likelihood, 1e-11), "{label}");
        assert!(close(f.null_log_likelihood, base.null_log_likelihood, 1e-13), "{label}");
        assert!(normwise(&f.covariance, &base.covariance) <= 1e-8, "{label}");
        assert!(
            normwise(
                f.cluster_covariance.as_deref().unwrap(),
                base.cluster_covariance.as_deref().unwrap()
            ) <= 1e-8,
            "{label}"
        );
    }
}

fn log_likelihood(data: &Data, beta: &[f64]) -> f64 {
    conditional_logit_log_likelihood(&data.design(), beta).unwrap()
}

#[test]
fn gradient_vanishes_at_the_returned_coefficients() {
    let data = generate(0x1234_5678_9ABC_DEF1, 150, &[1.2, -0.5, 0.3, 0.0], None);
    let fit = fit(&data);
    let h = 1e-5;
    for k in 0..data.p {
        let mut up = fit.coefficients.clone();
        let mut down = fit.coefficients.clone();
        up[k] += h;
        down[k] -= h;
        let slope = (log_likelihood(&data, &up) - log_likelihood(&data, &down)) / (2.0 * h);
        assert!(slope.abs() <= 1e-7 * data.n_sets() as f64, "coordinate {k}: {slope}");
    }
    assert!(close(log_likelihood(&data, &fit.coefficients), fit.log_likelihood, 1e-14));
}

#[test]
fn information_matches_a_finite_difference_hessian() {
    let data = generate(0x0F0F_1234_ABCD_0001, 120, &[0.6, -1.1, 0.8], None);
    let fit = fit(&data);
    let p = data.p;
    let h = 1e-3;
    let mut hessian = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let at = |si: f64, sj: f64| {
                let mut b = fit.coefficients.clone();
                b[i] += si * h;
                b[j] += sj * h;
                log_likelihood(&data, &b)
            };
            let second =
                (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h);
            hessian[i * p + j] = -second;
        }
    }
    assert!(normwise(&fit.information, &hessian) <= 1e-4, "{:?} vs {hessian:?}", fit.information);
    // The covariance is the inverse of the information.
    for i in 0..p {
        for j in 0..p {
            let product: f64 =
                (0..p).map(|k| fit.covariance[i * p + k] * fit.information[k * p + j]).sum();
            let identity = if i == j { 1.0 } else { 0.0 };
            assert!((product - identity).abs() <= 1e-10);
        }
    }
}

#[test]
fn cluster_sandwich_matches_the_hand_formula_in_the_two_option_case() {
    // Sets 0..6 choose option 1 (score (1 - p) d), sets 6..8 choose option 0 (score -p d).
    let (a, b, d) = (6, 2, 0.5);
    let n = (a + b) as f64;
    let p = a as f64 / n;
    let info = n * p * (1.0 - p) * d * d;
    let s1 = (1.0 - p) * d;
    let s0 = -p * d;

    // Every set its own cluster: the meat a s1^2 + b s0^2 equals the information, so the
    // sandwich equals the model covariance.
    let own = fit(&binary(a, b, 0.25, d, Some((0..8).collect())));
    let meat = a as f64 * s1 * s1 + b as f64 * s0 * s0;
    assert!(close(meat, info, 1e-14));
    let sandwich = own.cluster_covariance.unwrap();
    assert!(close(sandwich[0], meat / (info * info), 1e-10));
    assert!(close(sandwich[0], own.covariance[0], 1e-10));

    // Repeated clusters: {0, 6}, {1, 7}, {2, 3}, {4, 5}.
    let paired = fit(&binary(a, b, 0.25, d, Some(vec![0, 1, 2, 2, 3, 3, 0, 1])));
    let sums = [s1 + s0, s1 + s0, 2.0 * s1, 2.0 * s1];
    let meat: f64 = sums.iter().map(|s| s * s).sum();
    let sandwich = paired.cluster_covariance.unwrap();
    assert!(close(sandwich[0], meat / (info * info), 1e-10));
    assert!((sandwich[0] - paired.covariance[0]).abs() > 0.1 * paired.covariance[0]);
}

#[test]
fn cluster_sandwich_matches_scores_recomputed_from_the_features() {
    let data = generate(0x5555_AAAA_0123_4567, 140, &[0.7, -0.4], Some(23));
    let fit = fit(&data);
    let p = data.p;
    let mut cluster_scores = std::collections::BTreeMap::<u32, Vec<f64>>::new();
    for s in 0..data.n_sets() {
        let rows = data.set_rows(s);
        let eta: Vec<f64> = rows
            .iter()
            .map(|r| r.iter().zip(&fit.coefficients).map(|(x, b)| x * b).sum())
            .collect();
        let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = eta.iter().map(|e| (e - top).exp()).collect();
        let total: f64 = w.iter().sum();
        let entry = cluster_scores
            .entry(data.clusters.as_ref().unwrap()[s])
            .or_insert_with(|| vec![0.0; p]);
        for k in 0..p {
            let mean: f64 = rows.iter().zip(&w).map(|(r, wj)| wj / total * r[k]).sum();
            entry[k] += rows[data.chosen[s]][k] - mean;
        }
    }
    let mut meat = vec![0.0; p * p];
    for score in cluster_scores.values() {
        for i in 0..p {
            for j in 0..p {
                meat[i * p + j] += score[i] * score[j];
            }
        }
    }
    let c = &fit.covariance;
    let mut want = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            for k in 0..p {
                for l in 0..p {
                    want[i * p + j] += c[i * p + k] * meat[k * p + l] * c[l * p + j];
                }
            }
        }
    }
    assert!(normwise(fit.cluster_covariance.as_deref().unwrap(), &want) <= 1e-9);
    assert!(normwise(&fit.covariance, &want) > 1e-3);
}

#[test]
fn varying_set_sizes_fit_and_duplication_halves_the_covariance() {
    let data = generate(0xDEAD_BEEF_0000_0042, 200, &[1.0, -0.8, 0.5], None);
    let sizes: Vec<usize> = data.starts.windows(2).map(|w| w[1] - w[0]).collect();
    for k in 2..=5 {
        assert!(sizes.contains(&k), "size {k} missing");
    }
    let once = fit(&data);
    assert!(once.log_likelihood > once.null_log_likelihood);
    let expected_null: f64 = sizes.iter().map(|&k| -(k as f64).ln()).sum();
    assert!(close(once.null_log_likelihood, expected_null, 1e-14));

    let mut doubled = data.sets();
    doubled.extend(data.sets());
    let twice = fit(&Data::from_sets(data.p, &doubled, None));
    assert!(normwise(&twice.coefficients, &once.coefficients) <= 1e-9);
    assert!(close(twice.log_likelihood, 2.0 * once.log_likelihood, 1e-12));
    let half: Vec<f64> = once.covariance.iter().map(|v| 0.5 * v).collect();
    assert!(normwise(&twice.covariance, &half) <= 1e-8);

    // Deterministic: a second fit is identical.
    assert_eq!(fit(&data), once);
}

#[test]
fn shape_refusals_name_their_cause() {
    let good = binary(2, 1, 0.0, 1.0, None);
    let with = |edit: &dyn Fn(&mut Data)| {
        let mut d = good.clone();
        edit(&mut d);
        refusal(&d)
    };
    let cases: Vec<(ConditionalLogitError, &str)> = vec![
        (with(&|d| d.p = 0), "conditional_logit.no_features"),
        (with(&|d| d.p = 4), "conditional_logit.feature_length"),
        (
            with(&|d| {
                d.features.clear();
                d.starts = vec![0];
                d.chosen.clear();
            }),
            "conditional_logit.no_sets",
        ),
        (with(&|d| d.starts = vec![]), "conditional_logit.no_sets"),
        (with(&|d| d.starts[0] = 1), "conditional_logit.set_starts_not_zero"),
        (with(&|d| d.starts = vec![0, 1, 4, 6]), "conditional_logit.set_too_small"),
        (with(&|d| d.starts = vec![0, 4, 2, 6]), "conditional_logit.set_too_small"),
        (with(&|d| d.starts = vec![0, 2, 4]), "conditional_logit.set_starts_row_mismatch"),
        (with(&|d| d.chosen.push(0)), "conditional_logit.chosen_length"),
        (with(&|d| d.chosen[1] = 2), "conditional_logit.chosen_out_of_range"),
        (with(&|d| d.clusters = Some(vec![0, 1])), "conditional_logit.cluster_length"),
        (with(&|d| d.features[3] = f64::NAN), "conditional_logit.non_finite_feature"),
        (with(&|d| d.features[0] = f64::INFINITY), "conditional_logit.non_finite_feature"),
    ];
    for (error, detail) in &cases {
        assert_eq!(error.detail(), *detail, "{error}");
    }
    assert_eq!(
        with(&|d| d.starts = vec![0, 1, 4, 6]),
        ConditionalLogitError::SetTooSmall { set: 0, options: 1 }
    );
    assert_eq!(
        with(&|d| d.chosen[1] = 2),
        ConditionalLogitError::ChosenOutOfRange { set: 1, index: 2, options: 2 }
    );
    assert_eq!(
        with(&|d| d.features[3] = f64::NAN),
        ConditionalLogitError::NonFiniteFeature { row: 3, column: 0 }
    );
    // Overflowing within-set difference of two finite features.
    assert_eq!(
        with(&|d| {
            d.features[0] = f64::MAX;
            d.features[1] = -f64::MAX;
        }),
        ConditionalLogitError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[test]
fn log_likelihood_refusals_and_options_refusal() {
    let data = binary(2, 1, 0.0, 1.0, None);
    assert_eq!(
        conditional_logit_log_likelihood(&data.design(), &[0.0, 1.0]),
        Err(ConditionalLogitError::CoefficientLength { len: 2, n_features: 1 })
    );
    assert_eq!(
        conditional_logit_log_likelihood(&data.design(), &[f64::NAN]).map_err(|e| e.detail()),
        Err("conditional_logit.non_finite_coefficient")
    );
    for options in [
        ConditionalLogitOptions { gradient_tolerance: 0.0, ..ConditionalLogitOptions::default() },
        ConditionalLogitOptions {
            separation_bound: f64::NAN,
            ..ConditionalLogitOptions::default()
        },
    ] {
        assert_eq!(
            fit_conditional_logit(&data.design(), options),
            Err(ConditionalLogitError::InvalidOptions)
        );
    }
}

#[test]
fn rank_deficiency_names_the_dependent_columns() {
    // Column 1 is constant within each set; column 3 is 2 * column 0 - column 2.
    let base = generate(0x7777_0000_1111_2222, 40, &[0.5, 0.0, -0.5, 0.0], None);
    let mut data = base.clone();
    for s in 0..data.n_sets() {
        for r in data.starts[s]..data.starts[s + 1] {
            let row = &mut data.features[r * 4..(r + 1) * 4];
            row[1] = s as f64 * 0.75 - 3.0;
            row[3] = 2.0 * row[0] - row[2];
        }
    }
    let error = refusal(&data);
    assert_eq!(error, ConditionalLogitError::RankDeficient { columns: vec![1, 3] });
    assert_eq!(error.detail(), "conditional_logit.rank_deficient");
    // A single all-equal feature is a zero column.
    assert_eq!(
        refusal(&binary(2, 2, 1.0, 0.0, None)),
        ConditionalLogitError::RankDeficient { columns: vec![0] }
    );
}

#[test]
fn complete_separation_is_refused_with_its_direction() {
    // The chosen option always has the larger feature.
    let data = Data::from_sets(
        1,
        &[
            (vec![vec![0.0], vec![1.0]], 1),
            (vec![vec![2.0], vec![0.0], vec![-1.0]], 0),
            (vec![vec![0.5], vec![3.0]], 1),
        ],
        None,
    );
    match refusal(&data) {
        ConditionalLogitError::Separated { direction } => {
            assert_eq!(direction.len(), 1);
            assert!((direction[0] - 1.0).abs() <= 1e-12);
        }
        other => panic!("expected separation, got {other:?}"),
    }
}

#[test]
fn quasi_separation_is_refused_with_its_direction() {
    // Feature 0 strictly favours the chosen option in sets 0 and 1 and ties in the rest, where
    // feature 1 has no separating sign.
    let data = Data::from_sets(
        2,
        &[
            (vec![vec![0.0, 0.0], vec![1.0, 0.3]], 1),
            (vec![vec![0.0, 1.0], vec![2.0, -0.5]], 1),
            (vec![vec![0.0, 0.0], vec![0.0, 1.0]], 1),
            (vec![vec![0.0, 0.0], vec![0.0, 1.0]], 0),
            (vec![vec![0.0, 0.0], vec![0.0, -1.0]], 1),
            (vec![vec![0.0, 1.0], vec![0.0, 2.0]], 0),
        ],
        None,
    );
    match refusal(&data) {
        ConditionalLogitError::Separated { direction } => {
            assert!((direction[0] - 1.0).abs() <= 1e-6, "{direction:?}");
            assert!(direction[1].abs() <= 1e-6, "{direction:?}");
            let norm: f64 = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() <= 1e-12);
        }
        other => panic!("expected separation, got {other:?}"),
    }
}

#[test]
fn an_exhausted_iteration_budget_is_refused() {
    let data = generate(0x4242_4242_4242_4242, 60, &[0.8, -0.3], None);
    let options =
        ConditionalLogitOptions { max_iterations: 1, ..ConditionalLogitOptions::default() };
    match fit_conditional_logit(&data.design(), options) {
        Err(ConditionalLogitError::NotConverged { iterations, gradient_norm }) => {
            assert_eq!(iterations, 1);
            assert!(gradient_norm > 0.0);
        }
        other => panic!("expected no convergence, got {other:?}"),
    }
}

#[test]
fn detail_ids_are_distinct_and_in_the_family() {
    let all = [
        ConditionalLogitError::NoFeatures,
        ConditionalLogitError::FeatureLength { len: 0, n_features: 0 },
        ConditionalLogitError::NoSets,
        ConditionalLogitError::SetStartsNotZero { first: 0 },
        ConditionalLogitError::SetTooSmall { set: 0, options: 0 },
        ConditionalLogitError::SetStartsRowMismatch { last: 0, rows: 0 },
        ConditionalLogitError::ChosenLength { len: 0, n_sets: 0 },
        ConditionalLogitError::ChosenOutOfRange { set: 0, index: 0, options: 0 },
        ConditionalLogitError::ClusterLength { len: 0, n_sets: 0 },
        ConditionalLogitError::NonFiniteFeature { row: 0, column: 0 },
        ConditionalLogitError::CoefficientLength { len: 0, n_features: 0 },
        ConditionalLogitError::NonFiniteCoefficient { index: 0 },
        ConditionalLogitError::InvalidOptions,
        ConditionalLogitError::RankDeficient { columns: vec![] },
        ConditionalLogitError::Separated { direction: vec![] },
        ConditionalLogitError::SingularInformation,
        ConditionalLogitError::NotConverged { iterations: 0, gradient_norm: 0.0 },
        ConditionalLogitError::Cancelled,
    ];
    let mut ids: Vec<&str> = all.iter().map(ConditionalLogitError::detail).collect();
    assert!(ids.iter().all(|id| id.starts_with("conditional_logit.")));
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), all.len());
}

#[test]
fn a_cancelled_context_stops_the_fit_before_or_between_newton_steps() {
    let data = generate(0x4242_4242_4242_4242, 60, &[0.8, -0.3], None);
    let options = ConditionalLogitOptions::default();
    let mut ctx = ExecutionContext::production(0, 1);
    ctx.cancellation.cancel();
    assert_eq!(
        fit_conditional_logit_with_context(&data.design(), options, &ctx).unwrap_err(),
        ConditionalLogitError::Cancelled
    );
    // The first observation passes, so the fit starts and stops before its second step.
    ctx.cancellation = CancellationToken::cancel_after_checks(2);
    assert_eq!(
        fit_conditional_logit_with_context(&data.design(), options, &ctx).unwrap_err(),
        ConditionalLogitError::Cancelled
    );
    // An uncancelled context gives the context-free fit exactly.
    ctx.cancellation = CancellationToken::new();
    let within = fit_conditional_logit_with_context(&data.design(), options, &ctx).unwrap();
    assert_eq!(within.coefficients, fit(&data).coefficients);
}
