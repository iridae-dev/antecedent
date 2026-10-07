//! Independent Gaussian experiment for F18's Q/Wald and Holm composition.
//! Known covariance only: this does not calibrate a caller's estimated SEs,
//! non-Gaussian effect estimator, or a different sampling design.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::effect_constancy::{
    ConstancyConclusion, ContrastFamily, EffectEstimandIdentity, PartitionDependence,
    PartitionEstimate, PartitionSupport, test_effect_constancy,
};

// An independent Box-Muller generator; no estimator RNG or tail code is reused.
fn normal(state: &mut u64) -> f64 {
    let mut uniform = || {
        *state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    };
    let u = uniform();
    let v = uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

#[test]
#[allow(clippy::too_many_lines)] // One declared experiment and its per-cell acceptance rules.
fn f18_known_gaussian_null_power_and_holm_grid() {
    const REPLICATES: usize = 2000;
    for k in [2_usize, 3, 8] {
        for rho in [0.0_f64, 0.6, -0.5 / (k - 1) as f64] {
            let mut last_power = 0.0;
            for n in [20_u32, 80, 320] {
                let mut state = 0x18ca_11b0 + u64::from(n) + k as u64;
                let se = 1.0 / f64::from(n).sqrt();
                let variance = se * se;
                let covariance: Vec<f64> = (0..k)
                    .flat_map(|i| {
                        (0..k).map(move |j| if i == j { variance } else { rho * variance })
                    })
                    .collect();
                let dependence = if rho.abs() < f64::EPSILON {
                    PartitionDependence::Independent
                } else {
                    PartitionDependence::Covariance(covariance)
                };
                let (mut rejected, mut family_rejected, mut reference_rejected, mut power) =
                    (0, 0, 0, 0);
                for _ in 0..REPLICATES {
                    let z: Vec<f64> = (0..k).map(|_| normal(&mut state)).collect();
                    let zbar = z.iter().sum::<f64>() / k as f64;
                    // Equicorrelation square root from the common-mean and
                    // orthogonal-contrast eigenspaces (not production Cholesky).
                    let noise: Vec<f64> = z
                        .iter()
                        .map(|x| {
                            (1.0 - rho).sqrt() * (x - zbar)
                                + (1.0 + (k - 1) as f64 * rho).sqrt() * zbar
                        })
                        .collect();
                    let mut partitions: Vec<PartitionEstimate> = noise
                        .iter()
                        .enumerate()
                        .map(|(i, z)| PartitionEstimate {
                            label: format!("p{i}"),
                            coordinate: format!("period:{i}"),
                            support: PartitionSupport::Supported,
                            estimand: EffectEstimandIdentity {
                                estimand: "mean_difference".into(),
                                units: "units".into(),
                                regime: "treat_vs_control".into(),
                                population: "gaussian_reference".into(),
                            },
                            effect: 1.0 + se * z,
                            standard_error: se,
                        })
                        .collect();
                    let null = test_effect_constancy(
                        &partitions,
                        &dependence,
                        &ContrastFamily::AllPairs,
                        0.05,
                    )
                    .unwrap();
                    rejected += usize::from(null.conclusion == ConstancyConclusion::Rejected);
                    family_rejected += usize::from(null.contrasts.iter().any(|c| c.rejected));
                    let reference = test_effect_constancy(
                        &partitions,
                        &dependence,
                        &ContrastFamily::AgainstReference("p0".into()),
                        0.05,
                    )
                    .unwrap();
                    reference_rejected +=
                        usize::from(reference.contrasts.iter().any(|c| c.rejected));
                    partitions[0].effect += 0.8;
                    let alternative = test_effect_constancy(
                        &partitions,
                        &dependence,
                        &ContrastFamily::AllPairs,
                        0.05,
                    )
                    .unwrap();
                    power += usize::from(alternative.conclusion == ConstancyConclusion::Rejected);
                }
                let rate = |count: usize| count as f64 / REPLICATES as f64;
                let (size, fwer, reference_fwer, power) =
                    (rate(rejected), rate(family_rejected), rate(reference_rejected), rate(power));
                println!(
                    "f18-gaussian n={n} k={k} rho={rho:.6} replicates={REPLICATES} type_i={size:.4} holm_all={fwer:.4} holm_reference={reference_fwer:.4} power={power:.4}"
                );
                // Fixed in advance: alpha=.05, roughly five binomial SEs at
                // 2000 simulations. Holm may be conservative as k increases.
                assert!((0.025..=0.075).contains(&size), "null size {size}");
                assert!(fwer <= 0.075 && reference_fwer <= 0.075, "Holm family error");
                assert!(power + 0.03 >= last_power, "power should increase with precision");
                if n == 320 {
                    assert!(power >= 0.95, "large-sample power {power}");
                }
                last_power = power;
            }
        }
    }
}
