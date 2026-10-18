//! Repeated-sampling coverage of the pooled-mean intervals of the evidence
//! ledger, whole method: independent studies generate raw rows, each study is
//! summarized (estimate and estimated sampling variance), the summaries are
//! registered as disjoint ledger sources, pooled through the session and the
//! interval is read from the session's own construction.
//!
//! Two constructions are measured over the advertised family of 5, 10, 15, 20,
//! 25 and 30 studies, each at three per-study sample sizes (`grid_n(100)`). The
//! records are keyed by the study-count range and by the observations of the
//! smallest study, so a count inside the range is licensed only because every
//! count visited here passed:
//!
//! * fixed-effect inverse-variance pooling, Wald `z` interval, on homogeneous
//!   studies with unequal noise scales;
//! * equal-study Gaussian pooling with a study-level Student `t` interval,
//!   on studies whose true effects vary around one mean: moderate
//!   declared heterogeneity (between-study standard deviation half the typical
//!   study standard error), strong heterogeneity with a four-fold spread of
//!   study noise scales, and homogeneous studies (heterogeneity absent, where
//!   the interval must stay valid).
//!
//! The Gaussian study-level construction guarantees at least nominal coverage
//! for independent marginally Gaussian estimates with a common superpopulation
//! mean and arbitrary unequal variances (Ibragimov--Mueller 2010, section 2.1).
//! Its distinct `conservative_point` role tests the existing lower coverage
//! floor, retaining all original DGPs, seeds and reporting requirements. It
//! does not relabel the old DL/HKSJ construction as calibrated.
//!
//! The moderate-heterogeneity and fixed-effect designs also assert interval
//! usefulness: the median half-width stays within the acceptance ratio to the
//! root-mean-square study standard error at that study count.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::analysis::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};
use antecedent::{BetweenStudy, PoolingDeclaration, PoolingRequest, StudyBuilder, StudySummary};
use antecedent_core::ExecutionContext;
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed, map_replicates, n_sim,
};

const LEVEL: f64 = 0.95;
const MEAN: f64 = 0.5;
/// Per-study sample size at the base grid point.
const STUDY_ROWS: usize = 100;

/// Normal measurements use the new method. Attestation replay derives this
/// override from a stored legacy construction, never from a desired outcome.
fn protocol_version() -> u32 {
    static VERSION: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| {
        let raw =
            std::env::var("ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION").unwrap_or_else(|_| "2".into());
        raw.parse::<u32>()
            .ok()
            .filter(|version| matches!(version, 1 | 2))
            .unwrap_or_else(|| panic!("ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION={raw}: want 1 or 2"))
    })
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// How the studies' true effects and noise scales are generated.
#[derive(Clone, Copy)]
struct Design {
    /// Fixed-effect Wald `z` (`true`) or conservative Gaussian study-level `t`.
    fixed_effect: bool,
    /// Between-study standard deviation as a multiple of the root-mean-square
    /// study standard error; zero is homogeneous.
    heterogeneity: f64,
    /// Study noise scales cycle through `1 + step * (0..cycle)`.
    scale_cycle: usize,
    scale_step: f64,
}

impl Design {
    /// Homogeneous studies, noise scales 1, 1.5 and 2.
    const FIXED_EFFECT: Self =
        Self { fixed_effect: true, heterogeneity: 0.0, scale_cycle: 3, scale_step: 0.5 };
    /// Declared heterogeneity of half the typical study standard error.
    const MODERATE: Self =
        Self { fixed_effect: false, heterogeneity: 0.5, scale_cycle: 3, scale_step: 0.5 };
    /// Heterogeneity twice the typical study standard error, noise scales 1 to 4.
    const STRONG: Self =
        Self { fixed_effect: false, heterogeneity: 2.0, scale_cycle: 4, scale_step: 1.0 };
    /// No heterogeneity under the random-effects construction.
    const ABSENT: Self =
        Self { fixed_effect: false, heterogeneity: 0.0, scale_cycle: 3, scale_step: 0.5 };

    fn scale(self, study: usize) -> f64 {
        1.0 + (study % self.scale_cycle) as f64 * self.scale_step
    }
}

/// One study's raw rows summarized: sample mean and estimated variance of it.
fn summarize(rng: &mut Rng, effect: f64, scale: f64, n: usize) -> (f64, f64) {
    let rows: Vec<f64> = (0..n).map(|_| effect + scale * rng.normal()).collect();
    let mean = rows.iter().sum::<f64>() / n as f64;
    let ss: f64 = rows.iter().map(|y| (y - mean) * (y - mean)).sum();
    (mean, ss / (n as f64 - 1.0) / n as f64)
}

/// One replicate: the interval construction and its half-width relative to the
/// root-mean-square study standard error.
fn replicate(
    design: Design,
    studies: usize,
    n: usize,
    seed: u64,
    replicate: u64,
) -> Option<(antecedent::analysis::evidence_ledger::PooledIntervalCandidate, f64)> {
    replicate_with_protocol(design, studies, n, seed, replicate, protocol_version())
}

fn replicate_with_protocol(
    design: Design,
    studies: usize,
    n: usize,
    seed: u64,
    replicate: u64,
    protocol: u32,
) -> Option<(antecedent::analysis::evidence_ledger::PooledIntervalCandidate, f64)> {
    let mut rng = Rng(grid_seed(seed + replicate));
    let mut session = StudyBuilder::evidence_ledger();
    let mut summaries = Vec::with_capacity(studies);
    let rms_scale =
        ((0..studies).map(|s| design.scale(s).powi(2)).sum::<f64>() / studies as f64).sqrt();
    let tau = design.heterogeneity * rms_scale / (n as f64).sqrt();
    for study in 0..studies {
        let rows: Vec<ObservationId> = (0..n)
            .map(|i| {
                ObservationId::from_key("study", &(((study as u64) << 32) | i as u64).to_le_bytes())
            })
            .collect();
        let source = session
            .add_source(
                SourceDeclaration::new(format!("study_{study}"), SourceKind::ObservationSet)
                    .with_observations(rows),
            )
            .ok()?;
        let effect = MEAN + tau * rng.normal();
        let (estimate, variance) = summarize(&mut rng, effect, design.scale(study), n);
        summaries.push(StudySummary { source, estimate, variance });
    }
    let rms_se =
        (summaries.iter().map(|s| s.variance).sum::<f64>() / summaries.len() as f64).sqrt();
    let ctx = ExecutionContext::for_tests(seed + replicate);
    // A predeclared consumer witness at the first counter of each legacy grid
    // exercises the actual artifact implementation without adding a serialization
    // workload to every Monte Carlo draw. The native producer/tally remains the
    // stored construction; this is not a new consumer-coverage claim.
    let witness =
        ((design.fixed_effect || protocol == 1) && replicate == 0).then(|| summaries.clone());
    let request = PoolingRequest { label: "pooled".to_owned(), summaries, replaces: None };
    let result = if design.fixed_effect {
        session.pool_summaries(request, PoolingDeclaration::Independent, &ctx).ok()?.result
    } else if protocol == 1 {
        session.random_effects(request, BetweenStudy::DerSimonianLaird, &ctx).ok()?.result
    } else {
        session.pool_gaussian_studies(request, &ctx).ok()?.result
    };
    let candidate = session.pooled_interval_candidate(result).ok()?;
    if let Some(summaries) = witness {
        let export = session
            .export("legacy-synthesis-calibration-replay")
            .expect("legacy synthesis producer must seal its real artifact");
        let consumed = antecedent::analysis::evidence_ledger::consume_evidence_ledger_artifact(
            &export.bytes,
            Some(&export.identity),
        )
        .expect("legacy synthesis artifact must independently replay");
        assert_eq!(consumed.identity(), export.identity);
        let recomputed = consumed
            .synthesis_of(result)
            .expect("consumer must recompute the original synthesis result");
        let mean = recomputed.values[0];
        let half = if design.fixed_effect {
            assert_eq!(recomputed.method, "fixed_effect_inverse_variance");
            antecedent_stats::normal_ppf(0.5 + LEVEL / 2.0) * recomputed.values[1].sqrt()
        } else {
            assert_eq!(recomputed.method, "random_effects_dersimonian_laird");
            let tau_squared = recomputed.values[2];
            let precision: Vec<f64> =
                summaries.iter().map(|s| 1.0 / (s.variance + tau_squared)).collect();
            let total: f64 = precision.iter().sum();
            let spread: f64 = precision
                .iter()
                .zip(&summaries)
                .map(|(w, s)| w * (s.estimate - mean) * (s.estimate - mean))
                .sum();
            let variance = (spread / (studies - 1) as f64).max(1.0) / total;
            antecedent_stats::student_t_ppf(0.5 + LEVEL / 2.0, (studies - 1) as f64)
                * variance.sqrt()
        };
        assert_eq!(mean.to_bits(), candidate.estimate.to_bits(), "consumer mean changed");
        assert_eq!(
            (mean - half).to_bits(),
            candidate.lower.to_bits(),
            "consumer lower endpoint changed"
        );
        assert_eq!(
            (mean + half).to_bits(),
            candidate.upper.to_bits(),
            "consumer upper endpoint changed"
        );
        eprintln!(
            "calibration-replay synthesis_artifact_verified method={} studies={studies} rows={n} first_counter=0",
            recomputed.method,
        );
    }
    let half_width = (candidate.upper - candidate.lower) / 2.0;
    Some((candidate, half_width / rms_se))
}

#[test]
fn legacy_synthesis_first_counter_roundtrips_the_real_artifact_without_changing_intervals() {
    for design in [Design::FIXED_EFFECT, Design::MODERATE, Design::STRONG] {
        for studies in [5, 15, 30] {
            let (candidate, _) = replicate_with_protocol(design, studies, 75, 91_100, 0, 1)
                .expect("legacy producer and artifact consumer");
            assert_eq!(candidate.method, if design.fixed_effect { "wald_z" } else { "hksj_t" });
        }
    }
}

/// Largest median ratio of the interval half-width to the root-mean-square
/// study standard error accepted at a study count.
fn usefulness_bound(studies: usize) -> f64 {
    match studies {
        5 => 1.68,
        10 => 1.19,
        15 => 0.97,
        20 => 0.84,
        25 => 0.75,
        30 => 0.68,
        other => unreachable!("{other} studies are not in the measured family"),
    }
}

fn measure(
    design: Design,
    studies: usize,
    useful: bool,
    test: &'static str,
    dgp: &'static str,
    seed: u64,
) {
    eprintln!("calibration-protocol synthesis_source_version={}", protocol_version());
    let n = grid_n(STUDY_ROWS);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp,
            interval: if design.fixed_effect {
                "wald_z"
            } else if protocol_version() == 1 {
                "hksj_t"
            } else {
                "independent_gaussian_study_t"
            },
        },
        LEVEL,
    );
    let replicates = map_replicates(n_sim(), |r| replicate(design, studies, n, seed, r));
    let mut ratios = Vec::with_capacity(replicates.len());
    for outcome in replicates {
        match outcome {
            Some((candidate, ratio)) => {
                let basis = &candidate.basis;
                let construction = Construction {
                    query: basis.query.to_string(),
                    graph_class: basis.graph_class.to_string(),
                    structure: basis.structure.to_string(),
                    modality: basis.modality.to_string(),
                    inference: basis.inference.to_string(),
                    estimator: basis.estimator.to_string(),
                    interval_method: basis.interval_method.to_string(),
                    se_kind: basis.se_kind.to_string(),
                    dependence: basis.dependence.to_string(),
                    posterior: basis.posterior.to_string(),
                    functional: basis.functional.to_string(),
                    identification: basis.identification.to_string(),
                    reported_level: candidate.level,
                };
                tally.bind(
                    &construction,
                    ScopeFacts {
                        row_count: basis.row_count,
                        replicates_ok: None,
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                );
                tally.record(Some((candidate.lower, candidate.upper)), MEAN);
                ratios.push(ratio);
            }
            None => tally.skip(),
        }
    }
    if design.fixed_effect || protocol_version() == 1 {
        tally.assert();
    } else {
        tally.assert_gaussian_study_t();
    }
    if useful {
        ratios.sort_by(f64::total_cmp);
        let median = ratios[ratios.len() / 2];
        let bound = usefulness_bound(studies);
        assert!(
            median <= bound,
            "{test}: median half-width is {median:.3} study standard errors, above {bound}"
        );
    }
}

// Claim: the 95% Wald `z` interval of the fixed-effect pooled mean covers the
// common effect of homogeneous studies of estimated variance.
#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_five_studies() {
    measure(
        Design::FIXED_EFFECT,
        5,
        true,
        "pooled_fixed_effect_five_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_ten_studies() {
    measure(
        Design::FIXED_EFFECT,
        10,
        true,
        "pooled_fixed_effect_ten_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000 + 0x10000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_fifteen_studies() {
    measure(
        Design::FIXED_EFFECT,
        15,
        true,
        "pooled_fixed_effect_fifteen_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000 + 0x20000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_twenty_studies() {
    measure(
        Design::FIXED_EFFECT,
        20,
        true,
        "pooled_fixed_effect_twenty_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000 + 0x30000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_twenty_five_studies() {
    measure(
        Design::FIXED_EFFECT,
        25,
        true,
        "pooled_fixed_effect_twenty_five_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000 + 0x40000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_thirty_studies() {
    measure(
        Design::FIXED_EFFECT,
        30,
        true,
        "pooled_fixed_effect_thirty_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000 + 0x50000,
    );
}

// Claim: the 95% Hartung-Knapp-Sidik-Jonkman interval of the random-effects
// pooled mean covers the mean of the study effects under declared heterogeneity.
#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_five_studies() {
    measure(
        Design::MODERATE,
        5,
        true,
        "pooled_random_effects_five_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_ten_studies() {
    measure(
        Design::MODERATE,
        10,
        true,
        "pooled_random_effects_ten_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000 + 0x10000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_fifteen_studies() {
    measure(
        Design::MODERATE,
        15,
        true,
        "pooled_random_effects_fifteen_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000 + 0x20000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_twenty_studies() {
    measure(
        Design::MODERATE,
        20,
        true,
        "pooled_random_effects_twenty_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000 + 0x30000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_twenty_five_studies() {
    measure(
        Design::MODERATE,
        25,
        true,
        "pooled_random_effects_twenty_five_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000 + 0x40000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_thirty_studies() {
    measure(
        Design::MODERATE,
        30,
        true,
        "pooled_random_effects_thirty_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000 + 0x50000,
    );
}

// Strong heterogeneity and a four-fold spread of noise scales: coverage only.
#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_strong_heterogeneity_five_studies() {
    measure(
        Design::STRONG,
        5,
        false,
        "pooled_random_effects_strong_heterogeneity_five_studies",
        "strongly_heterogeneous_gaussian_studies_wide_scale_spread",
        0x5e30_0000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_strong_heterogeneity_fifteen_studies() {
    measure(
        Design::STRONG,
        15,
        false,
        "pooled_random_effects_strong_heterogeneity_fifteen_studies",
        "strongly_heterogeneous_gaussian_studies_wide_scale_spread",
        0x5e30_0000 + 0x20000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_strong_heterogeneity_thirty_studies() {
    measure(
        Design::STRONG,
        30,
        false,
        "pooled_random_effects_strong_heterogeneity_thirty_studies",
        "strongly_heterogeneous_gaussian_studies_wide_scale_spread",
        0x5e30_0000 + 0x50000,
    );
}

// No heterogeneity under the random-effects construction: the interval stays valid.
#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_homogeneous_five_studies() {
    measure(
        Design::ABSENT,
        5,
        false,
        "pooled_random_effects_homogeneous_five_studies",
        "homogeneous_gaussian_studies_unequal_scale_random_effects",
        0x5e40_0000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_homogeneous_fifteen_studies() {
    measure(
        Design::ABSENT,
        15,
        false,
        "pooled_random_effects_homogeneous_fifteen_studies",
        "homogeneous_gaussian_studies_unequal_scale_random_effects",
        0x5e40_0000 + 0x20000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_homogeneous_thirty_studies() {
    measure(
        Design::ABSENT,
        30,
        false,
        "pooled_random_effects_homogeneous_thirty_studies",
        "homogeneous_gaussian_studies_unequal_scale_random_effects",
        0x5e40_0000 + 0x50000,
    );
}
