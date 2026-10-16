//! Repeated-sampling coverage of the pooled-mean intervals of the evidence
//! ledger, whole method: independent studies generate raw rows, each study is
//! summarized (estimate and estimated sampling variance), the summaries are
//! registered as disjoint ledger sources, pooled through the session and the
//! interval is read from the session's own construction.
//!
//! Two constructions are measured, each at 5 and 10 studies and at three
//! per-study sample sizes (`grid_n(100)`):
//!
//! * fixed-effect inverse-variance pooling, Wald `z` interval, on homogeneous
//!   studies with unequal noise scales;
//! * DerSimonian-Laird random effects, Hartung-Knapp-Sidik-Jonkman `t`
//!   interval, on studies whose true effects vary around one mean.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::analysis::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};
use antecedent::{BetweenStudy, PoolingDeclaration, PoolingRequest, StudyBuilder, StudySummary};
use antecedent_core::ExecutionContext;
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_point, grid_seed,
    map_replicates, n_sim,
};

const LEVEL: f64 = 0.95;
const MEAN: f64 = 0.5;
/// Between-study standard deviation of the random-effects design.
const TAU: f64 = 0.3;

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

#[derive(Clone, Copy)]
enum Design {
    FixedEffect,
    RandomEffects,
}

/// One study's raw rows summarized: sample mean and estimated variance of it.
fn summarize(rng: &mut Rng, effect: f64, scale: f64, n: usize) -> (f64, f64) {
    let rows: Vec<f64> = (0..n).map(|_| effect + scale * rng.normal()).collect();
    let mean = rows.iter().sum::<f64>() / n as f64;
    let ss: f64 = rows.iter().map(|y| (y - mean) * (y - mean)).sum();
    (mean, ss / (n as f64 - 1.0) / n as f64)
}

fn measure(design: Design, studies: usize, test: &'static str, dgp: &'static str, seed: u64) {
    let n = grid_n(100);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp,
            interval: match design {
                Design::FixedEffect => "wald_z",
                Design::RandomEffects => "hksj_t",
            },
        },
        LEVEL,
    );
    let replicates = map_replicates(n_sim(), |replicate| {
        let mut rng = Rng(grid_seed(seed + replicate));
        let mut session = StudyBuilder::evidence_ledger();
        let mut summaries = Vec::with_capacity(studies);
        for study in 0..studies {
            let rows: Vec<ObservationId> = (0..n)
                .map(|i| {
                    ObservationId::from_key(
                        "study",
                        &(((study as u64) << 32) | i as u64).to_le_bytes(),
                    )
                })
                .collect();
            let source = session
                .add_source(
                    SourceDeclaration::new(format!("study_{study}"), SourceKind::ObservationSet)
                        .with_observations(rows),
                )
                .ok()?;
            let scale = 1.0 + (study % 3) as f64 * 0.5;
            let effect = match design {
                Design::FixedEffect => MEAN,
                Design::RandomEffects => MEAN + TAU * rng.normal(),
            };
            let (estimate, variance) = summarize(&mut rng, effect, scale, n);
            summaries.push(StudySummary { source, estimate, variance });
        }
        let ctx = ExecutionContext::for_tests(seed + replicate);
        let request = PoolingRequest { label: "pooled".to_owned(), summaries, replaces: None };
        let result = match design {
            Design::FixedEffect => {
                session.pool_summaries(request, PoolingDeclaration::Independent, &ctx).ok()?.result
            }
            Design::RandomEffects => {
                session.random_effects(request, BetweenStudy::DerSimonianLaird, &ctx).ok()?.result
            }
        };
        session.pooled_interval_candidate(result).ok()
    });
    for candidate in replicates {
        match candidate {
            Some(candidate) => {
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
            }
            None => tally.skip(),
        }
    }
    let _ = grid_point();
    tally.assert();
}

/// Claim: the 95% Wald `z` interval of the fixed-effect pooled mean covers the
/// common effect of five homogeneous studies of estimated variance.
#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_five_studies() {
    measure(
        Design::FixedEffect,
        5,
        "pooled_fixed_effect_five_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e10_0000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_fixed_effect_ten_studies() {
    measure(
        Design::FixedEffect,
        10,
        "pooled_fixed_effect_ten_studies",
        "homogeneous_gaussian_studies_unequal_scale",
        0x5e11_0000,
    );
}

/// Claim: the 95% Hartung-Knapp-Sidik-Jonkman interval of the random-effects
/// pooled mean covers the mean of the study effects.
#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_five_studies() {
    measure(
        Design::RandomEffects,
        5,
        "pooled_random_effects_five_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e20_0000,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn pooled_random_effects_ten_studies() {
    measure(
        Design::RandomEffects,
        10,
        "pooled_random_effects_ten_studies",
        "heterogeneous_gaussian_studies_unequal_scale",
        0x5e21_0000,
    );
}
