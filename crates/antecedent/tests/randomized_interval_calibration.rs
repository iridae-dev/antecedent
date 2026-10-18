//! Coverage designs for the pointwise 95% intervals of randomized trials at 300 participants.
//!
//! Each design runs the public randomized-effect route on a simulated trial and scores the
//! interval it publishes against the known finite-population average effect of the simulated
//! units (the effect is `1.4 + 0.2 * x1`, so it varies by unit, and the two baseline covariates
//! are fixed deterministic sequences). Only the assignment is random, which is the randomness
//! the intervals are about:
//!
//! * the independent-assignment score interval and the ANCOVA (HC0) interval, each under a
//!   balanced allocation and a 2:1 allocation, on 300 participants (the grid halves and doubles
//!   the size);
//! * the Neyman interval of a complete randomization with 150 of 300 units treated;
//! * the small-arm interval: a trial whose smaller arm realizes 20 of 300 participants gets the
//!   HC2 variance with a Bell-McCaffrey Student reference, for the difference in means of a
//!   complete randomization and for the ANCOVA. Its construction is read from the runtime's own
//!   calibration key, so its record is keyed apart from the normal intervals above.
//!
//! The designs are `#[ignore]`d measurement designs registered in `scripts/gate_calibration.sh`
//! (`run_ri`). The fixed-seed checks of the same routes are
//! `crates/antecedent/tests/randomized_precision_intervals.rs` and
//! `crates/antecedent/tests/randomized_trial_route.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{RandomizedEffectQuery, Study, StudyResult};
use antecedent_core::{ExecutionContext, RandomizationDesign, StreamDomain, VariableId};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::constructions;

const INTERVAL: &str = "analytic_se";
const OUTCOME: VariableId = VariableId::from_raw(0);
const BASELINE_A: VariableId = VariableId::from_raw(1);
const BASELINE_B: VariableId = VariableId::from_raw(2);

/// How the interval under test adjusts the outcome.
#[derive(Clone, Copy)]
enum Adjustment {
    /// The independent-assignment inverse-probability score contrast.
    Score,
    /// ANCOVA on both baseline covariates with the HC0 variance.
    Ancova,
}

/// How a replicate assigns the units.
#[derive(Clone, Copy)]
enum Assignment {
    /// Independent Bernoulli assignment with this treatment probability.
    Bernoulli(f64),
    /// Complete randomization of exactly this many treated units.
    Complete(usize),
    /// A trial that declares independent assignment with this probability and realizes exactly
    /// `treated` treated units: the allocation a weak arm leaves.
    DeclaredBernoulli { probability: f64, treated: usize },
    /// A smaller arm of exactly this many treated units: a complete randomization for the score
    /// contrast, a Bernoulli trial declaring `treated / n` for the ANCOVA.
    WeakArm { treated: usize },
}

/// The simulated units: baseline covariates and the known average effect.
struct Units {
    x1: Vec<f64>,
    x2: Vec<f64>,
    effect: f64,
}

fn units(n: usize) -> Units {
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let x2 = (0..n).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
    let effect = x1.iter().map(|x| 1.4 + 0.2 * x).sum::<f64>() / n as f64;
    Units { x1, x2, effect }
}

/// Draw the assignment of one replicate.
fn assign(n: usize, assignment: Assignment, seed: u64) -> Vec<bool> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5201);
    match assignment {
        Assignment::Bernoulli(probability) => {
            (0..n).map(|_| rng.next_f64() < probability).collect()
        }
        Assignment::Complete(treated)
        | Assignment::DeclaredBernoulli { treated, .. }
        | Assignment::WeakArm { treated } => {
            let keys = (0..n).map(|_| rng.next_f64()).collect::<Vec<_>>();
            let mut order = (0..n).collect::<Vec<_>>();
            order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
            let mut assigned = vec![false; n];
            for &unit in &order[..treated] {
                assigned[unit] = true;
            }
            assigned
        }
    }
}

/// The prepared study and result of one simulated trial and the known average effect; a hard
/// refusal is a bug in the design and panics.
fn run_trial(
    n: usize,
    adjustment: Adjustment,
    assignment: Assignment,
    seed: u64,
) -> (Study, StudyResult, f64) {
    let units = units(n);
    let assignment = match (assignment, adjustment) {
        (Assignment::WeakArm { treated }, Adjustment::Score) => Assignment::Complete(treated),
        (Assignment::WeakArm { treated }, Adjustment::Ancova) => {
            Assignment::DeclaredBernoulli { probability: treated as f64 / n as f64, treated }
        }
        (other, _) => other,
    };
    let assigned = assign(n, assignment, seed);
    let outcome = (0..n)
        .map(|i| {
            2.0 + 1.2 * units.x1[i] - 0.7 * units.x2[i]
                + 0.5 * (0.29 * i as f64).sin()
                + f64::from(assigned[i]) * (1.4 + 0.2 * units.x1[i])
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()),
        ("baseline_a", units.x1.as_slice()),
        ("baseline_b", units.x2.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned");
    let ids = (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let rows = (0..n).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>();
    let query = match assignment {
        Assignment::Bernoulli(probability) | Assignment::DeclaredBernoulli { probability, .. } => {
            RandomizedEffectQuery::bernoulli_itt(
                OUTCOME,
                assigned,
                vec![probability; n],
                ids,
                rows,
                ("control", "treated"),
            )
        }
        Assignment::Complete(treated) | Assignment::WeakArm { treated } => {
            RandomizedEffectQuery::with_design(
                RandomizationDesign::Complete { treated_units: treated },
                OUTCOME,
                assigned,
                vec![treated as f64 / n as f64; n],
                ids,
                rows,
                ("control", "treated"),
            )
        }
    };
    let query = match adjustment {
        Adjustment::Score => query,
        Adjustment::Ancova => query.with_ancova(vec![BASELINE_A, BASELINE_B]),
    };
    let study =
        Study::tabular(data).query(query).build().expect("the trial declares a valid design");
    let result = study
        .run(&ExecutionContext::for_tests(seed))
        .expect("the route runs on the simulated trial");
    (study, result, units.effect)
}

/// The interval the route publishes for one simulated trial (`None` when it withholds it), the
/// diagnostics it reports and the known average effect.
fn published(
    n: usize,
    adjustment: Adjustment,
    assignment: Assignment,
    seed: u64,
) -> (Option<[f64; 2]>, Vec<Arc<str>>, f64) {
    let (_, result, effect) = run_trial(n, adjustment, assignment, seed);
    let fit = result.randomized_effect().expect("a randomized effect result");
    let diagnostics = result.diagnostics.iter().map(|d| Arc::<str>::from(&*d.message)).collect();
    (fit.interval_95, diagnostics, effect)
}

fn construction(estimator: &str, se_kind: &str) -> Construction {
    Construction {
        query: "RandomizedEffect".into(),
        graph_class: "RandomizedTrial".into(),
        structure: "design".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: INTERVAL.into(),
        se_kind: se_kind.into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "randomized_itt".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn measure(
    test: &'static str,
    dgp: &'static str,
    construction: &Construction,
    adjustment: Adjustment,
    assignment: Assignment,
    seed_base: u64,
) {
    let n = grid_n(300);
    // The treated count is declared for 300 participants and keeps its share of the grid's size,
    // so a halved grid point is still a proper complete randomization.
    let assignment = match assignment {
        Assignment::Complete(treated) => Assignment::Complete(treated * n / 300),
        other => other,
    };
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    let runs = map_replicates(n_sim(), |rep| {
        published(n, adjustment, assignment, grid_seed(seed_base + rep))
    });
    for (interval, _, truth) in runs {
        // An allocation that realizes too few rows in an arm publishes no interval; the route
        // withholds it, so the replicate is a skip (a miss), never a free pass.
        let Some([lower, upper]) = interval else {
            tally.skip();
            continue;
        };
        tally.bind(
            construction,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some((lower, upper)), truth);
    }
    tally.assert();
}

/// A balanced trial publishes a finite ordered interval around the known effect.
#[test]
fn the_scored_intervals_are_ordered_and_centred_near_the_known_effect() {
    for adjustment in [Adjustment::Score, Adjustment::Ancova] {
        let (interval, _, truth) = published(300, adjustment, Assignment::Bernoulli(0.5), 7);
        let [lower, upper] = interval.expect("300 balanced rows publish an interval");
        assert!(lower.is_finite() && upper.is_finite() && lower < upper);
        assert!(((lower + upper) / 2.0 - truth).abs() < 0.5);
    }
    let (interval, _, truth) = published(300, Adjustment::Score, Assignment::Complete(150), 7);
    let [lower, upper] = interval.expect("a complete randomization publishes an interval");
    assert!(lower < truth + 0.5 && truth - 0.5 < upper);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_score_balanced_interval() {
    measure(
        "randomized_score_balanced_interval",
        "bernoulli_trial_balanced_known_finite_population_effect",
        &construction("randomized_independent_action_score", "influence"),
        Adjustment::Score,
        Assignment::Bernoulli(0.5),
        131_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_score_two_to_one_interval() {
    measure(
        "randomized_score_two_to_one_interval",
        "bernoulli_trial_two_to_one_known_finite_population_effect",
        &construction("randomized_independent_action_score", "influence"),
        Adjustment::Score,
        Assignment::Bernoulli(1.0 / 3.0),
        132_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_ancova_balanced_interval() {
    measure(
        "randomized_ancova_balanced_interval",
        "bernoulli_trial_balanced_known_finite_population_effect",
        &construction("randomized_ancova_hc0", "hc0"),
        Adjustment::Ancova,
        Assignment::Bernoulli(0.5),
        133_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_ancova_two_to_one_interval() {
    measure(
        "randomized_ancova_two_to_one_interval",
        "bernoulli_trial_two_to_one_known_finite_population_effect",
        &construction("randomized_ancova_hc0", "hc0"),
        Adjustment::Ancova,
        Assignment::Bernoulli(1.0 / 3.0),
        134_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn complete_randomization_neyman_interval() {
    // The Neyman variance is conservative under unit-level effect variation (here
    // `0.2 * x1`, a variance of 0.02 against an outcome variance near one), so the
    // design is held to nominal coverage from below and to its precision ceiling.
    measure(
        "complete_randomization_neyman_interval",
        "complete_randomization_half_treated_known_finite_population_effect",
        &construction("randomized_complete_neyman", "neyman"),
        Adjustment::Score,
        Assignment::Complete(150),
        135_000,
    );
}

/// Coverage of the small-arm interval: every replicate's construction and scope facts come from
/// the runtime's own calibration key, so the record is keyed by what the facade reports.
fn measure_small_arm(
    test: &'static str,
    dgp: &'static str,
    adjustment: Adjustment,
    treated: usize,
    seed_base: u64,
) {
    let n = grid_n(300);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    let runs = map_replicates(n_sim(), |rep| {
        let (study, result, truth) =
            run_trial(n, adjustment, Assignment::WeakArm { treated }, grid_seed(seed_base + rep));
        let interval = result.randomized_effect().expect("a randomized result").interval_95;
        let bound = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _)| construction.interval_method == INTERVAL);
        (interval, truth, bound)
    });
    for (interval, truth, bound) in runs {
        // A replicate that publishes no interval is a skip (a miss), never a free pass.
        let (Some([lower, upper]), Some((construction, scope))) = (interval, bound) else {
            tally.skip();
            continue;
        };
        tally.bind(&construction, scope);
        tally.record(Some((lower, upper)), truth);
    }
    tally.assert();
}

/// A smaller arm of 20 of 300 participants is below the normal interval's support. The
/// difference in means of a complete randomization publishes the HC2 Bell-McCaffrey Student
/// interval, held to nominal coverage and to the precision ceiling.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weak_arm_difference_in_means_hc2_student_interval() {
    measure_small_arm(
        "weak_arm_difference_in_means_hc2_student_interval",
        "complete_randomization_twenty_treated_known_finite_population_effect",
        Adjustment::Score,
        20,
        136_000,
    );
}

/// The ANCOVA of a trial whose smaller arm realizes 20 of 300 participants publishes the same
/// construction over the two baseline covariates.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weak_arm_ancova_hc2_student_interval() {
    measure_small_arm(
        "weak_arm_ancova_hc2_student_interval",
        "bernoulli_trial_twenty_treated_known_finite_population_effect",
        Adjustment::Ancova,
        20,
        137_000,
    );
}

/// Fixed-seed check of the small-arm routes: a published interval, its own construction label,
/// the Student reference (wider than the normal), and a calibration key apart from the normal
/// interval's.
#[test]
fn the_small_arm_interval_is_published_under_its_own_construction() {
    for (adjustment, label) in [
        (Adjustment::Score, "complete_difference_in_means_hc2_bell_mccaffrey_student_interval"),
        (Adjustment::Ancova, "bernoulli_ancova_hc2_bell_mccaffrey_student_interval"),
    ] {
        let (study, result, truth) =
            run_trial(300, adjustment, Assignment::WeakArm { treated: 20 }, 7);
        let fit = result.randomized_effect().expect("a randomized effect result");
        let [lower, upper] = fit.interval_95.expect("a 20-participant arm publishes an interval");
        assert_eq!(fit.uncertainty.as_ref(), label);
        assert!(lower < truth + 0.5 && truth - 0.5 < upper);
        let standard_error = fit.standard_error.expect("the interval carries its standard error");
        assert!(
            (upper - lower) / (2.0 * standard_error) > 1.96,
            "a Student reference, not the normal"
        );
        let (construction, _) = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _)| construction.interval_method == INTERVAL)
            .expect("an analytic interval is reported");
        assert_eq!(construction.se_kind, "hc2");
    }
}
