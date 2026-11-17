//! Coverage of the intervals read from joint-response results.
//!
//! Each test repeats one whole procedure on independent data sets: data generation, design
//! compilation with its support and rank checks, the native frequentist or Bayesian fit, and
//! the interval the library publishes for a prespecified target. The interval and the
//! calibration key both come from the result (`contrast_interval_reading`,
//! `selected_value_reading`, `population_reference_region`, `updated_contrast_reading`), so
//! a record describes exactly the construction the runtime resolves. A replicate that
//! publishes no interval, or whose key falls outside the support condition the cell
//! measures, is a skip and counts as a miss.
//!
//! Sizes scale with the sample-size grid (one, two and four times the smallest design), so
//! the smallest design of each cell is the lower end of its record's measured range.
//!
//! * Native designs: a switchback with one period of carryover, a structural-feature
//!   instrument design, cohort difference-in-differences, a two-rollout synthetic control,
//!   a continuous treatment-affected longitudinal design, an unequally weighted geo
//!   experiment, a factorial experiment, a finite-history longitudinal design and a
//!   one-sided-receipt encouragement design, each in frequentist and Bayesian mode. Every
//!   Bayesian cell declares its likelihood scale at the generating value; the synthetic
//!   control declares an inverse-gamma variance prior whose mean is the generating residual
//!   variance, and the runtime checks that mean against the pre-period fit error.
//! * Held-out selection: the value of a program selected on training data and evaluated on
//!   reserved units, at exact ties, near ties and a separated winner, and a continuous
//!   structural dose with weak separation.
//! * Reference decomposition: the simultaneous region of the four Shapley contributions.
//! * Experiment update: the posterior interval of a contrast after an independent
//!   experiment, with the truth drawn from the declared prior.
//!
//! Two constructions are conservative by the union bound and are one-sided records: the
//! conformal contrast (projection of a Bonferroni rectangle over two rollouts) and the
//! decomposition regions in both modes (Bonferroni over four contributions).
//!
//! The tests at the end of the file are not simulations: each compares one reading with an
//! interval computed here from the published law and checks the key it derives.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices and design sizes in a calibration fixture"
)]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]
#![allow(clippy::too_many_lines, reason = "each design keeps its complete setup together")]

mod common;
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

use antecedent::analysis::joint_panel::designs::{
    JointDesignSourceBinding, NativeJointDesignResult, PreparedNativeJointDesign,
    joint_design_world_identity, replay_native_joint_design,
};
use antecedent::analysis::joint_panel::selection::instrumented::{
    InstrumentedDomainGeneration, InstrumentedDoseControl, InstrumentedSelectionRequest,
};
use antecedent::analysis::joint_panel::selection::{
    PreparedRandomizedSelection, RandomizedCandidateGeneration, RandomizedSelectionPolicy,
};
use antecedent::analysis::joint_panel::{
    ChangeComponent, JointDesignProof, JointDesignResult, JointDesignWorldBinding,
    JointLawExperimentRequest, PanelExperimentEvidence, PanelExperimentMapping,
    PopulationReferenceBindings, PreparedJointDesign, ReferenceComponentBindings,
};
use antecedent_contract::bridge::*;
use antecedent_contract::evidence_ledger::*;
use antecedent_core::execution::CausalRng;
use antecedent_core::{
    AssumptionSet, CalibrationBasis, Conditioning, ExecutionContext, GraphFamily,
    InferenceSelection, Intervention, InterventionSequence, InterventionWorld, ModelAssumptions,
    ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId,
    SequencedIntervention, TargetForm, TargetPopulation, TemporalPolicy, TimeFrame, Value,
    VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_design::program_search::certified::CertificateOptions;
use antecedent_design::scientific_source::{ScientificConsumptionScope, ScientificSourcePayload};
use antecedent_estimate::joint_design::compiled::{JointDesignInference, JointDesignRecipe};
use antecedent_estimate::joint_design::randomized::{
    JointRandomizedDesign, JointRandomizedUnit, RandomizedFinitePopulationPrior,
    RandomizedGaussianPrior,
};
use antecedent_estimate::joint_design::{instrumented::*, rollout::*, switchback::*, synthetic::*};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, grid_point, grid_seed,
    map_replicates, n_sim,
};

const LEVEL: f64 = 0.95;

/// One, two and four times the smallest design, by grid point.
fn scale() -> usize {
    [1, 2, 4][grid_point()]
}

/// One replicate's reading in one inference mode.
enum Reading {
    /// A published interval, its truth and the key the result derived.
    Interval { bounds: [f64; 2], truth: f64, basis: CalibrationBasis },
    /// A published simultaneous region: whether it holds every truth, and its bounds.
    Region { covered: bool, bounds: Vec<f64>, basis: CalibrationBasis },
    /// No interval was published, and why.
    Refused(String),
}

fn bind_basis(tally: &mut CoverageTally, basis: &CalibrationBasis) {
    tally.bind_validation(
        &Construction {
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
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
        &basis.validation,
    );
}

/// What one record scores: its interval construction, the support condition it measures and,
/// for an interval that is conservative by construction, the target it covers at least
/// nominally.
struct Target {
    mode: &'static str,
    interval: &'static str,
    support: &'static str,
    conservative: Option<&'static str>,
}

/// Fold one mode's readings into its record and hold it to the gate.
fn score<'a>(
    test: &'static str,
    dgp: &'static str,
    target: &Target,
    readings: impl Iterator<Item = &'a Reading>,
) {
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: target.interval }, LEVEL)
            .named(format!("{test} {}", target.mode));
    let mut refusals = std::collections::BTreeMap::<String, usize>::new();
    for reading in readings {
        let (basis, refusal) = match reading {
            Reading::Interval { basis, .. } | Reading::Region { basis, .. } => {
                (Some(basis), format!("outside the measured support: {}", basis.dependence))
            }
            Reading::Refused(reason) => (None, reason.clone()),
        };
        let Some(basis) = basis.filter(|basis| &*basis.dependence == target.support) else {
            tally.skip();
            *refusals.entry(refusal).or_default() += 1;
            continue;
        };
        bind_basis(&mut tally, basis);
        match reading {
            Reading::Interval { bounds, truth, .. } => {
                tally.record(Some((bounds[0], bounds[1])), *truth);
            }
            Reading::Region { covered, bounds, .. } => {
                for bound in bounds {
                    tally.absorb(bound);
                }
                // A flag scored as coverage: the unit interval holds 0.5 and not 2.
                tally.record(Some((0.0, 1.0)), if *covered { 0.5 } else { 2.0 });
            }
            Reading::Refused(_) => unreachable!("a refusal carries no key"),
        }
    }
    for (reason, count) in &refusals {
        eprintln!("{test} {}: {count} replicate(s) scored no interval: {reason}", target.mode);
    }
    match target.conservative {
        Some(covered) => tally.assert_one_sided(CoverageSide::Upper, covered),
        None => tally.assert(),
    }
}

// ---------------------------------------------------------------------------------------
// Native designs.
// ---------------------------------------------------------------------------------------

/// A design template, its Bayesian procedure and the prespecified contrast.
type Fixture = (JointDesignRecipe, JointDesignInference, Vec<f64>);

/// Publish the contrast interval of one inference mode, replaying the artifact as a caller
/// would.
fn native_reading(
    prepared: &PreparedNativeJointDesign,
    inference: JointDesignInference,
    contrast: &[f64],
    truth: f64,
    ctx: &ExecutionContext,
) -> Reading {
    let result = match prepared.execute(inference, ctx) {
        Ok(result) => result,
        Err(error) => return Reading::Refused(format!("execute: {error}")),
    };
    let bytes = result.export_artifact().expect("the executed design exports");
    assert_eq!(*result.law(), replay_native_joint_design(&bytes, ctx).expect("replay"));
    prepared.consume_current(&bytes, ctx).expect("current consumption");
    match result.contrast_interval_reading(contrast, LEVEL) {
        Ok(reading) => Reading::Interval { bounds: reading.bounds, truth, basis: reading.basis },
        Err(error) => Reading::Refused(format!("interval: {error}")),
    }
}

/// Measure one native design in both inference modes. `index` is the cell's seed offset;
/// `data` draws one data set into the template and returns a realized truth when the target
/// is a finite-population quantity.
#[allow(clippy::too_many_arguments, reason = "one call states a whole measured cell")]
#[allow(
    clippy::needless_pass_by_value,
    reason = "each call states its two targets as an array literal"
)]
fn native_cell(
    test: &'static str,
    dgp: &'static str,
    index: u64,
    (template, bayesian, contrast): &Fixture,
    truth: f64,
    data: fn(&mut JointDesignRecipe, &mut CausalRng) -> Option<f64>,
    frequentist: impl Fn(u64) -> JointDesignInference + Sync,
    targets: [Target; 2],
) {
    let readings = map_replicates(n_sim(), |replicate| {
        let mut rng = CausalRng::from_seed(grid_seed(60_000 + index * 1000 + replicate));
        let mut recipe = template.clone();
        let truth = data(&mut recipe, &mut rng).unwrap_or(truth);
        let ctx = ExecutionContext::for_tests(grid_seed(70_000 + replicate));
        match PreparedNativeJointDesign::compile(recipe, &ctx) {
            Ok(prepared) => [frequentist(replicate), bayesian.clone()]
                .map(|inference| native_reading(&prepared, inference, contrast, truth, &ctx)),
            Err(error) => [0, 1].map(|_| Reading::Refused(format!("compile: {error}"))),
        }
    });
    for (mode, target) in targets.iter().enumerate() {
        score(test, dgp, target, readings.iter().map(|pair| &pair[mode]));
    }
}

fn plain(_: u64) -> JointDesignInference {
    JointDesignInference::Frequentist
}

fn modes(
    frequentist: (&'static str, Option<&'static str>),
    support: [&'static str; 2],
) -> [Target; 2] {
    [
        Target {
            mode: "frequentist",
            interval: frequentist.0,
            support: support[0],
            conservative: frequentist.1,
        },
        Target {
            mode: "bayesian",
            interval: "equal_tailed_posterior",
            support: support[1],
            conservative: None,
        },
    ]
}

/// `repeats` independent sequences at each of the sixteen two-period schedules.
fn switchback_fixture(repeats: usize) -> Fixture {
    let schedules = (0..4).flat_map(|a| (0..4).map(move |b| vec![a, b])).collect::<Vec<_>>();
    let mut sequences = Vec::new();
    for _ in 0..repeats {
        for realized in 0..16 {
            sequences.push(antecedent_estimate::switchback::HistoryAssignment {
                schedules: schedules.clone(),
                probabilities: vec![1.0 / 16.0; 16],
                realized,
                outcomes: vec![0.0, 0.0],
            });
        }
    }
    let design = JointSwitchbackDesign {
        worlds: schedules.iter().enumerate().map(|(i, h)| (format!("h{i}"), h.clone())).collect(),
        carryover: 1,
        no_anticipation: "future assignments do not affect current outcomes".into(),
        finite_carryover: "response depends only on current and preceding joint arm".into(),
        sequence_isolation: "independently randomized noninterfering sequences".into(),
        population: "eligible terminal periods over these independent sequences".into(),
    };
    let prior = SwitchbackGaussianPrior {
        mean: vec![0.0; 16],
        variance: vec![100.0; 16],
        sequence_variance: vec![1.0; sequences.len()],
        sequence_rho: vec![0.0; sequences.len()],
        draws: 8000,
        seed: 445,
    };
    let mut contrast = vec![0.0; 16];
    contrast[..4].copy_from_slice(&[1.0, -1.0, -1.0, 1.0]);
    (
        JointDesignRecipe::Switchback { design, sequences },
        JointDesignInference::SwitchbackGaussian { prior },
        contrast,
    )
}

/// Each sequence draws its schedule uniformly; the current arm acts through both factors
/// and their interaction, the preceding arm through carryover.
fn switchback_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::Switchback { sequences, .. } = recipe else {
        unreachable!("switchback template")
    };
    for sequence in sequences {
        let realized = (rng.next_f64() * sequence.schedules.len() as f64) as usize;
        sequence.realized = realized;
        let history = &sequence.schedules[realized];
        let (past, arm) = (history[0], history[1]);
        sequence.outcomes = vec![
            standard_normal(rng),
            (2 * (arm & 1) + 3 * (arm >> 1) + 4 * (arm & 1) * (arm >> 1)) as f64
                + 1.5 * (past & 1) as f64
                + 2.0 * (past >> 1) as f64
                + standard_normal(rng),
        ];
    }
    None
}

/// The switchback interaction contrast. Every one of the sixteen histories must be observed,
/// and the unnormalized inverse-probability estimator is skewed at few sequences per
/// history, so the smallest design is 512 sequences (32 per history).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_switchback_contrast_l95() {
    native_cell(
        "joint_response_switchback_contrast_l95",
        "switchback_data",
        0,
        &switchback_fixture(32 * scale()),
        4.0,
        switchback_data,
        plain,
        modes(("normal_approximation", None), ["sequences_per_history_at_least_32"; 2]),
    );
}

/// `repeats` copies of a two-instrument sign design with three structural features.
fn structural_iv_fixture(repeats: usize) -> Fixture {
    let mut units = Vec::new();
    for repeat in 0..repeats {
        for a in [-1.0, 1.0] {
            for b in [-1.0, 1.0] {
                for cell in 0..8 {
                    units.push(JointInstrumentedUnit {
                        id: format!("{repeat}/{a}/{b}/{cell}"),
                        endogenous: vec![a, b, a * b],
                        instruments: vec![a, b, a * b],
                        exogenous: vec![1.0],
                        outcome: 0.0,
                    });
                }
            }
        }
    }
    let design = JointInstrumentedDesign {
        contrasts: vec![
            ("a".into(), vec![1.0, 0.0, 0.0]),
            ("ab".into(), vec![0.0, 0.0, 1.0]),
            ("all".into(), vec![1.0, 1.0, 1.0]),
        ],
        structural_model: "homogeneous structural response with actual D1,D2,D1D2 features".into(),
        moment_restrictions:
            "Z=(A,B,AB) independent raw-dose/outcome disturbances; excluded structural instruments"
                .into(),
        population: "original independent IV population".into(),
    };
    let prior = JointIvGaussianPrior {
        raw_feature_indices: vec![0, 1],
        structural_features: vec![vec![1, 0], vec![0, 1], vec![1, 1]],
        parameter_mean: vec![0.0; 12],
        parameter_variance: vec![1000.0; 12],
        disturbance_covariance: vec![1.0, 0.0, 0.6, 0.0, 1.0, 0.4, 0.6, 0.4, 1.52],
        draws: 8000,
        seed: 446,
    };
    (
        JointDesignRecipe::Instrumented { design, units },
        JointDesignInference::InstrumentedGaussian { prior },
        vec![0.0, 0.0, 1.0],
    )
}

/// Doses are the instruments plus unit-variance first-stage noise that also enters the
/// outcome, so the doses are endogenous.
fn structural_iv_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::Instrumented { units, .. } = recipe else {
        unreachable!("instrumented template")
    };
    for unit in units {
        let (v1, v2) = (standard_normal(rng), standard_normal(rng));
        let (d1, d2) = (unit.instruments[0] + v1, unit.instruments[1] + v2);
        unit.endogenous = vec![d1, d2, d1 * d2];
        unit.outcome =
            2.0 * d1 + 3.0 * d2 + 4.0 * d1 * d2 + 5.0 + 0.6 * v1 + 0.4 * v2 + standard_normal(rng);
    }
    None
}

/// The sum of the three structural coefficients, from 128 units.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_structural_iv_contrast_l95() {
    native_cell(
        "joint_response_structural_iv_contrast_l95",
        "structural_iv_data",
        1,
        &structural_iv_fixture(4 * scale()),
        9.0,
        structural_iv_data,
        plain,
        modes(("normal_approximation", None), ["independent_units"; 2]),
    );
}

/// Four joint regimes of `per_regime` units observed before and after adoption.
fn cohort_did_fixture(per_regime: usize) -> Fixture {
    let mut units = Vec::new();
    for (regime, effect) in [("00", 0.0), ("10", 2.0), ("01", 3.0), ("11", 9.0)] {
        for i in 0..per_regime {
            units.push(JointRolloutUnit {
                id: format!("{regime}/{i}"),
                regime: regime.into(),
                adoption: (regime != "00").then_some(2),
                outcomes: vec![(1, (i % 4) as f64), (2, (i % 4) as f64 + 5.0 + effect)],
            });
        }
    }
    let design = JointRolloutDesign {
        cells: ["10", "01", "11"]
            .into_iter()
            .map(|r| JointRolloutCell {
                key: r.into(),
                regime: r.into(),
                adoption: 2,
                time: 2,
                baseline: 1,
            })
            .collect(),
        no_anticipation: "no effects before period2".into(),
        parallel_trends: "common untreated calendar increment".into(),
        consistency_and_isolation: "complete joint regimes, independent units".into(),
        population: "adopting regime cohorts".into(),
    };
    let prior = JointRolloutGaussianPrior {
        unit_mean: 0.0,
        unit_variance: 1e6,
        time_mean: 0.0,
        time_variance: 1e6,
        effect_mean: vec![0.0; 3],
        effect_variance: vec![1e6; 3],
        outcome_variance: 1.0,
        ar1_rho: 0.0,
        draws: 10000,
        seed: 447,
    };
    (
        JointDesignRecipe::DifferenceInDifferences { design, units },
        JointDesignInference::RolloutGaussian { prior },
        vec![-1.0, -1.0, 1.0],
    )
}

/// Unit-variance noise on every unit-period outcome.
fn cohort_did_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::DifferenceInDifferences { units, .. } = recipe else {
        unreachable!("rollout template")
    };
    for unit in units {
        for (_, outcome) in &mut unit.outcomes {
            *outcome += standard_normal(rng);
        }
    }
    None
}

/// The joint-regime interaction of three cohort effects, from four units per regime.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_cohort_did_contrast_l95() {
    native_cell(
        "joint_response_cohort_did_contrast_l95",
        "cohort_did_data",
        2,
        &cohort_did_fixture(4 * scale()),
        4.0,
        cohort_did_data,
        plain,
        modes(("normal_approximation", None), ["independent_units"; 2]),
    );
}

/// Three unaffected donors and two treated units; both rollouts start after `pre` periods
/// and are followed for two.
fn synthetic_rollout_fixture(pre: i64) -> Fixture {
    let (periods, intervention) = (pre + 2, pre + 1);
    let mut observations = Vec::new();
    for time in 1..=periods {
        let x = time as f64;
        let donors =
            [0.2 * x + (0.5 * x).sin(), 0.1 * x + (0.3 * x).cos(), 0.3 * x + (time % 5) as f64];
        for (j, outcome) in donors.iter().enumerate() {
            observations.push(JointSyntheticObservation {
                unit: format!("d{j}"),
                time,
                outcome: *outcome,
            });
        }
        for (index, effect) in [4.0, 8.0].iter().enumerate() {
            observations.push(JointSyntheticObservation {
                unit: format!("treated{index}"),
                time,
                outcome: donors.iter().sum::<f64>() / 3.0
                    + if time >= intervention { *effect } else { 0.0 },
            });
        }
    }
    let design = JointSyntheticDesign {
        worlds: (0..2)
            .map(|index| JointSyntheticWorld {
                key: format!("joint{index}"),
                treated_unit: format!("treated{index}"),
                intervention,
                donors: (0..3).map(|j| format!("d{j}")).collect(),
            })
            .collect(),
        counterfactual_model: "untreated response remains the preperiod simplex donor combination"
            .into(),
        donor_isolation: "all donors unaffected; no anticipation or interunit contamination".into(),
        population: format!(
            "each original treated-unit rollout, postperiod{intervention}:{periods}"
        ),
    };
    // The inverse-gamma prior mean 0.03 / (4 - 1) is the residual variance 0.1^2 the treated
    // rows are generated with: the declared prior is consistent with the data scale.
    let prior = JointSyntheticPrior {
        dirichlet_concentration: 10.0,
        variance_shape: 4.0,
        variance_scale: 0.03,
        ar1_rho: 0.0,
        burn_in: 2000,
        draws: 8000,
        seed: 448,
        independent_treated_residuals:
            "independent treated counterfactual errors conditional on observed donors".into(),
    };
    (
        JointDesignRecipe::SyntheticControl { design, observations },
        JointDesignInference::SyntheticBayesian { prior },
        vec![-1.0, 1.0],
    )
}

/// Independent Gaussian residuals of standard deviation 0.1 on every treated row.
fn synthetic_rollout_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::SyntheticControl { observations, .. } = recipe else {
        unreachable!("synthetic template")
    };
    for row in observations {
        if row.unit.starts_with("treated") {
            row.outcome += 0.1 * standard_normal(rng);
        }
    }
    None
}

/// The difference of two rollout effects. A joint 95% conformal region over two rollouts
/// tests each at 0.025 and needs at least 39 pre-periods; the smallest design has 40. The
/// conformal contrast is the projection of a Bonferroni rectangle, so it is held to at
/// least nominal coverage.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_synthetic_rollout_contrast_l95() {
    native_cell(
        "joint_response_synthetic_rollout_contrast_l95",
        "synthetic_rollout_data",
        3,
        &synthetic_rollout_fixture(
            40 * i64::try_from(scale()).expect("the grid scale fits in i64"),
        ),
        4.0,
        synthetic_rollout_data,
        |_| {
            JointDesignInference::SyntheticConformal {
            joint_level: LEVEL,
            residual_exchangeability:
                "independent Gaussian treated residuals exchangeable across all observed time periods"
                    .into(),
        }
        },
        modes(
            (
                "conformal_projection",
                Some(
                    "the difference of the two rollout effects, covered whenever the joint \
                     conformal rectangle covers both",
                ),
            ),
            ["exchangeable_residuals", "variance_prior_within_2x_preperiod_error"],
        ),
    );
}

/// `repeats` copies of a two-period continuous design with a treatment-affected covariate.
fn continuous_longitudinal_fixture(repeats: usize) -> Fixture {
    use antecedent_estimate::joint_design::gaussian_longitudinal::*;
    let episodes = (0..repeats * 32)
        .map(|i| GaussianLongitudinalEpisode {
            id: format!("episode{i}"),
            covariates: vec![vec![0.0], vec![0.0]],
            treatments: vec![vec![0.0], vec![0.0]],
            outcome: 0.0,
        })
        .collect();
    let mut model = GaussianLongitudinalModel::full_history(2, 1, 1);
    model.outcome.push(GaussianHistoryFeature::TreatmentProduct {
        first_time: 0,
        first_channel: 0,
        second_time: 1,
        second_channel: 0,
    });
    model.outcome.push(GaussianHistoryFeature::CovariateTreatment {
        covariate_time: 1,
        coordinate: 0,
        treatment_time: 1,
        channel: 0,
    });
    let design = GaussianLongitudinalDesign {
        channel_variables: vec![1],
        covariate_variables: vec![2],
        worlds: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .enumerate()
            .map(|(i, (a, b))| GaussianLongitudinalWorld {
                key: i.to_string(),
                treatments: vec![vec![a], vec![b]],
            })
            .collect(),
        assignment_support: vec![GaussianAssignmentSupport::Real],
        sequential_exchangeability: "full recorded histories suffice before each joint assignment"
            .into(),
        consistency: "stable continuous intervention versions".into(),
        population: "original independent fullhistory source episodes".into(),
        conditional_model:
            "conditional Gaussian full-history state/outcome and treatment-policy mechanisms".into(),
        model,
    };
    (
        JointDesignRecipe::GaussianLongitudinal { design, episodes },
        JointDesignInference::GaussianLongitudinalBayesian {
            prior: GaussianLongitudinalPrior {
                coefficient_scale: 1e6,
                variance_shape: 2.0,
                variance_scale: 1.0,
                draws: 512,
                seed: 556,
            },
        },
        vec![1.0, -1.0, -1.0, 1.0],
    )
}

/// Linear Gaussian state, policy and outcome mechanisms with unit-variance disturbances.
fn continuous_longitudinal_data(
    recipe: &mut JointDesignRecipe,
    rng: &mut CausalRng,
) -> Option<f64> {
    let JointDesignRecipe::GaussianLongitudinal { episodes, .. } = recipe else {
        unreachable!("continuous longitudinal template")
    };
    for episode in episodes {
        let l0 = standard_normal(rng);
        let a0 = 0.6 * l0 + standard_normal(rng);
        let l1 = 1.0 + 2.0 * a0 + 0.5 * l0 + standard_normal(rng);
        let a1 = 0.4 * l1 + 0.2 * a0 + standard_normal(rng);
        episode.covariates = vec![vec![l0], vec![l1]];
        episode.treatments = vec![vec![a0], vec![a1]];
        episode.outcome = 3.0
            + a0
            + 2.0 * a1
            + 4.0 * l1
            + 0.3 * l0
            + 5.0 * a0 * a1
            + 0.2 * l1 * a1
            + standard_normal(rng);
    }
    None
}

/// The two-period interaction of regime means, from 96 episodes; the frequentist interval
/// uses the covariance of 128 whole-procedure bootstrap refits.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_continuous_longitudinal_contrast_l95() {
    native_cell(
        "joint_response_continuous_longitudinal_contrast_l95",
        "continuous_longitudinal_data",
        4,
        &continuous_longitudinal_fixture(3 * scale()),
        5.4,
        continuous_longitudinal_data,
        |replicate| JointDesignInference::GaussianLongitudinalBootstrap {
            replicates: 128,
            seed: replicate + 80_000,
        },
        modes(("bootstrap_covariance_normal", None), ["independent_episodes"; 2]),
    );
}

/// A two-arm geo census with population weights 1, 1, 2, 2, 3, 3, 4, 4, repeated.
fn weighted_geo_fixture(geos: usize) -> Fixture {
    let design = JointRandomizedDesign {
        worlds: vec!["reference".into(), "joint".into()],
        sampling_unit: "geo".into(),
        population: "original unequal weighted geo census".into(),
        consistency: "wholeprogram assignment within each geo".into(),
        between_unit_isolation: "no spillovers between independently randomized geos".into(),
    };
    let units = (0..geos)
        .map(|i| JointRandomizedUnit {
            id: format!("geo{i}"),
            arm: i % 2,
            probabilities: vec![0.5; 2],
            outcome: 0.0,
            population_weight: ((i % 8) / 2 + 1) as f64,
        })
        .collect();
    let prior = RandomizedFinitePopulationPrior {
        arm: RandomizedGaussianPrior {
            mean: vec![2.0, 5.0],
            variance: vec![4.0; 2],
            outcome_variance: vec![1.0; geos],
            draws: 1024,
            seed: 557,
        },
        heterogeneity_variance: vec![2.0; 2],
    };
    (
        JointDesignRecipe::Randomized { design, units },
        JointDesignInference::RandomizedFinitePopulationGaussian { prior },
        vec![-1.0, 1.0],
    )
}

/// The whole census of potential responses is drawn first; its realized weighted contrast,
/// not its expectation, is the target. Each geo is then assigned by a fair coin.
#[allow(
    clippy::unnecessary_wraps,
    reason = "every data generator shares the fn-pointer signature of `native_cell`, and this one always draws a realized target"
)]
fn weighted_geo_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::Randomized { units, .. } = recipe else {
        unreachable!("randomized template")
    };
    let (mut contrast, mut total) = (0.0, 0.0);
    for unit in units {
        let potential = [
            2.0 + 2.0_f64.sqrt() * standard_normal(rng),
            5.0 + 2.0_f64.sqrt() * standard_normal(rng),
        ];
        contrast += unit.population_weight * (potential[1] - potential[0]);
        total += unit.population_weight;
        unit.arm = usize::from(rng.next_f64() >= 0.5);
        unit.outcome = potential[unit.arm] + standard_normal(rng);
    }
    Some(contrast / total)
}

/// The weighted census contrast of two programs from eight geos. A data set with an empty
/// arm is refused and counts as a miss.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_weighted_geo_contrast_l95() {
    native_cell(
        "joint_response_weighted_geo_contrast_l95",
        "weighted_geo_data",
        5,
        &weighted_geo_fixture(8 * scale()),
        3.0,
        weighted_geo_data,
        plain,
        modes(("satterthwaite_t", None), ["unequal_weights_4_units_per_program"; 2]),
    );
}

/// A two-by-two factorial with equal weights and assignment probabilities.
fn factorial_fixture(units: usize) -> Fixture {
    (
        JointDesignRecipe::Randomized {
            design: JointRandomizedDesign {
                worlds: (0..4).map(|i| format!("arm{i}")).collect(),
                sampling_unit: "participant".into(),
                population: "factorial original population".into(),
                consistency: "complete factorial joint arms".into(),
                between_unit_isolation: "independent participants".into(),
            },
            units: (0..units)
                .map(|i| JointRandomizedUnit {
                    id: format!("participant{i}"),
                    arm: i % 4,
                    probabilities: vec![0.25; 4],
                    outcome: 0.0,
                    population_weight: 1.0,
                })
                .collect(),
        },
        JointDesignInference::RandomizedGaussian {
            prior: RandomizedGaussianPrior {
                mean: vec![0.0; 4],
                variance: vec![100.0; 4],
                outcome_variance: vec![1.0; units],
                draws: 1024,
                seed: 559,
            },
        },
        vec![1.0, -1.0, -1.0, 1.0],
    )
}

/// Each participant draws one of four arms uniformly; arm means 0, 1, 2, 4, unit noise.
fn factorial_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::Randomized { units, .. } = recipe else {
        unreachable!("randomized template")
    };
    for unit in units {
        unit.arm = (rng.next_f64() * 4.0) as usize;
        unit.outcome = [0.0, 1.0, 2.0, 4.0][unit.arm] + standard_normal(rng);
    }
    None
}

/// The factorial interaction from 128 participants.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_factorial_contrast_l95() {
    native_cell(
        "joint_response_factorial_contrast_l95",
        "factorial_data",
        6,
        &factorial_fixture(128 * scale()),
        1.0,
        factorial_data,
        plain,
        modes(("satterthwaite_t", None), ["equal_weights_32_units_per_program"; 2]),
    );
}

/// Binary two-period episodes with a treatment-affected binary covariate.
fn finite_longitudinal_fixture(episodes: usize) -> Fixture {
    use antecedent_estimate::joint_design::longitudinal::{
        LongitudinalDirichletPrior, LongitudinalEpisode, LongitudinalJointDesign,
    };
    (
        JointDesignRecipe::Longitudinal {
            design: LongitudinalJointDesign {
                worlds: vec![("00".into(), vec![0, 0]), ("11".into(), vec![1, 1])],
                sequential_exchangeability:
                    "recorded past states suffice before each randomized joint arm".into(),
                consistency: "complete versions".into(),
                sampling_population: "independent binary source episodes".into(),
            },
            episodes: (0..episodes)
                .map(|i| LongitudinalEpisode {
                    id: format!("binary{i}"),
                    covariates: vec![0, 0],
                    assignments: vec![0, 0],
                    outcome: 0.0,
                })
                .collect(),
        },
        JointDesignInference::LongitudinalDirichlet {
            prior: LongitudinalDirichletPrior {
                covariate_states: vec![vec![0, 1]; 2],
                outcome_values: vec![0.0, 1.0],
                concentration: 1.0,
                draws: 1024,
                seed: 561,
            },
        },
        vec![-1.0, 1.0],
    )
}

/// Fair-coin assignments; the second covariate depends on the first assignment and the
/// outcome on both covariates and the second assignment.
fn finite_longitudinal_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::Longitudinal { episodes, .. } = recipe else {
        unreachable!("longitudinal template")
    };
    for episode in episodes {
        let l0 = usize::from(rng.next_f64() < 0.5);
        let a0 = usize::from(rng.next_f64() < 0.5);
        let l1 = usize::from(rng.next_f64() < 0.25 + 0.5 * a0 as f64);
        let a1 = usize::from(rng.next_f64() < 0.5);
        let outcome = f64::from(
            rng.next_f64() < 0.125 + 0.25 * l0 as f64 + 0.25 * l1 as f64 + 0.25 * a1 as f64,
        );
        episode.covariates = vec![l0, l1];
        episode.assignments = vec![a0, a1];
        episode.outcome = outcome;
    }
    None
}

/// The always-treated against never-treated regime contrast from 512 episodes.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_finite_longitudinal_contrast_l95() {
    native_cell(
        "joint_response_finite_longitudinal_contrast_l95",
        "finite_longitudinal_data",
        7,
        &finite_longitudinal_fixture(512 * scale()),
        0.375,
        finite_longitudinal_data,
        plain,
        modes(("normal_approximation", None), ["independent_episodes"; 2]),
    );
}

/// A randomized encouragement with one-sided receipt.
fn local_complier_fixture(units: usize) -> Fixture {
    use antecedent_estimate::joint_design::local_instrument::{
        LocalInstrumentDesign, LocalInstrumentUnit,
    };
    (
        JointDesignRecipe::InstrumentedLocal {
            design: LocalInstrumentDesign {
                key: "local".into(),
                population: "encouragement source compliers".into(),
                exclusion: "no direct encouragement effect".into(),
                monotonicity: "one-sided receipt; no defiers".into(),
                consistency_and_isolation: "independent complete versions".into(),
            },
            units: (0..units)
                .map(|i| LocalInstrumentUnit {
                    id: format!("local{i}"),
                    assigned: false,
                    received: false,
                    probability: 0.5,
                    outcome: 0.0,
                })
                .collect(),
        },
        JointDesignInference::LocalComplierBayesian { draws: 1024, seed: 562 },
        vec![1.0],
    )
}

/// Half the units are encouraged, half of those receive the program; receipt adds 3.
fn local_complier_data(recipe: &mut JointDesignRecipe, rng: &mut CausalRng) -> Option<f64> {
    let JointDesignRecipe::InstrumentedLocal { units, .. } = recipe else {
        unreachable!("encouragement template")
    };
    for unit in units {
        unit.assigned = rng.next_f64() < 0.5;
        unit.received = unit.assigned && rng.next_f64() < 0.5;
        unit.outcome = 3.0 * f64::from(u8::from(unit.received)) + standard_normal(rng);
    }
    None
}

/// The complier average effect from 800 units at a first stage of one half.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_local_complier_effect_l95() {
    native_cell(
        "joint_response_local_complier_effect_l95",
        "local_complier_data",
        8,
        &local_complier_fixture(800 * scale()),
        3.0,
        local_complier_data,
        plain,
        modes(("wald_ratio", None), ["first_stage_at_least_0_4"; 2]),
    );
}

// ---------------------------------------------------------------------------------------
// Held-out selection.
// ---------------------------------------------------------------------------------------

const HELDOUT: Target = Target {
    mode: "heldout",
    interval: "heldout_normal_approximation",
    support: "independent_heldout_units",
    conservative: None,
};

/// `n` training and `n` held-out units, each assigned one of three programs uniformly; the
/// second program's mean is `signal`, the others zero, with unit noise.
fn randomized_selection_data(
    seed: u64,
    n: usize,
    signal: f64,
) -> (Vec<JointRandomizedUnit>, Vec<JointRandomizedUnit>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut units = |stage: &str| {
        (0..n)
            .map(|i| {
                let arm = (rng.next_u64() % 3) as usize;
                let noise = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt()
                    * (std::f64::consts::TAU * rng.next_f64()).cos();
                JointRandomizedUnit {
                    id: format!("{stage}:{i}"),
                    arm,
                    probabilities: vec![1.0 / 3.0; 3],
                    outcome: if arm == 1 { signal + noise } else { noise },
                    population_weight: 1.0,
                }
            })
            .collect::<Vec<_>>()
    };
    let training = units("train");
    let validation = units("validation");
    (training, validation)
}

/// The whole procedure on one data set: the two best programs on training data are
/// generated, one is selected, and its value is read from the held-out units; ten
/// whole-procedure resamples run alongside. Returns the reading and the held-out error.
fn randomized_selection_replicate(seed: u64, n: usize, signal: f64) -> (Reading, f64) {
    let prepared = randomized_selection(seed, n, signal);
    let result = match prepared.execute(10, 0.01, &ExecutionContext::for_tests(seed)) {
        Ok(result) => result,
        Err(error) => return (Reading::Refused(format!("execute: {error}")), 0.0),
    };
    let selected = &result.procedure().observed;
    let truth = if selected.action == 1 { signal } else { 0.0 };
    let error = selected.heldout.value - truth;
    match result.selected_value_reading(LEVEL) {
        Ok(reading) => {
            (Reading::Interval { bounds: reading.bounds, truth, basis: reading.basis }, error)
        }
        Err(refusal) => (Reading::Refused(format!("interval: {refusal}")), error),
    }
}

/// The compiled selection of one data set: three programs of equal cost, two of which fit
/// the budget screen of the two best on training data.
fn randomized_selection(seed: u64, n: usize, signal: f64) -> PreparedRandomizedSelection {
    let (training, validation) = randomized_selection_data(seed, n, signal);
    PreparedRandomizedSelection::compile(
        JointRandomizedDesign {
            worlds: vec!["a".into(), "b".into(), "c".into()],
            sampling_unit: "independently sampled randomized cluster".into(),
            population: "common sampling frame".into(),
            consistency: "exact complete program and common outcome measure".into(),
            between_unit_isolation: "no interference across sampled clusters".into(),
        },
        training,
        validation,
        RandomizedSelectionPolicy {
            independent_sampling_frame: "IID clusters from the same predeclared frame".into(),
            utility_units: "outcome units".into(),
            benefit_multiplier: 1.0,
            utility_costs: vec![0.0; 3],
            resource_costs: vec![1.0; 3],
            budget: 2.0,
            generation: RandomizedCandidateGeneration::TrainingScreen { top_k: 2 },
        },
    )
    .expect("the selection compiles")
}

fn randomized_selection_cell(test: &'static str, signal: f64) {
    let n = 240 * scale();
    let outcomes = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(antecedent_core::derive_stream(0x5830_3453_454c_4543, replicate));
        randomized_selection_replicate(seed, n, signal)
    });
    score(test, "randomized_selection_data", &HELDOUT, outcomes.iter().map(|outcome| &outcome.0));
    // The held-out estimate of the selected program is unbiased.
    let bias = outcomes.iter().map(|outcome| outcome.1).sum::<f64>() / outcomes.len() as f64;
    assert!(
        bias.abs() <= 4.0 * (3.0 / (n as f64 * outcomes.len() as f64)).sqrt(),
        "{test}: held-out bias {bias}"
    );
}

/// Three programs with equal values: the selection is a coin flip among exact ties.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_randomized_selection_exact_ties_l95() {
    randomized_selection_cell("joint_response_randomized_selection_exact_ties_l95", 0.0);
}

/// One program better by 0.02 standard deviations: a near tie.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_randomized_selection_near_ties_l95() {
    randomized_selection_cell("joint_response_randomized_selection_near_ties_l95", 0.02);
}

/// One program better by 0.2 standard deviations: a separated winner.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_randomized_selection_separated_l95() {
    randomized_selection_cell("joint_response_randomized_selection_separated_l95", 0.2);
}

/// `n` training and `n` held-out units with a uniform instrument, an endogenous dose and a
/// nearly flat quadratic structural response `0.05 d - 0.01 d^2`.
fn structural_selection_data(seed: u64, n: usize) -> InstrumentedSelectionRequest {
    let mut rng = CausalRng::from_seed(seed);
    let mut units = |stage: &str| {
        (0..n)
            .map(|i| {
                let z = 2.0 * rng.next_f64() - 1.0;
                let disturbance = 0.3 * (2.0 * rng.next_f64() - 1.0);
                let dose = 1.5 + z + 0.1 * disturbance;
                JointInstrumentedUnit {
                    id: format!("{stage}:{i}"),
                    endogenous: vec![dose, dose * dose],
                    instruments: vec![z, z * z],
                    exogenous: vec![1.0],
                    outcome: 4.0 * dose - dose * dose + disturbance - 3.95 * dose
                        + 0.99 * dose * dose,
                }
            })
            .collect()
    };
    let training = units("training");
    let validation = units("heldout");
    InstrumentedSelectionRequest {
        design: JointInstrumentedDesign {
            contrasts: vec![("dose_increment".into(), vec![1.0, 1.0])],
            structural_model: "Y = beta1*dose + beta2*dose^2 + intercept + error".into(),
            moment_restrictions:
                "independent uniform Z; E[Z error]=E[Z^2 error]=0; homogeneous coefficients".into(),
            population: "independent units from the declared common moment population".into(),
        },
        features: JointStructuralFeatureBinding {
            raw_feature_indices: vec![0],
            powers: vec![vec![1], vec![2]],
        },
        training,
        validation,
        independent_sampling_frame: "original independently sampled units reserved before fitting"
            .into(),
        controls: vec![InstrumentedDoseControl {
            lower: 0.0,
            upper: 3.0,
            unit_cost: 1.0,
            utility_cost: 0.0,
        }],
        restrictions: vec![],
        budget: None,
        reference: vec![0.0],
        initial: vec![1.0],
        benefit_multiplier: 1.0,
        utility_units: "outcome units".into(),
        generation: InstrumentedDomainGeneration::TrainingRange { margin: 0.1 },
        certificate: CertificateOptions { absolute_gap: 0.001, max_splits: 256 },
    }
}

/// The structural value of a dose selected by certified search on the training fit over the
/// training dose range and evaluated by an independent held-out fit, from 64 units a side.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_structural_selection_weak_separation_l95() {
    let test = "joint_response_structural_selection_weak_separation_l95";
    let n = 64 * scale();
    let readings = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(replicate);
        let request = structural_selection_data(seed, n);
        let result = match request.execute(10, 0.01, &ExecutionContext::for_tests(seed)) {
            Ok(result) => result,
            Err(error) => return Reading::Refused(format!("execute: {error}")),
        };
        let dose = result.procedure.observed.action.controls[0];
        let truth = 0.05 * dose - 0.01 * dose * dose;
        match result.selected_value_reading(LEVEL) {
            Ok(reading) => {
                Reading::Interval { bounds: reading.bounds, truth, basis: reading.basis }
            }
            Err(refusal) => Reading::Refused(format!("interval: {refusal}")),
        }
    });
    score(test, "structural_selection_data", &HELDOUT, readings.iter());
}

// ---------------------------------------------------------------------------------------
// Reference decomposition.
// ---------------------------------------------------------------------------------------

/// A sixteen-arm randomized experiment over exposure, saturation, mechanism and a baseline
/// stratum, `units` participants assigned uniformly with response noise of standard
/// deviation 0.2, compiled against static or two-period worlds and executed with the
/// Gaussian arm-mean posterior at the generating outcome variance.
fn reference_decomposition_design(
    temporal: bool,
    seed: u64,
    units: usize,
) -> (JointDesignResult, ReferenceComponentBindings) {
    const ARMS: usize = 16;
    let ctx = ExecutionContext::for_tests(311);
    let levels = |mask: usize| {
        [
            if mask & 1 == 0 { 1.0 } else { 4.0 },
            if mask & 2 == 0 { 1.0 } else { 2.0 },
            if mask & 4 == 0 { 1.0 } else { 3.0 },
            if mask & 8 == 0 { 0.0 } else { 1.0 },
        ]
    };
    let worlds = (0..ARMS)
        .map(|mask| {
            let values = levels(mask);
            let interventions = (0..4)
                .map(|j| Intervention::set(VariableId::from_raw(j as u32), Value::f64(values[j])))
                .collect::<Vec<_>>();
            let action = if temporal {
                WorldAction::schedule(
                    Intervention::Sequence(InterventionSequence::new(
                        (0..2)
                            .flat_map(|offset| {
                                interventions.iter().cloned().map(move |intervention| {
                                    SequencedIntervention::new(
                                        intervention,
                                        TemporalPolicy::pulse(offset),
                                    )
                                })
                            })
                            .collect::<Vec<_>>(),
                    )),
                    TemporalPolicy::sustained(0, 1),
                )
                .unwrap()
            } else {
                WorldAction::at_decision(interventions).unwrap()
            };
            let time = if temporal {
                TimeFrame::temporal(
                    2,
                    antecedent_core::HistoryWindow::max_lag(0),
                    antecedent_core::InitialLaw::named("shared-zero-history").unwrap(),
                )
                .unwrap()
            } else {
                TimeFrame::static_decision()
            };
            InterventionWorld::new(
                ScientificWorldId::from_raw(mask as u32),
                action,
                mask == 0,
                WorldBinding::new(
                    TargetPopulation::AllObserved,
                    ObservationUnit::new("independent participant").unwrap(),
                    OutcomeQuantity::mean(
                        VariableId::from_raw(4),
                        QuantityUnits::new("response").unwrap(),
                    )
                    .unwrap(),
                    Conditioning::unconditional(),
                )
                .unwrap(),
                time,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let design = JointRandomizedDesign {
        worlds: (0..ARMS).map(|arm| format!("parameter-arm-{arm}")).collect(),
        sampling_unit: "independent participant".into(),
        population: "randomized observed participant population".into(),
        consistency: "physical interventions jointly set dose A, saturation-control node S and \
                      mechanism-control node M; final response measured under each full program"
            .into(),
        between_unit_isolation: "participants have isolated responses".into(),
    };
    let mut rng = CausalRng::from_seed(seed);
    let sample = (0..units)
        .map(|unit| {
            let arm = (rng.next_u64() % ARMS as u64) as usize;
            let values = levels(arm);
            let noise = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt()
                * (std::f64::consts::TAU * rng.next_f64()).cos();
            JointRandomizedUnit {
                id: format!("unit-{unit}"),
                arm,
                probabilities: vec![1.0 / ARMS as f64; ARMS],
                outcome: (1.0 + values[3]) * values[2] * values[0] / (values[0] + values[1])
                    + 0.2 * noise,
                population_weight: 1.0,
            }
        })
        .collect::<Vec<_>>();
    let proof = JointDesignProof {
        version: 1,
        source_binding: JointDesignSourceBinding::from_world(
            &worlds[0],
            "randomized observed participant population".into(),
        )
        .unwrap(),
        longitudinal_arms: vec![],
        recipe: JointDesignRecipe::Randomized { design, units: sample },
        worlds: worlds
            .iter()
            .enumerate()
            .map(|(arm, world)| JointDesignWorldBinding {
                key: format!("parameter-arm-{arm}"),
                world: arm as u32,
                target_identity: joint_design_world_identity(world).to_hex(),
            })
            .collect(),
    };
    let target = ScientificTarget::new(
        TargetForm::Worlds(WorldCollection::new(worlds).unwrap()),
        ModelAssumptions::new(GraphFamily::Unspecified, AssumptionSet::default()),
        InferenceSelection::Bayesian,
    );
    let prepared = PreparedJointDesign::compile(target, proof, &ctx).unwrap();
    let result = prepared
        .execute(
            JointDesignInference::RandomizedGaussian {
                prior: RandomizedGaussianPrior {
                    mean: vec![0.0; ARMS],
                    variance: vec![1e6; ARMS],
                    outcome_variance: vec![0.04; units],
                    draws: 2048,
                    seed: seed ^ 0x0025_A931,
                },
            },
            &ctx,
        )
        .unwrap();
    let bindings = ReferenceComponentBindings {
        exposure: vec![VariableId::from_raw(0)],
        saturation: vec![VariableId::from_raw(1)],
        mechanism: vec![VariableId::from_raw(2)],
        hybrids: std::array::from_fn(|mask| {
            ScientificWorldId::from_raw(((mask & 3) | ((mask & 8) >> 1)) as u32)
        }),
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    (result, bindings)
}

/// The four true Shapley contributions of the change from a population with 20% of the
/// second stratum to one with 80%, computed from the generating response alone.
fn reference_decomposition_truth() -> [f64; 4] {
    let table: [f64; 16] = std::array::from_fn(|mask| {
        let a = if mask & 1 == 0 { 1.0 } else { 4.0 };
        let s = if mask & 2 == 0 { 1.0 } else { 2.0 };
        let m = if mask & 8 == 0 { 1.0 } else { 3.0 };
        (if mask & 4 == 0 { 1.2 } else { 1.8 }) * m * a / (a + s)
    });
    std::array::from_fn(|component| {
        (0_usize..16)
            .filter(|mask| mask & (1 << component) == 0)
            .map(|mask| {
                let weight = match mask.count_ones() {
                    0 | 3 => 0.25,
                    1 | 2 => 1.0 / 12.0,
                    _ => unreachable!(),
                };
                weight * (table[mask | (1 << component)] - table[mask])
            })
            .sum()
    })
}

/// The population change the decomposition explains: the second baseline stratum moves
/// from 20% to 80% of the population, under a diagram whose only selection node is that
/// root baseline.
fn population_change(
    bindings: &ReferenceComponentBindings,
) -> (PopulationReferenceBindings, antecedent_graph::SelectionDiagram) {
    let mut graph = antecedent_graph::Admg::with_variables(5);
    for variable in 0..4 {
        graph
            .insert_directed(
                antecedent_graph::DenseNodeId::from_raw(variable),
                antecedent_graph::DenseNodeId::from_raw(4),
            )
            .unwrap();
    }
    let diagram =
        antecedent_graph::SelectionDiagram::try_new(graph, [VariableId::from_raw(3)]).unwrap();
    let population = PopulationReferenceBindings {
        baseline: VariableId::from_raw(3),
        strata: vec![0.0, 1.0],
        stratum_worlds: vec![
            bindings.hybrids,
            bindings.hybrids.map(|id| ScientificWorldId::from_raw(id.raw() + 8)),
        ],
        before_weights: vec![0.8, 0.2],
        after_weights: vec![0.2, 0.8],
        components: bindings.clone(),
    };
    (population, diagram)
}

const REGION_TARGET: &str =
    "all four Shapley contributions, covered whenever each Bonferroni interval covers its own";

fn reference_decomposition_cell(test: &'static str, temporal: bool) {
    let truth = reference_decomposition_truth();
    let units = 512 * scale();
    let readings = map_replicates(n_sim(), |replicate| {
        let (posterior, bindings) =
            reference_decomposition_design(temporal, grid_seed(0x0028_F100 + replicate), units);
        let ctx = ExecutionContext::for_tests(grid_seed(replicate));
        let (population, diagram) = population_change(&bindings);
        let frequentist =
            posterior.prepared().execute(JointDesignInference::Frequentist, &ctx).unwrap();
        [&frequentist, &posterior].map(|result| {
            let decomposition = match result.decompose_population_reference(
                &population,
                &diagram,
                ScientificConsumptionScope::IdentifiedCausal,
                &ctx,
            ) {
                Ok(decomposition) => decomposition,
                Err(error) => return Reading::Refused(format!("decompose: {error}")),
            };
            match result.population_reference_region(&decomposition, LEVEL) {
                Ok(reading) => Reading::Region {
                    covered: reading
                        .bounds
                        .iter()
                        .zip(truth)
                        .all(|(bound, value)| bound[0] <= value && value <= bound[1]),
                    bounds: reading.bounds.iter().flatten().copied().collect(),
                    basis: reading.basis,
                },
                Err(error) => Reading::Refused(format!("region: {error}")),
            }
        })
    });
    let targets = [
        Target {
            mode: "frequentist",
            interval: "bonferroni_normal_approximation",
            support: "equal_weights_32_units_per_program",
            conservative: Some(REGION_TARGET),
        },
        Target {
            mode: "bayesian",
            interval: "bonferroni_equal_tailed_posterior",
            support: "equal_weights_32_units_per_program",
            conservative: Some(REGION_TARGET),
        },
    ];
    for (mode, target) in targets.iter().enumerate() {
        score(
            test,
            "reference_decomposition_design",
            target,
            readings.iter().map(|pair| &pair[mode]),
        );
    }
}

/// Static decision worlds, 512 participants over sixteen arms.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_reference_decomposition_static_l95() {
    reference_decomposition_cell("joint_response_reference_decomposition_static_l95", false);
}

/// The same experiment compiled against two-period scheduled worlds. The fitted arm means
/// are those of the static cell: only the compiled target differs.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_reference_decomposition_temporal_l95() {
    reference_decomposition_cell("joint_response_reference_decomposition_temporal_l95", true);
}

// ---------------------------------------------------------------------------------------
// Experiment update.
// ---------------------------------------------------------------------------------------

/// The declared identity translation of an experimental sample mean to the target contrast.
fn experiment_bridge(target: &antecedent_contract::ScientificQuantity) -> BridgeContract {
    let mut experiment_quantity = target.clone();
    experiment_quantity.functional_id = "experimental_sample_mean".into();
    let setting = ExposureSetting {
        population_id: target.population_id.clone(),
        time_start: "program_period:23".into(),
        time_end: "program_period:23".into(),
        dose: 1.0,
        dose_units: "declared_joint_program".into(),
        duration: 1.0,
        duration_units: "panel_period".into(),
        exposure_regime: target.regime_id.clone(),
    };
    let bridge = BridgeContract {
        kind: BridgeKind::ParameterInterventionResponse,
        source: BridgeEndpoint {
            quantity: experiment_quantity,
            class: QuantityClass::ModelParameter,
        },
        target: BridgeEndpoint {
            quantity: target.clone(),
            class: QuantityClass::InterventionResponse,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: target.population_id.clone(),
            regime_id: target.regime_id.clone(),
        },
        mapping: BridgeMapping::Affine { intercept: 0.0, slope: 1.0 },
        assumptions: vec![BridgeAssumption {
            id: "declared_experiment_likelihood".into(),
            kind: AssumptionKind::ParameterInterpretation,
            statement: "The experimental Gaussian mean maps to this exact future joint-response \
                        contrast under frozen modular conditioning"
                .into(),
            evidence: Some("geo_experiment".into()),
        }],
        transport: None,
        mapping_uncertainty: Some(MappingUncertainty {
            id: "paired_mapping".into(),
            intercept_variance: 0.0,
            slope_variance: 0.0,
            intercept_slope_covariance: 0.0,
        }),
        source_dependence: Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.0,
            evidence: "retained_joint_slots".into(),
        }),
        estimand_bridge: Some(EstimandBridge {
            id: "exact_program_translation".into(),
            experiment_id: "geo_experiment".into(),
            experiment: setting.clone(),
            target: setting,
            recency_delta: 0.0,
            recency_units: "panel_period".into(),
            spend_delta: 0.0,
            spend_units: "declared_joint_program".into(),
        }),
    };
    bridge.check_support().unwrap();
    bridge
}

/// Posterior draws retained by the source fit of the experiment chain.
const CHAIN_DRAWS: usize = 2048;

/// One executed experiment chain.
struct Chain {
    ctx: ExecutionContext,
    prepared: PreparedJointDesign,
    fit: JointDesignResult,
    updated: antecedent::analysis::joint_panel::JointLawExperimentResult,
    ledger: EvidenceLedger,
    coefficients: Vec<f64>,
    true_effect: f64,
}

/// One whole chain: four arm means drawn from the declared prior N(0, 4), `units`
/// participants randomized uniformly with unit noise, the Gaussian arm-mean posterior, an
/// independent experiment of eight observations of the target contrast with observation
/// variance 0.3, and the frozen importance update.
fn experiment_update_replicate(replicate: u64, units: usize) -> Chain {
    const DRAWS: usize = CHAIN_DRAWS;
    // The target contrast and its four whole-program worlds are the workflow fixture's.
    let panel = fixture::source(false);
    let template = fixture::request(
        &panel,
        false,
        antecedent_estimate::joint_panel::PanelInference::Frequentist { draws: 2 },
    );
    let TargetForm::Contrast(contrast) = template.target.form() else {
        panic!("the workflow fixture targets a contrast");
    };
    let worlds = contrast.worlds().worlds();
    let keys = (0..4).map(|i| format!("whole_program_{i}")).collect::<Vec<_>>();
    let ctx = ExecutionContext::for_tests(grid_seed(13_000 + replicate));
    let mut rng = ctx.rng.stream(0x5E1E_C710_0000 + replicate);
    let truth = (0..4).map(|_| 2.0 * standard_normal(&mut rng)).collect::<Vec<_>>();
    let sample = (0..units)
        .map(|i| {
            let arm = antecedent_kernels::sample_categorical(&mut rng, &[0.25; 4]).unwrap();
            JointRandomizedUnit {
                id: format!("original:{replicate}:{i}"),
                arm,
                probabilities: vec![0.25; 4],
                outcome: truth[arm] + standard_normal(&mut rng),
                population_weight: 1.0,
            }
        })
        .collect::<Vec<_>>();
    let source_binding = JointDesignSourceBinding::from_world(
        &worlds[0],
        "prespecified finite Gaussian arm-mean target population".into(),
    )
    .unwrap();
    let recipe = JointDesignRecipe::Randomized {
        design: JointRandomizedDesign {
            worlds: keys.clone(),
            sampling_unit: source_binding.sampling_unit.clone(),
            population: "prespecified finite Gaussian arm-mean target population".into(),
            consistency: "Each known arm code is the complete retained multichannel world program"
                .into(),
            between_unit_isolation:
                "Independent randomized clusters, no between-cluster interference".into(),
        },
        units: sample,
    };
    let proof = JointDesignProof {
        version: 1,
        recipe,
        source_binding,
        worlds: worlds
            .iter()
            .zip(&keys)
            .map(|(world, key)| JointDesignWorldBinding {
                key: key.clone(),
                world: world.id().raw(),
                target_identity: joint_design_world_identity(world).to_hex(),
            })
            .collect(),
        longitudinal_arms: vec![],
    };
    let prepared = PreparedJointDesign::compile(template.target.clone(), proof, &ctx).unwrap();
    let fit = prepared
        .execute(
            JointDesignInference::RandomizedGaussian {
                prior: RandomizedGaussianPrior {
                    mean: vec![0.0; 4],
                    variance: vec![4.0; 4],
                    outcome_variance: vec![1.0; units],
                    draws: DRAWS,
                    seed: 19_000 + replicate,
                },
            },
            &ctx,
        )
        .unwrap();
    let source = fit.posterior_source().unwrap();
    let quantity =
        antecedent::analysis::joint_panel::JointLawExperimentResult::quantity(&source).unwrap();
    let coefficients = worlds
        .iter()
        .map(|w| contrast.terms().iter().find(|t| t.world == w.id()).unwrap().coefficient)
        .collect::<Vec<_>>();
    let true_effect = truth.iter().zip(&coefficients).map(|(v, c)| v * c).sum::<f64>();
    let observed = true_effect + (0.3_f64 / 8.0).sqrt() * standard_normal(&mut rng);
    let mut ledger = EvidenceLedger::new();
    let model = ledger
        .add_source(
            SourceDeclaration::new(source.dependency_identity().to_hex(), SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("original", &replicate.to_le_bytes())]),
        )
        .unwrap();
    let experiment = ledger
        .add_source(
            SourceDeclaration::new("independent_new_experiment", SourceKind::Experiment)
                .with_observations([ObservationId::from_key(
                    "new-experiment",
                    &replicate.to_le_bytes(),
                )]),
        )
        .unwrap();
    let mapping = ledger
        .add_source(
            SourceDeclaration::new("identity_translation_reference", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key(
                    "identity-reference",
                    b"fixed-external-model",
                )]),
        )
        .unwrap();
    let request = JointLawExperimentRequest {
        bridge: experiment_bridge(&quantity),
        observed_mean: observed,
        observation_variance: 0.3,
        sample_size: 8,
        mapping: PanelExperimentMapping {
            dependency: source.dependency_identity(),
            intercepts: vec![0.0; DRAWS],
            slopes: vec![1.0; DRAWS],
        },
        evidence: PanelExperimentEvidence { model, experiment, mapping, prior_origins: vec![] },
    };
    let updated = fit.condition_experiment(request, &ledger, &ctx).unwrap();
    Chain { ctx, prepared, fit, updated, ledger, coefficients, true_effect }
}

/// The contrast of every draw of a four-world law, in draw order.
fn chain_contrasts(draws: &[f64], coefficients: &[f64]) -> Vec<f64> {
    draws
        .chunks_exact(4)
        .map(|row| row.iter().zip(coefficients).map(|(v, c)| v * c).sum::<f64>())
        .collect()
}

/// The reading of one chain and whether the update changed the sign of the posterior mean
/// contrast.
fn experiment_update_reading(chain: &Chain) -> (Reading, bool) {
    let law = chain.updated.numerical_law_source(&chain.ledger).unwrap();
    let source = chain.fit.posterior_source().unwrap();
    let ScientificSourcePayload::Law(prior) = source.as_numerical_source() else {
        panic!("the source carries its posterior law");
    };
    let before = chain_contrasts(prior.draws(), &chain.coefficients).iter().sum::<f64>()
        / CHAIN_DRAWS as f64;
    let weights = law.metadata().weights.as_ref().expect("the updated law is weighted");
    let after = chain_contrasts(law.draws(), &chain.coefficients)
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>();
    // The updated law and the source design both replay for a current consumer.
    let artifact = chain.updated.export(&chain.ledger, &chain.ctx).unwrap();
    chain.updated.consume_current(&artifact, &chain.ledger, &chain.ctx).unwrap();
    chain.prepared.consume_current(&chain.fit.export_artifact().unwrap(), &chain.ctx).unwrap();
    let reading = match chain.fit.updated_contrast_reading(
        &chain.updated,
        &chain.ledger,
        &chain.coefficients,
        LEVEL,
    ) {
        Ok(reading) => Reading::Interval {
            bounds: reading.bounds,
            truth: chain.true_effect,
            basis: reading.basis,
        },
        Err(error) => Reading::Refused(format!("interval: {error}")),
    };
    (reading, (before > 0.0) != (after > 0.0))
}

/// The posterior interval of the target contrast after the experiment, from 64 participants
/// over four programs. The truth is drawn from the declared prior and the likelihood scales
/// are declared at their generating values, so this is coverage averaged over that prior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn joint_response_experiment_update_contrast_l95() {
    let test = "joint_response_experiment_update_contrast_l95";
    let units = 64 * scale();
    let outcomes = map_replicates(n_sim(), |replicate| {
        experiment_update_reading(&experiment_update_replicate(replicate, units))
    });
    score(
        test,
        "experiment_update_replicate",
        &Target {
            mode: "bayesian",
            interval: "weighted_equal_tailed_posterior",
            support: "equal_weights_16_units_per_program",
            conservative: None,
        },
        outcomes.iter().map(|outcome| &outcome.0),
    );
    assert!(
        outcomes.iter().any(|outcome| outcome.1),
        "{test}: independent evidence must sometimes change the sign of the posterior contrast"
    );
}

// ---------------------------------------------------------------------------------------
// Known-value checks of each reading. These are not calibration simulations: each compares
// one reading with an interval computed here from the published law, and checks its key.
// ---------------------------------------------------------------------------------------

/// Standard normal 0.975 quantile.
const Z975: f64 = 1.959_963_984_540_054;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} is not within {tolerance} of {expected}"
    );
}

fn eight_unit_geo_result(ctx: &ExecutionContext) -> (NativeJointDesignResult, Vec<f64>) {
    let (mut recipe, _, contrast) = weighted_geo_fixture(8);
    let JointDesignRecipe::Randomized { units, .. } = &mut recipe else {
        unreachable!("randomized template")
    };
    for (unit, outcome) in units.iter_mut().zip([1.0, 3.0, 2.0, 5.0, 0.0, 4.0, 3.0, 6.0]) {
        unit.population_weight = 1.0;
        unit.outcome = outcome;
    }
    let prepared = PreparedNativeJointDesign::compile(recipe, ctx).unwrap();
    (prepared.execute(JointDesignInference::Frequentist, ctx).unwrap(), contrast)
}

fn switchback_history_result(
    repeats: usize,
    ctx: &ExecutionContext,
) -> (NativeJointDesignResult, Vec<f64>) {
    let (mut recipe, _, contrast) = switchback_fixture(repeats);
    let JointDesignRecipe::Switchback { sequences, .. } = &mut recipe else {
        unreachable!("switchback template")
    };
    for sequence in sequences.iter_mut() {
        let history = &sequence.schedules[sequence.realized];
        let (past, arm) = (history[0], history[1]);
        sequence.outcomes = vec![
            0.0,
            (2 * (arm & 1) + 3 * (arm >> 1) + 4 * (arm & 1) * (arm >> 1)) as f64
                + 1.5 * (past & 1) as f64
                + 2.0 * (past >> 1) as f64,
        ];
    }
    let prepared = PreparedNativeJointDesign::compile(recipe, ctx).unwrap();
    (prepared.execute(JointDesignInference::Frequentist, ctx).unwrap(), contrast)
}

/// Eight equally weighted units alternate between two programs at probability one half. The
/// inverse-probability scores of the contrast are -2, 6, -4, 10, 0, 8, -6, 12: mean 3,
/// sample variance 328 / 7, so the variance of the mean is 328 / 56 and the interval is
/// 3 -/+ t(0.975, 7) sqrt(328 / 56) with t(0.975, 7) = 2.364624.
#[test]
fn native_contrast_reading_is_the_hand_computed_interval_and_is_keyed_by_its_support() {
    let ctx = ExecutionContext::for_tests(901);
    let (result, contrast) = eight_unit_geo_result(&ctx);
    let reading = result.contrast_interval_reading(&contrast, LEVEL).unwrap();
    let radius = 2.364_624_251_592_785 * (328.0_f64 / 56.0).sqrt();
    assert_close(reading.bounds[0], 3.0 - radius, 1e-6);
    assert_close(reading.bounds[1], 3.0 + radius, 1e-6);
    assert_eq!(&*reading.basis.interval_method, "satterthwaite_t");
    assert_eq!(&*reading.basis.estimator, "joint_design.whole_program_ht");
    assert_eq!(reading.basis.row_count, 8);
    // Four units per program is below the 32 the equal-weight interval was measured at.
    assert_eq!(&*reading.basis.dependence, "equal_weights_sparse_programs");
    assert!(result.contrast_interval_reading(&contrast, 1.5).is_err());
    let bytes = result.export_artifact().unwrap();
    assert_eq!(*result.law(), replay_native_joint_design(&bytes, &ctx).unwrap());

    // A switchback is keyed by its sequences per requested history.
    for (repeats, support) in
        [(4, "sequences_per_history_below_32"), (32, "sequences_per_history_at_least_32")]
    {
        let (result, contrast) = switchback_history_result(repeats, &ctx);
        let reading = result.contrast_interval_reading(&contrast, LEVEL).unwrap();
        assert_close(0.5 * (reading.bounds[0] + reading.bounds[1]), 4.0, 1e-6);
        assert_eq!(&*reading.basis.dependence, support);
        assert_eq!(reading.basis.row_count, 16 * repeats as u64);
    }
}

/// The eight-unit geo experiment and the switchback below its measured support are
/// annotated as not calibrated; the annotation is read from the result, not from the
/// interval.
#[test]
fn native_contrast_reading_carries_its_calibration_annotation() {
    let ctx = ExecutionContext::for_tests(901);
    // Four units per program is below the 32 the equal-weight interval was measured at.
    let (result, contrast) = eight_unit_geo_result(&ctx);
    let reading = result.contrast_interval_reading(&contrast, LEVEL).unwrap();
    assert_ne!(reading.calibration.status.as_str(), "calibrated");
    // A switchback below 32 sequences per requested history is not calibrated.
    let (result, contrast) = switchback_history_result(4, &ctx);
    let reading = result.contrast_interval_reading(&contrast, LEVEL).unwrap();
    assert_ne!(reading.calibration.status.as_str(), "calibrated");
}

/// The synthetic posterior is keyed by whether the declared prior mean of the residual
/// variance is within a factor of two of the pre-period fit error, and a joint 95%
/// conformal region over two rollouts is refused below 39 pre-periods.
#[test]
fn synthetic_readings_check_the_declared_prior_and_the_pre_period_count() {
    use antecedent_estimate::joint_design::compiled::JointNativeDesignLaw;
    let ctx = ExecutionContext::for_tests(902);
    let (mut recipe, bayesian, contrast) = synthetic_rollout_fixture(14);
    let JointDesignRecipe::SyntheticControl { observations, .. } = &mut recipe else {
        unreachable!("synthetic template")
    };
    for (row, observation) in observations.iter_mut().enumerate() {
        if observation.unit.starts_with("treated") {
            observation.outcome += 0.1 * (1.7 * row as f64).sin();
        }
    }
    let prepared = PreparedNativeJointDesign::compile(recipe, &ctx).unwrap();
    let point = prepared.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    let JointNativeDesignLaw::SyntheticPoint(points) = point.law() else {
        panic!("the frequentist synthetic law is a point fit");
    };
    let fit_error =
        points.iter().map(|p| p.preperiod_rmse.powi(2)).sum::<f64>() / points.len() as f64;
    assert!(fit_error > 0.0);
    let JointDesignInference::SyntheticBayesian { prior } = bayesian else {
        unreachable!("synthetic posterior")
    };
    for (ratio, support) in [
        (1.0, "variance_prior_within_2x_preperiod_error"),
        (4.0, "variance_prior_outside_2x_preperiod_error"),
    ] {
        let mut declared = prior.clone();
        declared.variance_scale = ratio * fit_error * (declared.variance_shape - 1.0);
        let result = prepared
            .execute(JointDesignInference::SyntheticBayesian { prior: declared }, &ctx)
            .unwrap();
        let reading = result.contrast_interval_reading(&contrast, LEVEL).unwrap();
        assert!(reading.bounds[0] < reading.bounds[1]);
        assert_eq!(&*reading.basis.interval_method, "equal_tailed_posterior");
        assert_eq!(&*reading.basis.dependence, support);
        assert_eq!(reading.basis.row_count, 14);
        assert_eq!(reading.basis.posterior_draws, Some(8000));
        assert_ne!(reading.calibration.status.as_str(), "calibrated");
    }
    let conformal = prepared
        .execute(
            JointDesignInference::SyntheticConformal {
                joint_level: LEVEL,
                residual_exchangeability: "exchangeable treated residuals across periods".into(),
            },
            &ctx,
        )
        .and_then(|result| result.contrast_interval_reading(&contrast, LEVEL));
    assert!(conformal.is_err(), "fourteen pre-periods cannot resolve a joint 95% region");
}

/// Both selected-value readings are the held-out estimate -/+ 1.959964 standard errors, are
/// keyed by the smaller of the two samples, and replay.
#[test]
fn selected_value_readings_are_the_heldout_normal_interval() {
    use antecedent::analysis::joint_panel::selection::instrumented::replay_instrumented_selection;
    let ctx = ExecutionContext::for_tests(903);
    let prepared = randomized_selection(7, 240, 0.2);
    let result = prepared.execute(10, 0.01, &ctx).unwrap();
    let heldout = result.procedure().observed.heldout.clone();
    let reading = result.selected_value_reading(LEVEL).unwrap();
    assert_close(reading.bounds[0], heldout.value - Z975 * heldout.standard_error, 1e-6);
    assert_close(reading.bounds[1], heldout.value + Z975 * heldout.standard_error, 1e-6);
    assert_eq!(&*reading.basis.interval_method, "heldout_normal_approximation");
    assert_eq!(&*reading.basis.functional, "selected_program_value");
    assert_eq!(reading.basis.row_count, 240);
    let refusal = result.selected_value_reading(1.5).unwrap_err().to_string();
    assert!(
        refusal.contains(antecedent_core::reason_details::joint_response::READING_LEVEL_INVALID),
        "{refusal}"
    );
    let bytes = result.export().unwrap();
    assert_eq!(prepared.consume_current(&bytes, &ctx).unwrap().report(), result.report());

    let request = structural_selection_data(17, 64);
    let result = request.execute(3, 0.01, &ctx).unwrap();
    let heldout = result.procedure.observed.heldout.clone();
    let reading = result.selected_value_reading(LEVEL).unwrap();
    assert_close(reading.bounds[0], heldout.value - Z975 * heldout.standard_error, 1e-6);
    assert_close(reading.bounds[1], heldout.value + Z975 * heldout.standard_error, 1e-6);
    assert_eq!(&*reading.basis.functional, "selected_structural_increment");
    assert_eq!(reading.basis.row_count, 64);
    assert!(result.selected_value_reading(0.0).is_err());
    let bytes = result.export().unwrap();
    assert_eq!(
        replay_instrumented_selection(&bytes, Some(&request), &ctx).unwrap().report(),
        result.report()
    );
}

/// A selected-value reading at a 50% level is annotated as not calibrated: no record is
/// measured there.
#[test]
fn selected_value_reading_carries_its_calibration_annotation() {
    let ctx = ExecutionContext::for_tests(903);
    let result = randomized_selection(7, 240, 0.2).execute(10, 0.01, &ctx).unwrap();
    assert_ne!(
        result.selected_value_reading(0.5).unwrap().calibration.status.as_str(),
        "calibrated"
    );
}

/// The frequentist region is each contribution -/+ 2.497705 propagated standard errors
/// (the normal 1 - 0.05 / 8 quantile); the Bayesian region cuts 12 of 2048 draws from each
/// tail of each contribution.
#[test]
fn population_reference_region_is_the_bonferroni_region_of_each_mode() {
    let ctx = ExecutionContext::for_tests(904);
    let (posterior, bindings) = reference_decomposition_design(false, 0x0028_F100, 512);
    let (population, diagram) = population_change(&bindings);
    let scope = ScientificConsumptionScope::IdentifiedCausal;
    let frequentist =
        posterior.prepared().execute(JointDesignInference::Frequentist, &ctx).unwrap();
    let decomposition =
        frequentist.decompose_population_reference(&population, &diagram, scope, &ctx).unwrap();
    let reading = frequentist.population_reference_region(&decomposition, LEVEL).unwrap();
    let covariance = decomposition.covariance.as_ref().unwrap();
    for component in 0..4 {
        let point = decomposition.decomposition.point.shapley[component];
        let radius = 2.497_705_474_412_37 * covariance.shapley[component][component].sqrt();
        assert_close(reading.bounds[component][0], point - radius, 1e-6);
        assert_close(reading.bounds[component][1], point + radius, 1e-6);
    }
    assert_eq!(&*reading.basis.interval_method, "bonferroni_normal_approximation");
    assert_eq!(&*reading.basis.dependence, "equal_weights_32_units_per_program");
    assert_eq!(reading.basis.row_count, 512);
    assert!(frequentist.population_reference_region(&decomposition, 1.0).is_err());

    let decomposition =
        posterior.decompose_population_reference(&population, &diagram, scope, &ctx).unwrap();
    let reading = posterior.population_reference_region(&decomposition, LEVEL).unwrap();
    for component in 0..4 {
        let mut values = decomposition
            .decomposition
            .draws
            .iter()
            .flatten()
            .map(|draw| draw.shapley[component])
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        assert_eq!(values.len(), 2048);
        assert_eq!(
            reading.bounds[component].map(f64::to_bits),
            [values[12], values[2035]].map(f64::to_bits)
        );
    }
    assert_eq!(&*reading.basis.interval_method, "bonferroni_equal_tailed_posterior");
    assert_eq!(reading.basis.posterior_draws, Some(2048));
    posterior.prepared().consume_current(&posterior.export_artifact().unwrap(), &ctx).unwrap();
}

/// A region at a joint 50% level is annotated as not calibrated: no record is measured
/// there.
#[test]
fn population_reference_region_carries_its_calibration_annotation() {
    let ctx = ExecutionContext::for_tests(904);
    let (posterior, bindings) = reference_decomposition_design(false, 0x0028_F100, 512);
    let (population, diagram) = population_change(&bindings);
    let scope = ScientificConsumptionScope::IdentifiedCausal;
    let decomposition =
        posterior.decompose_population_reference(&population, &diagram, scope, &ctx).unwrap();
    assert_ne!(
        posterior
            .population_reference_region(&decomposition, 0.5)
            .unwrap()
            .calibration
            .status
            .as_str(),
        "calibrated"
    );
}

/// The updated interval is the weighted 2.5% and 97.5% quantiles of the contrast over the
/// draws of the updated law, recomputed here from the law.
#[test]
fn updated_contrast_reading_is_the_weighted_equal_tailed_interval() {
    let chain = experiment_update_replicate(3, 64);
    let reading = chain
        .fit
        .updated_contrast_reading(&chain.updated, &chain.ledger, &chain.coefficients, LEVEL)
        .unwrap();
    let law = chain.updated.numerical_law_source(&chain.ledger).unwrap();
    let weights = law.metadata().weights.as_ref().unwrap();
    let mut ordered = chain_contrasts(law.draws(), &chain.coefficients)
        .into_iter()
        .zip(weights.iter().copied())
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
    let quantile = |p: f64| {
        let mut total = 0.0;
        for (value, weight) in &ordered {
            total += weight;
            if total >= p {
                return *value;
            }
        }
        ordered[ordered.len() - 1].0
    };
    assert_close(reading.bounds[0], quantile(0.025), 1e-12);
    assert_close(reading.bounds[1], quantile(0.975), 1e-12);
    assert_eq!(&*reading.basis.interval_method, "weighted_equal_tailed_posterior");
    assert_eq!(&*reading.basis.dependence, "equal_weights_16_units_per_program");
    assert_eq!(reading.basis.row_count, 64);
    assert_eq!(reading.basis.posterior_draws, Some(2048));
    let refusal = chain
        .fit
        .updated_contrast_reading(&chain.updated, &chain.ledger, &chain.coefficients[..3], LEVEL)
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains(antecedent_core::reason_details::joint_response::READING_LEVEL_INVALID),
        "{refusal}"
    );
    // The updated law and its source design replay for a current consumer.
    let _ = experiment_update_reading(&chain);
}

/// An updated contrast reading at a 50% level is annotated as not calibrated: no record is
/// measured there.
#[test]
fn updated_contrast_reading_carries_its_calibration_annotation() {
    let chain = experiment_update_replicate(3, 64);
    assert_ne!(
        chain
            .fit
            .updated_contrast_reading(&chain.updated, &chain.ledger, &chain.coefficients, 0.5)
            .unwrap()
            .calibration
            .status
            .as_str(),
        "calibrated"
    );
}
