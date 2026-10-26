//! Retained existing two-step history mechanisms and empirical source joint.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::EstimationError;
use crate::temporal_dependent_interval::{SequenceTallies, TemporalUnitPanel, UnitHistories};
use crate::temporal_initial_state::{InitialStateLaw, InitialStatePopulation, StateContribution};
use antecedent_core::{RegimeId, Value, VariableId, reason_code};
use antecedent_expr::execution_counts::{StaticWork, note_static_work};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use std::sync::Arc;

/// Actual retained sequence-specific covariate/outcome counts from the existing estimator.
/// No response value is substituted for these mechanisms.
#[derive(Debug)]
pub struct FittedSequenceHistory {
    sequence: [u32; 2],
    tallies: SequenceTallies,
}
impl FittedSequenceHistory {
    /// Fit the existing whole-history mechanism tallies once.
    #[must_use]
    pub fn fit(units: &[&UnitHistories], sequence: [u32; 2]) -> Self {
        let attempt = antecedent_core::execution_attempt::OperationGuard::begin(
            antecedent_core::execution_attempt::Operation::FactorConstruction,
        );
        let tallies = SequenceTallies::of(units, sequence);
        note_static_work(StaticWork::FactorBuild);
        attempt.complete();
        Self { sequence, tallies }
    }
    /// Construct several existing sequence mechanisms in one history traversal.
    /// The counter records each constructed mechanism, not each raw traversal.
    #[must_use]
    pub fn fit_many(units: &[&UnitHistories], sequences: &[[u32; 2]]) -> Vec<Self> {
        let attempts: Vec<_> = sequences
            .iter()
            .map(|_| {
                antecedent_core::execution_attempt::OperationGuard::begin(
                    antecedent_core::execution_attempt::Operation::FactorConstruction,
                )
            })
            .collect();
        SequenceTallies::of_many(units, sequences)
            .into_iter()
            .zip(sequences)
            .zip(attempts)
            .map(|((tallies, &sequence), attempt)| {
                note_static_work(StaticWork::FactorBuild);
                attempt.complete();
                Self { sequence, tallies }
            })
            .collect()
    }
    /// The fitted whole-sequence intervention history.
    #[must_use]
    pub const fn sequence(&self) -> [u32; 2] {
        self.sequence
    }
    /// Project the existing mechanism response onto a declared target initial law.
    /// This reads retained counts and validates every positive-mass target history.
    /// # Errors
    /// A positive-mass state or action/history cell lacks observed source support.
    pub fn contributions(
        &self,
        law: &InitialStateLaw,
    ) -> Result<Vec<StateContribution>, EstimationError> {
        if law.population() != InitialStatePopulation::Target {
            return Err(EstimationError::refused(
                reason_code!("transport_missing_evidence"),
                "initial_state.target_law_missing: only a source initial-state law was supplied",
            ));
        }
        law.states()
            .iter()
            .filter(|(_, mass)| *mass > 0.)
            .map(|&(s0, mass)| {
                let response = self
                    .tallies
                    .response(s0)
                    .map_err(|gap| crate::temporal_initial_state::support_gap(s0, gap))?;
                Ok(StateContribution { s0, mass, response })
            })
            .collect()
    }
}

/// Levels a binary history route accepts per coordinate.
const BINARY_LEVELS: u32 = 2;

/// Fit one empirical joint source factor from actual complete binary histories.
/// The joint is labeled empirical and carries reconciled counts, never supplied-exact.
/// Whole-unit ownership remains in the input panel for the dependent candidate.
/// # Errors
/// No histories, unsupported nonbinary coordinates/outcome, or invalid frequency law.
pub fn fit_binary_history_joint(
    panel: &TemporalUnitPanel,
) -> Result<ExactDiscreteLaw, EstimationError> {
    fit_binary_history_units(panel.snapshot_id(), &panel.units().iter().collect::<Vec<_>>())
}

/// Fit the same empirical history joint on a whole-unit draw, including repeated units.
/// # Errors
/// Empty draw, nonbinary history coordinates or invalid source frequency law.
pub fn fit_binary_history_units(
    snapshot: &str,
    units: &[&UnitHistories],
) -> Result<ExactDiscreteLaw, EstimationError> {
    fit_history_units(snapshot, units, Some(BINARY_LEVELS))
}

/// Fit one empirical joint source factor from complete histories whose state, actions,
/// covariate and outcome are categorical codes `0..k` (each coordinate has as many levels as its
/// largest observed code plus one, and at least two). The joint is labeled empirical and
/// carries reconciled counts. The cell index is the mixed radix of the five level counts, so
/// binary histories reproduce [`fit_binary_history_joint`] exactly.
/// # Errors
/// No histories, an outcome that is not a non-negative integer code, a joint cell table the
/// admission pool cannot fund, or an invalid frequency law.
pub fn fit_categorical_history_joint(
    panel: &TemporalUnitPanel,
) -> Result<ExactDiscreteLaw, EstimationError> {
    fit_categorical_history_units(panel.snapshot_id(), &panel.units().iter().collect::<Vec<_>>())
}

/// [`fit_categorical_history_joint`] on a whole-unit draw, including repeated units.
/// # Errors
/// As [`fit_categorical_history_joint`].
pub fn fit_categorical_history_units(
    snapshot: &str,
    units: &[&UnitHistories],
) -> Result<ExactDiscreteLaw, EstimationError> {
    fit_history_units(snapshot, units, None)
}

/// The category code of an outcome value: a non-negative integer.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "the value is checked to be a non-negative integer below u32::MAX before the cast"
)]
fn outcome_code(y: f64) -> Option<u32> {
    (y.is_finite() && y >= 0.0 && y.fract() == 0.0 && y < f64::from(u32::MAX)).then(|| y as u32)
}

/// Coordinates of a history in axis order: state, first action, covariate, second action,
/// outcome code.
fn history_coordinates(
    h: &crate::temporal_dependent_interval::SequenceHistory,
) -> Option<[u32; 5]> {
    Some([h.s0, h.a1, h.l2, h.a2, outcome_code(h.y)?])
}

/// `cap` bounds the levels of every coordinate (`Some(2)` is the binary route); `None` takes
/// as many as the data carries.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the panel is bounded far below 2^53, and a coordinate index is one of five"
)]
fn fit_history_units(
    snapshot: &str,
    units: &[&UnitHistories],
    cap: Option<u32>,
) -> Result<ExactDiscreteLaw, EstimationError> {
    let scope = || {
        EstimationError::refused(
            reason_code!("route_not_supported"),
            "temporal_recalc.binary_history_scope: only binary state, actions, covariate and outcome",
        )
    };
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::FactorConstruction,
        || {
            // Level count of each coordinate: its largest code plus one, and at least two.
            let mut levels = [BINARY_LEVELS; 5];
            for h in units.iter().flat_map(|u| &u.histories) {
                let Some(coordinates) = history_coordinates(h) else {
                    return Err(if cap.is_some() {
                        scope()
                    } else {
                        EstimationError::data_msg("temporal_recalc.invalid_history_outcome")
                    });
                };
                for (level, &code) in levels.iter_mut().zip(&coordinates) {
                    *level = (*level).max(code.saturating_add(1));
                }
            }
            if cap.is_some_and(|cap| levels.iter().any(|&level| level > cap)) {
                return Err(scope());
            }
            let cells = levels
                .iter()
                .try_fold(1_usize, |cells, &level| cells.checked_mul(level as usize))
                .ok_or_else(|| EstimationError::data_msg("temporal_recalc.history_cells_overflow"))?;
            let _lease = crate::util::admit_bound_array::<u64>(cells, 2)?;
            let mut counts = vec![0_u64; cells];
            for h in units.iter().flat_map(|u| &u.histories) {
                let Some(coordinates) = history_coordinates(h) else {
                    return Err(EstimationError::data_msg("temporal_recalc.invalid_history_fit"));
                };
                let cell = coordinates
                    .iter()
                    .zip(&levels)
                    .fold(0_usize, |cell, (&code, &level)| cell * level as usize + code as usize);
                counts[cell] += 1;
            }
            let n: u64 = counts.iter().sum();
            if n == 0 {
                return Err(EstimationError::data_msg("temporal_recalc.empty_histories"));
            }
            let axes = levels
                .iter()
                .enumerate()
                .map(|(id, &level)| DiscreteAxis {
                    variable: VariableId::from_raw(id as u32),
                    values: (0..level).map(|code| Value::f64(f64::from(code))).collect(),
                })
                .collect::<Vec<_>>();
            let joint = ExactDiscreteLaw::try_empirical(
                "source",
                RegimeId::from_raw(0),
                [],
                axes,
                counts.iter().map(|&n_cell| n_cell as f64 / n as f64).collect::<Vec<_>>(),
                snapshot,
                LawTolerance::default(),
            )
            .and_then(|law| law.with_empirical_counts(counts))
            .map_err(|e| {
                EstimationError::data_msg(format!("temporal_recalc.invalid_source_law: {e}"))
            })?;
            note_static_work(StaticWork::FactorBuild);
            Ok(joint)
        },
    )
}

/// Closed whole-unit candidate: both effect arms use the same resampled units.
/// This route is built only for the internal acceptance/calibration harness;
/// its empirical interval has no calibrated coverage claim.
/// # Errors
/// Invalid configuration, insufficient units, cancellation or unsupported resampled histories.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn history_interval_candidate(
    panel: &TemporalUnitPanel,
    active: [u32; 2],
    control: Option<[u32; 2]>,
    law: &InitialStateLaw,
    config: &crate::temporal_dependent_interval::DependentIntervalConfig,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<crate::temporal_dependent_interval::DependentInterval, EstimationError> {
    struct PairedHistory<'a> {
        active: [u32; 2],
        control: Option<[u32; 2]>,
        law: &'a InitialStateLaw,
    }
    impl crate::temporal_dependent_interval::TemporalEstimator for PairedHistory<'_> {
        fn label(&self) -> &'static str {
            if self.control.is_some() { "paired_sequence_effect" } else { "sequence_response" }
        }
        fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
            let sequences =
                [Some(self.active), self.control].into_iter().flatten().collect::<Vec<_>>();
            let fits = FittedSequenceHistory::fit_many(units, &sequences);
            let project = |fit: &FittedSequenceHistory| -> Result<f64, EstimationError> {
                Ok(fit.contributions(self.law)?.iter().map(|c| c.mass * c.response).sum())
            };
            let active = project(&fits[0])?;
            Ok(active - fits.get(1).map(project).transpose()?.unwrap_or(0.))
        }
    }
    crate::temporal_dependent_interval::dependent_unit_interval(
        panel,
        &PairedHistory { active, control, law },
        config,
        ctx,
    )
}

#[cfg(all(test, feature = "calibration-internal"))]
mod tests {
    use super::*;
    use crate::temporal_dependent_interval::{DependentIntervalConfig, SequenceHistory};
    use antecedent_core::ExecutionContext;
    fn candidate_pin(key: &str) -> f64 {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/recalculation/temporal_history/expected.json"
        ))
        .unwrap();
        oracle["paired_additive_candidate"][key].as_f64().unwrap()
    }

    fn panel() -> TemporalUnitPanel {
        let units = (0..24_u32)
            .map(|unit| {
                let mut histories = Vec::new();
                for s0 in 0..2 {
                    for a1 in 0..2 {
                        for l2 in 0..2 {
                            for a2 in 0..2 {
                                let y = 1.
                                    + f64::from(s0)
                                    + 2. * f64::from(a1)
                                    + 3. * f64::from(a2)
                                    + 0.5 * f64::from(l2)
                                    + 0.01 * f64::from(unit);
                                for _ in 0..2 {
                                    histories.push(SequenceHistory {
                                        time_id: u64::try_from(histories.len()).unwrap() * 2,
                                        s0,
                                        a1,
                                        l2,
                                        a2,
                                        y,
                                    });
                                }
                            }
                        }
                    }
                }
                UnitHistories { unit_id: u64::from(unit), histories }
            })
            .collect();
        TemporalUnitPanel::new("correlated-within-unit", Some(units)).unwrap()
    }
    #[test]
    fn shared_history_accumulator_matches_each_existing_single_sequence_tally() {
        let panel = panel();
        let units = panel.units().iter().collect::<Vec<_>>();
        let sequences = [[0, 0], [0, 1], [1, 0], [1, 1]];
        let law = InitialStateLaw::new(
            InitialStatePopulation::Target,
            "target",
            vec![(0, 0.2), (1, 0.8)],
        )
        .unwrap();
        let fitted = FittedSequenceHistory::fit_many(&units, &sequences);
        for (fit, &sequence) in fitted.iter().zip(&sequences) {
            let original = SequenceTallies::of(&units, sequence);
            for c in fit.contributions(&law).unwrap() {
                assert_eq!(c.response.to_bits(), original.response(c.s0).unwrap().to_bits());
                let independent = 1.
                    + f64::from(c.s0)
                    + 2. * f64::from(sequence[0])
                    + 3. * f64::from(sequence[1])
                    + 0.25
                    + 0.115;
                assert!((c.response - independent).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn history_candidate_pairs_both_sequences_on_identical_whole_unit_draws() {
        let panel = panel();
        let law = InitialStateLaw::new(
            InitialStatePopulation::Target,
            "target",
            vec![(0, 0.2), (1, 0.8)],
        )
        .unwrap();
        // The retained history estimator has no certified studentization. Its
        // former generic interval must refuse rather than borrow the licensed
        // balanced-unit estimator's authority.
        let config = DependentIntervalConfig {
            replicates: 20,
            seed: 19,
            ..DependentIntervalConfig::default()
        };
        let ctx = ExecutionContext::for_tests(19);
        for control in [Some([0, 0]), None] {
            let error = history_interval_candidate(&panel, [1, 1], control, &law, &config, &ctx)
                .unwrap_err();
            assert!(matches!(
                error,
                EstimationError::Refused { code: "route_not_supported", ref message }
                    if message == "temporal_interval.studentized_estimator_not_certified"
            ));
        }

        // Specify shared whole-unit selections independently of the interval
        // engine, including duplicates and reordering. Each fit sees exactly
        // the same histories in both arms, so the unit disturbance cancels.
        for selection in [vec![0, 23, 23, 7], vec![7, 0, 23, 23], (0..24).collect()] {
            let units = selection.iter().map(|&index| &panel.units()[index]).collect::<Vec<_>>();
            let fits = FittedSequenceHistory::fit_many(&units, &[[1, 1], [0, 0]]);
            let project = |fit: &FittedSequenceHistory| {
                fit.contributions(&law).unwrap().iter().map(|c| c.mass * c.response).sum::<f64>()
            };
            let response = project(&fits[0]);
            let control = project(&fits[1]);
            assert!((response - control - candidate_pin("effect_11_vs_00")).abs() < 1e-10);
            if selection.len() == panel.units().len() {
                assert!((response - candidate_pin("response_11")).abs() < 1e-10);
            }
        }
    }
}

#[cfg(test)]
mod categorical_tests {
    use super::*;
    use crate::temporal_dependent_interval::SequenceHistory;

    fn unit(histories: &[(u32, u32, u32, u32, f64)]) -> UnitHistories {
        UnitHistories {
            unit_id: 0,
            histories: histories
                .iter()
                .enumerate()
                .map(|(i, &(s0, a1, l2, a2, y))| SequenceHistory {
                    time_id: i as u64 * 2,
                    s0,
                    a1,
                    l2,
                    a2,
                    y,
                })
                .collect(),
        }
    }

    #[test]
    fn binary_histories_keep_the_binary_cell_order_in_both_routes() {
        let u = unit(&[(1, 0, 1, 1, 1.0), (0, 1, 0, 0, 0.0), (1, 0, 1, 1, 0.0)]);
        let units = [&u];
        let binary = fit_binary_history_units("s", &units).unwrap();
        let general = fit_categorical_history_units("s", &units).unwrap();
        assert_eq!(binary.empirical_counts(), general.empirical_counts());
        let counts = binary.empirical_counts().unwrap();
        // cell = s0*16 + a1*8 + l2*4 + a2*2 + y
        assert_eq!((counts.len(), counts[16 + 4 + 2 + 1], counts[8], counts[16 + 4 + 2]), (32, 1, 1, 1));
    }

    #[test]
    fn a_three_level_coordinate_indexes_the_mixed_radix_cell() {
        let u = unit(&[(2, 0, 1, 1, 1.0), (0, 1, 0, 0, 2.0)]);
        let units = [&u];
        // The binary route refuses a level it does not carry.
        assert!(fit_binary_history_units("s", &units).is_err());
        let law = fit_categorical_history_units("s", &units).unwrap();
        // Levels per coordinate: state 3, a1 2, l2 2, a2 2, outcome 3.
        let radix = [3_usize, 2, 2, 2, 3];
        assert_eq!(law.axes().iter().map(|axis| axis.values.len()).collect::<Vec<_>>(), radix);
        let counts = law.empirical_counts().unwrap();
        assert_eq!(counts.len(), 3 * 2 * 2 * 2 * 3);
        let cell = |c: [usize; 5]| c.iter().zip(radix).fold(0, |cell, (&code, level)| cell * level + code);
        assert_eq!(counts[cell([2, 0, 1, 1, 1])], 1);
        assert_eq!(counts[cell([0, 1, 0, 0, 2])], 1);
        assert_eq!(counts.iter().sum::<u64>(), 2);
        assert!(fit_categorical_history_units("s", &[&unit(&[(0, 0, 0, 0, 0.5)])]).is_err());
    }
}
