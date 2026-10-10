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

/// Fit one empirical joint source factor from actual complete binary histories.
/// The joint is labeled empirical and carries reconciled counts, never supplied-exact.
/// Whole-unit ownership remains in the input panel for the dependent candidate.
/// # Errors
/// No histories, unsupported nonbinary coordinates/outcome, or invalid frequency law.
#[allow(clippy::cast_precision_loss, reason = "the panel is bounded far below 2^53")]
#[allow(clippy::float_cmp, reason = "binary outcome membership requires exactly zero or one")]
pub fn fit_binary_history_joint(
    panel: &TemporalUnitPanel,
) -> Result<ExactDiscreteLaw, EstimationError> {
    fit_binary_history_units(panel.snapshot_id(), &panel.units().iter().collect::<Vec<_>>())
}
/// Fit the same empirical history joint on a whole-unit draw, including repeated units.
/// # Errors
/// Empty draw, nonbinary history coordinates or invalid source frequency law.
#[allow(clippy::cast_precision_loss, reason = "the panel is bounded far below 2^53")]
#[allow(clippy::float_cmp, reason = "binary outcome membership requires exactly zero or one")]
pub fn fit_binary_history_units(
    snapshot: &str,
    units: &[&UnitHistories],
) -> Result<ExactDiscreteLaw, EstimationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::FactorConstruction,
        || {
            let mut counts = vec![0_u64; 32];
            for h in units.iter().flat_map(|u| &u.histories) {
                if [h.s0, h.a1, h.l2, h.a2].iter().any(|v| *v > 1) || !(h.y == 0. || h.y == 1.) {
                    return Err(EstimationError::refused(
                        reason_code!("route_not_supported"),
                        "temporal_recalc.binary_history_scope: only binary state, actions, covariate and outcome",
                    ));
                }
                let y = usize::from(h.y == 1.);
                let cell =
                    usize::try_from(h.s0 * 16 + h.a1 * 8 + h.l2 * 4 + h.a2 * 2).map_err(|_| {
                        EstimationError::data_msg("temporal_recalc.invalid_history_fit")
                    })? + y;
                counts[cell] += 1;
            }
            let n: u64 = counts.iter().sum();
            if n == 0 {
                return Err(EstimationError::data_msg("temporal_recalc.empty_histories"));
            }
            let axes = (0..5)
                .map(|id| DiscreteAxis {
                    variable: VariableId::from_raw(id),
                    values: Arc::from([Value::f64(0.), Value::f64(1.)]),
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
