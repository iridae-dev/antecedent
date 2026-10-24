//! The value of measuring a named confounder, under a prior the caller declares.
//!
//! A sensitivity analysis reports how a decision depends on an unmeasured
//! confounding coordinate: an invariant action, a switch at a tipping coordinate,
//! or no robust action ([`SensitivityOutcome`]). Its assumption range is not a
//! probability law, so by itself it cannot say what *learning* the coordinate is
//! worth. This module turns that outcome into a decision problem only through a
//! prior the caller declares and that is recorded with the result:
//!
//! * the **states** are the grid coordinates the sensitivity surface was evaluated
//!   at, the **actions** are the surface's declared actions and the **utilities**
//!   are the surface's own exact utilities at those coordinates (a
//!   [`GridUtility`]); nothing is interpolated between grid points;
//! * the **prior** is a [`ConfounderPriorDeclaration`]: equally weighted draws of
//!   the grid coordinates in the declared multiplicities, with who declared it and
//!   what it rests on. It is never inferred from the range;
//! * the **signal** (how well a study measures the confounder) is the caller's
//!   ordinary signal declaration on the design-ranking path, so the value is an
//!   expected value of sample information under the declared prior and signal,
//!   comparable to any other candidate's.
//!
//! A surface with an assumption range (several scenarios per grid point), or one
//! with an unevaluated grid point, is refused: those carry no single utility to
//! value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{blake3_hex, reason_code};

use crate::decision::{DecisionProblem, Utility};
use crate::error::DesignError;
use crate::preposterior::DecisionPrior;
use crate::sensitivity_decision::{
    ScenarioCoverage, SensitivityDecisionResult, SensitivityOutcome,
};

/// Most prior draws a declaration may expand to.
pub const MAX_PRIOR_DRAWS: u64 = 65_536;

/// A utility defined only on the grid coordinates it was evaluated at.
#[derive(Clone, Debug, PartialEq)]
pub struct GridUtility {
    states: Arc<[f64]>,
    rows: Arc<[Vec<f64>]>,
}

impl GridUtility {
    /// `rows[action][k]` is the utility of `action` at `states[k]`.
    ///
    /// # Errors
    /// No states or actions, a ragged table, repeated or non-finite states, or a
    /// non-finite utility.
    pub fn new(states: Vec<f64>, rows: Vec<Vec<f64>>) -> Result<Self, DesignError> {
        if states.is_empty() || rows.is_empty() || rows.iter().any(|r| r.len() != states.len()) {
            return Err(DesignError::Shape(
                "a grid utility needs states and one utility per state for every action".into(),
            ));
        }
        if states.iter().chain(rows.iter().flatten()).any(|x| !x.is_finite()) {
            return Err(DesignError::Numerical("grid utility values must be finite".into()));
        }
        for (k, state) in states.iter().enumerate() {
            if states[..k].iter().any(|other| other.to_bits() == state.to_bits()) {
                return Err(DesignError::Shape("grid coordinates must be distinct".into()));
            }
        }
        Ok(Self { states: states.into(), rows: rows.into() })
    }

    /// Number of actions.
    #[must_use]
    pub fn n_actions(&self) -> usize {
        self.rows.len()
    }

    /// The grid coordinates.
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.states
    }

    /// The utilities, `[action][state]`.
    #[must_use]
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }
}

impl Utility<usize, f64> for GridUtility {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        outcomes: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let n_o = outcomes.len();
        for (row, &action) in actions.iter().enumerate() {
            let utilities = self.rows.get(action).ok_or_else(|| {
                DesignError::Shape(format!("grid utility has no action index {action}"))
            })?;
            for (col, state) in outcomes.iter().enumerate() {
                let k = self
                    .states
                    .iter()
                    .position(|s| s.to_bits() == state.to_bits())
                    .ok_or_else(|| {
                        DesignError::Numerical(format!(
                            "state {state} is not a grid coordinate the surface was evaluated at; \
                             nothing is interpolated between grid points"
                        ))
                    })?;
                out[row * n_o + col] = utilities[k];
            }
        }
        Ok(())
    }
}

/// A refused frame: a registered reason code and a stable detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct ConfounderValueError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `confounder_value.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl ConfounderValueError {
    fn surface(detail: &'static str, message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("decision_contract_unsatisfied"),
            detail,
            message: message.into(),
        }
    }

    fn prior(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: "confounder_value.invalid_prior",
            message: message.into(),
        }
    }
}

/// The belief about the confounding coordinate, declared by the caller.
///
/// An assumption range says what is possible, not what is probable; this is the
/// only place a probability over the coordinate enters, it is the caller's, and it
/// is part of the value's identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfounderPriorDeclaration {
    /// Who asserts the prior.
    pub declared_by: Arc<str>,
    /// What the prior rests on (data, elicitation, a published bound).
    pub basis: Arc<str>,
    /// Equally weighted draws of each grid coordinate, in ascending coordinate
    /// order. Zero means the coordinate is given no prior mass.
    pub draws_per_coordinate: Vec<u32>,
}

impl ConfounderPriorDeclaration {
    /// Stable identity (`cp1:<digest>`) of the declaration.
    #[must_use]
    pub fn identity(&self) -> String {
        let counts = self.draws_per_coordinate.iter().map(ToString::to_string).collect::<Vec<_>>();
        let content =
            format!("by={};basis={};draws=[{}]", self.declared_by, self.basis, counts.join(","));
        format!("cp1:{}", &blake3_hex(content.as_bytes())[..32])
    }
}

/// A sensitivity surface read as a decision over the confounding coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfounderValueFrame {
    /// The evaluated grid coordinates, ascending.
    pub coordinates: Vec<f64>,
    /// The surface's action identities.
    pub action_ids: Vec<String>,
    /// Utility of each action at each coordinate, `[action][coordinate]`.
    pub utilities: Vec<Vec<f64>>,
    /// What the surface found along the coordinate.
    pub outcome: SensitivityOutcome,
    /// Identity of the sensitivity artifact the surface came from.
    pub artifact_identity: String,
    /// The declared prior.
    pub prior: ConfounderPriorDeclaration,
}

impl ConfounderValueFrame {
    /// Read the exact per-coordinate utilities off a sensitivity decision over a
    /// point surface and attach the declared prior.
    ///
    /// # Errors
    /// `confounder_value.range_surface` for a surface with an assumption range;
    /// `confounder_value.unevaluated_point` when a grid point carries no utility;
    /// `confounder_value.invalid_prior` for a prior whose draw counts do not match
    /// the grid, give no mass, or exceed [`MAX_PRIOR_DRAWS`].
    pub fn from_sensitivity(
        result: &SensitivityDecisionResult,
        prior: ConfounderPriorDeclaration,
    ) -> Result<Self, ConfounderValueError> {
        if result.coverage != ScenarioCoverage::PointSurface {
            return Err(ConfounderValueError::surface(
                "confounder_value.range_surface",
                "the surface carries an assumption range; a range is not a probability law, so \
                 it has no single utility to value (declare a point surface)",
            ));
        }
        let mut utilities = Vec::with_capacity(result.structural.actions.len());
        for action in &result.structural.actions {
            let row: Option<Vec<f64>> = action.per_atom.iter().copied().collect();
            let row = row.filter(|r| r.len() == result.coordinates.len()).ok_or_else(|| {
                ConfounderValueError::surface(
                    "confounder_value.unevaluated_point",
                    format!(
                        "action `{}` has no utility at some grid point; nothing is assumed there",
                        action.id
                    ),
                )
            })?;
            utilities.push(row);
        }
        Self::from_table(
            result.coordinates.clone(),
            result.structural.actions.iter().map(|a| a.id.clone()).collect(),
            utilities,
            result.outcome.clone(),
            result.artifact_identity.clone(),
            prior,
        )
    }

    /// Build a frame from an explicit utility table (`[action][coordinate]`).
    ///
    /// # Errors
    /// As [`Self::from_sensitivity`], and a ragged or non-finite table.
    pub fn from_table(
        coordinates: Vec<f64>,
        action_ids: Vec<String>,
        utilities: Vec<Vec<f64>>,
        outcome: SensitivityOutcome,
        artifact_identity: String,
        prior: ConfounderPriorDeclaration,
    ) -> Result<Self, ConfounderValueError> {
        GridUtility::new(coordinates.clone(), utilities.clone()).map_err(|e| {
            ConfounderValueError::surface("confounder_value.invalid_surface", e.to_string())
        })?;
        if action_ids.len() != utilities.len() {
            return Err(ConfounderValueError::surface(
                "confounder_value.invalid_surface",
                "one action identity per utility row",
            ));
        }
        let total: u64 = prior.draws_per_coordinate.iter().map(|c| u64::from(*c)).sum();
        if prior.draws_per_coordinate.len() != coordinates.len() {
            return Err(ConfounderValueError::prior(
                "declare one draw count per evaluated grid coordinate",
            ));
        }
        if total == 0 || total > MAX_PRIOR_DRAWS {
            return Err(ConfounderValueError::prior(format!(
                "a prior needs between 1 and {MAX_PRIOR_DRAWS} draws; {total} declared"
            )));
        }
        if prior.declared_by.trim().is_empty() || prior.basis.trim().is_empty() {
            return Err(ConfounderValueError::prior(
                "a prior names who declared it and what it rests on",
            ));
        }
        Ok(Self { coordinates, action_ids, utilities, outcome, artifact_identity, prior })
    }

    /// The decision problem over the grid: actions are indices into
    /// [`Self::action_ids`], the utility is the surface's.
    ///
    /// # Errors
    /// A table [`GridUtility`] refuses.
    pub fn problem(&self) -> Result<DecisionProblem<usize, f64>, DesignError> {
        let utility = GridUtility::new(self.coordinates.clone(), self.utilities.clone())?;
        Ok(DecisionProblem::new((0..self.action_ids.len()).collect(), Arc::new(utility), vec![]))
    }

    /// The declared prior as equally weighted draws of the grid coordinates.
    #[must_use]
    pub fn prior_draws(&self) -> Vec<f64> {
        self.coordinates
            .iter()
            .zip(&self.prior.draws_per_coordinate)
            .flat_map(|(coordinate, count)| std::iter::repeat_n(*coordinate, *count as usize))
            .collect()
    }

    /// The declared prior in the ranker's vocabulary.
    #[must_use]
    pub fn decision_prior(&self) -> DecisionPrior<f64> {
        DecisionPrior::Draws(self.prior_draws())
    }

    /// Digests of the sources the value rests on, to bind into the ranking: the
    /// sensitivity artifact and the declared prior.
    #[must_use]
    pub fn source_digests(&self) -> Vec<String> {
        let mut digests = vec![self.artifact_identity.clone(), self.prior.identity()];
        digests.sort();
        digests
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::evaluate_decision;
    use crate::decision_contract::StructuralPolicy;
    use crate::decision_structural::{
        ActionStructure, AtomStatus, AtomSummary, StructuralDecisionResult, StructuralVerdict,
    };
    use crate::sensitivity_decision::{DecisionSwitch, SamplingReport};

    fn prior(counts: Vec<u32>) -> ConfounderPriorDeclaration {
        ConfounderPriorDeclaration {
            declared_by: Arc::from("analyst"),
            basis: Arc::from("expert elicitation, 2026 panel"),
            draws_per_coordinate: counts,
        }
    }

    /// The treat action wins below the tipping coordinate 0.5, the withhold action above.
    fn switching() -> SensitivityDecisionResult {
        let coordinates = vec![0.0, 0.5, 1.0];
        let action = |id: &str, per_atom: Vec<Option<f64>>| ActionStructure {
            id: id.to_owned(),
            per_atom,
            range: None,
            weighted_value: None,
            mass_where_best: None,
            excluded_in: Vec::new(),
        };
        SensitivityDecisionResult {
            artifact_identity: "sens-artifact".to_owned(),
            coordinates: coordinates.clone(),
            structural: StructuralDecisionResult {
                contract_identity: "contract".to_owned(),
                policy: StructuralPolicy::ReportOnly,
                atoms: coordinates
                    .iter()
                    .map(|c| AtomSummary {
                        id: format!("c{c}"),
                        probability: None,
                        status: AtomStatus::Unidentified,
                    })
                    .collect(),
                actions: vec![
                    action("treat", vec![Some(1.0), Some(0.0), Some(-1.0)]),
                    action("withhold", vec![Some(-1.0), Some(0.0), Some(1.0)]),
                ],
                unidentified_mass: None,
                unevaluated_mass: None,
                evaluated_mass: None,
                verdict: StructuralVerdict::ReportOnly,
            },
            outcome: SensitivityOutcome::AssumptionDependent {
                switch: DecisionSwitch {
                    from: vec!["treat".to_owned()],
                    to: vec!["withhold".to_owned()],
                    lower: 0.0,
                    upper: 1.0,
                    exact: false,
                    interpolated: Some(0.5),
                },
            },
            coverage: ScenarioCoverage::PointSurface,
            sampling: SamplingReport::Withheld {
                reason_code: "not_executed".to_owned(),
                detail: "none".to_owned(),
            },
            interpretation: "test",
        }
    }

    #[test]
    fn the_value_of_learning_the_coordinate_is_zero_without_a_declared_spread() {
        // All declared mass on the first coordinate: the decision is already settled.
        let frame =
            ConfounderValueFrame::from_sensitivity(&switching(), prior(vec![10, 0, 0])).unwrap();
        let settled = evaluate_decision(&frame.problem().unwrap(), &frame.prior_draws()).unwrap();
        assert!(settled.posterior_regret.abs() < 1e-12);
        // A prior spread across the tipping coordinate leaves value to buy.
        let spread =
            ConfounderValueFrame::from_sensitivity(&switching(), prior(vec![5, 0, 5])).unwrap();
        let open = evaluate_decision(&spread.problem().unwrap(), &spread.prior_draws()).unwrap();
        assert!((open.posterior_regret - 1.0).abs() < 1e-12);
        assert_eq!(spread.action_ids, vec!["treat", "withhold"]);
    }

    #[test]
    fn the_prior_is_recorded_in_the_identity_and_never_inferred() {
        let a = ConfounderValueFrame::from_sensitivity(&switching(), prior(vec![1, 1, 1])).unwrap();
        let b = ConfounderValueFrame::from_sensitivity(&switching(), prior(vec![1, 2, 1])).unwrap();
        assert_ne!(a.prior.identity(), b.prior.identity());
        assert!(a.source_digests().contains(&"sens-artifact".to_owned()));
        assert!(a.source_digests().contains(&a.prior.identity()));
        assert_eq!(b.prior_draws(), vec![0.0, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn a_range_surface_an_unevaluated_point_and_a_mismatched_prior_are_refused() {
        let mut ranged = switching();
        ranged.coverage = ScenarioCoverage::VertexCertified;
        assert_eq!(
            ConfounderValueFrame::from_sensitivity(&ranged, prior(vec![1, 1, 1]))
                .unwrap_err()
                .detail,
            "confounder_value.range_surface"
        );
        let mut unevaluated = switching();
        unevaluated.structural.actions[0].per_atom[1] = None;
        assert_eq!(
            ConfounderValueFrame::from_sensitivity(&unevaluated, prior(vec![1, 1, 1]))
                .unwrap_err()
                .detail,
            "confounder_value.unevaluated_point"
        );
        for bad in [vec![1, 1], vec![0, 0, 0]] {
            assert_eq!(
                ConfounderValueFrame::from_sensitivity(&switching(), prior(bad))
                    .unwrap_err()
                    .detail,
                "confounder_value.invalid_prior"
            );
        }
        let mut anonymous = prior(vec![1, 1, 1]);
        anonymous.basis = Arc::from(" ");
        assert_eq!(
            ConfounderValueFrame::from_sensitivity(&switching(), anonymous).unwrap_err().detail,
            "confounder_value.invalid_prior"
        );
    }

    #[test]
    fn a_state_off_the_grid_is_never_interpolated() {
        let utility = GridUtility::new(vec![0.0, 1.0], vec![vec![1.0, 0.0]]).unwrap();
        let mut out = [0.0];
        let refused = utility.evaluate_batch(&[0], &[0.25], &mut out).unwrap_err();
        assert!(matches!(refused, DesignError::Numerical(_)));
        assert!(GridUtility::new(vec![0.0, 0.0], vec![vec![1.0, 0.0]]).is_err());
    }
}
