//! Two-decision target policy choice with finite structural and preference ambiguity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use antecedent_io_core::contract_section::CalibrationSlotWire;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) mod artifact;
pub use artifact::{
    CompiledScheduleEvidence, IntegratedScheduleBundle, ScheduleBundleError, ScheduleEvidence,
    ScheduleReplayInputs, compile_program_schedule_bundle, consume_program_schedule_bundle,
    inspect_program_schedule_bundle,
};

const MAX_SUBJECTS: usize = 1_000_000;
const PRIMITIVES: f64 = 13.0; // eight observed endpoint means, four second-action probabilities, one target prevalence
const ALPHA: f64 = 0.05;

/// Independent source subject with binary baseline and two sequential actions.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleSourceSubject {
    /// Pre-first-action baseline stratum.
    pub baseline: bool,
    /// The two realized actions, ordered as administered.
    pub actions: [bool; 2],
    /// Endpoint on the declared [0,1] utility outcome scale, absent after censoring.
    pub endpoint: Option<f64>,
}

/// Declared structural possibilities for the causal role of the two actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleStructure {
    /// Conditional on baseline and observed earlier action, both decisions are exchangeable.
    BothActionsCausal,
    /// The first decision is exchangeable given baseline. The second has no causal effect;
    /// its natural dependence on the endpoint can arise from post-first-action common causes.
    FirstActionOnly,
}

/// Source-fixed study law for which the final optimum set is measured.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleCalibrationDesign {
    /// Declared eight-cell Bernoulli experiment with a unique optimum within each world.
    ExampleUniqueOptima,
    /// Same experiment using the fixed cost table that produces tied optima.
    ExampleTiedOptima,
}

/// Finite structural and preference states fixed independently of the two samples.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleDecisionDeclaration {
    /// Candidate two-bit schedule codes, first action most significant, in ascending order.
    pub schedules: Vec<u8>,
    /// Exhaustive declared structural possibilities, unweighted.
    pub structures: Vec<ScheduleStructure>,
    /// Finite preference states. Each row gives known cost by schedule code on the endpoint scale.
    pub preference_costs: Vec<[f64; 4]>,
    /// Declare the source-fixed example study law when evaluating its calibration scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_design: Option<ScheduleCalibrationDesign>,
}

impl ScheduleDecisionDeclaration {
    /// Require the complete finite declaration used by this decision.
    ///
    /// # Errors
    /// Missing structural or preference states, or coordinates outside the bounded design.
    pub fn check(&self) -> Result<(), ScheduleDecisionError> {
        if self.structures.len() != 2
            || !self.structures.contains(&ScheduleStructure::BothActionsCausal)
            || !self.structures.contains(&ScheduleStructure::FirstActionOnly)
            || self.preference_costs.is_empty()
        {
            return Err(ScheduleDecisionError::MissingJointEvidence);
        }
        if !(2..=4).contains(&self.schedules.len())
            || self.schedules.iter().any(|&code| code >= 4)
            || self.schedules.windows(2).any(|w| w[0] >= w[1])
            || self.preference_costs.len() > 16
            || self
                .preference_costs
                .iter()
                .flatten()
                .any(|cost| !cost.is_finite() || !(0.0..=1.0).contains(cost))
        {
            return Err(ScheduleDecisionError::InvalidDeclaration);
        }
        Ok(())
    }
}

/// Setup or support refusal from the bounded schedule procedure.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ScheduleDecisionError {
    /// Combined evidence is required before the final set can be computed.
    #[error(
        "supply source trajectories with censor indicators, an independent target baseline sample, both structural alternatives and finite joint preference costs"
    )]
    MissingJointEvidence,
    /// An endpoint or utility coordinate lies outside the compact declared domain.
    #[error(
        "the integrated schedule decision requires sorted binary schedules, bounded samples, endpoints in [0,1] and finite nonnegative preference costs"
    )]
    InvalidDeclaration,
    /// Too little overlap or follow-up to estimate source primitives.
    #[error(
        "increase observed source support: each baseline-by-schedule cell needs at least 10 uncensored endpoints and the target needs at least 20 independent baseline rows"
    )]
    InadequateOverlap,
    /// The caller cancelled the procedure before its result was published.
    #[error("the integrated schedule decision was cancelled")]
    Cancelled,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Cell {
    assigned: usize,
    observed: usize,
    endpoint_sum: f64,
}

/// Sufficient source-study evidence from the common trajectory sample.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleSourceFit {
    cells: [[Cell; 4]; 2],
    // Only a fit from checked rows can establish this calibration requirement.
    #[serde(skip)]
    binary_observed_endpoints: bool,
}

impl ScheduleSourceFit {
    /// Fit counts, censoring risks and bounded endpoint means jointly from whole subjects.
    pub fn fit(rows: &[ScheduleSourceSubject]) -> Result<Self, ScheduleDecisionError> {
        if rows.is_empty() {
            return Err(ScheduleDecisionError::MissingJointEvidence);
        }
        if rows.len() > MAX_SUBJECTS {
            return Err(ScheduleDecisionError::InvalidDeclaration);
        }
        let mut cells = [[Cell::default(); 4]; 2];
        let mut binary_observed_endpoints = true;
        for row in rows {
            let action = 2 * usize::from(row.actions[0]) + usize::from(row.actions[1]);
            let cell = &mut cells[usize::from(row.baseline)][action];
            cell.assigned += 1;
            if let Some(endpoint) = row.endpoint {
                if !endpoint.is_finite() || !(0.0..=1.0).contains(&endpoint) {
                    return Err(ScheduleDecisionError::InvalidDeclaration);
                }
                cell.observed += 1;
                cell.endpoint_sum += endpoint;
                binary_observed_endpoints &= matches!(endpoint, 0.0 | 1.0);
            }
        }
        let fit = Self { cells, binary_observed_endpoints };
        fit.check()?;
        Ok(fit)
    }

    fn check(&self) -> Result<(), ScheduleDecisionError> {
        if self.cells.iter().flatten().any(|cell| {
            cell.assigned > MAX_SUBJECTS
                || cell.observed > cell.assigned
                || !cell.endpoint_sum.is_finite()
                || !(0.0..=cell.observed as f64).contains(&cell.endpoint_sum)
        }) || self.cells.iter().flatten().map(|cell| cell.assigned).sum::<usize>() > MAX_SUBJECTS
        {
            return Err(ScheduleDecisionError::InvalidDeclaration);
        }
        if self.cells.iter().flatten().any(|cell| cell.observed < 10) {
            return Err(ScheduleDecisionError::InadequateOverlap);
        }
        Ok(())
    }
}

/// Independent target sample reduced to the count of the two baseline strata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleTargetCohort {
    subjects: usize,
    positive: usize,
}

impl ScheduleTargetCohort {
    /// Construct from independent baseline rows in the named target population.
    pub fn fit(baseline: &[bool]) -> Result<Self, ScheduleDecisionError> {
        if baseline.is_empty() {
            return Err(ScheduleDecisionError::MissingJointEvidence);
        }
        let fit =
            Self { subjects: baseline.len(), positive: baseline.iter().filter(|&&x| x).count() };
        fit.check()?;
        Ok(fit)
    }

    fn check(self) -> Result<(), ScheduleDecisionError> {
        if self.subjects > MAX_SUBJECTS || self.positive > self.subjects {
            return Err(ScheduleDecisionError::InvalidDeclaration);
        }
        if self.subjects < 20 {
            return Err(ScheduleDecisionError::InadequateOverlap);
        }
        Ok(())
    }
}

/// Simultaneous limits for one candidate-minus-other utility difference.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleUtilityDifference {
    /// Candidate schedule code.
    pub candidate: u8,
    /// Compared schedule code.
    pub other: u8,
    /// Joint confidence bounds for the difference under this structural/preference state.
    pub bounds: [f64; 2],
}

/// A fixed structural and preference state before unioning the possible optima.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleScenarioResult {
    /// Declared graph interpretation.
    pub structure: ScheduleStructure,
    /// Index of the finite preference state.
    pub preference_state: usize,
    /// Plug-in utilities by schedule, for comparison with the confidence set.
    pub point_utilities: Vec<(u8, f64)>,
    /// Every candidate not shown inferior to another by a simultaneous bound.
    pub compatible_optima: Vec<u8>,
    /// Bounds from the common source and target confidence region.
    pub differences: Vec<ScheduleUtilityDifference>,
}

/// The final optimum confidence set and the precise scope of its coverage guarantee.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IntegratedScheduleDecision {
    /// Union of compatible optimal schedules across all graph and preference states.
    pub optimum_set: Vec<u8>,
    /// Separate structural and preference states, with every competing difference.
    pub scenarios: Vec<ScheduleScenarioResult>,
    /// Empirical treatment-one probability at the first decision, by baseline.
    pub first_assignment: [f64; 2],
    /// Empirical treatment-one probability at the second decision, by baseline then first action.
    pub second_assignment: [[f64; 2]; 2],
    /// Empirical conditional observation survival, by baseline and full schedule.
    pub observation_survival: [[f64; 4]; 2],
    /// Target stratum prevalence from the independent target sample.
    pub target_prevalence: f64,
    /// Simultaneous finite-sample lower bound on coverage of the entire set of true optima.
    pub minimum_coverage: f64,
    /// Conditions that delimit the bound, as stated by the compiled procedure.
    pub coverage_scope: String,
    /// Measurement key for the two prespecified example studies, when in their scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_basis: Option<CalibrationBasisWire>,
}

impl IntegratedScheduleDecision {
    /// Current measurement standing, resolved separately from artifact identity and the theorem.
    #[must_use]
    pub fn calibration(&self) -> CalibrationSlotWire {
        self.calibration_basis.as_ref().map_or_else(
            || {
                CalibrationSlotWire::not_measured(
                    "integrated_schedule_example_design_outside_measurement_scope",
                )
            },
            calibration_slot,
        )
    }
}

fn calibration_basis(
    declaration: &ScheduleDecisionDeclaration,
    source_rows: u64,
    target_rows: u64,
    binary_observed_endpoints: bool,
) -> Option<CalibrationBasisWire> {
    if !binary_observed_endpoints
        || declaration.schedules != [0, 1, 2, 3]
        || declaration.preference_costs.len() != 2
        || declaration.preference_costs[0] != [0.0, 0.0625, 0.125, 0.1875]
        || target_rows != source_rows / 3
    {
        return None;
    }
    let functional = match (declaration.calibration_design?, declaration.preference_costs[1]) {
        (ScheduleCalibrationDesign::ExampleUniqueOptima, [0.0, 0.125, 0.0625, 0.25]) => {
            "integrated_schedule_example_unique_optima"
        }
        (ScheduleCalibrationDesign::ExampleTiedOptima, [0.0, 0.1875, 0.460_937_5, 0.460_937_5]) => {
            "integrated_schedule_example_all_tied_optima"
        }
        _ => return None,
    };
    Some(CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "not_applicable".into(),
            query: "IntegratedScheduleDecision".into(),
            graph_class: "NotApplicable".into(),
            structure: "scenario_set".into(),
            modality: "tabular".into(),
            inference: "Frequentist".into(),
            estimator: "schedule_decision.joint_fit_and_transport".into(),
            interval_method: antecedent_core::IntervalMethod::HoeffdingOptimumSet.as_str().into(),
            se_kind: String::new(),
            dependence: "shared_source_trajectories_independent_target_sample".into(),
            posterior: String::new(),
            functional: functional.into(),
            level: 1.0 - ALPHA,
            identification: "partial".into(),
        },
        scope: CalibrationScopeWire {
            row_count: source_rows,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    })
}

#[derive(Clone, Copy)]
struct Region {
    point: f64,
    bounds: [f64; 2],
}

impl Region {
    fn bounded_mean(sum: f64, count: usize) -> Self {
        let point = sum / count as f64;
        let radius = ((2.0 * PRIMITIVES / ALPHA).ln() / (2.0 * count as f64)).sqrt();
        Self { point, bounds: [(point - radius).max(0.0), (point + radius).min(1.0)] }
    }
}

fn mix(weight: Region, baseline0: Region, baseline1: Region) -> Region {
    let projected =
        |p: f64, end: usize| (1.0 - p) * baseline0.bounds[end] + p * baseline1.bounds[end];
    Region {
        point: (1.0 - weight.point) * baseline0.point + weight.point * baseline1.point,
        bounds: [
            projected(weight.bounds[0], 0).min(projected(weight.bounds[1], 0)),
            projected(weight.bounds[0], 1).max(projected(weight.bounds[1], 1)),
        ],
    }
}

fn difference(a: Region, b: Region) -> Region {
    Region {
        point: a.point - b.point,
        bounds: [a.bounds[0] - b.bounds[1], a.bounds[1] - b.bounds[0]],
    }
}

fn zero() -> Region {
    Region { point: 0.0, bounds: [0.0; 2] }
}

/// Compiled three-stage decision: joint source fit, target standardization, and optimal-set projection.
#[derive(Clone, Debug)]
pub struct PreparedIntegratedScheduleDecision {
    source: ScheduleSourceFit,
    target: ScheduleTargetCohort,
    declaration: ScheduleDecisionDeclaration,
}

impl PreparedIntegratedScheduleDecision {
    /// Seal the two distinct samples and the finite scientific declaration before ranking schedules.
    pub fn compile(
        source: ScheduleSourceFit,
        target: ScheduleTargetCohort,
        declaration: ScheduleDecisionDeclaration,
    ) -> Result<Self, ScheduleDecisionError> {
        source.check()?;
        target.check()?;
        declaration.check()?;
        Ok(Self { source, target, declaration })
    }

    /// Project a joint primitive confidence region through the same transport and decision stages.
    pub fn execute(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<IntegratedScheduleDecision, ScheduleDecisionError> {
        if ctx.cancellation.is_cancelled() {
            return Err(ScheduleDecisionError::Cancelled);
        }
        let groups = self.source.cells;
        let target = Region::bounded_mean(self.target.positive as f64, self.target.subjects);
        // Each bound is valid conditional on its realized source count. Bonferroni covers all
        // primitives together without treating the dependent source estimators as independent.
        let terminal = groups
            .map(|cells| cells.map(|cell| Region::bounded_mean(cell.endpoint_sum, cell.observed)));
        let second = groups.map(|cells| {
            [0, 2].map(|start| {
                Region::bounded_mean(
                    cells[start + 1].assigned as f64,
                    cells[start].assigned + cells[start + 1].assigned,
                )
            })
        });
        let observed =
            groups.map(|cells| cells.map(|cell| cell.observed as f64 / cell.assigned as f64));
        let first = groups.map(|cells| {
            (cells[2].assigned + cells[3].assigned) as f64
                / cells.iter().map(|cell| cell.assigned).sum::<usize>() as f64
        });
        let mut included = [false; 4];
        let mut scenarios = Vec::new();
        for &structure in &self.declaration.structures {
            let cells: [[Region; 4]; 2] = std::array::from_fn(|baseline| match structure {
                ScheduleStructure::BothActionsCausal => terminal[baseline],
                ScheduleStructure::FirstActionOnly => [0, 0, 2, 2].map(|first| {
                    mix(
                        second[baseline][first / 2],
                        terminal[baseline][first],
                        terminal[baseline][first + 1],
                    )
                }),
            });
            for (preference_state, costs) in self.declaration.preference_costs.iter().enumerate() {
                let mut row = ScheduleScenarioResult {
                    structure,
                    preference_state,
                    point_utilities: Vec::new(),
                    compatible_optima: Vec::new(),
                    differences: Vec::new(),
                };
                for &code in &self.declaration.schedules {
                    let a = code as usize;
                    let value = mix(target, cells[0][a], cells[1][a]).point - costs[a];
                    row.point_utilities.push((code, value));
                    let mut compatible = true;
                    for &other in self.declaration.schedules.iter().filter(|&&other| other != code)
                    {
                        let b = other as usize;
                        let shared_first =
                            structure == ScheduleStructure::FirstActionOnly && a / 2 == b / 2;
                        let strata = cells.map(|within| {
                            if shared_first { zero() } else { difference(within[a], within[b]) }
                        });
                        let contrast = mix(target, strata[0], strata[1]);
                        let bounds = contrast.bounds.map(|value| value - costs[a] + costs[b]);
                        compatible &= bounds[1] >= 0.0;
                        row.differences.push(ScheduleUtilityDifference {
                            candidate: code,
                            other,
                            bounds,
                        });
                    }
                    if compatible {
                        included[a] = true;
                        row.compatible_optima.push(code);
                    }
                }
                scenarios.push(row);
            }
        }
        Ok(IntegratedScheduleDecision {
            optimum_set: (0_u8..4).filter(|&code| included[code as usize]).collect(),
            scenarios,
            first_assignment: first,
            second_assignment: second.map(|baseline| baseline.map(|p| p.point)),
            observation_survival: observed,
            target_prevalence: target.point,
            minimum_coverage: 1.0 - ALPHA,
            coverage_scope: "Repeated-study probability of publishing a set that misses any member of the entire optimum set is at most 0.05. Bounds jointly cover independent identically distributed source trajectories and a separate independent identically distributed target baseline sample, bounded endpoints, positivity, consistency, and conditionally exchangeable censoring given baseline and both observed actions. The first action is exchangeable given baseline. The second is either exchangeable given baseline and the first action, or has no causal effect. Graph-specific baseline-conditional schedule values transport to the target. The prespecified structures and preference states are bounded by their declared finite sets.".into(),
            calibration_basis: calibration_basis(
                &self.declaration,
                groups.iter().flatten().map(|cell| cell.assigned as u64).sum(),
                self.target.subjects as u64,
                self.source.binary_observed_endpoints,
            ),
        })
    }
}
