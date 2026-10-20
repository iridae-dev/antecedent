//! Robust decisions over identified sets, coverage regions and scenarios.
//!
//! A robust conclusion is a worst case or an invariant over a set. The mean of
//! the endpoints is reported beside it and is never that conclusion. A coverage
//! level is not a probability of the region, and an identified set is not given
//! one. Hard exclusions remove actions; they do not change the stored interval.
//!
//! Incomplete model mass is the structural evaluator's own mass, unscaled.
//! A bounded scenario simulation is compared with the mean of a small
//! materialized law and carries that simulation's Monte Carlo error. The
//! conjugate future-study score is the existing expectation, not a robust
//! conclusion.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decision_adapters::{
    AdapterError, IdentifiedSetDecision, IdentifiedUtility, IdentifiedVerdict,
    evaluate_identified_sets,
};
use crate::decision_contract::AdmissibleDecisionContract;
use crate::decision_reliability::WithinStreamDependence;
use crate::decision_reliability::mc_error::{BatchPlan, Functional, McEstimate, estimate};
use crate::decision_structural::{
    StructuralAtom, StructuralDecisionResult, StructuralError, evaluate_structural,
};
use crate::evsi::{FutureStudyError, FutureStudyProgram, FutureStudyValue};

/// Why a robust evaluation refused.
#[derive(Clone, Debug, PartialEq)]
pub enum RobustDecisionError {
    /// The identified-set evaluator refused. Bayes weights are this error's
    /// [`AdapterError::BayesOverIdentifiedSet`] value.
    Adapter(AdapterError),
    /// The structural evaluator refused.
    Structural(StructuralError),
    /// The Monte Carlo receipt refused.
    Reliability(crate::decision_reliability::ReliabilityError),
    /// The future-study score refused.
    FutureStudy(FutureStudyError),
    /// A coverage level was asked to act as a probability.
    CoverageIsNotAProbability,
    /// Scenario weights were asked to act as probabilities.
    ScenariosHaveNoProbabilities,
    /// The materialized law is empty or not finite.
    EmptyLaw,
    /// The simulation used more draws than its bound.
    DrawBoundExceeded {
        /// Draws supplied.
        draws: usize,
        /// Bound.
        max_draws: usize,
    },
}

impl std::fmt::Display for RobustDecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Adapter(error) => write!(f, "identified-set evaluation refused: {error:?}"),
            Self::Structural(error) => write!(f, "structural evaluation refused: {error:?}"),
            Self::Reliability(error) => write!(f, "monte carlo receipt refused: {error:?}"),
            Self::FutureStudy(error) => write!(f, "{error}"),
            Self::CoverageIsNotAProbability => {
                write!(f, "a coverage level is not a probability of the region")
            }
            Self::ScenariosHaveNoProbabilities => {
                write!(f, "a scenario set has no probabilities to weight by")
            }
            Self::EmptyLaw => write!(f, "a materialized law needs at least one finite value"),
            Self::DrawBoundExceeded { draws, max_draws } => {
                write!(f, "{draws} draws exceed the bound of {max_draws}")
            }
        }
    }
}

impl std::error::Error for RobustDecisionError {}

/// A coverage region. `coverage` is a nominal level, not a probability.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageRegion {
    /// Action the region describes.
    pub action_id: String,
    /// Lower endpoint of the region.
    pub lower: f64,
    /// Upper endpoint of the region.
    pub upper: f64,
    /// Nominal coverage in `(0, 1)`.
    pub coverage: f64,
    /// Hard constraints that exclude the action.
    pub hard_exclusions: Vec<String>,
}

/// Robust verdict and the endpoint average, kept apart.
#[derive(Clone, Debug, PartialEq)]
pub struct SetConclusions {
    /// Worst case, invariant, or the identified-set evaluator's other verdict.
    pub robust: IdentifiedVerdict,
    /// Action with the best mean of endpoints, when that ranking is unique.
    pub endpoint_average_action: Option<String>,
    /// Always false: an endpoint average is not a robust conclusion.
    pub endpoint_average_is_robust: bool,
    /// The identified-set decision the robust verdict was read from.
    pub decision: IdentifiedSetDecision,
}

/// Coverage regions evaluated as sets, with the levels retained unused as weights.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageConclusions {
    /// Set conclusions on the endpoints.
    pub set: SetConclusions,
    /// Nominal coverage of each action, in contract order.
    pub coverages: Vec<(String, f64)>,
}

/// Model mass as the structural evaluator reported it.
#[derive(Clone, Debug, PartialEq)]
pub struct UnnormalizedMass {
    /// Mass of evaluated structures.
    pub evaluated_mass: f64,
    /// Mass of unidentified structures.
    pub unidentified_mass: f64,
    /// Mass of unevaluated structures.
    pub unevaluated_mass: f64,
    /// Probability-weighted values, not divided by the evaluated mass.
    pub weighted_values: Vec<(String, f64)>,
}

/// A bounded simulation against the mean of a materialized law.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationComparison {
    /// Mean of every materialized row.
    pub reference_mean: f64,
    /// Monte Carlo error of the bounded draws.
    pub receipt: McEstimate,
    /// Absolute gap between the receipt and the reference mean.
    pub absolute_gap: f64,
    /// The gap is at most two receipt standard errors.
    pub within_two_standard_errors: bool,
}

/// The conjugate future-study score, marked as an expectation.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectationNotRobust {
    /// Gross EVSI from [`FutureStudyProgram::evaluate_future_study`].
    pub gross_evsi: f64,
    /// Net value when the program declared a cost.
    pub net_value: Option<f64>,
    /// Always false.
    pub is_robust_conclusion: bool,
}

/// Evaluate identified-set intervals. Bayes weights are refused.
///
/// # Errors
/// Any refusal of [`evaluate_identified_sets`], including
/// [`AdapterError::BayesOverIdentifiedSet`].
pub fn evaluate_robust_set(
    contract: &AdmissibleDecisionContract,
    utilities: &[IdentifiedUtility],
) -> Result<SetConclusions, RobustDecisionError> {
    let decision =
        evaluate_identified_sets(contract, utilities).map_err(RobustDecisionError::Adapter)?;
    Ok(conclusions_from(decision))
}

/// Evaluate coverage regions as identified sets.
///
/// The coverage level is stored and is not a weight.
///
/// # Errors
/// A coverage outside `(0, 1)`, or any refusal of [`evaluate_robust_set`].
pub fn evaluate_coverage_regions(
    contract: &AdmissibleDecisionContract,
    regions: &[CoverageRegion],
) -> Result<CoverageConclusions, RobustDecisionError> {
    if regions.iter().any(|region| !(region.coverage > 0.0 && region.coverage < 1.0)) {
        return Err(RobustDecisionError::CoverageIsNotAProbability);
    }
    let utilities: Vec<IdentifiedUtility> = regions
        .iter()
        .map(|region| IdentifiedUtility {
            action_id: region.action_id.clone(),
            utility: antecedent_core::IdentifiedSet { lower: region.lower, upper: region.upper },
            hard_exclusions: region.hard_exclusions.clone(),
            support: crate::decision_robustness::AtomSupport::supported(),
        })
        .collect();
    let set = evaluate_robust_set(contract, &utilities)?;
    let coverages =
        regions.iter().map(|region| (region.action_id.clone(), region.coverage)).collect();
    Ok(CoverageConclusions { set, coverages })
}

/// Refuse to read a coverage level as a probability of the parameter.
///
/// # Errors
/// Always [`RobustDecisionError::CoverageIsNotAProbability`].
pub const fn probability_of_coverage_region(coverage: f64) -> Result<f64, RobustDecisionError> {
    let _ = coverage;
    Err(RobustDecisionError::CoverageIsNotAProbability)
}

/// Refuse to weight a scenario set.
///
/// # Errors
/// Always [`RobustDecisionError::ScenariosHaveNoProbabilities`].
pub const fn weight_scenarios(weights: &[f64]) -> Result<&[f64], RobustDecisionError> {
    let _ = weights;
    Err(RobustDecisionError::ScenariosHaveNoProbabilities)
}

/// Read model mass from a structural result without scaling it to one.
#[must_use]
pub fn unnormalized_mass(result: &StructuralDecisionResult) -> UnnormalizedMass {
    let value = |action: &crate::decision_structural::ActionStructure| {
        (action.id.clone(), action.weighted_value.unwrap_or(f64::NAN))
    };
    UnnormalizedMass {
        evaluated_mass: result.evaluated_mass.unwrap_or(0.0),
        unidentified_mass: result.unidentified_mass.unwrap_or(0.0),
        unevaluated_mass: result.unevaluated_mass.unwrap_or(0.0),
        weighted_values: result.actions.iter().map(value).collect(),
    }
}

/// Evaluate structures and return their mass unscaled.
///
/// # Errors
/// Any refusal of [`evaluate_structural`].
pub fn evaluate_model_mass(
    contract: &crate::decision_contract::DecisionContract,
    atoms: &[StructuralAtom],
) -> Result<UnnormalizedMass, RobustDecisionError> {
    let result = evaluate_structural(contract, atoms).map_err(RobustDecisionError::Structural)?;
    Ok(unnormalized_mass(&result))
}

/// Compare a draw-bounded simulation with the mean of a materialized law.
///
/// `materialized` is the small reference law, one row per value. `draws` is
/// the simulation and must be no longer than `max_draws`. The receipt is the
/// Monte Carlo error of `draws` alone.
///
/// # Errors
/// An empty or non-finite reference, a simulation past the bound, or a receipt
/// the reliability module refuses (including too few draws).
pub fn compare_bounded_simulation(
    materialized: &[f64],
    draws: &[f64],
    max_draws: usize,
) -> Result<SimulationComparison, RobustDecisionError> {
    if materialized.is_empty() || materialized.iter().any(|value| !value.is_finite()) {
        return Err(RobustDecisionError::EmptyLaw);
    }
    if draws.len() > max_draws {
        return Err(RobustDecisionError::DrawBoundExceeded { draws: draws.len(), max_draws });
    }
    let reference_mean = materialized.iter().sum::<f64>() / materialized.len() as f64;
    let receipt = estimate(
        draws,
        None,
        WithinStreamDependence::Independent,
        BatchPlan::SquareRoot,
        &Functional::ExpectedUtility,
    )
    .map_err(RobustDecisionError::Reliability)?
    .estimate;
    let absolute_gap = (receipt.value - reference_mean).abs();
    let within_two_standard_errors = absolute_gap <= 2.0 * receipt.standard_error + 1e-12;
    Ok(SimulationComparison { reference_mean, receipt, absolute_gap, within_two_standard_errors })
}

/// Score a future study with the conjugate expectation.
///
/// The number is [`FutureStudyProgram::evaluate_future_study`]. It is not a
/// robust conclusion.
///
/// # Errors
/// Any refusal of that score.
pub fn future_study_expectation(
    program: &FutureStudyProgram,
) -> Result<ExpectationNotRobust, RobustDecisionError> {
    let FutureStudyValue { gross_evsi, net_value, .. } =
        program.evaluate_future_study().map_err(RobustDecisionError::FutureStudy)?;
    Ok(ExpectationNotRobust { gross_evsi, net_value, is_robust_conclusion: false })
}

fn conclusions_from(decision: IdentifiedSetDecision) -> SetConclusions {
    let eligible = decision.actions.iter().filter(|action| {
        action.hard_exclusions.is_empty()
            && action.declared_exclusion.is_none()
            && action.support_shortfalls.is_empty()
    });
    let mut best: Option<(String, f64)> = None;
    let mut tied = false;
    for action in eligible {
        let score = (action.utility.0 + action.utility.1) / 2.0;
        match &best {
            None => best = Some((action.id.clone(), score)),
            Some((_, current)) if score > *current => {
                best = Some((action.id.clone(), score));
                tied = false;
            }
            Some((_, current)) if (score - *current).abs() <= 1e-12 => tied = true,
            Some(_) => {}
        }
    }
    let endpoint_average_action = if tied { None } else { best.map(|(id, _)| id) };
    SetConclusions {
        robust: decision.verdict.clone(),
        endpoint_average_action,
        endpoint_average_is_robust: false,
        decision,
    }
}

#[cfg(test)]
mod tests {
    use antecedent_contract::ScientificQuantity;
    use antecedent_core::IdentifiedSet;
    use antecedent_io_core::distribution_artifact::{
        DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
        DistributionProvenance, DistributionTrust, DrawAlignment,
    };
    use antecedent_io_core::quantity_wire::DistributionMeaningWire;

    use super::{
        CoverageRegion, RobustDecisionError, compare_bounded_simulation, evaluate_coverage_regions,
        evaluate_model_mass, evaluate_robust_set, future_study_expectation,
        probability_of_coverage_region, weight_scenarios,
    };
    use crate::decision_adapters::{AdapterError, IdentifiedUtility, IdentifiedVerdict};
    use crate::decision_contract::{
        ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction,
        DecisionContract, DecisionCriterion, StructuralPolicy, UncertaintyRequirement, UtilityExpr,
    };
    use crate::decision_robustness::AtomSupport;
    use crate::decision_structural::{AtomEvidence, StructuralAtom};
    use crate::evsi::{
        AffineStateUtility, FutureStudyProgram, NormalMeanFutureData,
        NormalMeanKnownVarianceUpdate, NormalStateBelief, StudyAction,
    };

    fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: variable.into(),
            role: antecedent_contract::QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn contract(
        policy: StructuralPolicy,
        criterion: DecisionCriterion,
    ) -> AdmissibleDecisionContract {
        let action = |id: &str, variable: &str, regime: &str| DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![quantity(variable, regime)],
            utility: UtilityExpr::Input(0),
        };
        AdmissibleDecisionContract {
            contract: DecisionContract {
                actions: vec![
                    action("stay", "stay", "do(stay)"),
                    action("treat", "treat", "do(treat)"),
                ],
                utility_units: "utility".into(),
                criterion,
                constraints: vec![],
                target_population: "target".into(),
                horizon: 0,
                structural_policy: policy,
            },
            rules: AdmissibilityRules {
                default_weakest_support: None,
                support_rules: vec![],
                declared_exclusions: vec![],
                uncertainty: UncertaintyRequirement::StructuralEnvelope,
            },
        }
    }

    fn interval(action: &str, lower: f64, upper: f64, excluded: bool) -> IdentifiedUtility {
        IdentifiedUtility {
            action_id: action.into(),
            utility: IdentifiedSet { lower, upper },
            hard_exclusions: if excluded { vec!["safety".into()] } else { vec![] },
            support: AtomSupport::supported(),
        }
    }

    fn worst_case() -> AdmissibleDecisionContract {
        contract(StructuralPolicy::ReportOnly, DecisionCriterion::MaximinOverStructures)
    }

    #[test]
    fn worst_case_and_endpoint_average_are_different_conclusions() {
        let utilities = [interval("stay", 0.0, 0.0, false), interval("treat", -2.0, 10.0, false)];
        let conclusions = evaluate_robust_set(&worst_case(), &utilities).unwrap();
        assert_eq!(conclusions.robust, IdentifiedVerdict::WorstCaseChoice("stay".into()));
        assert_eq!(conclusions.endpoint_average_action.as_deref(), Some("treat"));
        assert!(!conclusions.endpoint_average_is_robust);
    }

    #[test]
    fn endpoint_average_is_not_robust_when_it_matches_the_worst_case() {
        let utilities = [interval("stay", 1.0, 3.0, false), interval("treat", 0.0, 2.0, false)];
        let conclusions = evaluate_robust_set(&worst_case(), &utilities).unwrap();
        assert_eq!(conclusions.robust, IdentifiedVerdict::WorstCaseChoice("stay".into()));
        assert_eq!(conclusions.endpoint_average_action.as_deref(), Some("stay"));
        assert!(!conclusions.endpoint_average_is_robust);
    }

    #[test]
    fn hard_exclusion_removes_an_action_without_changing_its_interval() {
        let utilities = [interval("stay", -1.0, 5.0, false), interval("treat", 0.0, 1.0, true)];
        let conclusions = evaluate_robust_set(&worst_case(), &utilities).unwrap();
        assert_eq!(conclusions.robust, IdentifiedVerdict::WorstCaseChoice("stay".into()));
        let treat =
            conclusions.decision.actions.iter().find(|action| action.id == "treat").unwrap();
        assert_eq!(treat.utility, (0.0, 1.0));
        assert_eq!(treat.hard_exclusions, vec!["safety".to_owned()]);
        assert!(!treat.eligible);
        assert_eq!(conclusions.endpoint_average_action.as_deref(), Some("stay"));
    }

    #[test]
    fn identified_set_refuses_bayes_weights() {
        let bayes = contract(
            StructuralPolicy::BayesOverStructures,
            DecisionCriterion::PosteriorExpectedUtility,
        );
        let utilities = [interval("stay", 0.0, 0.0, false), interval("treat", -2.0, 10.0, false)];
        let error = evaluate_robust_set(&bayes, &utilities).unwrap_err();
        assert_eq!(error, RobustDecisionError::Adapter(AdapterError::BayesOverIdentifiedSet));
        assert!(weight_scenarios(&[0.5, 0.5]).is_err());
    }

    #[test]
    fn coverage_level_is_not_a_probability_of_the_region() {
        let regions = [
            CoverageRegion {
                action_id: "stay".into(),
                lower: 0.0,
                upper: 0.0,
                coverage: 0.99,
                hard_exclusions: vec![],
            },
            CoverageRegion {
                action_id: "treat".into(),
                lower: -2.0,
                upper: 10.0,
                coverage: 0.5,
                hard_exclusions: vec![],
            },
        ];
        let conclusions = evaluate_coverage_regions(&worst_case(), &regions).unwrap();
        assert_eq!(conclusions.set.robust, IdentifiedVerdict::WorstCaseChoice("stay".into()));
        assert_eq!(conclusions.coverages, vec![("stay".into(), 0.99), ("treat".into(), 0.5)]);
        // A coverage-weighted midpoint would prefer treat (0.5 * 4 > 0.99 * 0).
        assert_eq!(conclusions.set.endpoint_average_action.as_deref(), Some("treat"));
        assert!(probability_of_coverage_region(0.95).is_err());
    }

    fn law(a: f64, b: f64) -> Box<DistributionArtifact> {
        let columns = [quantity("stay", "do(stay)"), quantity("treat", "do(treat)")];
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::InterventionalPredictive,
            &columns,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: "structure".into(),
                provider_id: "exact-law".into(),
                rng_id: "deterministic_exact".into(),
                snapshot_id: "enumeration".into(),
                causal_contract_id: "checked".into(),
            },
        )
        .unwrap();
        Box::new(
            DistributionArtifact::new(
                DistributionMetadata {
                    version: 1,
                    identity,
                    axes: ["draw".into(), "quantity".into()],
                    shape: [2, 2],
                    weights: None,
                    supported: None,
                    calibration: DistributionCalibration::Exact,
                    trust: DistributionTrust::Unverified,
                    legacy_posterior: None,
                    legacy_bindings: None,
                },
                vec![a, b, a, b],
            )
            .unwrap(),
        )
    }

    #[test]
    fn incomplete_model_mass_is_not_renormalized() {
        let bayes = contract(
            StructuralPolicy::BayesOverStructures,
            DecisionCriterion::PosteriorExpectedUtility,
        );
        let atoms = [
            StructuralAtom {
                id: "s1".into(),
                probability: Some(0.5),
                evidence: AtomEvidence::Evaluated(law(5.0, 3.0)),
            },
            StructuralAtom {
                id: "s2".into(),
                probability: Some(0.3),
                evidence: AtomEvidence::Evaluated(law(1.0, 4.0)),
            },
            StructuralAtom {
                id: "s3".into(),
                probability: Some(0.2),
                evidence: AtomEvidence::Unidentified,
            },
        ];
        let mass = evaluate_model_mass(&bayes.contract, &atoms).unwrap();
        assert!((mass.evaluated_mass - 0.8).abs() < 1e-12);
        assert!((mass.unidentified_mass - 0.2).abs() < 1e-12);
        assert!(mass.unevaluated_mass.abs() < 1e-12);
        assert!((mass.weighted_values[0].1 - 2.8).abs() < 1e-12);
        assert!((mass.weighted_values[1].1 - 2.7).abs() < 1e-12);
        let renormalized = 2.8 / 0.8;
        assert!((mass.weighted_values[0].1 - renormalized).abs() > 0.1);
    }

    #[test]
    fn bounded_simulation_matches_the_materialized_mean_with_an_error_receipt() {
        let materialized = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let draws = [1.0, 3.0, 1.0, 3.0];
        let comparison = compare_bounded_simulation(&materialized, &draws, 4).unwrap();
        assert!((comparison.reference_mean - 2.0).abs() < 1e-12);
        assert!((comparison.receipt.value - 2.0).abs() < 1e-12);
        assert!((comparison.receipt.standard_error - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(comparison.receipt.n_draws, 4);
        assert!(comparison.absolute_gap < 1e-12);
        assert!(comparison.within_two_standard_errors);
    }

    #[test]
    fn a_bounded_sample_off_the_materialized_mean_keeps_its_receipt() {
        let comparison = compare_bounded_simulation(&[1.0, 3.0], &[1.0, 1.0, 1.0, 1.0], 4).unwrap();
        assert!((comparison.reference_mean - 2.0).abs() < 1e-12);
        assert!((comparison.receipt.value - 1.0).abs() < 1e-12);
        assert!(comparison.receipt.standard_error.abs() < 1e-12);
        assert!(!comparison.within_two_standard_errors);
        let error = compare_bounded_simulation(&[1.0], &[0.0; 5], 4).unwrap_err();
        assert_eq!(error, RobustDecisionError::DrawBoundExceeded { draws: 5, max_draws: 4 });
    }

    #[test]
    fn conjugate_future_study_score_is_the_expectation_not_a_robust_conclusion() {
        let program = FutureStudyProgram::try_new(
            NormalStateBelief::try_new(0.5, 1.0).unwrap(),
            NormalMeanFutureData::try_new(4, 12.0).unwrap(),
            NormalMeanKnownVarianceUpdate,
            vec![StudyAction::try_new("stay").unwrap(), StudyAction::try_new("treat").unwrap()],
            AffineStateUtility::try_new(vec![0.0, 0.0], vec![0.0, 1.0]).unwrap(),
            None,
        )
        .unwrap();
        let score = future_study_expectation(&program).unwrap();
        assert!((score.gross_evsi - 0.041_657_735_293_843_15).abs() < 1e-12);
        assert!(score.net_value.is_none());
        assert!(!score.is_robust_conclusion);
    }
}
