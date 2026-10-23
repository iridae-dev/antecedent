//! Lowering of the design-based and specialist cells: transport, interference, randomized
//! effects, policy value, dose response, difference-in-differences, synthetic control,
//! local-polynomial ratio, survival, longitudinal regimes, sequential contrasts and
//! history matching.
//!
//! A design identifies its target, not a graph. Layer two therefore retains the design's
//! own identification product: the assumption set of the assignment mechanism (known
//! randomization, parallel trends, overlap, carryover, the interference exposure model,
//! regime support), built once by the operation that executes the design and bound to the
//! scientific target when the target is world-indexed. The prepare-time caches compile
//! retains beside it are exactly the ones preparation would install, so preparation does
//! not identify again.
//!
//! Layer one is a world-indexed target where the design intervenes on a schema variable
//! (the sequential regimes, the local ratio at its cutoff, the dose grid, the transported
//! contrast) and a typed functional otherwise. Layer three carries the realization the
//! study declares and, for a frequentist study, whether its sampling law is design-based
//! (over the assignment of fixed units) or superpopulation (over sampled units); the model
//! slots, folds and randomization test the design fits. Layer five quotes the one design
//! matrix a cell fits (an ANCOVA randomized effect) and states for every other cell what
//! it fits instead.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
#[path = "design_based_tests.rs"]
mod tests;

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus, Conditioning,
    GraphFamily, IdentificationStatus, InferenceSelection, Intervention, InterventionContrast,
    InterventionWorld, ModelAssumptions, ObservationUnit, OrderedFloatBits, OutcomeQuantity,
    QuantityUnits, ScientificTarget, ScientificWorldId, ScoreFrequentist, ScoreModel, TargetForm,
    TargetPopulation, TimeFrame, Value, VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_data::TableView;
use antecedent_data::TabularData;
use antecedent_identify::compiled_product::IdentificationObligation;
use antecedent_identify::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome, IdentificationResult,
};

use super::{LoweringContext, RD_IDENTIFIES_PER_EXECUTION, RouteCell};
use crate::accepted::GraphClass;
use crate::analysis::IdentificationCache;
use crate::analysis::execute::{
    CheckedContinuousDoseOperation, CheckedHistoryMatchOperation, CheckedInterferenceOperation,
    CheckedLongitudinalRegimeOperation, CheckedPolicyValueOperation,
    CheckedSequentialContrastOperation, RouteInput, local_polynomial_ratio_identification,
    panel_did_graphless_key, panel_did_identification, randomized_identification,
    survival_identification, synthetic_control_identification,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, DesignShape, FunctionalKind, IdentificationLayer, LoweredParts, LoweredProgram,
    LoweringDepth, PhysicalLayer, ProblemTarget, RetainedIdentification, ScientificProblem,
    UncertaintyReading, WidthSource,
};
use crate::planner::statistical::{
    DeclaredProgram, FoldPlan, ModelRole, ModelSlot, Realization, ResampleUnit, ResamplingMethod,
    ResamplingPlan, StatisticalProgram,
};
use crate::{EstimatorId, IdentifierId, InferenceMode, Study};

/// Every cell this family lowers completely.
pub(super) const LOWERED: [RouteCell; 12] = [
    RouteCell::TransportMixed,
    RouteCell::InterferenceDag,
    RouteCell::RandomizedEffect,
    RouteCell::PolicyValue,
    RouteCell::PanelDid,
    RouteCell::SyntheticControl,
    RouteCell::ContinuousDoseResponse,
    RouteCell::LocalPolynomialRatio,
    RouteCell::Survival,
    RouteCell::LongitudinalRegime,
    RouteCell::SequentialContrast,
    RouteCell::HistoryMatch,
];

/// Replicates of a randomization test that enumerates every admissible reassignment
/// rather than sampling a fixed number of them.
const EXHAUSTIVE_REASSIGNMENT: u32 = 0;
/// World ids of a two-world contrast.
const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;
/// Declaration name used when the route record names no estimator.
const UNNAMED_ESTIMATOR: &str = "unnamed";
/// Why an ANCOVA design with a categorical covariate is not quoted.
const UNCODED_ANCOVA_COVARIATE: &str = "an ANCOVA covariate is categorical without a declared \
                                        contrast coding; the fit sizes the design at execution";
/// Why a design cell without a quotable design is not quoted.
const UNQUOTED_DESIGN: &str = "the design is sized at execution";

/// Which population a frequentist sampling law speaks for.
#[derive(Clone, Copy)]
enum SamplingBasis {
    /// Randomization law over the assignment of the fixed units in the study.
    DesignBased,
    /// Sampling law over units drawn from a population.
    Superpopulation,
}

impl SamplingBasis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::DesignBased => "design_based",
            Self::Superpopulation => "superpopulation",
        }
    }
}

/// What one design cell contributes beyond the declared program.
struct DesignCell {
    target: ProblemTarget,
    identification: IdentificationResult,
    models: Vec<ModelSlot>,
    folds: Option<FoldPlan>,
    /// A randomization test the design runs in place of the declared resampling.
    randomization_test: Option<ResamplingPlan>,
    basis: SamplingBasis,
    width: WidthSource,
    claim: ClaimKind,
}

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let Some(input) = cx.input.filter(|_| LOWERED.contains(&cx.cell)) else {
        return Err(not_lowered(cx));
    };
    let study = cx.study;
    let plan = cx.plan;
    let record = &plan.logical.record;
    let identifier =
        record.identifier.as_deref().and_then(|name| name.parse::<IdentifierId>().ok());
    let estimator = record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok());
    let caches = study.identify_caches(plan, cx.ctx)?;
    let cell = design_cell(cx, input, &caches)?;
    let identification =
        if matches!(estimator, Some(EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear)) {
            IdentificationLayer::PerExecution { reason: RD_IDENTIFIES_PER_EXECUTION }
        } else {
            IdentificationLayer::Retained(Arc::new(RetainedIdentification {
                product: Some(retain_product(&cell.identification, &cell.target)?),
                caches: Arc::from(caches),
                key: super::retention_key(study),
            }))
        };
    let design = design_shape(cx, input, cell.width);
    let statistical = statistical_program(study, identifier, estimator, design, &cell)?;
    let scientific = ScientificProblem {
        target: cell.target,
        plan_id: Arc::clone(&record.plan_id),
        data: record.data_classification,
        graph_class: study.graph.class().as_str(),
        query_variables: Arc::clone(&record.query_variables),
        source_digest: None,
        support_coordinate: crate::analysis::contract::matrix_coordinate(study).map(Arc::from),
        licensed: matches!(study.support_status, Some(crate::support::CellStatus::Licensed)),
    };
    let claim = ClaimDemand { uncertainty: uncertainty(&study.inference), kind: Some(cell.claim) };
    let evidence = super::decision::evidence_program(cx, None, &statistical)?;
    Ok(LoweredParts {
        cell: cx.cell,
        depth: LoweringDepth::Complete,
        scientific,
        identification,
        statistical,
        evidence,
        physical: PhysicalLayer { operators: Arc::from(Vec::new()), design },
        claim,
    }
    .finish())
}

/// Layer five. An ANCOVA randomized effect quotes its design `[1 | assignment |
/// covariates]` from the data; the assignment is the query's vector, not a source
/// column, so the design is never gathered directly. Every other cell states what it
/// fits instead, sized by the planner.
fn design_shape(
    cx: &LoweringContext<'_>,
    input: RouteInput<'_>,
    width: WidthSource,
) -> DesignShape {
    let planned = |source: WidthSource| DesignShape {
        rows: cx.plan.logical.row_count_hint,
        columns: cx.plan.logical.design_columns_hint,
        source,
        direct: false,
    };
    match (input, width) {
        (RouteInput::RandomizedEffect { query, .. }, WidthSource::Quoted) => {
            super::quote_adjustment_width(cx, &query.ancova_covariates, &[]).map_or_else(
                |_| planned(WidthSource::NotDesignBased(UNCODED_ANCOVA_COVARIATE)),
                |quoted| DesignShape { direct: false, ..quoted },
            )
        }
        // Only the ANCOVA design is quoted; a cell that names `Quoted` without one is
        // stated as unquoted rather than carrying the planner's sizing as a quote.
        (_, WidthSource::Quoted) => planned(WidthSource::NotDesignBased(UNQUOTED_DESIGN)),
        (_, width) => planned(width),
    }
}

fn not_lowered(cx: &LoweringContext<'_>) -> CausalError {
    crate::compile_reason!(
        "route_not_lowered",
        "the design-based lowering does not lower {} without its parsed design route",
        cx.cell.as_str()
    )
}

/// The target, design identification and statistical facts of one parsed design route.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per design cell so each cell's layers read together"
)]
fn design_cell(
    cx: &LoweringContext<'_>,
    input: RouteInput<'_>,
    caches: &[IdentificationCache],
) -> Result<DesignCell, CausalError> {
    let study = cx.study;
    let inference = inference_selection(&study.inference);
    Ok(match input {
        RouteInput::TransportMixed { data, query, class } => {
            let identified = caches
                .iter()
                .find_map(|cache| match cache {
                    IdentificationCache::Transport(identified) => Some(Arc::clone(identified)),
                    _ => None,
                })
                .ok_or_else(|| CausalError::Compile {
                    message: "transport compile retained no transport identification".into(),
                })?;
            let (identification, _) = Study::transport_design_identification(query, &identified)?;
            let (treatment, outcome) =
                query.response.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                    message: "transport response names no treatment/outcome pair".into(),
                })?;
            let mut assumptions = identification.required_assumptions.clone();
            assumptions.push(declared(
                "transport.populations",
                format!(
                    "the response is transported from population {} to population {}",
                    query.source_population, query.target_population
                ),
            ));
            let family = if class == GraphClass::TemporalAdmg {
                GraphFamily::TemporalAdmg
            } else {
                GraphFamily::Admg
            };
            let binding = binding(data, query.response.target_population.clone(), "row", outcome)?;
            let target = binary_contrast(
                &binding,
                vec![set(treatment, 1.0)],
                vec![set(treatment, 0.0)],
                ModelAssumptions::new(family, assumptions),
                inference,
            )?;
            let mut models =
                vec![slot(ModelRole::Outcome, "trial_to_target_ipw", &[treatment], Some(outcome))];
            if let Some(trial) = study.transport_trial.as_ref() {
                models.push(slot(
                    ModelRole::Nuisance,
                    "supplied_selection_probability",
                    &[trial.trial],
                    Some(trial.selection_probability),
                ));
                models.push(slot(
                    ModelRole::Propensity,
                    "supplied_treatment_probability",
                    &[treatment],
                    Some(trial.treatment_probability),
                ));
            }
            DesignCell {
                target: ProblemTarget::Target(target),
                identification,
                models,
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "trial-to-target weighting of outcome means fits no design matrix",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::InterferenceDag { query, .. } => {
            let partial = matches!(
                query.assignment,
                antecedent_core::AssignmentDesign::ClusterRandomization { .. }
                    | antecedent_core::AssignmentDesign::TwoStageSaturation { .. }
                    | antecedent_core::AssignmentDesign::ObservedExposure { .. }
            );
            // Dispatch runs the checked operation for these assignments and the
            // one-shot exposure route for the others; record the identification it runs.
            let (identification, _) = if partial {
                CheckedInterferenceOperation::design_identification(query, &study.inference)
            } else {
                Study::interference_route_identification(query, study.inference.is_bayesian())
            };
            let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
                query.functional;
            let observed = matches!(
                query.assignment,
                antecedent_core::AssignmentDesign::ObservedExposure { .. }
            );
            let mut models =
                vec![slot(ModelRole::Outcome, "exposure_contrast", &[], Some(outcome))];
            if observed {
                models.push(slot(
                    ModelRole::Propensity,
                    "supplied_exposure_probability",
                    &[],
                    None,
                ));
            }
            DesignCell {
                target: functional(study, FunctionalKind::ExposureContrast, outcome),
                identification,
                models,
                folds: None,
                randomization_test: None,
                basis: if observed {
                    SamplingBasis::Superpopulation
                } else {
                    SamplingBasis::DesignBased
                },
                width: WidthSource::NotDesignBased(
                    "exposure-weighted outcome means over a fixed network fit no design matrix",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::RandomizedEffect { query, .. } => {
            let (identification, _) = randomized_identification(query);
            let (_, _, method, assignment_unit, _) =
                antecedent_core::randomized_graphless_coordinate(
                    &query.design,
                    query.estimand,
                    query.received_treatment.is_some(),
                    query.fixed_cuped.is_some(),
                    !query.ancova_covariates.is_empty(),
                );
            let mut covariates: Vec<VariableId> = query.ancova_covariates.to_vec();
            covariates.extend(query.fixed_cuped.map(|(covariate, _)| covariate));
            let ancova = !query.ancova_covariates.is_empty();
            DesignCell {
                target: functional(study, FunctionalKind::AssignmentContrast, query.outcome),
                identification,
                models: vec![slot(ModelRole::Outcome, method, &covariates, Some(query.outcome))],
                folds: None,
                randomization_test: query.exact_randomization_test.then_some(ResamplingPlan {
                    method: ResamplingMethod::Permutation,
                    replicates: EXHAUSTIVE_REASSIGNMENT,
                    unit: assignment_resample_unit(assignment_unit),
                }),
                basis: if ancova {
                    SamplingBasis::Superpopulation
                } else {
                    SamplingBasis::DesignBased
                },
                // An ANCOVA fit is the design [1 | assignment | covariates], quoted from the
                // data in `design_shape`; the attached program re-plans the record from it.
                width: if ancova {
                    WidthSource::Quoted
                } else {
                    WidthSource::NotDesignBased("a difference of arm means fits no design matrix")
                },
                claim: ClaimKind::Point,
            }
        }
        RouteInput::PolicyValue { query, .. } => {
            let (identification, _) = CheckedPolicyValueOperation::design_identification(query)?;
            let ipw = query.mu0.is_empty();
            let (_, method, _) = antecedent_core::policy_graphless_coordinate(
                query.multi_action.is_some(),
                ipw,
                query.uplift_bin_count > 0,
                query.multi_action.as_ref().is_some_and(|policy| !policy.cate_groups.is_empty()),
                query.regret.is_some(),
                !ipw && query.crossfit_fold_ownership_valid && !query.disjoint_training_subjects,
            );
            let mut models = vec![
                slot(ModelRole::Outcome, method, &[], Some(query.outcome)),
                slot(ModelRole::Propensity, "known_randomized_propensity", &[], None),
            ];
            if !ipw {
                models.push(slot(
                    ModelRole::Nuisance,
                    "frozen_conditional_outcome_predictions",
                    &[],
                    Some(query.outcome),
                ));
            }
            DesignCell {
                target: functional(study, FunctionalKind::PolicyValue, query.outcome),
                identification,
                models,
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "a weighted policy value score fits no design matrix",
                ),
                claim: ClaimKind::PolicyValue,
            }
        }
        RouteInput::PanelDid { query, .. } => {
            let (identification, _) = panel_did_identification(query);
            let key = panel_did_graphless_key(query.design);
            let mut models = vec![slot(ModelRole::Outcome, key.method, &[], Some(query.outcome))];
            if let Some(augmented) = &query.augmented {
                models.push(slot(
                    ModelRole::Propensity,
                    "supplied_treatment_propensity",
                    &[augmented.outcome_pre],
                    Some(augmented.propensity),
                ));
                models.push(slot(
                    ModelRole::Nuisance,
                    "supplied_untreated_change_prediction",
                    &[augmented.outcome_pre],
                    Some(augmented.untreated_change_prediction),
                ));
            }
            DesignCell {
                target: functional(study, FunctionalKind::EffectOnTreated, query.outcome),
                identification,
                models,
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "a difference of cluster outcome changes fits no design matrix",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::SyntheticControl { query, .. } => {
            let (identification, _) = synthetic_control_identification(query);
            let family = match (query.method, query.augmentation_ridge.is_some()) {
                (antecedent_core::SyntheticPanelMethod::Control, false) => "simplex_donor_weights",
                (antecedent_core::SyntheticPanelMethod::Control, true) => {
                    "simplex_donor_weights_ridge_augmented"
                }
                (antecedent_core::SyntheticPanelMethod::DifferenceInDifferences, false) => {
                    "synthetic_did_unit_and_period_weights"
                }
                (antecedent_core::SyntheticPanelMethod::DifferenceInDifferences, true) => {
                    "synthetic_did_unit_and_period_weights_ridge_augmented"
                }
            };
            DesignCell {
                target: functional(study, FunctionalKind::EffectOnTreated, query.outcome),
                identification,
                models: vec![slot(ModelRole::Outcome, family, &[], Some(query.outcome))],
                folds: None,
                randomization_test: query.uniform_unit_randomization.then_some(ResamplingPlan {
                    method: ResamplingMethod::Permutation,
                    replicates: EXHAUSTIVE_REASSIGNMENT,
                    unit: ResampleUnit::Subject,
                }),
                basis: if query.uniform_unit_randomization {
                    SamplingBasis::DesignBased
                } else {
                    SamplingBasis::Superpopulation
                },
                width: WidthSource::NotDesignBased(
                    "simplex donor weights are fit on the pre-period panel, not a design matrix",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::ContinuousDoseResponse { data, query } => {
            let (identification, _) = CheckedContinuousDoseOperation::design_identification(query);
            let target = if query.target_doses.is_empty() {
                functional(study, FunctionalKind::PolicyValue, query.outcome)
            } else {
                let binding = binding(data, TargetPopulation::AllObserved, "row", query.outcome)?;
                let worlds = query
                    .target_doses
                    .iter()
                    .enumerate()
                    .map(|(index, dose)| {
                        world(
                            u32::try_from(index).map_err(|_| CausalError::Compile {
                                message: "the dose grid has more points than a world id holds"
                                    .into(),
                            })?,
                            vec![set(query.dose, *dose)],
                            false,
                            &binding,
                        )
                    })
                    .collect::<Result<Vec<_>, CausalError>>()?;
                ProblemTarget::Target(ScientificTarget::new(
                    TargetForm::Worlds(
                        WorldCollection::new(worlds).map_err(|error| target_error(&error))?,
                    ),
                    ModelAssumptions::new(
                        GraphFamily::Graphless,
                        identification.required_assumptions.clone(),
                    ),
                    inference,
                ))
            };
            let claim = if query.target_doses.is_empty() {
                ClaimKind::PolicyValue
            } else {
                ClaimKind::Response
            };
            DesignCell {
                target,
                identification,
                models: vec![
                    slot(
                        ModelRole::Propensity,
                        "supplied_conditional_dose_density",
                        &[query.dose],
                        Some(query.dose_density),
                    ),
                    slot(
                        ModelRole::Outcome,
                        "epanechnikov_kernel",
                        &[query.dose],
                        Some(query.outcome),
                    ),
                ],
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "a local kernel average per baseline group fits no design matrix",
                ),
                claim,
            }
        }
        RouteInput::LocalPolynomialRatio { data, query } => {
            let (identification, _) = local_polynomial_ratio_identification(query);
            let population = TargetPopulation::LocalAtCutoff {
                running: query.running,
                cutoff: OrderedFloatBits::from_f64(query.cutoff),
            };
            let binding = binding(data, population, "row", query.outcome)?;
            let target = binary_contrast(
                &binding,
                vec![set(query.treatment, 1.0)],
                vec![set(query.treatment, 0.0)],
                ModelAssumptions::new(
                    GraphFamily::Graphless,
                    identification.required_assumptions.clone(),
                ),
                inference,
            )?;
            let family = if query.kink { "local_linear_slope_change" } else { "local_linear_jump" };
            DesignCell {
                target: ProblemTarget::Target(target),
                identification,
                models: vec![
                    slot(ModelRole::Outcome, family, &[query.running], Some(query.outcome)),
                    slot(ModelRole::Instrument, family, &[query.running], Some(query.treatment)),
                ],
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "two one-sided local fits inside the bandwidth, not one adjustment design",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::Survival { query, .. } => {
            let (identification, _) = survival_identification(query);
            let mut inputs = vec![query.treatment];
            inputs.extend(query.delayed_entry);
            let family = if query.known_censoring.is_some() {
                "ipcw_product_limit"
            } else {
                "product_limit"
            };
            let mut models = vec![slot(ModelRole::Outcome, family, &inputs, Some(query.duration))];
            if let Some(known) = &query.known_censoring {
                models.push(slot(
                    ModelRole::Nuisance,
                    "known_censoring_survival",
                    &known.columns,
                    None,
                ));
            }
            DesignCell {
                target: functional(study, FunctionalKind::SurvivalContrast, query.duration),
                identification,
                models,
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "arm product-limit estimators fit no design matrix",
                ),
                claim: ClaimKind::Response,
            }
        }
        RouteInput::LongitudinalRegime { query, .. } => {
            let (identification, _) =
                CheckedLongitudinalRegimeOperation::design_identification(query);
            let mut models = vec![slot(
                ModelRole::Propensity,
                "supplied_sequential_treatment_probabilities",
                &[],
                None,
            )];
            let outcome_family = match query.method {
                antecedent_core::LongitudinalRegimeMethod::Ipw => None,
                antecedent_core::LongitudinalRegimeMethod::GFormula => {
                    Some("supplied_period_rewards")
                }
                antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => {
                    Some("supplied_sequential_q_scores")
                }
                antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                    Some("stabilized_iptw_additive_msm")
                }
            };
            models.extend(
                outcome_family
                    .map(|family| slot(ModelRole::Outcome, family, &[], Some(query.outcome))),
            );
            let folds = query.fold_ids.iter().collect::<BTreeSet<_>>().len();
            DesignCell {
                target: functional(study, FunctionalKind::RegimeValue, query.outcome),
                identification,
                models,
                folds: (folds > 0).then_some(FoldPlan {
                    folds,
                    unit: ResampleUnit::Subject,
                    embargo: None,
                }),
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "sequentially weighted subject values fit no design matrix",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::SequentialContrast { data, query } => {
            let (identification, _) =
                CheckedSequentialContrastOperation::design_identification(query);
            let binding = binding(data, TargetPopulation::AllObserved, "subject", query.outcome)?;
            let regime = |actions: [u8; 2]| {
                vec![
                    set(query.first_treatment, f64::from(actions[0])),
                    set(query.second_treatment, f64::from(actions[1])),
                ]
            };
            let target = binary_contrast(
                &binding,
                regime(query.active),
                regime(query.control),
                ModelAssumptions::new(
                    GraphFamily::Graphless,
                    identification.required_assumptions.clone(),
                ),
                inference,
            )?;
            let mut second_history: Vec<VariableId> = query.baseline.to_vec();
            second_history.push(query.first_treatment);
            second_history.extend(query.intermediate.iter().copied());
            let mut outcome_history = second_history.clone();
            outcome_history.push(query.second_treatment);
            DesignCell {
                target: ProblemTarget::Target(target),
                identification,
                models: vec![
                    slot(
                        ModelRole::Propensity,
                        "crossfit_first_decision_propensity",
                        &query.baseline,
                        Some(query.first_treatment),
                    ),
                    slot(
                        ModelRole::Propensity,
                        "crossfit_second_decision_propensity",
                        &second_history,
                        Some(query.second_treatment),
                    ),
                    slot(
                        ModelRole::Outcome,
                        "crossfit_sequential_outcome_regression",
                        &outcome_history,
                        Some(query.outcome),
                    ),
                ],
                folds: Some(FoldPlan {
                    folds: query.fold_count as usize,
                    unit: ResampleUnit::Subject,
                    embargo: None,
                }),
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "augmented sequential scores average per-subject nuisance fits",
                ),
                claim: ClaimKind::Point,
            }
        }
        RouteInput::HistoryMatch { query, .. } => {
            let (identification, _) = CheckedHistoryMatchOperation::design_identification(query);
            let mut match_inputs = vec![query.treated];
            match_inputs.extend(query.period);
            DesignCell {
                target: functional(study, FunctionalKind::HistoryMatch, query.outcome),
                identification,
                models: vec![
                    slot(ModelRole::Nuisance, "history_mean_score", &query.history, None),
                    slot(
                        ModelRole::Outcome,
                        "within_period_nearest_neighbor",
                        &match_inputs,
                        Some(query.outcome),
                    ),
                ],
                folds: None,
                randomization_test: None,
                basis: SamplingBasis::Superpopulation,
                width: WidthSource::NotDesignBased(
                    "nearest-neighbor matching on a history score fits no design matrix",
                ),
                claim: ClaimKind::Point,
            }
        }
        _ => return Err(not_lowered(cx)),
    })
}

/// The declared program, with the design's models, folds, randomization test and the
/// sampling basis of a frequentist realization.
fn statistical_program(
    study: &Study,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    design: DesignShape,
    cell: &DesignCell,
) -> Result<StatisticalProgram, CausalError> {
    let mut statistical = StatisticalProgram::declared(DeclaredProgram {
        identifier,
        estimator,
        inference: &study.inference,
        bootstrap_replicates: study.bootstrap_replicates,
        sampling_design: study.sampling_design,
        design,
        operators: Vec::new(),
    })?;
    if matches!(statistical.realization, Realization::Frequentist(_)) {
        let name = estimator.map_or(UNNAMED_ESTIMATOR, |estimator| estimator.as_str());
        let score = ScoreModel::new(format!("{name}.{}", cell.basis.as_str()))
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        statistical.realization = Realization::Frequentist(ScoreFrequentist::new(score));
    }
    statistical.models = Arc::from(cell.models.clone());
    statistical.folds = cell.folds;
    if let Some(test) = cell.randomization_test {
        statistical.resampling = Some(test);
    }
    Ok(statistical)
}

/// Retain the design identification, bound to a world-indexed target.
fn retain_product(
    identification: &IdentificationResult,
    target: &ProblemTarget,
) -> Result<CompiledIdentificationProduct, CausalError> {
    let (outcome, class) = identification_claim(identification.status);
    let mut parts = CompiledProductParts::new(outcome, class);
    let retained = match target {
        ProblemTarget::Target(target) => {
            let (worlds, assembled) = match target.form() {
                TargetForm::Contrast(contrast) => (contrast.worlds().worlds(), true),
                TargetForm::Worlds(worlds) => (worlds.worlds(), false),
                _ => {
                    return Err(CausalError::Compile {
                        message: "a design lowers only a world collection or a contrast".into(),
                    });
                }
            };
            parts.world_obligations = worlds
                .iter()
                .map(|world| {
                    IdentificationObligation::for_world(
                        world.id(),
                        outcome,
                        class,
                        Vec::new(),
                        None,
                    )
                })
                .collect();
            if assembled {
                parts.assembled =
                    Some(IdentificationObligation::assembled(outcome, class, Vec::new(), None));
            }
            CompiledIdentificationProduct::retain_for_target(identification, parts, target)
        }
        ProblemTarget::Targets(_) | ProblemTarget::QueryFunctional { .. } => {
            CompiledIdentificationProduct::retain(identification, parts)
        }
    };
    retained.map_err(|refusal| CausalError::Compile { message: refusal.to_string() })
}

/// Identified under the design alone is an identified claim; under a parametric model or a
/// prior it is evaluated inside that model.
const fn identification_claim(
    status: IdentificationStatus,
) -> (IdentificationOutcome, IdentificationClaimClass) {
    match status {
        IdentificationStatus::NonparametricallyIdentified => {
            (IdentificationOutcome::Identified, IdentificationClaimClass::IdentifiedClaim)
        }
        IdentificationStatus::IdentifiedUnderParametricRestrictions
        | IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            (IdentificationOutcome::Identified, IdentificationClaimClass::ModelConditional)
        }
        IdentificationStatus::PartiallyIdentified
        | IdentificationStatus::GraphDependent
        | IdentificationStatus::NotIdentified => {
            (IdentificationOutcome::Unidentified, IdentificationClaimClass::ModelConditional)
        }
    }
}

const fn uncertainty(inference: &InferenceMode) -> UncertaintyReading {
    match inference {
        InferenceMode::Frequentist => UncertaintyReading::Sampling,
        InferenceMode::Bayesian(_) => UncertaintyReading::Posterior,
    }
}

const fn inference_selection(inference: &InferenceMode) -> InferenceSelection {
    match inference {
        InferenceMode::Frequentist => InferenceSelection::Frequentist,
        InferenceMode::Bayesian(_) => InferenceSelection::Bayesian,
    }
}

/// The resample unit of a randomized design's assignment unit.
fn assignment_resample_unit(assignment_unit: &str) -> ResampleUnit {
    match assignment_unit {
        "cluster" => ResampleUnit::Cluster,
        "sequence" => ResampleUnit::Subject,
        _ => ResampleUnit::Row,
    }
}

/// The typed functional `kind` on `outcome`, keyed by the canonical digest of the
/// study's query so two designs of one kind lower to different programs.
fn functional(study: &Study, kind: FunctionalKind, outcome: VariableId) -> ProblemTarget {
    super::query_functional(study, kind, Some(outcome))
}

fn slot(
    role: ModelRole,
    family: &str,
    inputs: &[VariableId],
    output: Option<VariableId>,
) -> ModelSlot {
    ModelSlot { role, family: Arc::from(family), inputs: Arc::from(inputs), output }
}

fn set(variable: VariableId, value: f64) -> Intervention {
    Intervention::Set { variable, value: Value::Float64(value) }
}

/// A design premise recorded beside the design's own identification assumptions.
fn declared(id: &str, description: String) -> AssumptionRecord {
    AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(id),
            description: Arc::from(description),
        },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}

/// The unconditional mean of `outcome` per `unit` over `population`.
fn binding(
    data: &TabularData,
    population: TargetPopulation,
    unit: &str,
    outcome: VariableId,
) -> Result<WorldBinding, CausalError> {
    let spec = data.schema().get(outcome)?;
    let units = QuantityUnits::new(spec.unit.as_deref().unwrap_or("unspecified"))
        .map_err(|error| target_error(&error))?;
    WorldBinding::new(
        population,
        ObservationUnit::new(unit).map_err(|error| target_error(&error))?,
        OutcomeQuantity::mean(outcome, units).map_err(|error| target_error(&error))?,
        Conditioning::unconditional(),
    )
    .map_err(|error| target_error(&error))
}

fn world(
    raw: u32,
    interventions: Vec<Intervention>,
    reference: bool,
    binding: &WorldBinding,
) -> Result<InterventionWorld, CausalError> {
    InterventionWorld::new(
        ScientificWorldId::from_raw(raw),
        WorldAction::at_decision(interventions).map_err(|error| target_error(&error))?,
        reference,
        binding.clone(),
        TimeFrame::static_decision(),
    )
    .map_err(|error| target_error(&error))
}

/// `E[Y(active)] − E[Y(control)]`, with the control world as the reference.
fn binary_contrast(
    binding: &WorldBinding,
    active: Vec<Intervention>,
    control: Vec<Intervention>,
    assumptions: ModelAssumptions,
    inference: InferenceSelection,
) -> Result<ScientificTarget, CausalError> {
    let worlds = WorldCollection::new([
        world(CONTROL_WORLD, control, true, binding)?,
        world(ACTIVE_WORLD, active, false, binding)?,
    ])
    .map_err(|error| target_error(&error))?;
    let contrast = InterventionContrast::difference(
        worlds,
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
    )
    .map_err(|error| target_error(&error))?;
    Ok(ScientificTarget::new(TargetForm::Contrast(contrast), assumptions, inference))
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the error is consumed by map_err at every call site"
)]
fn target_error(error: &antecedent_core::ScientificTargetError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}

/// The design assumptions a lowered program retained, for inspection.
#[cfg(test)]
pub(super) fn retained_assumptions(
    program: &LoweredProgram,
) -> Option<antecedent_core::AssumptionSet> {
    program
        .retained_identification()
        .and_then(|retained| retained.product.as_ref())
        .map(|product| product.assumptions().clone())
}
