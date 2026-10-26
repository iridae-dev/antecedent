//! Lowering of the associational cells: a noncausal association of an outcome with predictors
//! over the rows of a table or the points of one series.
//!
//! Layer one is the association functional keyed by the query's canonical digest, so two
//! association queries that differ in any column, functional or recorded refusal are different
//! programs. Layer two claims no identification: the program describes the observed law and
//! states no causal contrast. When the query follows a causal query that was refused, the
//! refusal's reason code is retained on the layer. Layer three carries the analysis the declared
//! selection rule picked from the outcome scale and the functional (a stratified law, a
//! regression, a correlation or a predictive model), the resampling under the declared sampling
//! design (a row bootstrap for rows, circular blocks for a series) or the Bayesian posterior
//! construction (closed-form Normal-Inverse-Gamma marginals for a Gaussian outcome on independent
//! unweighted rows, the Bayesian bootstrap otherwise), and the held-out folds of a predictive
//! model. Layer four is a single observational source; a decision over an association is refused
//! here, because an association ranks no action. Layer five states that the program fits no
//! quoted adjustment design. Layer six demands an associational claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssociationQuery, CausalQuery, LikelihoodModel, OrdinaryBayesianLikelihood, Prior,
    ScoreFrequentist, ScoreModel, VariableId,
};
use antecedent_estimate::{AssociationSelection, Modality, bayesian_method, select_analysis};

use super::LoweringContext;
use super::static_response::{
    CellProgram, RealizationChoice, Resampling, assemble, declared_uncertainty, slot, union,
};
use crate::InferenceMode;
use crate::analysis::execute::RouteInput;
use crate::error::CausalError;
use crate::planner::lowered::{ClaimDemand, FunctionalKind, IdentificationLayer, LoweredProgram};
use crate::planner::statistical::{
    FoldPlan, ModelRole, PosteriorAlgorithm, PosteriorPlan, Realization, ResampleUnit,
    ResamplingMethod, ResamplingPlan,
};

/// Why an association quotes no adjustment-design width.
const ASSOCIATION_WIDTH: &str = "an association fits a stratified law, a correlation or a model \
                                 sized by its selected analysis; none is a quoted adjustment \
                                 design";

fn declaration_error(error: antecedent_core::DeclarationError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let (selection, query) = match cx.input {
        Some(RouteInput::AssociationStatic { data, query }) => {
            (select_analysis(data.storage(), query, Modality::CrossSection), query)
        }
        Some(RouteInput::AssociationTemporal { data, query }) => {
            (select_analysis(data.storage(), query, Modality::Series), query)
        }
        _ => return Err(super::not_lowered_by("associational", cx)),
    };
    let CausalQuery::Association(study_query) = &cx.study.query else {
        return Err(super::not_lowered_by("associational", cx));
    };
    if study_query != query {
        return Err(super::not_lowered_by("associational", cx));
    }
    if cx.goal.decision().is_some() || cx.goal.planning().is_some() {
        return Err(crate::compile_reason!(
            "quantity_semantics_mismatch",
            "an association describes the observed law and ranks no action, so a decision or an \
             experiment plan over it is refused. Next attempt: state the intervention as a causal \
             query, or compile the association without a decision"
        ));
    }
    let selection = selection.map_err(CausalError::from)?;
    lower_selected(cx, query, &selection)
}

/// The resampling a frequentist association draws under the declared design.
fn resampling_plan(cx: &LoweringContext<'_>, selection: &AssociationSelection) -> Resampling {
    let replicates = cx.study.bootstrap_replicates;
    if replicates == 0 {
        return Resampling::Off;
    }
    let (method, unit) = match selection.modality {
        Modality::CrossSection => (ResamplingMethod::Bootstrap, ResampleUnit::Row),
        Modality::Series => (ResamplingMethod::BlockBootstrap, ResampleUnit::CalendarBlock),
    };
    Resampling::Plan(ResamplingPlan { method, replicates, unit })
}

/// The statistical realization: a score-based frequentist reading, or the declared posterior
/// construction.
fn realization(
    cx: &LoweringContext<'_>,
    query: &AssociationQuery,
    selection: &AssociationSelection,
) -> Result<RealizationChoice, CausalError> {
    let analysis = selection.analysis.as_str();
    match &cx.study.inference {
        InferenceMode::Frequentist => Ok(RealizationChoice::Override {
            realization: Realization::Frequentist(ScoreFrequentist::new(
                ScoreModel::new(format!("association.{analysis}.{}", selection.modality.as_str()))
                    .map_err(declaration_error)?,
            )),
            posterior: None,
        }),
        InferenceMode::Bayesian(config) => {
            let method = bayesian_method(selection, query.weight.is_some());
            let prior =
                Prior::new(format!("association.prior.{method}")).map_err(declaration_error)?;
            let likelihood = LikelihoodModel::new(format!("association.{analysis}"))
                .map_err(declaration_error)?;
            let (algorithm, draws) = if method == "conjugate_normal_inverse_gamma" {
                (PosteriorAlgorithm::Conjugate, 0)
            } else {
                (PosteriorAlgorithm::BayesianBootstrap, config.n_draws)
            };
            Ok(RealizationChoice::Override {
                realization: Realization::Bayesian(OrdinaryBayesianLikelihood::new(
                    likelihood,
                    prior.clone(),
                )),
                posterior: Some(PosteriorPlan { algorithm, draws, prior }),
            })
        }
    }
}

fn lower_selected(
    cx: &LoweringContext<'_>,
    query: &AssociationQuery,
    selection: &AssociationSelection,
) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let inputs: Vec<VariableId> = union(&query.predictors, &query.conditioning);
    let models =
        vec![slot(ModelRole::Outcome, selection.analysis.as_str(), &inputs, Some(query.outcome))];
    let folds = selection.folds.map(|folds| FoldPlan {
        folds: folds as usize,
        unit: match selection.modality {
            Modality::CrossSection => ResampleUnit::Row,
            Modality::Series => ResampleUnit::CalendarBlock,
        },
        embargo: None,
    });
    let resampling = match &study.inference {
        InferenceMode::Frequentist => resampling_plan(cx, selection),
        // The posterior carries the uncertainty; no bootstrap is drawn.
        InferenceMode::Bayesian(_) => Resampling::Off,
    };
    assemble(
        cx,
        CellProgram {
            target: super::query_functional(
                study,
                FunctionalKind::Association,
                Some(query.outcome),
            ),
            identification: IdentificationLayer::NotClaimed {
                refusal: query.refused_after.clone(),
            },
            realization: realization(cx, query, selection)?,
            models,
            folds,
            resampling,
            intervenes: false,
            width: ASSOCIATION_WIDTH,
            claim: ClaimDemand {
                uncertainty: declared_uncertainty(&study.inference),
                kind: Some(ClaimKind::Associational),
            },
        },
    )
}

#[cfg(test)]
#[path = "associational_tests.rs"]
mod tests;

/// The lowered program's model slot of an association: the analysis name it fixed.
pub(crate) fn lowered_analysis(program: &LoweredProgram) -> Option<&str> {
    program.statistical().models().first().map(crate::planner::statistical::ModelSlot::family)
}

/// The refusal reason an association program retained, when it follows a refused causal query.
pub(crate) fn retained_refusal(program: &LoweredProgram) -> Option<Arc<str>> {
    match program.identification_layer() {
        IdentificationLayer::NotClaimed { refusal } => refusal.clone(),
        _ => None,
    }
}
