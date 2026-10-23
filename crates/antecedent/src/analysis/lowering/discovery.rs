//! Lowering of the structure-discovery cells: a static search over tabular data
//! and a temporal search over series, event, panel or multi-environment data.
//!
//! The scientific problem is the structure-search functional keyed by the
//! query's canonical digest (variables, method, bounds and the digest of the
//! executable configuration), so two searches that differ in any of them are
//! different programs. Identification retains the search's premises (the
//! observational law it reads and the assumptions it is correct under). The
//! statistical program declares the inference mode the method declares: a
//! frequentist search is an exact evaluation of its decision rule with no
//! sampling law, a structure posterior is a posterior under the declared prior
//! and score, computed by enumeration or by MCMC with the schedule's draw count.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ClaimKind;
use antecedent_core::{
    CausalQuery, DiscoveryInference, DiscoveryMethod, LikelihoodModel, OrdinaryBayesianLikelihood,
    Prior, ScoreFrequentist, ScoreModel,
};

use super::LoweringContext;
use super::static_response::{
    CellProgram, RealizationChoice, Resampling, assemble, retain_or_defer, slot,
};
use crate::analysis::execute::RouteInput;
use crate::error::CausalError;
use crate::planner::lowered::{ClaimDemand, FunctionalKind, LoweredProgram, UncertaintyReading};
use crate::planner::statistical::{ModelRole, PosteriorAlgorithm, PosteriorPlan, Realization};

/// Why a structure search quotes no adjustment-design width.
const SEARCH_WIDTH: &str = "a structure search reads the observational law of its variables \
                            through its own engine; it fits no adjustment design";

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    match cx.input {
        Some(
            RouteInput::DiscoverStatic { query, .. } | RouteInput::DiscoverTemporal { query, .. },
        ) => {
            let CausalQuery::Discover(study_query) = &cx.study.query else {
                return Err(super::not_lowered_by("discovery", cx));
            };
            if study_query != query {
                return Err(super::not_lowered_by("discovery", cx));
            }
            lower_search(cx)
        }
        _ => Err(super::not_lowered_by("discovery", cx)),
    }
}

fn declaration_error(error: antecedent_core::DeclarationError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}

fn lower_search(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let CausalQuery::Discover(query) = &study.query else {
        return Err(super::not_lowered_by("discovery", cx));
    };
    let id = query.method.id();
    let identification = retain_or_defer(cx)?;
    let (realization, posterior, uncertainty) = match query.inference() {
        DiscoveryInference::Frequentist => (
            Realization::Frequentist(ScoreFrequentist::new(
                ScoreModel::new(format!("discovery.{id}")).map_err(declaration_error)?,
            )),
            None,
            UncertaintyReading::None,
        ),
        DiscoveryInference::Bayesian => {
            let prior = Prior::new(format!("discovery.structural_prior.{id}"))
                .map_err(declaration_error)?;
            let likelihood = LikelihoodModel::new("discovery.gaussian_bic_marginal")
                .map_err(declaration_error)?;
            let (algorithm, draws) = match (&query.method, query.bounds.schedule) {
                (DiscoveryMethod::ExactDagPosterior | DiscoveryMethod::ExactDagMarginals, _) => {
                    (PosteriorAlgorithm::Enumeration, 0)
                }
                (_, Some(schedule)) => (
                    PosteriorAlgorithm::DiscreteMcmc,
                    (schedule.chains as usize)
                        .saturating_mul(schedule.draws as usize)
                        .checked_div(schedule.thin.max(1) as usize)
                        .unwrap_or(0),
                ),
                (_, None) => (PosteriorAlgorithm::Enumeration, 0),
            };
            (
                Realization::Bayesian(OrdinaryBayesianLikelihood::new(likelihood, prior.clone())),
                Some(PosteriorPlan { algorithm, draws, prior }),
                UncertaintyReading::Posterior,
            )
        }
    };
    let family = format!("discovery.{id}");
    let models = vec![slot(ModelRole::Nuisance, &family, &query.variables, None)];
    assemble(
        cx,
        CellProgram {
            target: super::query_functional(study, FunctionalKind::StructureSearch, None),
            identification,
            realization: RealizationChoice::Override { realization, posterior },
            models,
            folds: None,
            resampling: Resampling::Off,
            intervenes: false,
            width: SEARCH_WIDTH,
            claim: ClaimDemand { uncertainty, kind: Some(ClaimKind::Point) },
        },
    )
}

/// The candidate label the structural lowering names for a fitted-model study.
pub(super) fn mechanism_label(study: &crate::Study) -> String {
    study.mechanisms.as_deref().map_or_else(
        || crate::analysis::structure_route::MechanismProgram::standard().label(),
        crate::analysis::structure_route::MechanismProgram::label,
    )
}
