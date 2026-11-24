//! What the plan states about a decision across preferences, or about an ordinary decision
//! contract compiled on the same route.
//!
//! The decision program layer four carries names every quantity the actions read by its
//! canonical identity. Beside it the plan states each quantity's coordinates (the schema
//! variable when the source is a study, the regime, units, population, horizon, functional
//! and scale), the representation the route reads and the ones the contract accepts, the
//! trade-off parameters with their declared ranges, when separately estimated outcomes are
//! joined into one law, the propagation chain declared for them at compile and, when a compiled
//! design draws a resampling law of its arm means, that law and its sampling premise. An
//! ordinary contract has no trade-off parameters, so it states none.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalSchema;
use antecedent_design::decision_contract::SourceRepresentation;

use super::{CompiledContract, DemandedQuantity, PreferenceDemand};
use crate::planner::lowered::{
    PreferenceDesignLawFact, PreferenceJointLawFact, PreferenceParameterFact,
    PreferenceQuantityFact, PreferenceValueFacts,
};

/// Stable name of a source representation.
pub(super) const fn representation_name(representation: SourceRepresentation) -> &'static str {
    match representation {
        SourceRepresentation::Mean => "mean",
        SourceRepresentation::MeanAndCovariance => "mean_and_covariance",
        SourceRepresentation::Cdf => "cdf",
        SourceRepresentation::QuantileFunction => "quantile_function",
        SourceRepresentation::MarginalDraws => "marginal_draws",
        SourceRepresentation::JointDraws => "joint_draws",
    }
}

/// The facts of `quantities` (the ones one source targets) under `demand`, resolving each
/// quantity's variable in `schema` when the source is a study, with the joint law that joins
/// separate studies or the resampling law a compiled design draws.
pub(super) fn value_facts(
    contract: &CompiledContract,
    demand: &PreferenceDemand,
    quantities: &[&DemandedQuantity],
    schema: Option<&CausalSchema>,
    joint_law: Option<PreferenceJointLawFact>,
    design_law: Option<PreferenceDesignLawFact>,
) -> PreferenceValueFacts {
    let quantities = quantities
        .iter()
        .map(|demanded| {
            let quantity = &demanded.quantity;
            PreferenceQuantityFact {
                identity: Arc::from(quantity.canonical_identity()),
                variable: schema.and_then(|schema| schema.id_of(&quantity.variable_id).ok()),
                variable_id: Arc::from(quantity.variable_id.as_str()),
                regime: Arc::from(quantity.regime_id.as_str()),
                units: Arc::from(quantity.units.as_str()),
                population: Arc::from(quantity.population_id.as_str()),
                horizon: quantity.horizon,
                functional: Arc::from(quantity.functional_id.as_str()),
                transform: Arc::from(quantity.transform_id.as_str()),
                source: demanded.source,
            }
        })
        .collect::<Vec<_>>();
    let declared = contract.tradeoff().map_or(&[][..], |c| &c.model.preferences.parameters[..]);
    let parameters = declared
        .iter()
        .map(|parameter| PreferenceParameterFact {
            id: Arc::from(parameter.id.as_str()),
            units: Arc::from(parameter.units.as_str()),
            lower: parameter.lower,
            upper: parameter.upper,
            assumption: Arc::from(parameter.assumption_id.as_str()),
        })
        .collect::<Vec<_>>();
    let accepted = demand
        .requirement
        .any_of
        .iter()
        .map(|representation| Arc::from(representation_name(*representation)))
        .collect::<Vec<_>>();
    PreferenceValueFacts {
        quantities: Arc::from(quantities),
        representation: Arc::from(representation_name(demand.representation)),
        accepted: Arc::from(accepted),
        parameters: Arc::from(parameters),
        joint_law,
        design_law,
    }
}
