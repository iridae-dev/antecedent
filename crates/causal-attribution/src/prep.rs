//! Shared attribution prepare helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, PopulationSelector,
    ShapleyConfig, VariableId,
};
use causal_data::TabularData;
use causal_graph::DenseNodeId;
use causal_model::CompiledCausalModel;

use crate::error::AttributionError;
use crate::population::{resolve_rows, subset_table};

/// Require mechanism-only attribution components (rejects Inputs / Structure /
/// InputsAndMechanisms / All until joint attribution is implemented).
pub(crate) fn require_mechanism_components(
    components: AttributionComponents,
    message: &'static str,
) -> Result<(), AttributionError> {
    match components {
        AttributionComponents::Mechanisms => Ok(()),
        AttributionComponents::InputsAndMechanisms | AttributionComponents::All => {
            Err(AttributionError::unsupported(
                "InputsAndMechanisms/All not implemented for this path; use Mechanisms only",
            ))
        }
        AttributionComponents::Inputs | AttributionComponents::Structure => {
            Err(AttributionError::unsupported(message))
        }
        _ => Err(AttributionError::unsupported(
            "unsupported AttributionComponents for this attribution path",
        )),
    }
}

/// Require input-only attribution components (rejects Mechanisms / Structure /
/// InputsAndMechanisms / All until joint attribution is implemented).
pub(crate) fn require_input_components(
    components: AttributionComponents,
    message: &'static str,
) -> Result<(), AttributionError> {
    match components {
        AttributionComponents::Inputs => Ok(()),
        AttributionComponents::InputsAndMechanisms | AttributionComponents::All => {
            Err(AttributionError::unsupported(
                "InputsAndMechanisms/All not implemented for this path; use Inputs only",
            ))
        }
        AttributionComponents::Mechanisms | AttributionComponents::Structure => {
            Err(AttributionError::unsupported(message))
        }
        _ => Err(AttributionError::unsupported(
            "unsupported AttributionComponents for this attribution path",
        )),
    }
}

/// Require structure-only attribution components (rejects Inputs / Mechanisms /
/// InputsAndMechanisms / All until joint attribution is implemented).
pub(crate) fn require_structure_components(
    components: AttributionComponents,
    message: &'static str,
) -> Result<(), AttributionError> {
    match components {
        AttributionComponents::Structure => Ok(()),
        AttributionComponents::InputsAndMechanisms | AttributionComponents::All => {
            Err(AttributionError::unsupported(
                "InputsAndMechanisms/All not implemented for this path; use Structure only",
            ))
        }
        AttributionComponents::Inputs | AttributionComponents::Mechanisms => {
            Err(AttributionError::unsupported(message))
        }
        _ => Err(AttributionError::unsupported(
            "unsupported AttributionComponents for this attribution path",
        )),
    }
}

/// Resolve dense outcome id from a compiled model.
pub(crate) fn resolve_outcome_dense(
    model: &CompiledCausalModel,
    outcome: VariableId,
) -> Result<DenseNodeId, AttributionError> {
    model.dense_of(outcome).ok_or_else(|| AttributionError::missing_var("outcome", outcome))
}

/// Resolve and subset baseline/comparison populations; both must be non-empty.
pub(crate) fn resolve_population_pair(
    data: &TabularData,
    baseline: &PopulationSelector,
    comparison: &PopulationSelector,
) -> Result<(TabularData, TabularData), AttributionError> {
    let baseline_rows = resolve_rows(data, baseline)?;
    let comparison_rows = resolve_rows(data, comparison)?;
    if baseline_rows.is_empty() || comparison_rows.is_empty() {
        return Err(AttributionError::invalid_input(
            "baseline and comparison populations must be non-empty",
        ));
    }
    Ok((subset_table(data, &baseline_rows)?, subset_table(data, &comparison_rows)?))
}

/// Same as [`resolve_population_pair`] from a change-attribution query.
pub(crate) fn resolve_change_populations(
    data: &TabularData,
    query: &ChangeAttributionQuery,
) -> Result<(TabularData, TabularData), AttributionError> {
    resolve_population_pair(data, &query.baseline, &query.comparison)
}

/// Require Shapley allocation; return the approximation config.
pub(crate) fn require_shapley_config<'a>(
    allocation: &'a AllocationMethod,
    message: &'static str,
) -> Result<&'a ShapleyConfig, AttributionError> {
    match allocation {
        AllocationMethod::Shapley { approximation } => Ok(approximation),
        _ => Err(AttributionError::unsupported(message)),
    }
}
