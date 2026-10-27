//! Identifier / estimator strategy tables for plan compilation and static execution.
//! Incremental extraction from the analysis workflow — does not
//! replace [`crate::Study`] / plans / [`crate::StudyResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod dispatch;
mod ids;
#[cfg(test)]
mod workspace_tests;

pub use dispatch::{
    StaticEstimateInput, StaticEstimateWorkspaces, attach_static_bootstrap, identify_admg_query,
    identify_cpdag, identify_cpdag_in_context, identify_cpdag_response, identify_pag_response,
    identify_static_query, identify_static_query_in_context, identify_static_query_with_rd,
    identify_temporal_cpdag, identify_temporal_cpdag_configured, identify_temporal_pag,
    identify_temporal_pag_configured,
};
// The raw static estimate and identification entries run without a lowered program or a
// contract; only the compiled paths inside the crate reach them.
pub(crate) use dispatch::{
    check_class_search_stop, class_identifier_in_context, estimate_provenance_step,
    estimate_static_effect, estimate_static_effect_streamed, identify_admg,
    identify_admg_derivative, identify_cpdag_distribution_in_context,
    identify_cpdag_path_specific_in_context, identify_cpdag_response_in_context, identify_pag,
    identify_pag_distribution_in_context, identify_pag_in_context,
    identify_pag_path_specific_in_context, identify_pag_response_in_context,
    identify_provenance_step, identify_static, not_identified_admg, streams_static_effect,
};
pub use ids::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_ESTIMATOR_ID, DEFAULT_RESPONSE_IDENTIFIER_ID,
    EstimatorId, IdentifierId, UnknownStrategy, estimand_compatible_with_estimator,
    identification_status_acceptable, require_identified, select_claim, select_estimand,
    validate_class_response_pair, validate_distribution_pair, validate_path_specific_pair,
    validate_response_pair, validate_static_pair,
};
use ids::{estimator_data, identifier_data};
