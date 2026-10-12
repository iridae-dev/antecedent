//! Identifier / estimator strategy tables for plan compilation and static execution.
//! Incremental extraction from the analysis workflow — does not
//! replace [`crate::Study`] / plans / [`crate::StudyResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod dispatch;
mod ids;

pub use dispatch::{
    StaticEstimateInput, StaticEstimateWorkspaces, attach_static_bootstrap,
    estimate_provenance_step, estimate_static_effect, identify_admg, identify_admg_query,
    identify_cpdag, identify_cpdag_response, identify_pag, identify_pag_response,
    identify_provenance_step, identify_static, identify_static_query,
    identify_static_query_with_rd, identify_temporal_cpdag, identify_temporal_cpdag_configured,
    identify_temporal_pag, identify_temporal_pag_configured,
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
