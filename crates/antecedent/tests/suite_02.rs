//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "cancellation_expansion_cells.rs"]
mod cancellation_expansion_cells;
#[path = "checked_admg_graph_posterior_effect_route_evidence.rs"]
mod checked_admg_graph_posterior_effect_route_evidence;
#[path = "checked_attribution_route_evidence.rs"]
mod checked_attribution_route_evidence;
#[path = "checked_bayesian_class_conditional_route_evidence.rs"]
mod checked_bayesian_class_conditional_route_evidence;
#[path = "checked_bayesian_class_graph_posterior_effect_route_evidence.rs"]
mod checked_bayesian_class_graph_posterior_effect_route_evidence;
#[path = "checked_bayesian_conditional_multimodifier_lifecycle.rs"]
mod checked_bayesian_conditional_multimodifier_lifecycle;
#[path = "checked_bayesian_conditional_route_evidence.rs"]
mod checked_bayesian_conditional_route_evidence;
#[path = "checked_bayesian_counterfactual_route_evidence.rs"]
mod checked_bayesian_counterfactual_route_evidence;
#[path = "checked_bayesian_dag_graph_posterior_ate_route_evidence.rs"]
mod checked_bayesian_dag_graph_posterior_ate_route_evidence;
#[path = "checked_bayesian_specialist_route_evidence.rs"]
mod checked_bayesian_specialist_route_evidence;
#[path = "checked_bayesian_static_class_effect_route_evidence.rs"]
mod checked_bayesian_static_class_effect_route_evidence;
#[path = "checked_bayesian_static_dag_response_route_evidence.rs"]
mod checked_bayesian_static_dag_response_route_evidence;
#[path = "checked_bayesian_temporal_dag_lifecycle.rs"]
mod checked_bayesian_temporal_dag_lifecycle;
#[path = "checked_bayesian_temporal_response_route_evidence.rs"]
mod checked_bayesian_temporal_response_route_evidence;
#[path = "checked_cell_aipw_response_route_evidence.rs"]
mod checked_cell_aipw_response_route_evidence;
#[path = "checked_class_graph_posterior_effect_route_evidence.rs"]
mod checked_class_graph_posterior_effect_route_evidence;
#[path = "checked_codetermined_cell_aipw_response_route_evidence.rs"]
mod checked_codetermined_cell_aipw_response_route_evidence;
#[path = "checked_conditional_effect_route_evidence.rs"]
mod checked_conditional_effect_route_evidence;
#[path = "checked_counterfactual_route_evidence.rs"]
mod checked_counterfactual_route_evidence;
#[path = "checked_derivative_response_route_evidence.rs"]
mod checked_derivative_response_route_evidence;
#[path = "checked_graph_posterior_conditional_route_evidence.rs"]
mod checked_graph_posterior_conditional_route_evidence;
#[path = "checked_graph_posterior_effect_route_evidence.rs"]
mod checked_graph_posterior_effect_route_evidence;
#[path = "checked_graph_posterior_response_route_evidence.rs"]
mod checked_graph_posterior_response_route_evidence;
#[path = "checked_interference_lifecycle.rs"]
mod checked_interference_lifecycle;
#[path = "checked_propensity_route_evidence.rs"]
mod checked_propensity_route_evidence;
#[path = "checked_static_class_response_route_evidence.rs"]
mod checked_static_class_response_route_evidence;
#[path = "checked_static_dag_response_route_evidence.rs"]
mod checked_static_dag_response_route_evidence;
#[path = "checked_static_estimators_route_evidence.rs"]
mod checked_static_estimators_route_evidence;
#[path = "checked_static_mediation_route_evidence.rs"]
mod checked_static_mediation_route_evidence;
#[path = "checked_temporal_class_mediation_route_evidence.rs"]
mod checked_temporal_class_mediation_route_evidence;
