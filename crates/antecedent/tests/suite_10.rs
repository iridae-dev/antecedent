//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "admg_conditional_effect.rs"]
mod admg_conditional_effect;
#[path = "association_route.rs"]
mod association_route;
#[path = "attribute_mechanism_posterior.rs"]
mod attribute_mechanism_posterior;
#[path = "augmented_panel_did_bayesian.rs"]
mod augmented_panel_did_bayesian;
#[path = "bayesian_arm_effect_routes.rs"]
mod bayesian_arm_effect_routes;
#[path = "bayesian_draw_controls.rs"]
mod bayesian_draw_controls;
#[path = "bayesian_intervention_level_refutation.rs"]
mod bayesian_intervention_level_refutation;
#[path = "bayesian_policy_regime_routes.rs"]
mod bayesian_policy_regime_routes;
#[path = "bayesian_static_mediation_conjugate_truth.rs"]
mod bayesian_static_mediation_conjugate_truth;
#[path = "bayesian_temporal_glm_link_contrast_known_truth.rs"]
mod bayesian_temporal_glm_link_contrast_known_truth;
#[path = "bayesian_temporal_lagged_instrument_known_truth.rs"]
mod bayesian_temporal_lagged_instrument_known_truth;
#[path = "bayesian_temporal_white_innovation_known_truth.rs"]
mod bayesian_temporal_white_innovation_known_truth;
#[path = "build_small_gaps_class_nonfinite.rs"]
mod build_small_gaps_class_nonfinite;
#[path = "cells_00_iv_regime.rs"]
mod cells_00_iv_regime;
#[path = "cells_00_panel_did_serial.rs"]
mod cells_00_panel_did_serial;
#[path = "cells_00_sustained_declarations.rs"]
mod cells_00_sustained_declarations;
#[path = "cells_01_pulse_refuter_boundaries.rs"]
mod cells_01_pulse_refuter_boundaries;
#[path = "cells_03_quantile_missing.rs"]
mod cells_03_quantile_missing;
#[path = "cells_04_attribution_conditional.rs"]
mod cells_04_attribution_conditional;
#[path = "cells_04_joint_intervention.rs"]
mod cells_04_joint_intervention;
#[path = "cells_05_transport_projection.rs"]
mod cells_05_transport_projection;
#[path = "cells_06_adaptive.rs"]
mod cells_06_adaptive;
#[path = "cells_07_scenarios.rs"]
mod cells_07_scenarios;
#[path = "checked_claim_synthesis.rs"]
mod checked_claim_synthesis;
#[path = "claims_03_dbn_posterior_truth.rs"]
mod claims_03_dbn_posterior_truth;
#[path = "class_and_design_validation_suites.rs"]
mod class_and_design_validation_suites;
#[path = "class_derivative_response.rs"]
mod class_derivative_response;
#[path = "class_route_artifacts.rs"]
mod class_route_artifacts;
#[path = "compile_from_decision.rs"]
mod compile_from_decision;
#[path = "conditional_ipcw.rs"]
mod conditional_ipcw;
