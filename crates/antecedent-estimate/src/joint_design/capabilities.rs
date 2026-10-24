//! Machine-readable implementation inventory, including unfinished promised obligations.
// SPDX-License-Identifier: MIT OR Apache-2.0
/// One required X07 design capability. Missing implementations remain visible and
/// must never be converted into a scientific-boundary refusal in acceptance matrices.
#[derive(Clone, Debug, serde::Serialize)]
pub struct JointDesignCapability {
    /// Stable X07 design family.
    pub design: &'static str,
    /// Actual frequentist method owner, when implemented.
    pub frequentist_owner: Option<&'static str>,
    /// Actual Bayesian likelihood owner, when implemented.
    pub bayesian_owner: Option<&'static str>,
    /// Exact target retained by the native method.
    pub target: &'static str,
    /// Whether the public prepared-world compiler and source/artifact ports are integrated.
    pub compiled_world_port: bool,
    /// Whether method-native typed target execution, Python and versioned replay are integrated.
    pub compiled_native_port: bool,
    /// Explicit implementation obligations still outstanding; not scientific boundaries.
    pub implementation_gaps: &'static [&'static str],
}
/// Required catalogue, including every promised design rather than only passing rows.
pub const JOINT_DESIGN_CAPABILITIES: [JointDesignCapability; 7] = [
    JointDesignCapability {
        design: "factorial_multi_arm",
        frequentist_owner: Some("joint_design::randomized::randomized_joint_means"),
        bayesian_owner: Some("joint_design::randomized::randomized_joint_posterior"),
        target: "complete-program means and prespecified joint contrasts",
        compiled_world_port: true,
        compiled_native_port: true,
        implementation_gaps: &["complete release calibration remains candidate-bound"],
    },
    JointDesignCapability {
        design: "switchback",
        frequentist_owner: Some("joint_design::switchback::switchback_joint_means"),
        bayesian_owner: Some("joint_design::switchback::switchback_joint_posterior"),
        target: "finite-carryover joint-history means over eligible periods",
        compiled_world_port: false,
        compiled_native_port: true,
        implementation_gaps: &[
            "scientific world-source adapter for history-averaged targets",
            "independent all-stage chain reference matrix",
        ],
    },
    JointDesignCapability {
        design: "geo_experiment",
        frequentist_owner: Some("joint_design::randomized::randomized_joint_means"),
        bayesian_owner: Some("joint_design::randomized::randomized_weighted_population_posterior"),
        target: "fixed population-weighted cluster joint-program means with latent unit-response hierarchical posterior",
        compiled_world_port: true,
        compiled_native_port: true,
        implementation_gaps: &["complete release calibration remains candidate-bound"],
    },
    JointDesignCapability {
        design: "sequential_observational",
        frequentist_owner: Some("joint_design::longitudinal::longitudinal_joint_means"),
        bayesian_owner: Some("joint_design::longitudinal::longitudinal_joint_posterior"),
        target: "full-history longitudinal g-formula means with treatment-affected covariates",
        compiled_world_port: true,
        compiled_native_port: true,
        implementation_gaps: &[
            "continuous conditional-mechanism generalization",
            "complete release calibration remains candidate-bound",
        ],
    },
    JointDesignCapability {
        design: "instrumented_dose",
        frequentist_owner: Some("joint_design::instrumented::instrumented_joint_contrasts"),
        bayesian_owner: Some("joint_design::instrumented::instrumented_joint_posterior"),
        target: "joint structural-feature moment contrasts, distinct from local/complier effects",
        compiled_world_port: false,
        compiled_native_port: true,
        implementation_gaps: &[
            "scientific world-source and experiment consumer adapters for native contrast targets",
            "weak-identification robust uncertainty",
            "independent all-stage chain reference matrix",
        ],
    },
    JointDesignCapability {
        design: "difference_in_differences",
        frequentist_owner: Some("joint_design::rollout::rollout_joint_effects"),
        bayesian_owner: Some("joint_design::rollout::rollout_joint_posterior"),
        target: "joint-regime cohort-time ATT against supported untreated cohorts",
        compiled_world_port: false,
        compiled_native_port: true,
        implementation_gaps: &[
            "scientific world-source and experiment consumer adapters for native contrast targets",
            "independent all-stage chain reference matrix",
        ],
    },
    JointDesignCapability {
        design: "synthetic_control",
        frequentist_owner: Some("joint_design::synthetic::synthetic_joint_conformal"),
        bayesian_owner: Some("joint_design::synthetic::synthetic_joint_posterior"),
        target: "joint rollout contrasts against explicitly unaffected donor/reference worlds",
        compiled_world_port: false,
        compiled_native_port: true,
        implementation_gaps: &[
            "scientific world-source and experiment consumer adapters for native contrast targets",
            "independent all-stage chain reference matrix",
        ],
    },
];
