//! Versioned executable design proofs retaining identifying functionals and source data.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    JointDesignEstimate, JointDesignPosterior, instrumented::*, longitudinal::*, randomized::*,
    rollout::*, switchback::*, synthetic::*,
};

/// Versioned design-native execution recipe. Old panel artifacts retain their own decoder.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum JointDesignRecipe {
    /// Factorial, multi-arm or geographical whole-program randomization.
    Randomized {
        /// Checked complete-program assignment contract.
        design: JointRandomizedDesign,
        /// Original independent units or clusters.
        units: Vec<JointRandomizedUnit>,
    },
    /// Finite-carryover switchback histories with whole-schedule assignment laws.
    Switchback {
        /// Explicit finite-carryover joint-history contract.
        design: JointSwitchbackDesign,
        /// Original independently randomized schedules and outcomes.
        sequences: Vec<crate::switchback::HistoryAssignment>,
    },
    /// Joint synthetic rollout effects with unaffected donor/reference worlds.
    SyntheticControl {
        /// Explicit counterfactual and donor restrictions.
        design: JointSyntheticDesign,
        /// Original complete unit-time rows.
        observations: Vec<JointSyntheticObservation>,
    },
    /// Nonparametric sequential observational g-formula over complete histories.
    Longitudinal {
        /// Scientific history contract and regimes.
        design: LongitudinalJointDesign,
        /// Original independent longitudinal episodes.
        episodes: Vec<LongitudinalEpisode>,
    },
    /// Joint structural-feature IV, whose homogeneous moment target is explicit.
    Instrumented {
        /// Retained structural-feature moments.
        design: JointInstrumentedDesign,
        /// Original joint-feature observations.
        units: Vec<JointInstrumentedUnit>,
    },
    /// Regime- and cohort-specific rollout difference-in-differences.
    DifferenceInDifferences {
        /// No-anticipation, trend and comparison contract.
        design: JointRolloutDesign,
        /// Original whole-unit panels.
        units: Vec<JointRolloutUnit>,
    },
}
/// Statistical procedure consumed by the same retained design recipe.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "inference", rename_all = "snake_case", deny_unknown_fields)]
pub enum JointDesignInference {
    /// Native design point functional and complete-unit uncertainty.
    Frequentist,
    /// Proper Gaussian arm-mean likelihood, distinct from design-based uncertainty.
    RandomizedGaussian {
        /// Proper prior and known sampling variances.
        prior: RandomizedGaussianPrior,
    },
    /// Proper finite-population unit-response likelihood retaining unequal cluster weights.
    RandomizedFinitePopulationGaussian {
        /// Full hierarchical prior and known measurement/heterogeneity variances.
        prior: RandomizedFinitePopulationPrior,
    },
    /// Exact finite assignment inference for the sharp null of no joint-history effects.
    SwitchbackExactRandomization {
        /// Prespecified contrast in retained history order.
        contrast: Vec<f64>,
        /// Maximum whole-schedule combinations to enumerate; partial mass remains explicit.
        max_cases: usize,
    },
    /// Gaussian history means with known within-sequence AR1 residual covariance.
    SwitchbackGaussian {
        /// Proper mean priors and complete known covariance.
        prior: SwitchbackGaussianPrior,
    },
    /// Proper joint triangular raw-dose/outcome likelihood and all coefficient priors.
    InstrumentedGaussian {
        /// Exact feature binding and joint likelihood.
        prior: JointIvGaussianPrior,
    },
    /// Joint saturated cohort/time panel likelihood and proper priors.
    RolloutGaussian {
        /// Explicit unit/time/effect priors and known residual covariance.
        prior: JointRolloutGaussianPrior,
    },
    /// Joint family-wise conformal region under exchangeable counterfactual residuals.
    SyntheticConformal {
        /// Requested simultaneous level.
        joint_level: f64,
        /// Required statistical counterfactual residual restriction.
        residual_exchangeability: String,
    },
    /// Native simplex/AR1 likelihood for distinct treated outcomes conditional on shared donors.
    SyntheticBayesian {
        /// Proper simplex and variance priors plus conditional independence.
        prior: JointSyntheticPrior,
    },
    /// Proper categorical full-history likelihood and fixed-support Dirichlet priors.
    LongitudinalDirichlet {
        /// Fully declared finite-state prior.
        prior: LongitudinalDirichletPrior,
    },
}
/// Executed design law without changing the scientific target when inference mode changes.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum JointDesignLaw {
    /// Frequentist whole-unit uncertainty, retaining its actual method meaning.
    Frequentist(JointDesignEstimate),
    /// Aligned posterior world functionals under the explicit likelihood and prior.
    Bayesian(JointDesignPosterior),
}
impl JointDesignLaw {
    /// Joint world moments for either inference mode.
    #[must_use]
    pub fn estimate(&self) -> &JointDesignEstimate {
        match self {
            Self::Frequentist(e) => e,
            Self::Bayesian(p) => &p.estimate,
        }
    }
}
impl JointDesignRecipe {
    /// Stable method owner and identifying functional, independent of inference mode.
    #[must_use]
    pub fn functional(&self) -> &'static str {
        match self {
            Self::Randomized { .. } => {
                "E[w_i 1(A_i=a) Y_i / p_i(a)] / E[w_i] for each complete program a"
            }
            Self::Switchback { .. } => {
                "sum_{sequence,t eligible} 1(A_t-L:t=h) Y_t / P(A_t-L:t=h) divided by total eligible periods; whole-schedule probabilities"
            }
            Self::SyntheticControl { .. } => {
                "treated rollout mean minus unaffected donor-weighted reference mean, weights fitted on preperiods under persistent counterfactual model"
            }
            Self::Longitudinal { .. } => {
                "sum_l0..lT E[Y | L_0:T=l_0:T, A_0:T=a_0:T] P(L0=l0) product_t P(L_t | L_0:t-1, A_0:t-1=a_0:t-1)"
            }
            Self::Instrumented { .. } => {
                "c' beta where E[Z (Y-F(D)' beta-X'gamma)]=0 and rank(E[Z(F(D),X)']) is sufficient"
            }
            Self::DifferenceInDifferences { .. } => {
                "E[Y_t-Y_b | joint regime r, cohort g] - E[Y_t-Y_b | never/not-yet-treated at max(t,b)]"
            }
        }
    }
    /// Stable method owner, never an estimator alias for an incompatible design.
    #[must_use]
    pub fn method(&self) -> &'static str {
        match self {
            Self::Randomized { .. } => "joint_design.whole_program_ht",
            Self::Longitudinal { .. } => "joint_design.full_history_gformula",
            Self::Switchback { .. } => "joint_design.finite_carryover_switchback",
            Self::SyntheticControl { .. } => "joint_design.synthetic_rollout",
            Self::Instrumented { .. } => "joint_design.structural_feature_iv",
            Self::DifferenceInDifferences { .. } => "joint_design.regime_cohort_did",
        }
    }
    /// Execute the same identified functional with a selected, compatible statistical procedure.
    /// # Errors
    /// Missing design premises, unsupported histories/rank/cohorts or an incompatible posterior model.
    pub fn execute(&self, inference: &JointDesignInference) -> Result<JointDesignLaw, String> {
        match (self,inference){
            (Self::Randomized{design,units},JointDesignInference::Frequentist)=>randomized_joint_means(design,units).map(JointDesignLaw::Frequentist),
            (Self::Randomized{design,units},JointDesignInference::RandomizedGaussian{prior})=>randomized_joint_posterior(design,units,prior).map(JointDesignLaw::Bayesian),
            (Self::Randomized{design,units},JointDesignInference::RandomizedFinitePopulationGaussian{prior})=>randomized_weighted_population_posterior(design,units,prior).map(JointDesignLaw::Bayesian),
            (Self::Switchback{design,sequences},JointDesignInference::Frequentist)=>switchback_joint_means(design,sequences).map(JointDesignLaw::Frequentist),
            (Self::Switchback{design,sequences},JointDesignInference::SwitchbackGaussian{prior})=>switchback_joint_posterior(design,sequences,prior).map(JointDesignLaw::Bayesian),
            (Self::Instrumented{design,units},JointDesignInference::InstrumentedGaussian{prior})=>instrumented_joint_posterior(design,units,prior).map(JointDesignLaw::Bayesian),
            (Self::DifferenceInDifferences{design,units},JointDesignInference::RolloutGaussian{prior})=>rollout_joint_posterior(design,units,prior).map(JointDesignLaw::Bayesian),
            (Self::SyntheticControl{design,observations},JointDesignInference::SyntheticBayesian{prior})=>synthetic_joint_posterior(design,observations,prior).map(JointDesignLaw::Bayesian),
            (Self::Longitudinal{design,episodes},JointDesignInference::Frequentist)=>longitudinal_joint_means(design,episodes).map(JointDesignLaw::Frequentist),
            (Self::Longitudinal{design,episodes},JointDesignInference::LongitudinalDirichlet{prior})=>longitudinal_joint_posterior(design,episodes,prior).map(JointDesignLaw::Bayesian),
            (Self::Instrumented{design,units},JointDesignInference::Frequentist)=>instrumented_joint_contrasts(design,units).map(JointDesignLaw::Frequentist),
            (Self::DifferenceInDifferences{design,units},JointDesignInference::Frequentist)=>rollout_joint_effects(design,units).map(JointDesignLaw::Frequentist),
            _=>Err("requested posterior likelihood is not implemented for this design owner; no sampling-covariance Gaussian substitution is allowed".into()),
        }
    }
}

/// Native typed output: a synthetic point without uncertainty is not a zero-variance law.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum JointNativeDesignLaw {
    /// Native joint moments or a genuine aligned Bayesian mechanism law.
    Moments(JointDesignLaw),
    /// Supported donor-world point functionals with retained weights and fit evidence.
    SyntheticPoint(Vec<JointSyntheticCoordinate>),
    /// Exhaustive or explicitly partial sharp-null history randomization inference.
    SwitchbackExact(JointSwitchbackRandomizationTest),
    /// Joint family-wise conformal region, retaining its actual error-model premise.
    SyntheticConformal(JointSyntheticConformal),
}
impl JointDesignRecipe {
    /// Execute every design-native route without relabelling contrast targets as world means.
    /// # Errors
    /// Invalid scientific premises/support, incompatible inference, cancellation or admission failure.
    pub fn execute_native(
        &self,
        inference: &JointDesignInference,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<JointNativeDesignLaw, String> {
        if ctx.cancellation.is_cancelled() {
            return Err("joint-design execution cancelled".into());
        }
        let result = match (self, inference) {
            (
                Self::Switchback { design, sequences },
                JointDesignInference::SwitchbackExactRandomization { contrast, max_cases },
            ) => switchback_joint_sharp_null_test(design, sequences, contrast, *max_cases, ctx)
                .map(JointNativeDesignLaw::SwitchbackExact),

            (
                Self::SyntheticControl { design, observations },
                JointDesignInference::Frequentist,
            ) => synthetic_joint_point(design, observations)
                .map(JointNativeDesignLaw::SyntheticPoint),
            (
                Self::SyntheticControl { design, observations },
                JointDesignInference::SyntheticConformal { joint_level, residual_exchangeability },
            ) => synthetic_joint_conformal(
                design,
                observations,
                *joint_level,
                residual_exchangeability,
                ctx,
            )
            .map(JointNativeDesignLaw::SyntheticConformal),
            _ => self.execute(inference).map(JointNativeDesignLaw::Moments),
        }?;
        if ctx.cancellation.is_cancelled() {
            return Err("joint-design execution cancelled".into());
        }
        Ok(result)
    }
    /// Exact method-native target meaning, never a hidden conversion of local effects into ATEs.
    #[must_use]
    pub fn target_kind(&self) -> &'static str {
        match self {
            Self::Randomized { .. } => "whole_program_population_mean",
            Self::Switchback { .. } => "eligible_period_joint_history_mean",
            Self::Longitudinal { .. } => "longitudinal_regime_population_mean",
            Self::Instrumented { .. } => "homogeneous_structural_feature_contrast",
            Self::DifferenceInDifferences { .. } => "joint_regime_cohort_time_att",
            Self::SyntheticControl { .. } => "treated_unit_rollout_synthetic_reference_contrast",
        }
    }
}
