//! A sensitivity analysis reported as an assumption range tied to its target.
//!
//! Every sensitivity analysis of this crate beyond the marginal sensitivity model answers
//! the same three questions, and [`AssumptionRange`] carries exactly those answers:
//!
//! 1. **What is assumed.** [`SensitivityAssumption`] names the untestable assumption (no
//!    unmeasured confounding on a risk-ratio scale, on a partial-R² scale, no hidden bias in
//!    matched assignment, no mediator-outcome confounding).
//! 2. **Which range was tested.** [`AssumptionRange::tested`] lists the sensitivity
//!    parameter values probed and the target (or its test) at each.
//! 3. **Where the conclusion breaks.** [`AssumptionRange::break_even`] is the parameter value
//!    at which the target is explained away (or loses significance), when it is reached.
//!
//! The target's identity ([`RefutationTarget`]) is bound into the range's identity, so a range
//! computed for one target cannot be cited by a claim on another
//! ([`AssumptionRange::bound_to`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CanonicalWriter;

use crate::target_native::{
    RefutationTarget, TARGET_NATIVE_NEXT_ATTEMPT, TargetCertificationError,
};

/// The assumption a sensitivity range relaxes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SensitivityAssumption {
    /// No unmeasured confounder; relaxed to a confounder of given risk-ratio strength with
    /// both treatment and outcome (`VanderWeele` and Ding 2017).
    UnmeasuredConfoundingRiskRatio,
    /// No unmeasured confounder; relaxed to a confounder explaining given shares of the
    /// residual variance of treatment and outcome (Cinelli and Hazlett 2020).
    UnmeasuredConfoundingPartialR2,
    /// Random assignment within matched sets; relaxed to odds of treatment differing by at
    /// most `Γ` within a set (Rosenbaum 1987, 2002).
    HiddenBiasInMatchedAssignment,
    /// Sequential ignorability of the mediator; relaxed to a correlation `ρ` between the
    /// mediator and outcome errors (Imai, Keele and Yamamoto 2010).
    MediatorOutcomeConfounding,
    /// No unmeasured confounder of the mediator and the outcome given treatment and the
    /// measured covariates; relaxed to a confounder of given risk-ratio strength with the
    /// mediator and the outcome (E-value of a mediation component, with the treatment taken
    /// as unconfounded).
    MediatorOutcomeConfoundingRiskRatio,
    /// No unmeasured confounder of the mediator and the outcome in the outcome regression on
    /// treatment, mediator and covariates; relaxed to a confounder explaining given shares of
    /// the residual variance of the mediator and the outcome (Cinelli and Hazlett 2020, read
    /// on the mediator-outcome path).
    MediatorOutcomeConfoundingPartialR2,
    /// Sequential exchangeability of a marginal structural model; relaxed to a time-varying
    /// confounder of given risk-ratio strength with the exposure history and the outcome
    /// (E-value of an MSM parameter).
    SequentialExchangeabilityRiskRatio,
    /// Sequential exchangeability of a marginal structural model; relaxed to an omitted
    /// confounder explaining given shares of the residual variance of the exposure and of the
    /// outcome in the weighted outcome regression (Cinelli and Hazlett 2020 on the weighted
    /// fit with a robust standard error).
    SequentialExchangeabilityPartialR2,
    /// No unmeasured confounder of treatment and mediator given the measured covariates;
    /// relaxed to a confounder explaining given shares of the residual variance of the
    /// treatment and of the mediator (Cinelli and Hazlett 2020 on the mediator regression).
    TreatmentMediatorConfoundingPartialR2,
    /// No unmeasured confounder of treatment and outcome given the mediator and the measured
    /// covariates; relaxed to a confounder explaining given shares of the residual variance of
    /// the treatment and of the outcome in the outcome regression (the direct-effect path).
    TreatmentOutcomeConfoundingPartialR2,
    /// No unmeasured confounding of either path of the indirect effect (treatment-mediator and
    /// mediator-outcome); relaxed to confounders of both paths with the tested partial R²,
    /// the indirect effect being the product of the two path coefficients.
    IndirectEffectConfoundingPartialR2,
    /// Independence of the cross-world potential outcomes (and of the unit-level noise the
    /// fitted structural model abducts) from the treatment given the measured covariates;
    /// relaxed to a correlation `ρ` between the treatment and outcome errors.
    CrossWorldNoiseDependence,
}

impl SensitivityAssumption {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnmeasuredConfoundingRiskRatio => "unmeasured_confounding_risk_ratio",
            Self::UnmeasuredConfoundingPartialR2 => "unmeasured_confounding_partial_r2",
            Self::HiddenBiasInMatchedAssignment => "hidden_bias_in_matched_assignment",
            Self::MediatorOutcomeConfounding => "mediator_outcome_confounding",
            Self::MediatorOutcomeConfoundingRiskRatio => "mediator_outcome_confounding_risk_ratio",
            Self::MediatorOutcomeConfoundingPartialR2 => "mediator_outcome_confounding_partial_r2",
            Self::SequentialExchangeabilityRiskRatio => "sequential_exchangeability_risk_ratio",
            Self::SequentialExchangeabilityPartialR2 => "sequential_exchangeability_partial_r2",
            Self::TreatmentMediatorConfoundingPartialR2 => {
                "treatment_mediator_confounding_partial_r2"
            }
            Self::TreatmentOutcomeConfoundingPartialR2 => {
                "treatment_outcome_confounding_partial_r2"
            }
            Self::IndirectEffectConfoundingPartialR2 => "indirect_effect_confounding_partial_r2",
            Self::CrossWorldNoiseDependence => "cross_world_noise_dependence",
        }
    }

    /// Plain statement of the assumption and of how the range relaxes it.
    #[must_use]
    pub const fn statement(self) -> &'static str {
        match self {
            Self::UnmeasuredConfoundingRiskRatio => {
                "no unmeasured confounding; relaxed to a confounder associated with treatment \
                 and outcome by the tested risk ratio (each), conditional on the measured \
                 covariates"
            }
            Self::UnmeasuredConfoundingPartialR2 => {
                "no unmeasured confounding; relaxed to a confounder explaining the tested \
                 shares of residual treatment and outcome variance in the linear adjustment"
            }
            Self::HiddenBiasInMatchedAssignment => {
                "treatment is as if randomized within matched sets; relaxed to units in one set \
                 whose odds of treatment differ by at most the tested factor Gamma"
            }
            Self::MediatorOutcomeConfounding => {
                "no unmeasured mediator-outcome confounding (sequential ignorability); relaxed \
                 to the tested correlation rho between mediator and outcome model errors"
            }
            Self::MediatorOutcomeConfoundingRiskRatio => {
                "no unmeasured mediator-outcome confounding, with treatment unconfounded; \
                 relaxed to a confounder associated with the mediator and with the outcome by \
                 the tested risk ratio (each), conditional on treatment and the measured \
                 covariates"
            }
            Self::MediatorOutcomeConfoundingPartialR2 => {
                "no unmeasured mediator-outcome confounding in the outcome regression on \
                 treatment, mediator and covariates, with the treatment-mediator path \
                 unconfounded; relaxed to a confounder explaining the tested shares of the \
                 residual mediator and outcome variance"
            }
            Self::SequentialExchangeabilityRiskRatio => {
                "sequential exchangeability given the measured time-varying history (no \
                 unmeasured time-varying confounding); relaxed to a confounder associated with \
                 the exposure history and with the outcome by the tested risk ratio (each)"
            }
            Self::SequentialExchangeabilityPartialR2 => {
                "sequential exchangeability given the measured time-varying history, with the \
                 estimated weights held fixed; relaxed to an omitted confounder explaining the \
                 tested shares of the residual exposure and outcome variance of the weighted \
                 regression"
            }
            Self::TreatmentMediatorConfoundingPartialR2 => {
                "no unmeasured treatment-mediator confounding in the mediator regression on \
                 treatment and covariates; relaxed to a confounder explaining the tested shares \
                 of the residual treatment and mediator variance"
            }
            Self::TreatmentOutcomeConfoundingPartialR2 => {
                "no unmeasured treatment-outcome confounding in the outcome regression on \
                 treatment, mediator and covariates (the direct-effect path); relaxed to a \
                 confounder explaining the tested shares of the residual treatment and outcome \
                 variance"
            }
            Self::IndirectEffectConfoundingPartialR2 => {
                "no unmeasured confounding of the treatment-mediator and the mediator-outcome \
                 paths; relaxed to confounders of both paths with the tested partial R², each \
                 path coefficient moving toward zero by its worst-case omitted-variable bias"
            }
            Self::CrossWorldNoiseDependence => {
                "the treatment is independent of the outcome's noise given the measured \
                 covariates, so the cross-world and abducted-noise functionals read the same \
                 mechanism in both worlds; relaxed to the tested correlation rho between the \
                 treatment and outcome errors, the linear-Gaussian slope moving by \
                 rho times the ratio of the outcome and treatment residual deviations"
            }
        }
    }
}

/// One probed value of the sensitivity parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RangePoint {
    /// Sensitivity-parameter value.
    pub parameter: f64,
    /// Target (or bound on it) at this value.
    pub target: f64,
    /// Test at this value (an upper-bound p-value, the adjusted interval limit nearest the
    /// null), when the analysis reports one.
    pub test: Option<f64>,
}

/// A sensitivity analysis as an assumption range tied to its target.
#[derive(Clone, Debug, PartialEq)]
pub struct AssumptionRange {
    /// Target the range is about.
    pub target: RefutationTarget,
    /// The relaxed assumption.
    pub assumption: SensitivityAssumption,
    /// Name of the sensitivity parameter (`risk_ratio`, `partial_r2`, `gamma`, `rho`).
    pub parameter: Arc<str>,
    /// Value of the parameter at which the assumption holds exactly.
    pub null_parameter: f64,
    /// Probed values, ascending in departure from the assumption.
    pub tested: Arc<[RangePoint]>,
    /// Parameter value at which the conclusion is explained away, when reached.
    pub break_even: Option<f64>,
    /// Parameter value at which the conclusion loses significance at the analysis level,
    /// when the analysis reports one.
    pub break_even_significance: Option<f64>,
    /// Identity of the range: the target identity, the assumption, the parameter and every
    /// number above.
    pub identity: [u8; 32],
}

impl AssumptionRange {
    /// Assemble a range and compute its identity.
    #[must_use]
    pub fn new(
        target: RefutationTarget,
        assumption: SensitivityAssumption,
        parameter: &str,
        null_parameter: f64,
        tested: Vec<RangePoint>,
        break_even: Option<f64>,
        break_even_significance: Option<f64>,
    ) -> Self {
        let mut writer = CanonicalWriter::new("antecedent.assumption_range");
        writer
            .bytes(target.identity())
            .text(assumption.as_str())
            .text(parameter)
            .real(null_parameter)
            .word(tested.len() as u64);
        for point in &tested {
            writer.real(point.parameter).real(point.target).opt_real(point.test);
        }
        writer.opt_real(break_even).opt_real(break_even_significance);
        let identity = writer.finish();
        Self {
            target,
            assumption,
            parameter: Arc::from(parameter),
            null_parameter,
            tested: tested.into(),
            break_even,
            break_even_significance,
            identity,
        }
    }

    /// Lower-case hex of [`Self::identity`].
    #[must_use]
    pub fn identity_hex(&self) -> String {
        crate::target_native::hex(&self.identity)
    }

    /// This range, for a claim on `claim`.
    ///
    /// # Errors
    ///
    /// [`TargetCertificationError::TargetMismatch`] when the range was computed for another
    /// target.
    pub fn bound_to(&self, claim: &RefutationTarget) -> Result<&Self, TargetCertificationError> {
        if self.target.identity() == claim.identity() {
            Ok(self)
        } else {
            Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from(self.assumption.as_str()),
                claim_target: claim.identity_hex(),
                report_target: self.target.identity_hex(),
                next_attempt: TARGET_NATIVE_NEXT_ATTEMPT,
            })
        }
    }
}
