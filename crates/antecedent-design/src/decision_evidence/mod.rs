//! Experimental evidence for decision-relevant outcomes.
//!
//! The quantity that an experiment estimates best is often not the quantity that decides the
//! action: a randomized change moves transaction frequency, transaction value and variable cost,
//! and the decision turns on a margin built from all three. This module keeps what each
//! transition from estimate to decision is allowed to say.
//!
//! * [`JointExperimentEvidence`] is the coherent object for several effects of one experiment:
//!   their estimands, population, regimes and horizon, the assignment design, the shared
//!   observation identity, and either the full covariance of the arm-mean estimators or an
//!   explicit statement that only marginal variances exist. Marginals are never promoted to a
//!   joint law.
//! * A [`DerivedQuantity`] is a function of those outcomes written in the decision contract's own
//!   [`UtilityExpr`] language. Its inputs are measured components, components bounded by a
//!   declared assumption, or declared missing. [`derive`] answers with a point, a delta-method
//!   sampling interval, an assumption range, or an unresolved result, and never with a stronger
//!   standing than its weakest input: precision of one component confers none on the others, and
//!   an executable transformation confers no identification or calibration.
//! * [`decide`] compares two actions over the admissible set and classifies the decision as
//!   robust, assumption-dependent or unresolved, naming the unresolved quantity responsible. A
//!   sampling interval and an assumption range are reported side by side and never composed.
//! * [`plan_evidence`] ranks feasible next evidence by what it would do to that decision under a
//!   declared planning model.
//! * [`reuse_receipt`] states which stages a change invalidates and which it reuses.
//!
//! Compilation precedes execution: [`compile_joint_experiment`] states the design, the outcomes,
//! which variance-reduction strategies are admissible under that design, and the identification
//! and statistical obligations before anything is fitted.
//!
//! Nothing here estimates a nuisance, widens an interval, or turns a sensitivity bound into a
//! probability. A derived interval is a delta-method construction with calibration `unmeasured`.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::if_not_else,
    reason = "straight-line validation and enumeration that reads best in one place"
)]

mod derive;
mod expr;
mod plan;
mod reuse;

use antecedent_core::CanonicalWriter;

pub use derive::{
    Arm, AssumedRange, DecisionFrame, DecisionResult, DerivedQuantity, DerivedResult, InputBinding,
    SamplingInterval, Standing, UnresolvedReason, Verdict, decide, derive,
};
pub use plan::{Candidate, EvidencePlan, PlanRow, PlanValue, PlanningModel, plan_evidence};
pub use reuse::{
    EvidenceLedger, LineageStatus, ReuseReceipt, Stage, StageDigests, StageStatus, reuse_receipt,
};

pub use crate::decision_contract::UtilityExpr;

/// Normal quantile for a two-sided 95% interval.
pub(crate) const Z95: f64 = 1.959_963_984_540_054;

/// Independently assigned units required in each arm before a normal interval over the
/// covariance is supported; equal to the estimator's own threshold.
pub const MIN_UNITS_PER_ARM_FOR_INTERVAL: usize = 30;

/// The most unknown dimensions one evaluation enumerates (`2^n` vertices).
pub const MAX_UNKNOWN_DIMENSIONS: usize = 12;

/// Statement carried by every answer that has both a sampling and an assumption part.
pub const SEPARATION_STATEMENT: &str = "a sampling interval, an identified set and an assumption range are different objects; the assumption range is not a probability law and no sampling interval has been composed into it";

/// Statement carried by every delta-method interval.
pub const DELTA_METHOD_STATEMENT: &str = "first-order delta method through the conservative Neyman covariance of the arm-mean estimators; calibration is unmeasured and no support license is inherited from the components";

/// Why a request, a declaration or a combination of evidence is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EvidenceRefusal {
    /// A declaration is malformed.
    #[error("invalid declaration: {0}")]
    Invalid(String),
    /// The requested population is not the experiment's population.
    #[error("requested population `{requested}` is not the experiment population `{available}`")]
    IncompatiblePopulation {
        /// Population the request names.
        requested: String,
        /// Population the evidence covers.
        available: String,
    },
    /// The requested intervention contrast is not the one the experiment randomized.
    #[error("requested regimes `{requested}` are not the randomized regimes `{available}`")]
    IncompatibleIntervention {
        /// Regimes the request names.
        requested: String,
        /// Regimes the evidence covers.
        available: String,
    },
    /// The requested horizon is not the measured horizon.
    #[error("requested horizon {requested} is not the measured horizon {available}")]
    IncompatibleHorizon {
        /// Horizon the request names.
        requested: u32,
        /// Horizon the evidence covers.
        available: u32,
    },
    /// The assignment design is not one the joint route supports, or two designs differ.
    #[error("assignment design `{0}` has no joint route or does not match the other evidence")]
    UnsupportedAssignmentDesign(String),
    /// Evidence from different sources cannot be combined without a licensed dependence route.
    #[error(
        "evidence `{left}` and `{right}` share or have unknown dependence; independence was not declared"
    )]
    DependenceUnknown {
        /// First evidence identity.
        left: String,
        /// Second evidence identity.
        right: String,
    },
    /// The exposure assumption between assignment units is not stated.
    #[error("the exposure assumption between assignment units is unspecified")]
    InterferenceUnspecified,
    /// The request reaches beyond the experimental contrast without a declared bridge.
    #[error(
        "a deployment-policy quantity needs a declared structural bridge from the experimental contrast"
    )]
    DeploymentBridgeMissing,
    /// An adjustment strategy is not admissible under the declared design.
    #[error("adjustment `{strategy}` is not admissible under `{design}`: {reason}")]
    AdjustmentNotAdmissible {
        /// Adjustment strategy name.
        strategy: String,
        /// Assignment design name.
        design: String,
        /// Why not.
        reason: String,
    },
    /// More unknown dimensions than one evaluation enumerates.
    #[error("{0} unknown dimensions exceed the limit of {MAX_UNKNOWN_DIMENSIONS}")]
    TooManyUnknowns(usize),
    /// The frame does not name exactly two actions.
    #[error("a decision frame compares exactly two actions")]
    FrameShape,
}

impl EvidenceRefusal {
    /// Stable machine-readable reason.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "decision_evidence.invalid_declaration",
            Self::IncompatiblePopulation { .. } => "decision_evidence.incompatible_population",
            Self::IncompatibleIntervention { .. } => "decision_evidence.incompatible_intervention",
            Self::IncompatibleHorizon { .. } => "decision_evidence.incompatible_horizon",
            Self::UnsupportedAssignmentDesign(_) => {
                "decision_evidence.unsupported_assignment_design"
            }
            Self::DependenceUnknown { .. } => "decision_evidence.dependence_unknown",
            Self::InterferenceUnspecified => "decision_evidence.interference_unspecified",
            Self::DeploymentBridgeMissing => "decision_evidence.deployment_bridge_missing",
            Self::AdjustmentNotAdmissible { .. } => "decision_evidence.adjustment_not_admissible",
            Self::TooManyUnknowns(_) => "decision_evidence.too_many_unknowns",
            Self::FrameShape => "decision_evidence.frame_shape",
        }
    }

    /// The next thing to try; every refusal names one.
    #[must_use]
    pub fn next_attempt(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "correct the declaration named in the message",
            Self::IncompatiblePopulation { .. } => {
                "request the experiment's population, or bind evidence from the target population"
            }
            Self::IncompatibleIntervention { .. } => {
                "request the contrast the experiment randomized, or run the experiment for the requested contrast"
            }
            Self::IncompatibleHorizon { .. } => {
                "request the measured horizon, or measure the requested one"
            }
            Self::UnsupportedAssignmentDesign(_) => {
                "estimate each design separately and combine only under a declared independence"
            }
            Self::DependenceUnknown { .. } => {
                "declare the evidence independent if it shares no units or observations, or derive the joint law from one experiment"
            }
            Self::InterferenceUnspecified => {
                "declare no between-unit interference, or a bound on spillover, as an assumption"
            }
            Self::DeploymentBridgeMissing => {
                "declare the structural assumption that carries the experimental contrast to the deployment, or ask for the experimental contrast"
            }
            Self::AdjustmentNotAdmissible { .. } => {
                "use a strategy listed as admissible by compile_joint_experiment"
            }
            Self::TooManyUnknowns(_) => {
                "reduce the number of assumed components or tie them across arms"
            }
            Self::FrameShape => "name exactly two actions",
        }
    }
}

/// One outcome whose effect the experiment estimates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeComponent {
    /// Name derived quantities refer to.
    pub name: String,
    /// Stable variable identity.
    pub variable_id: String,
    /// Declared units; never converted.
    pub units: String,
}

/// How an outcome was adjusted for pretreatment information.
#[derive(Clone, Debug, PartialEq)]
pub enum AdjustmentKind {
    /// No adjustment.
    None,
    /// A pretreatment covariate with a coefficient fixed outside this experiment's outcomes.
    FixedCuped {
        /// Covariate name.
        covariate: String,
        /// Coefficient, fixed before the outcomes were observed.
        coefficient: f64,
    },
}

/// The adjustment of one component.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentAdjustment {
    /// Component name.
    pub component: String,
    /// Adjustment applied to it.
    pub kind: AdjustmentKind,
}

/// A refusal-or-admission verdict for one adjustment strategy under the declared design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdjustmentVerdict {
    /// Strategy name.
    pub strategy: &'static str,
    /// Whether its coefficient is fixed independently of the experiment or learned from it.
    pub coefficient: &'static str,
    /// Whether the strategy is admissible and, if not, why.
    pub admissible: Result<&'static str, &'static str>,
}

/// What is assumed about exposure between assignment units.
#[derive(Clone, Debug, PartialEq)]
pub enum Exposure {
    /// No outcome of a unit depends on another unit's assignment. An assumption, not a finding.
    NoBetweenUnitInterference {
        /// Identity of the declared assumption.
        assumption_id: String,
    },
    /// Spillover may bias the contrast of each component by at most the given absolute amount.
    BoundedSpillover {
        /// Identity of the declared assumption.
        assumption_id: String,
        /// Per-component bound on the bias of the contrast, aligned with the outcomes.
        bias_bounds: Vec<f64>,
    },
    /// Nothing is declared; derived quantities refuse.
    Unspecified,
}

/// What a request asks about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// The contrast the experiment randomized, in the experiment's population.
    ExperimentalContrast,
    /// A broader deployment policy, reached only through a declared structural assumption.
    Deployment {
        /// Identity of the structural bridge, when one is declared.
        bridge_assumption_id: Option<String>,
    },
}

/// What a derived quantity or decision is asked about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// Population the answer is for.
    pub population_id: String,
    /// Treated regime.
    pub treated_regime_id: String,
    /// Control regime.
    pub control_regime_id: String,
    /// Horizon in ordered causal steps.
    pub horizon: u32,
    /// Experimental contrast or a broader deployment.
    pub target: Target,
}

/// Dependence between the arm-mean estimators of the components.
#[derive(Clone, Debug, PartialEq)]
pub enum Dependence {
    /// Full `K x K` covariances (row-major) of the treated- and control-arm mean estimators.
    Joint {
        /// Treated arm.
        treated: Vec<f64>,
        /// Control arm.
        control: Vec<f64>,
    },
    /// Only marginal variances exist; the covariance between components is unknown.
    MarginalsOnly {
        /// Treated-arm variances.
        treated: Vec<f64>,
        /// Control-arm variances.
        control: Vec<f64>,
    },
}

/// The assignment design the estimates came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DesignRecord {
    /// `cluster` or `complete`.
    pub assignment_design: String,
    /// Independently assigned units (clusters) in the treated arm.
    pub treated_units: usize,
    /// Independently assigned units (clusters) in the control arm.
    pub control_units: usize,
    /// Outcome rows.
    pub outcome_rows: usize,
}

/// Several effects of one experiment, kept together.
#[derive(Clone, Debug, PartialEq)]
pub struct JointExperimentEvidence {
    /// Identity of the observations and assignment these estimates came from.
    pub evidence_id: String,
    /// Population the experiment sampled.
    pub population_id: String,
    /// Treated regime identity.
    pub treated_regime_id: String,
    /// Control regime identity.
    pub control_regime_id: String,
    /// Horizon in ordered causal steps.
    pub horizon: u32,
    /// The outcomes, in estimator order.
    pub components: Vec<OutcomeComponent>,
    /// Per-row mean of each outcome in the treated arm.
    pub treated_means: Vec<f64>,
    /// Per-row mean of each outcome in the control arm.
    pub control_means: Vec<f64>,
    /// What is known about the dependence between components.
    pub dependence: Dependence,
    /// The design.
    pub design: DesignRecord,
    /// Adjustment of each component.
    pub adjustments: Vec<ComponentAdjustment>,
    /// Exposure between assignment units.
    pub exposure: Exposure,
    /// Route license of each component's marginal effect, when one was earned; informational.
    pub marginal_licenses: Vec<Option<String>>,
}

impl JointExperimentEvidence {
    /// Validate and assemble the evidence.
    ///
    /// # Errors
    ///
    /// [`EvidenceRefusal::Invalid`] for misaligned or non-finite arrays, a covariance that is
    /// not symmetric with a nonnegative diagonal, an arm with fewer than two units, or an
    /// exposure bound list that does not align with the components;
    /// [`EvidenceRefusal::UnsupportedAssignmentDesign`] for a design other than `cluster` or
    /// `complete`.
    #[allow(clippy::too_many_arguments, reason = "one coherent record; every field is required")]
    pub fn new(
        evidence_id: impl Into<String>,
        population_id: impl Into<String>,
        regimes: (impl Into<String>, impl Into<String>),
        horizon: u32,
        components: Vec<OutcomeComponent>,
        means: (Vec<f64>, Vec<f64>),
        dependence: Dependence,
        design: DesignRecord,
        adjustments: Vec<ComponentAdjustment>,
        exposure: Exposure,
    ) -> Result<Self, EvidenceRefusal> {
        let k = components.len();
        let invalid = |m: &str| Err(EvidenceRefusal::Invalid(m.to_owned()));
        if k == 0 {
            return invalid("at least one outcome component is required");
        }
        let names: std::collections::BTreeSet<_> =
            components.iter().map(|c| c.name.as_str()).collect();
        if names.len() != k
            || components.iter().any(|c| c.name.trim().is_empty() || c.units.trim().is_empty())
        {
            return invalid("component names must be unique and non-empty, and units declared");
        }
        let (treated_means, control_means) = means;
        if treated_means.len() != k
            || control_means.len() != k
            || treated_means.iter().chain(&control_means).any(|v| !v.is_finite())
        {
            return invalid("arm means must be finite and aligned with the components");
        }
        let check_cov = |m: &[f64], full: bool| -> bool {
            let expected = if full { k * k } else { k };
            if m.len() != expected || m.iter().any(|v| !v.is_finite()) {
                return false;
            }
            (0..k).all(|i| {
                let d = m[if full { i * k + i } else { i }];
                d >= 0.0
                    && (!full
                        || (0..k).all(|j| {
                            (m[i * k + j] - m[j * k + i]).abs() <= 1e-9 * (1.0 + m[i * k + j].abs())
                                && m[i * k + j].abs()
                                    <= (m[i * k + i] * m[j * k + j]).sqrt() * (1.0 + 1e-9) + 1e-300
                        }))
            })
        };
        let ok = match &dependence {
            Dependence::Joint { treated, control } => {
                check_cov(treated, true) && check_cov(control, true)
            }
            Dependence::MarginalsOnly { treated, control } => {
                check_cov(treated, false) && check_cov(control, false)
            }
        };
        if !ok {
            return invalid(
                "covariances must be finite, symmetric, with nonnegative variances and |covariance| <= sqrt(variance product)",
            );
        }
        if !matches!(design.assignment_design.as_str(), "cluster" | "complete") {
            return Err(EvidenceRefusal::UnsupportedAssignmentDesign(design.assignment_design));
        }
        if design.treated_units < 2 || design.control_units < 2 || design.outcome_rows == 0 {
            return invalid("each arm needs at least two independently assigned units");
        }
        if adjustments.iter().any(|a| !names.contains(a.component.as_str())) {
            return invalid("an adjustment names an unknown component");
        }
        if let Exposure::BoundedSpillover { bias_bounds, .. } = &exposure {
            if bias_bounds.len() != k || bias_bounds.iter().any(|b| !b.is_finite() || *b < 0.0) {
                return invalid(
                    "spillover bounds must be finite, nonnegative and aligned with the components",
                );
            }
        }
        let (treated_regime, control_regime) = (regimes.0.into(), regimes.1.into());
        if treated_regime == control_regime {
            return invalid("the treated and control regimes must differ");
        }
        Ok(Self {
            evidence_id: evidence_id.into(),
            population_id: population_id.into(),
            treated_regime_id: treated_regime,
            control_regime_id: control_regime,
            horizon,
            components,
            treated_means,
            control_means,
            dependence,
            design,
            adjustments,
            exposure,
            marginal_licenses: vec![None; k],
        })
    }

    /// Attach the route license each component's marginal effect earned (informational: a
    /// derived quantity never inherits it).
    #[must_use]
    pub fn with_marginal_licenses(mut self, licenses: Vec<Option<String>>) -> Self {
        if licenses.len() == self.components.len() {
            self.marginal_licenses = licenses;
        }
        self
    }

    /// Index of the component named `name`.
    #[must_use]
    pub fn component_index(&self, name: &str) -> Option<usize> {
        self.components.iter().position(|c| c.name == name)
    }

    /// Combine evidence from two experiments that were declared independent.
    ///
    /// The result is block-diagonal in its dependence. Evidence that shares an identity (the
    /// same observations), differs in population, regimes, horizon or design, or was not
    /// declared independent is refused.
    ///
    /// # Errors
    ///
    /// The matching [`EvidenceRefusal`].
    pub fn merge_independent(
        &self,
        other: &Self,
        declared_independent: bool,
    ) -> Result<Self, EvidenceRefusal> {
        if self.population_id != other.population_id {
            return Err(EvidenceRefusal::IncompatiblePopulation {
                requested: other.population_id.clone(),
                available: self.population_id.clone(),
            });
        }
        if (&self.treated_regime_id, &self.control_regime_id)
            != (&other.treated_regime_id, &other.control_regime_id)
        {
            return Err(EvidenceRefusal::IncompatibleIntervention {
                requested: format!("{}/{}", other.treated_regime_id, other.control_regime_id),
                available: format!("{}/{}", self.treated_regime_id, self.control_regime_id),
            });
        }
        if self.horizon != other.horizon {
            return Err(EvidenceRefusal::IncompatibleHorizon {
                requested: other.horizon,
                available: self.horizon,
            });
        }
        if self.design.assignment_design != other.design.assignment_design {
            return Err(EvidenceRefusal::UnsupportedAssignmentDesign(
                other.design.assignment_design.clone(),
            ));
        }
        if self.evidence_id == other.evidence_id || !declared_independent {
            return Err(EvidenceRefusal::DependenceUnknown {
                left: self.evidence_id.clone(),
                right: other.evidence_id.clone(),
            });
        }
        let (ka, kb) = (self.components.len(), other.components.len());
        if other.components.iter().any(|c| self.component_index(&c.name).is_some()) {
            return Err(EvidenceRefusal::Invalid(
                "merged evidence repeats a component name".into(),
            ));
        }
        let k = ka + kb;
        let embed = |a: &Dependence, b: &Dependence, treated: bool| -> Vec<f64> {
            let pick = |d: &Dependence, n: usize| -> Vec<f64> {
                match d {
                    Dependence::Joint { treated: t, control: c } => {
                        if treated {
                            t.clone()
                        } else {
                            c.clone()
                        }
                    }
                    Dependence::MarginalsOnly { treated: t, control: c } => {
                        let v = if treated { t } else { c };
                        let mut m = vec![0.0; n * n];
                        for (i, x) in v.iter().enumerate() {
                            m[i * n + i] = *x;
                        }
                        m
                    }
                }
            };
            let (ma, mb) = (pick(a, ka), pick(b, kb));
            let mut m = vec![0.0; k * k];
            for i in 0..ka {
                for j in 0..ka {
                    m[i * k + j] = ma[i * ka + j];
                }
            }
            for i in 0..kb {
                for j in 0..kb {
                    m[(ka + i) * k + ka + j] = mb[i * kb + j];
                }
            }
            m
        };
        let both_joint = matches!(self.dependence, Dependence::Joint { .. })
            && matches!(other.dependence, Dependence::Joint { .. });
        let dependence = if both_joint {
            Dependence::Joint {
                treated: embed(&self.dependence, &other.dependence, true),
                control: embed(&self.dependence, &other.dependence, false),
            }
        } else {
            let diag = |m: &[f64]| (0..k).map(|i| m[i * k + i]).collect::<Vec<_>>();
            let (t, c) = (
                embed(&self.dependence, &other.dependence, true),
                embed(&self.dependence, &other.dependence, false),
            );
            Dependence::MarginalsOnly { treated: diag(&t), control: diag(&c) }
        };
        let exposure = match (&self.exposure, &other.exposure) {
            (
                Exposure::NoBetweenUnitInterference { .. },
                Exposure::NoBetweenUnitInterference { .. },
            ) => self.exposure.clone(),
            _ => Exposure::Unspecified,
        };
        Ok(Self {
            evidence_id: format!("{}+{}", self.evidence_id, other.evidence_id),
            population_id: self.population_id.clone(),
            treated_regime_id: self.treated_regime_id.clone(),
            control_regime_id: self.control_regime_id.clone(),
            horizon: self.horizon,
            components: self.components.iter().chain(&other.components).cloned().collect(),
            treated_means: self.treated_means.iter().chain(&other.treated_means).copied().collect(),
            control_means: self.control_means.iter().chain(&other.control_means).copied().collect(),
            dependence,
            design: DesignRecord {
                treated_units: self.design.treated_units.min(other.design.treated_units),
                control_units: self.design.control_units.min(other.design.control_units),
                ..self.design.clone()
            },
            adjustments: self.adjustments.iter().chain(&other.adjustments).cloned().collect(),
            exposure,
            marginal_licenses: self
                .marginal_licenses
                .iter()
                .chain(&other.marginal_licenses)
                .cloned()
                .collect(),
        })
    }

    /// Identity of the scientific premise: what was estimated, in whom, under which contrast and
    /// design and exposure assumption. Independent of the observations and the adjustment.
    #[must_use]
    pub fn scientific_identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.scientific.v1");
        w.text(&self.population_id).text(&self.treated_regime_id).text(&self.control_regime_id);
        w.word(u64::from(self.horizon)).text(&self.design.assignment_design);
        for c in &self.components {
            w.text(&c.name).text(&c.variable_id).text(&c.units);
        }
        match &self.exposure {
            Exposure::NoBetweenUnitInterference { assumption_id } => {
                w.word(1).text(assumption_id);
            }
            Exposure::BoundedSpillover { assumption_id, bias_bounds } => {
                w.word(2).text(assumption_id).reals(bias_bounds);
            }
            Exposure::Unspecified => {
                w.word(0);
            }
        }
        w.finish_hex()
    }

    /// Identity of the adjustment specification.
    #[must_use]
    pub fn adjustment_identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.adjustment.v1");
        let mut sorted = self.adjustments.clone();
        sorted.sort_by(|a, b| a.component.cmp(&b.component));
        for a in &sorted {
            w.text(&a.component);
            match &a.kind {
                AdjustmentKind::None => {
                    w.word(0);
                }
                AdjustmentKind::FixedCuped { covariate, coefficient } => {
                    w.word(1).text(covariate).real(*coefficient);
                }
            }
        }
        w.finish_hex()
    }

    /// Identity of the fitted joint estimate: the observations, the premise and the adjustment.
    #[must_use]
    pub fn estimate_identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.estimate.v1");
        w.text(&self.evidence_id)
            .text(&self.scientific_identity())
            .text(&self.adjustment_identity());
        w.finish_hex()
    }
}

/// An obligation a compiled experiment states before any fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledObligation {
    /// Stable id.
    pub id: &'static str,
    /// Scientific vocabulary kind.
    pub kind: antecedent_contract::EvidenceObligationKind,
    /// What is owed.
    pub statement: &'static str,
}

/// Declaration of a joint experiment before any outcome is read.
#[derive(Clone, Debug, PartialEq)]
pub struct JointExperimentSpec {
    /// Outcomes whose effects will be estimated together.
    pub outcomes: Vec<OutcomeComponent>,
    /// `cluster`, `complete`, or another design name (which will be refused).
    pub assignment_design: String,
    /// Population the experiment samples.
    pub population_id: String,
    /// Treated regime identity.
    pub treated_regime_id: String,
    /// Control regime identity.
    pub control_regime_id: String,
    /// Horizon in ordered causal steps.
    pub horizon: u32,
    /// Requested adjustment of each component.
    pub adjustments: Vec<ComponentAdjustment>,
    /// Exposure assumption between assignment units.
    pub exposure: Exposure,
}

/// A joint experiment compiled for inspection before fitting.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledJointExperiment {
    /// The declaration.
    pub spec: JointExperimentSpec,
    /// Every adjustment strategy and whether the design admits it.
    pub adjustment_verdicts: Vec<AdjustmentVerdict>,
    /// Identification and statistical obligations.
    pub obligations: Vec<CompiledObligation>,
    /// Statistical requirements for a published interval.
    pub statistical_requirements: Vec<&'static str>,
    /// Unit of analysis the design fixes.
    pub unit_of_analysis: &'static str,
    /// Identity of the scientific premise.
    pub scientific_identity: String,
    /// Identity of the requested adjustment.
    pub adjustment_identity: String,
}

/// Adjustment strategies and their admissibility under `design`.
#[must_use]
pub fn adjustment_verdicts(design: &str) -> Vec<AdjustmentVerdict> {
    let supported = matches!(design, "cluster" | "complete");
    vec![
        AdjustmentVerdict {
            strategy: "none",
            coefficient: "not applicable",
            admissible: if supported {
                Ok("the design's own estimator over the assigned units")
            } else {
                Err("no joint route for this assignment design")
            },
        },
        AdjustmentVerdict {
            strategy: "fixed_cuped",
            coefficient: "fixed independently of the experiment",
            admissible: if supported {
                Ok(if design == "cluster" {
                    "applied to cluster totals; the cluster stays the unit of analysis and the estimand stays the unadjusted per-row contrast"
                } else {
                    "applied per assigned unit; the estimand stays the unadjusted contrast"
                })
            } else {
                Err("no joint route for this assignment design")
            },
        },
        AdjustmentVerdict {
            strategy: "learned_regression",
            coefficient: "learned from the experiment's observations",
            admissible: Err(if design == "cluster" {
                "a coefficient fitted on rows treats rows as independent and would change the unit of analysis; no cluster-level learned adjustment is implemented"
            } else {
                "a learned coefficient needs its own variance construction, which the joint route does not provide"
            }),
        },
    ]
}

/// Compile a joint experiment: state the design, the outcomes, the admissible adjustments and the
/// obligations before anything is fitted.
///
/// # Errors
///
/// [`EvidenceRefusal::UnsupportedAssignmentDesign`] for a design with no joint route;
/// [`EvidenceRefusal::AdjustmentNotAdmissible`] for an unsupported adjustment strategy;
/// [`EvidenceRefusal::Invalid`] for malformed declarations.
pub fn compile_joint_experiment(
    spec: JointExperimentSpec,
) -> Result<CompiledJointExperiment, EvidenceRefusal> {
    if !matches!(spec.assignment_design.as_str(), "cluster" | "complete") {
        return Err(EvidenceRefusal::UnsupportedAssignmentDesign(spec.assignment_design));
    }
    if spec.outcomes.is_empty() {
        return Err(EvidenceRefusal::Invalid("at least one outcome is required".into()));
    }
    let names: std::collections::BTreeSet<_> =
        spec.outcomes.iter().map(|o| o.name.as_str()).collect();
    if names.len() != spec.outcomes.len() {
        return Err(EvidenceRefusal::Invalid("outcome names must be unique".into()));
    }
    for adjustment in &spec.adjustments {
        if !names.contains(adjustment.component.as_str()) {
            return Err(EvidenceRefusal::Invalid(format!(
                "adjustment names unknown outcome `{}`",
                adjustment.component
            )));
        }
        if let AdjustmentKind::FixedCuped { coefficient, covariate } = &adjustment.kind {
            if !coefficient.is_finite() || covariate.trim().is_empty() {
                return Err(EvidenceRefusal::Invalid(
                    "fixed CUPED needs a covariate and a finite coefficient".into(),
                ));
            }
        }
    }
    let design = spec.assignment_design.clone();
    let mut obligations = vec![
        CompiledObligation {
            id: "randomized_assignment_known",
            kind: antecedent_contract::EvidenceObligationKind::EstablishAssumption,
            statement: "the realized assignment follows the declared complete randomization of independent units",
        },
        CompiledObligation {
            id: "between_unit_exposure",
            kind: antecedent_contract::EvidenceObligationKind::EstablishAssumption,
            statement: "exposure between assignment units is declared: no interference, or a bound on spillover",
        },
        CompiledObligation {
            id: "joint_covariance",
            kind: antecedent_contract::EvidenceObligationKind::ProvideJointLaw,
            statement: "dependence between the outcomes is estimated from the same assigned units, not assumed",
        },
    ];
    if spec.adjustments.iter().any(|a| a.kind != AdjustmentKind::None) {
        obligations.push(CompiledObligation {
            id: "fixed_pre_assignment_coefficient",
            kind: antecedent_contract::EvidenceObligationKind::EstablishAssumption,
            statement: "each adjustment covariate was measured before assignment and its coefficient was fixed independently of these outcomes",
        });
    }
    let unit_of_analysis = if design == "cluster" { "assignment cluster" } else { "assigned unit" };
    let mut compiled_spec = spec;
    compiled_spec.adjustments.sort_by(|a, b| a.component.cmp(&b.component));
    let mut writer = CanonicalWriter::new("antecedent.decision_evidence.scientific.v1");
    writer
        .text(&compiled_spec.population_id)
        .text(&compiled_spec.treated_regime_id)
        .text(&compiled_spec.control_regime_id)
        .word(u64::from(compiled_spec.horizon))
        .text(&compiled_spec.assignment_design);
    for o in &compiled_spec.outcomes {
        writer.text(&o.name).text(&o.variable_id).text(&o.units);
    }
    let mut adj = CanonicalWriter::new("antecedent.decision_evidence.adjustment.v1");
    for a in &compiled_spec.adjustments {
        adj.text(&a.component);
        match &a.kind {
            AdjustmentKind::None => {
                adj.word(0);
            }
            AdjustmentKind::FixedCuped { covariate, coefficient } => {
                adj.word(1).text(covariate).real(*coefficient);
            }
        }
    }
    Ok(CompiledJointExperiment {
        adjustment_verdicts: adjustment_verdicts(&design),
        obligations,
        statistical_requirements: vec![
            "at least 30 independently assigned units in each arm for a normal interval over the covariance",
            "the number of independent units, not the number of rows, limits precision; extra rows inside existing clusters do not remove between-cluster variance",
            "a function of the outcomes carries a delta-method interval with calibration unmeasured",
        ],
        unit_of_analysis,
        scientific_identity: String::new(),
        adjustment_identity: adj.finish_hex(),
        spec: compiled_spec,
    })
    .map(|mut c| {
        // The premise identity of a compiled experiment matches the identity of the evidence
        // built from it, which also hashes the exposure assumption.
        let mut w = writer;
        match &c.spec.exposure {
            Exposure::NoBetweenUnitInterference { assumption_id } => {
                w.word(1).text(assumption_id);
            }
            Exposure::BoundedSpillover { assumption_id, bias_bounds } => {
                w.word(2).text(assumption_id).reals(bias_bounds);
            }
            Exposure::Unspecified => {
                w.word(0);
            }
        }
        c.scientific_identity = w.finish_hex();
        c
    })
}

#[cfg(test)]
mod tests {
    use super::derive::tests::{evidence, no_interference};
    use super::*;

    fn spec(design: &str, adjustments: Vec<ComponentAdjustment>) -> JointExperimentSpec {
        JointExperimentSpec {
            outcomes: vec![OutcomeComponent {
                name: "value".into(),
                variable_id: "v".into(),
                units: "usd".into(),
            }],
            assignment_design: design.into(),
            population_id: "pop".into(),
            treated_regime_id: "ship".into(),
            control_regime_id: "hold".into(),
            horizon: 4,
            adjustments,
            exposure: no_interference(),
        }
    }

    fn cuped() -> Vec<ComponentAdjustment> {
        vec![ComponentAdjustment {
            component: "value".into(),
            kind: AdjustmentKind::FixedCuped { covariate: "pre".into(), coefficient: 0.5 },
        }]
    }

    #[test]
    fn compile_states_design_and_admissibility_before_any_fit() {
        let c = compile_joint_experiment(spec("cluster", cuped())).unwrap();
        assert_eq!(c.unit_of_analysis, "assignment cluster");
        let learned =
            c.adjustment_verdicts.iter().find(|v| v.strategy == "learned_regression").unwrap();
        assert!(learned.admissible.is_err());
        assert!(
            c.adjustment_verdicts
                .iter()
                .find(|v| v.strategy == "fixed_cuped")
                .unwrap()
                .admissible
                .is_ok()
        );
        assert!(c.obligations.iter().any(|o| o.id == "fixed_pre_assignment_coefficient"));
        assert!(
            compile_joint_experiment(spec("cluster", vec![]))
                .unwrap()
                .obligations
                .iter()
                .all(|o| o.id != "fixed_pre_assignment_coefficient")
        );
        assert!(matches!(
            compile_joint_experiment(spec("stepped_wedge", vec![])),
            Err(EvidenceRefusal::UnsupportedAssignmentDesign(_))
        ));
    }

    #[test]
    fn compiled_premise_identity_matches_the_evidence_built_from_it() {
        let ev = evidence(1.0, no_interference());
        let spec = JointExperimentSpec {
            outcomes: ev.components.clone(),
            assignment_design: "cluster".into(),
            population_id: "pop".into(),
            treated_regime_id: "ship".into(),
            control_regime_id: "hold".into(),
            horizon: 4,
            adjustments: vec![],
            exposure: no_interference(),
        };
        assert_eq!(
            compile_joint_experiment(spec).unwrap().scientific_identity,
            ev.scientific_identity()
        );
    }

    #[test]
    fn merging_requires_declared_independence_and_matching_premise() {
        let a = evidence(1.0, no_interference());
        let mut b = evidence(1.0, no_interference());
        b.evidence_id = "exp-2".into();
        b.components = vec![OutcomeComponent {
            name: "refunds".into(),
            variable_id: "r".into(),
            units: "u".into(),
        }];
        b.treated_means = vec![0.1];
        b.control_means = vec![0.1];
        b.dependence = Dependence::Joint { treated: vec![0.01], control: vec![0.01] };
        assert!(matches!(
            a.merge_independent(&b, false),
            Err(EvidenceRefusal::DependenceUnknown { .. })
        ));
        assert!(matches!(
            a.merge_independent(&a, true),
            Err(EvidenceRefusal::DependenceUnknown { .. })
        ));
        let m = a.merge_independent(&b, true).unwrap();
        assert_eq!(m.components.len(), 4);
        let Dependence::Joint { treated, .. } = &m.dependence else { panic!() };
        assert_eq!(treated[3 * 4 + 0], 0.0, "cross-experiment covariance is the declared zero");
        let mut c = b.clone();
        c.horizon = 9;
        assert!(matches!(
            a.merge_independent(&c, true),
            Err(EvidenceRefusal::IncompatibleHorizon { .. })
        ));
    }

    #[test]
    fn few_independent_units_leave_no_interval_however_many_rows() {
        let mut ev = evidence(1.0, no_interference());
        ev.design = DesignRecord {
            assignment_design: "cluster".into(),
            treated_units: 8,
            control_units: 8,
            outcome_rows: 100_000,
        };
        let q = DerivedQuantity {
            name: "v".into(),
            units: "u".into(),
            expr: UtilityExpr::Input(0),
            inputs: vec![InputBinding::Measured("value".into())],
        };
        let req = derive::tests::request();
        let r = derive(&ev, &req, &q).unwrap();
        assert!(r.sampling.is_none() && r.contrast.is_some());
        assert!(matches!(
            r.unresolved[0],
            UnresolvedReason::TooFewIndependentUnits { required: 30, .. }
        ));
    }

    #[test]
    fn invalid_covariance_is_refused() {
        let mut ev = evidence(1.0, no_interference());
        if let Dependence::Joint { treated, .. } = &mut ev.dependence {
            treated[1] = 5.0;
            treated[3] = 5.0;
        }
        let rebuilt = JointExperimentEvidence::new(
            "x",
            "pop",
            ("a", "b"),
            1,
            ev.components.clone(),
            (ev.treated_means.clone(), ev.control_means.clone()),
            ev.dependence.clone(),
            ev.design.clone(),
            vec![],
            no_interference(),
        );
        assert!(matches!(rebuilt, Err(EvidenceRefusal::Invalid(_))));
    }
}
