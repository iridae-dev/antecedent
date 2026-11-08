//! Compilation of joint experiment declarations and their obligations.

use super::{
    AdjustmentKind, AdjustmentVerdict, ComponentAdjustment, EvidenceRefusal, Exposure,
    OutcomeComponent,
};
use antecedent_core::CanonicalWriter;

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
        if let AdjustmentKind::FixedCuped { coefficient, covariate }
        | AdjustmentKind::CenteredFixedCuped { coefficient, covariate } = &adjustment.kind
            && (!coefficient.is_finite() || covariate.trim().is_empty())
        {
            return Err(EvidenceRefusal::Invalid(
                "fixed CUPED needs a covariate and a finite coefficient".into(),
            ));
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
            AdjustmentKind::CenteredFixedCuped { covariate, coefficient } => {
                adj.word(2).text(covariate).real(*coefficient);
            }
        }
    }
    Ok(CompiledJointExperiment {
        adjustment_verdicts: adjustment_verdicts(&design),
        obligations,
        statistical_requirements: vec![
            "at least 30 independently assigned units in each arm for a normal interval over the covariance",
            "the number of independent units, not the number of rows, limits precision; extra rows inside existing clusters do not remove between-cluster variance",
            "a function of the outcomes carries a delta-method interval with calibration not_measured",
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
            Exposure::BoundedArmSpillover { assumption_id, treated_bias_bounds, control_bias_bounds } => {
                w.word(3).text(assumption_id).reals(treated_bias_bounds).reals(control_bias_bounds);
            }
            Exposure::Unspecified => {
                w.word(0);
            }
        }
        c.scientific_identity = w.finish_hex();
        c
    })
}
