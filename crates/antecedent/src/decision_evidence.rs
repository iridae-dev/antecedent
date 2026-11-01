//! Experimental evidence for decision-relevant outcomes.
//!
//! The estimator side ([`antecedent_estimate::complete_cluster_joint_contrast`]) and the
//! decision side ([`antecedent_design::decision_evidence`]) are separate layers; this module is
//! the one place they meet, so the design crate stays independent of the estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::decision_evidence::*;
pub use antecedent_estimate::randomized_joint::{
    JointClusterContrast, centered_fixed_cuped_totals, complete_cluster_joint_contrast,
};

/// Wrap a joint cluster estimate as decision evidence.
///
/// The covariance of the arm-mean estimators travels with the means; nothing is recomputed or
/// widened. The unit of analysis stays the assignment cluster.
///
/// # Errors
///
/// [`EvidenceRefusal`] from [`JointExperimentEvidence::new`].
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the evidence record; every field is required"
)]
pub fn evidence_from_cluster_contrast(
    contrast: &JointClusterContrast,
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    outcome_rows: usize,
    adjustments: Vec<ComponentAdjustment>,
    exposure: Exposure,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    if components.len() != contrast.outcomes {
        return Err(EvidenceRefusal::Invalid(
            "components must align with the estimated outcomes".into(),
        ));
    }
    JointExperimentEvidence::new(
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        (contrast.treated_means.clone(), contrast.control_means.clone()),
        Dependence::Joint {
            treated: contrast.treated_covariance.clone(),
            control: contrast.control_covariance.clone(),
        },
        DesignRecord {
            assignment_design: "cluster".into(),
            treated_units: contrast.treated_clusters,
            control_units: contrast.control_clusters,
            outcome_rows,
        },
        adjustments,
        exposure,
    )
}

/// Matched-pair assignment declared before outcomes or treatment assignment are observed.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchedPairDesign {
    /// Stable cluster identities, aligned with outcome totals and assignment.
    pub cluster_ids: Vec<String>,
    /// Fixed membership of each independently randomized pair, as cluster indexes.
    pub pairs: Vec<(usize, usize)>,
    /// Pretreatment variables used to form the pairs.
    pub matching_variables: Vec<String>,
    /// Declared probability of treatment for each cluster; currently exactly one half.
    pub treatment_probabilities: Vec<f64>,
    /// Fixed target row counts of each cluster.
    pub cluster_sizes: Vec<usize>,
    /// Identity of the assumption that matching and sizes were fixed before assignment.
    pub preassignment_assumption_id: String,
    /// Identity of the assumption that pairs are assigned independently.
    pub independent_pairs_assumption_id: String,
}

/// Inspected assignment design, prepared before fitting outcomes.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledMatchedPairDesign {
    /// The validated design.
    pub design: MatchedPairDesign,
    /// Stable identity of the declared pairing and assignment law.
    pub identity: String,
    /// Identification and uncertainty obligations, not empirical findings.
    pub obligations: Vec<&'static str>,
}

/// Validate a pretreatment matched-pair design before estimation.
///
/// # Errors
/// Refuses malformed pairs, unequal or undeclared assignment probabilities, and missing
/// preassignment information. Only independent 1:1 randomization within pairs is supported.
pub fn compile_matched_pair_design(
    design: MatchedPairDesign,
) -> Result<CompiledMatchedPairDesign, EvidenceRefusal> {
    use std::collections::BTreeSet;
    let n = design.cluster_ids.len();
    let invalid = |message: &str| EvidenceRefusal::Invalid(message.into());
    if n < 4
        || design.pairs.len().checked_mul(2) != Some(n)
        || design.cluster_sizes.len() != n
        || design.cluster_sizes.contains(&0)
        || design.treatment_probabilities.len() != n
        || design.treatment_probabilities.iter().any(|p| *p != 0.5)
        || design.preassignment_assumption_id.trim().is_empty()
        || design.independent_pairs_assumption_id.trim().is_empty()
        || design.matching_variables.is_empty()
        || design.matching_variables.iter().any(|v| v.trim().is_empty())
        || design.cluster_ids.iter().any(|id| id.trim().is_empty())
        || design.cluster_ids.iter().collect::<BTreeSet<_>>().len() != n
    {
        return Err(invalid(
            "matched pairs require unique clusters, fixed positive sizes, declared pretreatment matching, and treatment probability 0.5",
        ));
    }
    let mut seen = vec![false; n];
    for &(a, b) in &design.pairs {
        if a >= n || b >= n || a == b || seen[a] || seen[b] {
            return Err(invalid("each cluster must occur in exactly one pair"));
        }
        seen[a] = true;
        seen[b] = true;
    }
    let mut w = antecedent_core::CanonicalWriter::new(
        "antecedent.decision_evidence.matched_pair_design.v1",
    );
    w.text(&design.preassignment_assumption_id)
        .text(&design.independent_pairs_assumption_id)
        .word(design.matching_variables.len() as u64)
        .word(design.cluster_ids.len() as u64)
        .word(design.pairs.len() as u64);
    for v in &design.matching_variables {
        w.text(v);
    }
    for ((id, n), p) in
        design.cluster_ids.iter().zip(&design.cluster_sizes).zip(&design.treatment_probabilities)
    {
        w.text(id).word(*n as u64).real(*p);
    }
    for (a, b) in &design.pairs {
        w.word(*a as u64).word(*b as u64);
    }
    Ok(CompiledMatchedPairDesign {
        identity: w.finish_hex(),
        design,
        obligations: vec![
            "pair membership, matching variables and cluster sizes are fixed before assignment",
            "exactly one cluster per pair is independently assigned treatment with probability one half",
            "there is no interference between assignment units unless separately bounded",
            "nonlinear delta-method intervals require independent procedure-specific verification and calibration",
        ],
    })
}

/// Execute a compiled matched-pair joint experiment and bind its evidence to the design.
///
/// # Errors
/// Refuses nonaligned outcomes, assignments not realizing one treated cluster per pair,
/// unsupported exposure, or invalid evidence declarations.
#[allow(clippy::too_many_arguments)]
pub fn evidence_from_matched_pairs(
    design: &CompiledMatchedPairDesign,
    totals: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    if compile_matched_pair_design(design.design.clone())?.identity != design.identity {
        return Err(EvidenceRefusal::Invalid("compiled matched-pair design was modified".into()));
    }
    if matches!(exposure, Exposure::Unspecified) {
        return Err(EvidenceRefusal::InterferenceUnspecified);
    }
    let rows = design
        .design
        .cluster_sizes
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(|| EvidenceRefusal::Invalid("cluster row count overflow".into()))?;
    let fit = antecedent_estimate::randomized_joint::matched_pair_joint_contrast(totals, assigned, &design.design.pairs, rows)
        .ok_or_else(|| EvidenceRefusal::Invalid("outcomes and assignments must align with the compiled pairs, with one treated cluster per pair".into()))?;
    let mut result = JointExperimentEvidence::new(
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        (fit.treated_means, fit.control_means),
        Dependence::JointArms { covariance: fit.arm_covariance },
        DesignRecord {
            assignment_design: "matched_pairs".into(),
            treated_units: fit.pairs,
            control_units: fit.pairs,
            outcome_rows: rows,
        },
        vec![],
        exposure,
    )?;
    result.design_contract_id = Some(design.identity.clone());
    result.design_details = Some(DesignDetails::MatchedPairs {
        cluster_ids: design.design.cluster_ids.clone(),
        pairs: design.design.pairs.clone(),
        matching_variables: design.design.matching_variables.clone(),
        treatment_probabilities: design.design.treatment_probabilities.clone(),
        cluster_sizes: design.design.cluster_sizes.clone(),
        realized_assignment: assigned.to_vec(),
        preassignment_assumption_id: design.design.preassignment_assumption_id.clone(),
        independent_pairs_assumption_id: design.design.independent_pairs_assumption_id.clone(),
    });
    Ok(result)
}

/// Declared switchback schedule; this route supports a zero-washout design only.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSwitchbackDesign {
    /// Sequence identity for each retained period.
    pub sequence_ids: Vec<String>,
    /// Period identity for each retained row; unique within its sequence.
    pub period_ids: Vec<String>,
    /// Known treatment probability for each retained period.
    pub treatment_probabilities: Vec<f64>,
    /// Identity of the assumption that carryover is absent after the declared washout.
    pub no_carryover_assumption_id: String,
    /// Identity of the declared washout rule, including a rule declaring zero washout.
    pub washout_rule_id: String,
    /// Number of post-switch periods excluded; only zero is currently supported.
    pub washout_periods: usize,
    /// Identity of the assumption that sequences have independent assignment and sampling.
    pub independent_sequences_assumption_id: String,
}

/// A validated switchback schedule and its pre-execution obligations.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledJointSwitchback {
    /// Declared schedule.
    pub design: JointSwitchbackDesign,
    /// Identity of schedule, assignment probabilities and temporal assumptions.
    pub identity: String,
    /// Obligations retained for inspection before estimation.
    pub obligations: Vec<&'static str>,
}

/// Compile the schedule and identification assumptions for a joint switchback.
///
/// # Errors
/// Refuses invalid or repeated sequence-period identities, extreme assignment probabilities,
/// unequal sequence lengths, nonzero washout, and missing temporal assumptions.
pub fn compile_joint_switchback(
    design: JointSwitchbackDesign,
) -> Result<CompiledJointSwitchback, EvidenceRefusal> {
    use std::collections::{BTreeMap, BTreeSet};
    let n = design.sequence_ids.len();
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut seen = BTreeSet::new();
    if n == 0
        || design.period_ids.len() != n
        || design.treatment_probabilities.len() != n
        || design.no_carryover_assumption_id.trim().is_empty()
        || design.washout_rule_id.trim().is_empty()
        || design.washout_periods != 0
        || design.independent_sequences_assumption_id.trim().is_empty()
    {
        return Err(EvidenceRefusal::Invalid("switchback needs aligned periods, declared temporal and independent-sequence assumptions, and zero washout".into()));
    }
    for ((sequence, period), p) in
        design.sequence_ids.iter().zip(&design.period_ids).zip(&design.treatment_probabilities)
    {
        if sequence.trim().is_empty()
            || period.trim().is_empty()
            || !p.is_finite()
            || *p < 0.2
            || *p > 0.8
            || !seen.insert((sequence, period))
        {
            return Err(EvidenceRefusal::Invalid("switchback periods must be unique within sequences with assignment probabilities from 0.2 to 0.8".into()));
        }
        *counts.entry(sequence).or_default() += 1;
    }
    if counts.len() < 30
        || counts.values().any(|c| *c < 2 || *c != *counts.values().next().unwrap())
    {
        return Err(EvidenceRefusal::Invalid("joint switchback requires at least 30 independent equal-length sequences of two or more retained periods".into()));
    }
    let mut w = antecedent_core::CanonicalWriter::new("antecedent.decision_evidence.switchback.v1");
    w.text(&design.no_carryover_assumption_id)
        .text(&design.washout_rule_id)
        .word(design.washout_periods as u64)
        .text(&design.independent_sequences_assumption_id)
        .word(design.sequence_ids.len() as u64);
    for ((sequence, period), p) in
        design.sequence_ids.iter().zip(&design.period_ids).zip(&design.treatment_probabilities)
    {
        w.text(sequence).text(period).real(*p);
    }
    Ok(CompiledJointSwitchback {
        identity: w.finish_hex(),
        design,
        obligations: vec![
            "sequences are independently assigned and sampled",
            "declared washout removes carryover",
            "no interference between sequences unless separately bounded",
            "the composed interval has no inherited support or calibration",
        ],
    })
}

/// Estimate joint switchback arm means using the existing sequence-level HT estimator.
///
/// The covariance is recovered from the existing estimator by polarization of its sequence
/// sandwich variance, including cross-arm and cross-outcome covariance. This does not license
/// the derived interval. The same schedule and assignments are used for every component.
///
/// # Errors
/// Refuses a modified design, missing exposure, misaligned data or inadequate switchback support.
#[allow(clippy::too_many_arguments)]
pub fn evidence_from_joint_switchback(
    design: &CompiledJointSwitchback,
    outcomes: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    if compile_joint_switchback(design.design.clone())?.identity != design.identity {
        return Err(EvidenceRefusal::Invalid("compiled switchback design was modified".into()));
    }
    if matches!(exposure, Exposure::Unspecified) {
        return Err(EvidenceRefusal::InterferenceUnspecified);
    }
    let n = assigned.len();
    let k = outcomes.len();
    if n != design.design.sequence_ids.len()
        || k == 0
        || outcomes.iter().any(|y| y.len() != n || y.iter().any(|v| !v.is_finite()))
    {
        return Err(EvidenceRefusal::Invalid(
            "switchback outcomes and assignments must align with the schedule".into(),
        ));
    }
    let ids: Vec<&str> = design.design.sequence_ids.iter().map(String::as_str).collect();
    let mut channels = Vec::with_capacity(2 * k);
    for treated in [true, false] {
        for y in outcomes {
            channels.push(
                y.iter()
                    .zip(assigned)
                    .map(|(v, a)| if *a == treated { if treated { *v } else { -v } } else { 0.0 })
                    .collect::<Vec<f64>>(),
            );
        }
    }
    let fit = |y: &[f64]| {
        antecedent_estimate::switchback::switchback_itt(
            y,
            assigned,
            &design.design.treatment_probabilities,
            &ids,
        )
    };
    let mut means = Vec::with_capacity(2 * k);
    let mut vars = Vec::with_capacity(2 * k);
    for y in &channels {
        let v = fit(y).ok_or_else(|| {
            EvidenceRefusal::Invalid("switchback estimator refused the schedule".into())
        })?;
        if v.interval_95.is_none() {
            return Err(EvidenceRefusal::Invalid(
                "switchback sequence support is insufficient for the existing estimator's interval"
                    .into(),
            ));
        }
        means.push(v.effect);
        vars.push(v.variance);
    }
    let d = 2 * k;
    let mut covariance = vec![0.0; d * d];
    for i in 0..d {
        covariance[i * d + i] = vars[i];
        for j in 0..i {
            let sum: Vec<f64> = channels[i].iter().zip(&channels[j]).map(|(a, b)| a + b).collect();
            let v = fit(&sum)
                .ok_or_else(|| {
                    EvidenceRefusal::Invalid("switchback joint covariance unavailable".into())
                })?
                .variance;
            let c = (v - vars[i] - vars[j]) / 2.0;
            covariance[i * d + j] = c;
            covariance[j * d + i] = c;
        }
    }
    let sequences = ids.iter().collect::<std::collections::BTreeSet<_>>().len();
    let mut ev = JointExperimentEvidence::new(
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        (means[..k].to_vec(), means[k..].to_vec()),
        Dependence::JointArms { covariance },
        DesignRecord {
            assignment_design: "switchback".into(),
            treated_units: sequences,
            control_units: sequences,
            outcome_rows: n,
        },
        vec![],
        exposure,
    )?;
    ev.design_contract_id = Some(design.identity.clone());
    ev.design_details = Some(DesignDetails::Switchback {
        sequence_ids: design.design.sequence_ids.clone(),
        period_ids: design.design.period_ids.clone(),
        realized_assignment: assigned.to_vec(),
        treatment_probabilities: design.design.treatment_probabilities.clone(),
        no_carryover_assumption_id: design.design.no_carryover_assumption_id.clone(),
        washout_rule_id: design.design.washout_rule_id.clone(),
        washout_periods: design.design.washout_periods,
        independent_sequences_assumption_id: design
            .design
            .independent_sequences_assumption_id
            .clone(),
    });
    Ok(ev)
}
