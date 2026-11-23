//! Declaration, validation and compatibility evidence for experimental designs.

use super::{
    EvidenceRefusal, Exposure, JointExperimentEvidence, OutcomeComponent,
    compile_and_evidence_from_joint_excursion, compile_and_evidence_from_joint_static_regime,
    compile_and_evidence_from_joint_switchback, compile_and_evidence_from_matched_pairs,
};

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
    /// Consistency and treatment-version assumption.
    pub consistency_assumption_id: String,
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
#[allow(clippy::float_cmp, reason = "the design requires exactly equal assignment probabilities")]
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
        || design.consistency_assumption_id.trim().is_empty()
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
        .text(&design.consistency_assumption_id)
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
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let retained = compile_and_evidence_from_matched_pairs(
        design,
        totals,
        assigned,
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        exposure,
        &ctx,
    )?;
    Ok(retained.evidence.clone())
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
    /// Consistency and treatment-version assumption.
    pub consistency_assumption_id: String,
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
        || design.consistency_assumption_id.trim().is_empty()
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
        || counts.values().next().is_none_or(|first| counts.values().any(|c| *c < 2 || c != first))
    {
        return Err(EvidenceRefusal::Invalid("joint switchback requires at least 30 independent equal-length sequences of two or more retained periods".into()));
    }
    let mut w = antecedent_core::CanonicalWriter::new("antecedent.decision_evidence.switchback.v1");
    w.text(&design.no_carryover_assumption_id)
        .text(&design.washout_rule_id)
        .word(design.washout_periods as u64)
        .text(&design.independent_sequences_assumption_id)
        .text(&design.consistency_assumption_id)
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
/// sandwich variance, including cross-arm and cross-outcome covariance. This does not support
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
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let retained = compile_and_evidence_from_joint_switchback(
        design,
        outcomes,
        assigned,
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        exposure,
        &ctx,
    )?;
    Ok(retained.evidence.clone())
}

/// Sequentially randomized unit-period schedule for one lagged excursion estimand.
#[derive(Clone, Debug, PartialEq)]
pub struct JointExcursionDesign {
    /// Independent unit identity for each observation row.
    pub unit_ids: Vec<String>,
    /// Integer period for each row; each unit-period pair must be unique.
    pub periods: Vec<i64>,
    /// Recorded probability of assigning treatment at each decision.
    pub treatment_probabilities: Vec<f64>,
    /// Outcome lag in periods; past and future treatment follow the recorded design.
    pub lag: u32,
    /// Declared positivity floor for both treatment arms.
    pub min_probability: f64,
    /// Identity of the known sequential randomization law.
    pub assignment_law_id: String,
    /// Assumption that units are independent for the uncertainty calculation.
    pub independent_units_assumption_id: String,
    /// Assumption that decisions without a lagged outcome are administrative end-of-follow-up.
    pub administrative_followup_assumption_id: String,
    /// Consistency and treatment-version assumption.
    pub consistency_assumption_id: String,
}

/// Prepared excursion schedule, identification obligations and compatible decision target.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledJointExcursion {
    /// Validated schedule.
    pub design: JointExcursionDesign,
    /// Identity of the schedule, assignment law, lag and assumptions.
    pub identity: String,
    /// Number of observed decision/outcome pairs.
    pub paired_decisions: usize,
    /// Number of independently sampled units with at least one paired decision.
    pub independent_units: usize,
    /// Identification and statistical obligations, not findings.
    pub obligations: Vec<&'static str>,
}

/// Compile a sequentially randomized excursion before fitting any outcome.
///
/// # Errors
/// Refuses invalid probability laws, repeated unit-periods, internal missing follow-up and
/// missing assignment or identification assumptions. A lagged excursion is not a sustained
/// intervention or a whole-regime contrast.
pub fn compile_joint_excursion(
    design: JointExcursionDesign,
) -> Result<CompiledJointExcursion, EvidenceRefusal> {
    use antecedent_estimate::temporal_design::{PanelRows, dense_labels, lagged_pairs};
    use std::collections::{BTreeMap, BTreeSet};
    let n = design.unit_ids.len();
    let invalid = |m: &str| EvidenceRefusal::Invalid(m.into());
    if n == 0
        || design.periods.len() != n
        || design.treatment_probabilities.len() != n
        || !design.min_probability.is_finite()
        || design.min_probability <= 0.0
        || design.min_probability > 0.5
        || design.assignment_law_id.trim().is_empty()
        || design.independent_units_assumption_id.trim().is_empty()
        || design.administrative_followup_assumption_id.trim().is_empty()
        || design.consistency_assumption_id.trim().is_empty()
        || design.unit_ids.iter().any(|id| id.trim().is_empty())
        || design.treatment_probabilities.iter().any(|p| {
            !p.is_finite() || *p < design.min_probability || 1.0 - *p < design.min_probability
        })
    {
        return Err(invalid(
            "excursion needs aligned units, periods and probabilities, positivity, and declared assignment, independence and follow-up assumptions",
        ));
    }
    let (dense, _) = dense_labels(&design.unit_ids);
    let pairs = lagged_pairs(PanelRows { unit: &dense, period: &design.periods }, design.lag)
        .map_err(|e| invalid(&e.to_string()))?;
    let mut max = BTreeMap::new();
    let mut observed = BTreeSet::new();
    for (unit, period) in design.unit_ids.iter().zip(&design.periods) {
        observed.insert((unit.as_str(), *period));
        max.entry(unit.as_str())
            .and_modify(|v: &mut i64| *v = (*v).max(*period))
            .or_insert(*period);
    }
    for (unit, period) in design.unit_ids.iter().zip(&design.periods) {
        let target = period
            .checked_add(i64::from(design.lag))
            .ok_or_else(|| invalid("outcome lag overflows the period"))?;
        if !observed.contains(&(unit.as_str(), target))
            && max.get(unit.as_str()).is_some_and(|m| target <= *m)
        {
            return Err(invalid(
                "missing internal lagged outcomes cannot be treated as administrative follow-up",
            ));
        }
    }
    let independent_units = pairs.decision.iter().map(|i| dense[*i]).collect::<BTreeSet<_>>().len();
    if independent_units < 2 {
        return Err(invalid(
            "excursion requires at least two independent units with paired decisions",
        ));
    }
    let mut w =
        antecedent_core::CanonicalWriter::new("antecedent.decision_evidence.excursion_design.v1");
    w.word(u64::from(design.lag))
        .real(design.min_probability)
        .text(&design.assignment_law_id)
        .text(&design.independent_units_assumption_id)
        .text(&design.administrative_followup_assumption_id)
        .text(&design.consistency_assumption_id)
        .word(n as u64);
    for ((u, t), p) in
        design.unit_ids.iter().zip(&design.periods).zip(&design.treatment_probabilities)
    {
        w.text(u).text(&t.to_string()).real(*p);
    }
    Ok(CompiledJointExcursion {
        design,
        identity: w.finish_hex(),
        paired_decisions: pairs.decision.len(),
        independent_units,
        obligations: vec![
            "the recorded law sequentially randomizes each decision conditional on history, with positivity for both arms",
            "units are independent and interference between units is absent",
            "unpaired outcomes occur only because of administrative end-of-follow-up",
            "the target is an excursion at the declared lag, with other decisions following the recorded law; it is not a sustained regime",
            "joint nonlinear uncertainty needs procedure-specific verification and calibration",
        ],
    })
}

/// Estimate several outcomes of the same sequentially randomized excursion jointly.
///
/// This compatibility function returns a historical numerical component. Use
/// [`compile_and_evidence_from_joint_excursion`] to retain the compiler source and admission.
///
/// This uses Antecedent's existing `lagged_pairs`, `two_arm_scores`, and cluster-robust ratio
/// covariance machinery. The independent unit, not the number of periods, determines interval
/// support. The joint transformation inherits neither support nor calibration from the inputs.
///
/// # Errors
/// Refuses modified designs, misaligned outcomes or assignments, missing arms, and interference.
#[allow(clippy::too_many_arguments)]
pub fn evidence_from_joint_excursion(
    design: &CompiledJointExcursion,
    outcomes: &[&[f64]],
    assigned: &[bool],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    horizon: u32,
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let retained = compile_and_evidence_from_joint_excursion(
        design,
        outcomes,
        assigned,
        evidence_id,
        population_id,
        regimes,
        horizon,
        components,
        exposure,
        &ctx,
    )?;
    Ok(retained.evidence.clone())
}

/// A declared observational comparison of two complete static treatment paths.
#[derive(Clone, Debug, PartialEq)]
pub struct JointStaticRegimeDesign {
    /// Subject identity for each observation row.
    pub unit_ids: Vec<String>,
    /// Period for each observation row.
    pub periods: Vec<i64>,
    /// Observed treatment at each decision row.
    pub realized_assignment: Vec<bool>,
    /// Identities of the measured predecision confounder columns.
    pub confounder_ids: Vec<String>,
    /// Full active intervention path.
    pub active_path: Vec<bool>,
    /// Full reference intervention path.
    pub reference_path: Vec<bool>,
    /// Existing temporal regime estimator: `g_formula`, `inverse_probability_weighting` or `doubly_robust`.
    pub method: String,
    /// Minimum treatment-action probability.
    pub min_probability: f64,
    /// Declared sequential positivity of both paths.
    pub positivity_assumption_id: String,
    /// Declared sequential exchangeability given the measured history.
    pub exchangeability_assumption_id: String,
    /// Declared consistency assumption.
    pub consistency_assumption_id: String,
    /// Declared complete-follow-up assumption.
    pub followup_assumption_id: String,
    /// Model-correctness premise for the selected g-formula, propensity or doubly robust method.
    pub model_assumption_id: String,
}

/// Prepared static-regime comparison and obligations, before fitting outcomes.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledJointStaticRegime {
    /// Validated temporal design.
    pub design: JointStaticRegimeDesign,
    /// Scientific design identity.
    pub identity: String,
    /// Independent subjects.
    pub units: usize,
    /// Identification and uncertainty obligations.
    pub obligations: Vec<&'static str>,
}

/// Compile two static treatment paths and their measured history.
///
/// # Errors
/// Refuses incomplete or mismatched subject schedules, invalid paths, methods or assumptions.
pub fn compile_joint_static_regime(
    design: JointStaticRegimeDesign,
) -> Result<CompiledJointStaticRegime, EvidenceRefusal> {
    use std::collections::{BTreeMap, BTreeSet};
    let invalid = |m: &str| EvidenceRefusal::Invalid(m.into());
    let n = design.unit_ids.len();
    if n == 0
        || design.periods.len() != n
        || design.realized_assignment.len() != n
        || design.active_path.is_empty()
        || design.active_path.len() != design.reference_path.len()
        || design.active_path == design.reference_path
        || !matches!(
            design.method.as_str(),
            "g_formula" | "inverse_probability_weighting" | "doubly_robust"
        )
        || !design.min_probability.is_finite()
        || design.min_probability <= 0.0
        || design.min_probability > 0.5
        || design.confounder_ids.is_empty()
        || design.confounder_ids.iter().any(|x| x.trim().is_empty())
        || design.confounder_ids.iter().collect::<BTreeSet<_>>().len()
            != design.confounder_ids.len()
        || design.positivity_assumption_id.trim().is_empty()
        || design.exchangeability_assumption_id.trim().is_empty()
        || design.consistency_assumption_id.trim().is_empty()
        || design.followup_assumption_id.trim().is_empty()
        || design.model_assumption_id.trim().is_empty()
    {
        return Err(invalid(
            "static regimes need aligned complete histories, distinct full paths, a supported method, positivity and declared identification assumptions",
        ));
    }
    let mut schedules = BTreeMap::<&str, Vec<i64>>::new();
    for (u, t) in design.unit_ids.iter().zip(&design.periods) {
        if u.trim().is_empty() {
            return Err(invalid("regime subject identity cannot be empty"));
        }
        schedules.entry(u).or_default().push(*t);
    }
    for s in schedules.values_mut() {
        s.sort_unstable();
    }
    if schedules.len() < 2
        || schedules.values().next().is_none_or(|first| {
            first.len() != design.active_path.len()
                || first.windows(2).any(|w| w[0] >= w[1])
                || schedules.values().any(|s| s != first)
        })
    {
        return Err(invalid("every subject must have the same complete, unique period schedule"));
    }
    let units = schedules.len();
    let mut w =
        antecedent_core::CanonicalWriter::new("antecedent.decision_evidence.static_regime.v1");
    w.text(&design.method)
        .real(design.min_probability)
        .text(&design.positivity_assumption_id)
        .text(&design.exchangeability_assumption_id)
        .text(&design.consistency_assumption_id)
        .text(&design.followup_assumption_id)
        .text(&design.model_assumption_id);
    w.word(n as u64);
    for (u, t) in design.unit_ids.iter().zip(&design.periods) {
        w.text(u).text(&t.to_string());
    }
    w.word(design.confounder_ids.len() as u64);
    for c in &design.confounder_ids {
        w.text(c);
    }
    w.word(design.active_path.len() as u64);
    for a in &design.active_path {
        w.word(u64::from(*a));
    }
    for a in &design.reference_path {
        w.word(u64::from(*a));
    }
    Ok(CompiledJointStaticRegime {
        design,
        identity: w.finish_hex(),
        units,
        obligations: vec![
            "sequential exchangeability given the complete measured treatment and confounder history",
            "sequential positivity, consistency, no between-subject interference and complete follow-up",
            "the target compares the declared whole static paths, not a single-decision excursion",
            "this point-only composition does not inherit marginal uncertainty, identification or support",
        ],
    })
}

/// Evaluate the existing static-regime point estimator for each declared outcome.
///
/// This compatibility function returns a historical numerical component. Use
/// [`compile_and_evidence_from_joint_static_regime`] to retain the compiler source and admission.
///
/// Results retain the actual whole paths and identification assumptions. Since separately
/// fitted regime outcomes have no aligned joint sampling distribution here, derived decisions
/// remain unresolved rather than inheriting marginal intervals.
///
/// # Errors
/// Refuses modified designs, misaligned trajectories, invalid probabilities and unsupported models.
#[allow(clippy::too_many_arguments)]
pub fn evidence_from_joint_static_regime(
    design: &CompiledJointStaticRegime,
    outcomes: &[&[f64]],
    confounders: &[Vec<f64>],
    evidence_id: &str,
    population_id: &str,
    regimes: (&str, &str),
    components: Vec<OutcomeComponent>,
    exposure: Exposure,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let retained = compile_and_evidence_from_joint_static_regime(
        design,
        outcomes,
        confounders,
        evidence_id,
        population_id,
        regimes,
        components,
        exposure,
        &ctx,
    )?;
    Ok(retained.evidence.clone())
}
