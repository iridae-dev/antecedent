//! Derived decision quantities and decision classification over joint experimental evidence.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

use super::expr;
use super::{
    DELTA_METHOD_STATEMENT, Dependence, EvidenceRefusal, Exposure, JointExperimentEvidence,
    MAX_UNKNOWN_DIMENSIONS, Request, SEPARATION_STATEMENT, Target, UtilityExpr, Z95,
};

/// What one input of a derived quantity's expression is, in each arm.
#[derive(Clone, Debug, PartialEq)]
pub enum InputBinding {
    /// The arm mean of a measured component.
    Measured(String),
    /// A value known without uncertainty.
    Fixed(f64),
    /// A value not measured, bounded by a declared assumption.
    Assumed {
        /// Lower end of the declared range.
        lower: f64,
        /// Upper end of the declared range.
        upper: f64,
        /// Identity of the declared assumption.
        assumption_id: String,
        /// Whether the arms share one value (one unknown) or each arm has its own (two).
        shared_across_arms: bool,
    },
    /// A value that is needed and unavailable.
    Missing {
        /// What is missing.
        what: String,
    },
}

/// A function of experimental outcomes, written in the decision contract's expression language.
///
/// It is evaluated in each arm at that arm's component means and the contrast is the treated
/// value minus the control value. This is a function of means, not the mean of a function.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedQuantity {
    /// Name.
    pub name: String,
    /// Declared units of the result.
    pub units: String,
    /// The expression; `Input(k)` reads `inputs[k]`.
    pub expr: UtilityExpr,
    /// Binding of each input.
    pub inputs: Vec<InputBinding>,
}

fn hash_expr(w: &mut CanonicalWriter, e: &UtilityExpr) {
    match e {
        UtilityExpr::Const(c) => {
            w.word(0).real(*c);
        }
        UtilityExpr::Input(k) => {
            w.word(1).word(*k as u64);
        }
        UtilityExpr::Neg(a) => {
            w.word(2);
            hash_expr(w, a);
        }
        UtilityExpr::Add(a, b) => {
            w.word(3);
            hash_expr(w, a);
            hash_expr(w, b);
        }
        UtilityExpr::Sub(a, b) => {
            w.word(4);
            hash_expr(w, a);
            hash_expr(w, b);
        }
        UtilityExpr::Mul(a, b) => {
            w.word(5);
            hash_expr(w, a);
            hash_expr(w, b);
        }
        UtilityExpr::Min(a, b) => {
            w.word(6);
            hash_expr(w, a);
            hash_expr(w, b);
        }
        UtilityExpr::Max(a, b) => {
            w.word(7);
            hash_expr(w, a);
            hash_expr(w, b);
        }
    }
}

impl DerivedQuantity {
    /// Identity of the transformation: expression, bindings and units.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.derived_quantity.v1");
        w.text(&self.name).text(&self.units);
        hash_expr(&mut w, &self.expr);
        for b in &self.inputs {
            match b {
                InputBinding::Measured(n) => {
                    w.word(0).text(n);
                }
                InputBinding::Fixed(v) => {
                    w.word(1).real(*v);
                }
                InputBinding::Assumed { lower, upper, assumption_id, shared_across_arms } => {
                    w.word(2)
                        .real(*lower)
                        .real(*upper)
                        .text(assumption_id)
                        .word(u64::from(*shared_across_arms));
                }
                InputBinding::Missing { what } => {
                    w.word(3).text(what);
                }
            }
        }
        w.finish_hex()
    }
}

/// Why a derived quantity or a decision is unresolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnresolvedReason {
    /// A needed input is neither measured nor bounded by an assumption.
    MissingInput {
        /// What is missing.
        what: String,
    },
    /// Dependence required for sampling propagation is unavailable, including between
    /// components or between the two regime values; it is not assumed away.
    JointLawMissing,
    /// Too few independently assigned units in an arm for a normal interval over the
    /// covariance. Adding rows inside existing units does not change this.
    TooFewIndependentUnits {
        /// Independently assigned units in the treated arm.
        treated: usize,
        /// Independently assigned units in the control arm.
        control: usize,
        /// Required in each arm.
        required: usize,
    },
    /// A kink or numerical instability prevents a reliable delta-method interval.
    SamplingUnavailable,
    /// Sampling uncertainty alone leaves the decision threshold inside the interval for some
    /// admissible assumption values.
    SamplingStraddlesThreshold,
}

/// A delta-method sampling interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingInterval {
    /// Value of the contrast the interval is centered on.
    pub estimate: f64,
    /// Standard error through the joint covariance.
    pub standard_error: f64,
    /// Lower end of the nominal 95% interval.
    pub lower: f64,
    /// Upper end of the nominal 95% interval.
    pub upper: f64,
}

/// The range of the contrast over a declared assumption set. Not a probability law.
#[derive(Clone, Debug, PartialEq)]
pub struct AssumedRange {
    /// Smallest contrast over the admissible assumptions.
    pub lower: f64,
    /// Largest contrast over the admissible assumptions.
    pub upper: f64,
    /// Exact (vertex enumeration of a multilinear function) or a guaranteed outer bound.
    pub exact: bool,
    /// Declared assumptions the range depends on.
    pub assumption_ids: Vec<String>,
    /// Number of unknown dimensions enumerated.
    pub dimensions: usize,
}

/// Result of a derived quantity.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedResult {
    /// Quantity name.
    pub name: String,
    /// Units.
    pub units: String,
    /// Arm values at the center of the assumption set (the measured point when none).
    pub treated_value: Option<f64>,
    /// Control value at the same point.
    pub control_value: Option<f64>,
    /// Contrast at that point.
    pub contrast: Option<f64>,
    /// Sampling interval at that point, if defined.
    pub sampling: Option<SamplingInterval>,
    /// Sampling interval at every vertex of the assumption set, when defined.
    pub sampling_at_vertices: Vec<SamplingInterval>,
    /// Range over declared assumptions, when any are in play.
    pub assumption_range: Option<AssumedRange>,
    /// Why no resolved answer exists, if so.
    pub unresolved: Vec<UnresolvedReason>,
    /// Standing of the result.
    pub standing: Standing,
    /// Support inherited: always none.
    pub support: &'static str,
    /// Calibration of the interval: always not_measured.
    pub calibration: &'static str,
    /// Statement about the interval construction.
    pub interval_statement: &'static str,
    /// Statement keeping sampling and assumption ranges apart.
    pub separation: &'static str,
    /// Identity of the transformation.
    pub quantity_identity: String,
}

/// How strong an answer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Standing {
    /// Sampling interval only.
    Sampled,
    /// A sampling interval and an assumption range.
    AssumptionDependent,
    /// Neither can be stated.
    Unresolved,
}

#[allow(
    clippy::float_cmp,
    reason = "the calculation or fixture checks exact discrete or boundary values"
)]
fn check_request(
    e: &JointExperimentEvidence,
    r: &Request,
) -> Result<Option<String>, EvidenceRefusal> {
    let valid_compiled_design = match (
        &*e.design.assignment_design,
        &e.design_details,
        &e.dependence,
    ) {
        (
            "matched_pairs",
            Some(super::DesignDetails::MatchedPairs {
                cluster_ids,
                pairs,
                matching_variables,
                treatment_probabilities,
                cluster_sizes,
                realized_assignment,
                preassignment_assumption_id,
                independent_pairs_assumption_id,
                consistency_assumption_id,
            }),
            Dependence::JointArms { .. },
        ) => {
            let n = cluster_ids.len();
            let mut seen = vec![false; n];
            let pairs_ok = pairs.iter().all(|&(a, b)| {
                if a >= n || b >= n || a == b || seen[a] || seen[b] {
                    return false;
                }
                seen[a] = true;
                seen[b] = true;
                realized_assignment
                    .get(a)
                    .zip(realized_assignment.get(b))
                    .is_some_and(|(x, y)| x != y)
            });
            n >= 4
                && pairs.len().checked_mul(2) == Some(n)
                && pairs_ok
                && cluster_ids.iter().all(|s| !s.trim().is_empty())
                && cluster_ids.iter().collect::<std::collections::BTreeSet<_>>().len() == n
                && !matching_variables.is_empty()
                && matching_variables.iter().all(|s| !s.trim().is_empty())
                && !preassignment_assumption_id.trim().is_empty()
                && !independent_pairs_assumption_id.trim().is_empty()
                && !consistency_assumption_id.trim().is_empty()
                && treatment_probabilities.len() == n
                && treatment_probabilities.iter().all(|p| *p == 0.5)
                && cluster_sizes.len() == n
                && !cluster_sizes.contains(&0)
                && realized_assignment.len() == n
                && cluster_sizes.iter().try_fold(0usize, |a, b| a.checked_add(*b))
                    == Some(e.design.outcome_rows)
                && e.design.treated_units == pairs.len()
                && e.design.control_units == pairs.len()
        }
        (
            "switchback",
            Some(super::DesignDetails::Switchback {
                sequence_ids,
                period_ids,
                treatment_probabilities,
                realized_assignment,
                no_carryover_assumption_id,
                washout_rule_id,
                washout_periods,
                independent_sequences_assumption_id,
                consistency_assumption_id,
            }),
            Dependence::JointArms { .. },
        ) => {
            let n = sequence_ids.len();
            let mut counts = std::collections::BTreeMap::new();
            let mut seen = std::collections::BTreeSet::new();
            let aligned = n > 0
                && period_ids.len() == n
                && treatment_probabilities.len() == n
                && realized_assignment.len() == n;
            let rows_ok = aligned
                && sequence_ids.iter().zip(period_ids).zip(treatment_probabilities).all(
                    |((s, p), prob)| {
                        if s.trim().is_empty()
                            || p.trim().is_empty()
                            || !prob.is_finite()
                            || *prob < 0.2
                            || *prob > 0.8
                            || !seen.insert((s, p))
                        {
                            return false;
                        }
                        *counts.entry(s).or_insert(0usize) += 1;
                        true
                    },
                );
            rows_ok
                && !no_carryover_assumption_id.trim().is_empty()
                && !washout_rule_id.trim().is_empty()
                && *washout_periods == 0
                && !independent_sequences_assumption_id.trim().is_empty()
                && !consistency_assumption_id.trim().is_empty()
                && counts
                    .values()
                    .next()
                    .is_some_and(|first| *first >= 2 && counts.values().all(|v| v == first))
                && e.design.treated_units == counts.len()
                && e.design.control_units == counts.len()
                && e.design.outcome_rows == n
                && realized_assignment.iter().filter(|a| **a).count() >= 30
                && realized_assignment.iter().filter(|a| !**a).count() >= 30
        }
        (
            "temporal_excursion",
            Some(super::DesignDetails::TemporalExcursion {
                unit_ids,
                periods,
                treatment_probabilities,
                realized_assignment,
                lag,
                min_probability,
                assignment_law_id,
                independent_units_assumption_id,
                administrative_followup_assumption_id,
                consistency_assumption_id,
                paired_decisions,
            }),
            Dependence::JointArms { .. },
        ) => {
            let n = unit_ids.len();
            let mut rows = std::collections::BTreeSet::new();
            let mut maxima = std::collections::BTreeMap::<&str, i64>::new();
            let aligned = n > 0
                && periods.len() == n
                && treatment_probabilities.len() == n
                && realized_assignment.len() == n;
            let valid = aligned
                && unit_ids.iter().zip(periods).zip(treatment_probabilities).all(|((u, t), p)| {
                    if u.trim().is_empty()
                        || !p.is_finite()
                        || *p < *min_probability
                        || (1.0 - *p) < *min_probability
                        || !rows.insert((u.as_str(), *t))
                    {
                        return false;
                    }
                    maxima.entry(u.as_str()).and_modify(|v| *v = (*v).max(*t)).or_insert(*t);
                    true
                });
            let mut paired = 0usize;
            let mut arm_counts = [0usize; 2];
            let mut units = std::collections::BTreeSet::new();
            let followup_ok = valid
                && unit_ids.iter().zip(periods).zip(realized_assignment).all(
                    |((u, t), assigned)| {
                        let Some(next) = t.checked_add(i64::from(*lag)) else { return false };
                        if rows.contains(&(u.as_str(), next)) {
                            paired += 1;
                            arm_counts[usize::from(*assigned)] += 1;
                            units.insert(u.as_str());
                            true
                        } else {
                            maxima.get(u.as_str()).is_some_and(|max| next > *max)
                        }
                    },
                );
            valid
                && followup_ok
                && min_probability.is_finite()
                && *min_probability > 0.0
                && *min_probability <= 0.5
                && !assignment_law_id.trim().is_empty()
                && !independent_units_assumption_id.trim().is_empty()
                && !administrative_followup_assumption_id.trim().is_empty()
                && !consistency_assumption_id.trim().is_empty()
                && paired == *paired_decisions
                && paired > 0
                && !arm_counts.contains(&0)
                && e.design.outcome_rows == n
                && e.design.treated_units == units.len()
                && e.design.control_units == units.len()
                && matches!(&e.exposure, Exposure::NoBetweenUnitInterference { assumption_id } if !assumption_id.trim().is_empty())
        }
        (
            "adaptive_regime",
            Some(super::DesignDetails::AdaptiveRegime {
                subject_ids,
                periods,
                policy_prespecification_assumption_id,
                positivity_assumption_id,
                exchangeability_assumption_id,
                consistency_assumption_id,
                followup_assumption_id,
                failures,
            }),
            Dependence::PointOnly | Dependence::JointArms { .. },
        ) => {
            subject_ids.len() >= 3
                && subject_ids.iter().all(|s| !s.trim().is_empty())
                && subject_ids.iter().collect::<std::collections::BTreeSet<_>>().len()
                    == subject_ids.len()
                && *periods == e.horizon
                && *periods > 0
                && e.design.treated_units == subject_ids.len()
                && e.design.control_units == subject_ids.len()
                && e.design.outcome_rows == subject_ids.len()
                && failures.len() == subject_ids.len()
                && (!matches!(e.dependence, Dependence::JointArms { .. })
                    || failures.iter().all(Option::is_none))
                && [
                    policy_prespecification_assumption_id,
                    positivity_assumption_id,
                    exchangeability_assumption_id,
                    consistency_assumption_id,
                    followup_assumption_id,
                ]
                .iter()
                .all(|s| !s.trim().is_empty())
                && matches!(&e.exposure, Exposure::NoBetweenUnitInterference { assumption_id } if !assumption_id.trim().is_empty())
        }
        (
            "temporal_regime",
            Some(super::DesignDetails::TemporalRegime {
                unit_ids,
                periods,
                realized_assignment,
                observed_minimum_probabilities,
                confounder_ids,
                active_path,
                reference_path,
                method,
                min_probability,
                positivity_assumption_id,
                exchangeability_assumption_id,
                consistency_assumption_id,
                followup_assumption_id,
                model_assumption_id,
                bootstrap_replicates,
            }),
            Dependence::PointOnly | Dependence::JointArms { .. },
        ) => {
            let n = unit_ids.len();
            let mut by_unit = std::collections::BTreeMap::<&str, Vec<i64>>::new();
            let aligned = n > 0 && periods.len() == n && realized_assignment.len() == n;
            if aligned {
                for (unit, period) in unit_ids.iter().zip(periods) {
                    by_unit.entry(unit).or_default().push(*period);
                }
            }
            for schedule in by_unit.values_mut() {
                schedule.sort_unstable();
            }
            let same = by_unit.values().next().is_some_and(|first| {
                first.len() == active_path.len()
                    && first.windows(2).all(|w| w[0] < w[1])
                    && by_unit.values().all(|v| v == first)
            });
            aligned
                && same
                && unit_ids.iter().all(|v| !v.trim().is_empty())
                && by_unit.len() >= 2
                && !confounder_ids.is_empty()
                && confounder_ids.iter().all(|v| !v.trim().is_empty())
                && active_path != reference_path
                && active_path.len() == reference_path.len()
                && active_path.len() == e.horizon as usize
                && matches!(
                    method.as_str(),
                    "g_formula" | "inverse_probability_weighting" | "doubly_robust"
                )
                && min_probability.is_finite()
                && *min_probability > 0.0
                && *min_probability <= 0.5
                && observed_minimum_probabilities.len() == e.components.len()
                && observed_minimum_probabilities.iter().all(|v| {
                    if method == "g_formula" {
                        v.is_none()
                    } else {
                        v.is_some_and(|p| p.is_finite() && p >= *min_probability)
                    }
                })
                && !positivity_assumption_id.trim().is_empty()
                && !exchangeability_assumption_id.trim().is_empty()
                && !consistency_assumption_id.trim().is_empty()
                && !followup_assumption_id.trim().is_empty()
                && !model_assumption_id.trim().is_empty()
                && match &e.dependence {
                    Dependence::PointOnly => bootstrap_replicates.is_none_or(|r| r >= 99),
                    Dependence::JointArms { .. } => bootstrap_replicates.is_some_and(|r| r >= 99),
                    _ => false,
                }
                && e.design.outcome_rows == n
                && e.design.treated_units == by_unit.len()
                && e.design.control_units == by_unit.len()
                && matches!(&e.exposure, Exposure::NoBetweenUnitInterference { assumption_id } if !assumption_id.trim().is_empty())
        }
        (
            "matched_pairs" | "switchback" | "temporal_excursion" | "temporal_regime"
            | "adaptive_regime",
            ..,
        ) => false,
        _ => true,
    };
    if !valid_compiled_design
        || (matches!(
            e.design.assignment_design.as_str(),
            "matched_pairs"
                | "switchback"
                | "temporal_excursion"
                | "temporal_regime"
                | "adaptive_regime"
        ) && e.design_contract_id.as_deref().is_none_or(|id| id.trim().is_empty()))
    {
        return Err(EvidenceRefusal::Invalid("experimental evidence must retain a consistent compiled design and joint arm dependence".into()));
    }
    if r.population_id != e.population_id {
        return Err(EvidenceRefusal::IncompatiblePopulation {
            requested: r.population_id.clone(),
            available: e.population_id.clone(),
        });
    }
    if (&r.treated_regime_id, &r.control_regime_id) != (&e.treated_regime_id, &e.control_regime_id)
    {
        return Err(EvidenceRefusal::IncompatibleIntervention {
            requested: format!("{}/{}", r.treated_regime_id, r.control_regime_id),
            available: format!("{}/{}", e.treated_regime_id, e.control_regime_id),
        });
    }
    if r.horizon != e.horizon {
        return Err(EvidenceRefusal::IncompatibleHorizon {
            requested: r.horizon,
            available: e.horizon,
        });
    }
    if matches!(e.exposure, Exposure::Unspecified) {
        return Err(EvidenceRefusal::InterferenceUnspecified);
    }
    match (&e.design_details, &r.target) {
        (Some(super::DesignDetails::AdaptiveRegime { periods, exchangeability_assumption_id, .. }), Target::AdaptiveRegime { periods: requested_periods, exchangeability_assumption_id: requested }) if periods == requested_periods && exchangeability_assumption_id == requested => Ok(None),
        (Some(super::DesignDetails::AdaptiveRegime { .. }), _) | (_, Target::AdaptiveRegime { .. }) => Err(EvidenceRefusal::Invalid("adaptive regime evidence requires the same policy horizon and identification assumption".into())),
        (Some(super::DesignDetails::TemporalExcursion { lag, assignment_law_id, .. }), Target::Excursion { lag: requested_lag, assignment_law_id: requested_law }) if lag == requested_lag && assignment_law_id == requested_law => Ok(None),
        (Some(super::DesignDetails::TemporalRegime {active_path,exchangeability_assumption_id,..}), Target::Regime {periods,exchangeability_assumption_id:requested}) if active_path.len()==*periods as usize && exchangeability_assumption_id==requested => Ok(None),
        (Some(super::DesignDetails::TemporalRegime { .. }), _) | (_, Target::Regime { .. }) => Err(EvidenceRefusal::Invalid("whole-regime evidence requires the same regime horizon and identification assumption".into())),
        (Some(super::DesignDetails::TemporalExcursion { .. }), _) | (_, Target::Excursion { .. }) => Err(EvidenceRefusal::Invalid("excursion evidence requires the same lag and sequential assignment law; a different temporal estimand needs an explicit bridge".into())),
        (_, Target::ExperimentalContrast) => Ok(None),
        (_, Target::Deployment { bridge_assumption_id: Some(id) }) => Ok(Some(id.clone())),
        (_, Target::Deployment { bridge_assumption_id: None }) => {
            Err(EvidenceRefusal::DeploymentBridgeMissing)
        }
    }
}

/// One unknown dimension.
struct Dim {
    lower: f64,
    upper: f64,
    /// Inputs (index) affected; for a shared assumption both arms; spillover: treated only.
    kind: DimKind,
}
enum DimKind {
    Shared(usize),
    Treated(usize),
    Control(usize),
    /// Additive offset on the treated arm mean of a component.
    Spill(usize),
    ControlSpill(usize),
}

/// Evaluate a derived quantity over joint evidence.
///
/// # Errors
///
/// [`EvidenceRefusal`] when the request does not match the evidence, exposure is unspecified, a
/// deployment target has no declared bridge, a binding is malformed, or there are too many
/// unknown dimensions.
pub fn derive(
    evidence: &JointExperimentEvidence,
    request: &Request,
    quantity: &DerivedQuantity,
) -> Result<DerivedResult, EvidenceRefusal> {
    let bridge = check_request(evidence, request)?;
    // Bounds on differences alone do not bound either arm mean. A nonlinear
    // function of arm means therefore cannot be recovered from such bounds.
    if matches!(evidence.exposure, Exposure::BoundedSpillover { .. })
        && !expr::is_affine(&quantity.expr)
    {
        return Err(EvidenceRefusal::Invalid(
            "nonlinear quantities require bounds on arm means, not only spillover bias in contrasts".into(),
        ));
    }
    // Adjusted contrasts preserve differences, but uncentered adjusted arm
    // means need not estimate the original arm means used by nonlinear functions.
    if evidence
        .adjustments
        .iter()
        .any(|a| matches!(a.kind, super::AdjustmentKind::FixedCuped { .. }))
        && !expr::is_affine(&quantity.expr)
    {
        return Err(EvidenceRefusal::Invalid(
            "nonlinear quantities require arm means restored to the original outcome scale after adjustment".into(),
        ));
    }
    let n = quantity.inputs.len();
    let bad = |m: String| Err(EvidenceRefusal::Invalid(m));
    let mut max_input = 0usize;
    fn scan(e: &UtilityExpr, m: &mut usize) {
        match e {
            UtilityExpr::Input(k) => *m = (*m).max(*k + 1),
            UtilityExpr::Const(_) => {}
            UtilityExpr::Neg(a) => scan(a, m),
            UtilityExpr::Add(a, b)
            | UtilityExpr::Sub(a, b)
            | UtilityExpr::Mul(a, b)
            | UtilityExpr::Min(a, b)
            | UtilityExpr::Max(a, b) => {
                scan(a, m);
                scan(b, m);
            }
        }
    }
    scan(&quantity.expr, &mut max_input);
    if max_input > n {
        return bad(format!("expression reads input {} but only {n} are bound", max_input - 1));
    }

    let mut unresolved = Vec::new();
    let mut assumption_ids: Vec<String> = bridge.into_iter().collect();
    let mut dims: Vec<Dim> = Vec::new();
    // Component slot read by each input (None when not measured).
    let mut slot: Vec<Option<usize>> = vec![None; n];
    let (mut base_t, mut base_c) = (vec![0.0; n], vec![0.0; n]);
    for (k, b) in quantity.inputs.iter().enumerate() {
        match b {
            InputBinding::Measured(name) => {
                let Some(i) = evidence.component_index(name) else {
                    return bad(format!("input {k} names unknown component `{name}`"));
                };
                slot[k] = Some(i);
                base_t[k] = evidence.treated_means[i];
                base_c[k] = evidence.control_means[i];
            }
            InputBinding::Fixed(v) => {
                if !v.is_finite() {
                    return bad(format!("input {k} is not finite"));
                }
                base_t[k] = *v;
                base_c[k] = *v;
            }
            InputBinding::Assumed { lower, upper, assumption_id, shared_across_arms } => {
                if !(lower.is_finite() && upper.is_finite()) || lower > upper {
                    return bad(format!("input {k} has an invalid assumed range"));
                }
                assumption_ids.push(assumption_id.clone());
                let mid = 0.5 * (lower + upper);
                base_t[k] = mid;
                base_c[k] = mid;
                if upper > lower {
                    if *shared_across_arms {
                        dims.push(Dim { lower: *lower, upper: *upper, kind: DimKind::Shared(k) });
                    } else {
                        dims.push(Dim { lower: *lower, upper: *upper, kind: DimKind::Treated(k) });
                        dims.push(Dim { lower: *lower, upper: *upper, kind: DimKind::Control(k) });
                    }
                }
            }
            InputBinding::Missing { what } => {
                unresolved.push(UnresolvedReason::MissingInput { what: what.clone() })
            }
        }
    }
    if let Exposure::BoundedSpillover { assumption_id, bias_bounds } = &evidence.exposure {
        let mut used = false;
        for (c, b) in bias_bounds.iter().enumerate() {
            if *b > 0.0 && slot.contains(&Some(c)) {
                dims.push(Dim { lower: -b, upper: *b, kind: DimKind::Spill(c) });
                used = true;
            }
        }
        if used {
            assumption_ids.push(assumption_id.clone());
        }
    }
    if let Exposure::BoundedArmSpillover {
        assumption_id,
        treated_bias_bounds,
        control_bias_bounds,
    } = &evidence.exposure
    {
        let mut used = false;
        for (c, (&tb, &cb)) in treated_bias_bounds.iter().zip(control_bias_bounds).enumerate() {
            if slot.contains(&Some(c)) {
                if tb > 0.0 {
                    dims.push(Dim { lower: -tb, upper: tb, kind: DimKind::Spill(c) });
                    used = true;
                }
                if cb > 0.0 {
                    dims.push(Dim { lower: -cb, upper: cb, kind: DimKind::ControlSpill(c) });
                    used = true;
                }
            }
        }
        if used {
            assumption_ids.push(assumption_id.clone());
        }
    }
    assumption_ids.sort();
    assumption_ids.dedup();
    if dims.len() > MAX_UNKNOWN_DIMENSIONS {
        return Err(EvidenceRefusal::TooManyUnknowns(dims.len()));
    }
    let identity = quantity.identity();
    let finish = |unresolved: Vec<UnresolvedReason>,
                  point: Option<(f64, f64)>,
                  sampling: Option<SamplingInterval>,
                  vertices: Vec<SamplingInterval>,
                  range: Option<AssumedRange>| {
        let standing = if !unresolved.is_empty() {
            Standing::Unresolved
        } else if range.is_some() {
            Standing::AssumptionDependent
        } else {
            Standing::Sampled
        };
        DerivedResult {
            name: quantity.name.clone(),
            units: quantity.units.clone(),
            treated_value: point.map(|p| p.0),
            control_value: point.map(|p| p.1),
            contrast: point.map(|p| p.0 - p.1),
            sampling,
            sampling_at_vertices: vertices,
            assumption_range: range,
            unresolved,
            standing,
            support: "none",
            calibration: "not_measured",
            interval_statement: match evidence.design.assignment_design.as_str() {
                "matched_pairs" => {
                    "first-order delta method through the paired-cluster covariance, with a Student reference using pairs minus one degrees of freedom; calibration is not_measured and no component support is inherited"
                }
                "switchback" => {
                    "first-order delta method through the sequence-level switchback covariance, with a Student reference using sequences minus one degrees of freedom; calibration is not_measured and no component support is inherited"
                }
                "temporal_regime" => {
                    "first-order delta method through aligned whole-subject bootstrap covariance with a normal reference; calibration is not_measured and no component support is inherited"
                }
                "adaptive_regime" => {
                    "first-order delta method through aligned delete-subject jackknife covariance with a normal reference; calibration is not_measured and no component support is inherited"
                }
                _ => DELTA_METHOD_STATEMENT,
            },
            separation: SEPARATION_STATEMENT,
            quantity_identity: identity.clone(),
        }
    };
    if !unresolved.is_empty() {
        return Ok(finish(unresolved, None, None, Vec::new(), None));
    }

    // Apply a vertex (bit mask) or the midpoint to the arm input vectors.
    let apply = |mask: Option<usize>| -> (Vec<f64>, Vec<f64>) {
        let (mut t, mut c) = (base_t.clone(), base_c.clone());
        for (d, dim) in dims.iter().enumerate() {
            let v = match mask {
                Some(m) => {
                    if (m >> d) & 1 == 1 {
                        dim.upper
                    } else {
                        dim.lower
                    }
                }
                None => 0.5 * (dim.lower + dim.upper),
            };
            match dim.kind {
                DimKind::Shared(k) => {
                    t[k] = v;
                    c[k] = v;
                }
                DimKind::Treated(k) => t[k] = v,
                DimKind::Control(k) => c[k] = v,
                DimKind::Spill(comp) | DimKind::ControlSpill(comp) => {
                    for (k, s) in slot.iter().enumerate() {
                        if *s == Some(comp) {
                            if matches!(dim.kind, DimKind::Spill(_)) {
                                t[k] += v;
                            } else {
                                c[k] += v;
                            }
                        }
                    }
                }
            }
        }
        (t, c)
    };
    let sampling_at = |t: &[f64], c: &[f64]| -> Option<SamplingInterval> {
        let (vt, gt) = expr::value_and_gradient(&quantity.expr, t)?;
        let (vc, gc) = expr::value_and_gradient(&quantity.expr, c)?;
        let k = evidence.components.len();
        let (mut ht, mut hc) = (vec![0.0; k], vec![0.0; k]);
        for (i, s) in slot.iter().enumerate() {
            if let Some(s) = s {
                ht[*s] += gt[i];
                hc[*s] += gc[i];
            }
        }
        let quad = |h: &[f64], m: &[f64], full: bool| -> f64 {
            if full {
                (0..k).map(|i| (0..k).map(|j| h[i] * m[i * k + j] * h[j]).sum::<f64>()).sum()
            } else {
                (0..k).map(|i| h[i] * h[i] * m[i]).sum()
            }
        };
        let var = match &evidence.dependence {
            Dependence::Joint { treated, control } => {
                quad(&ht, treated, true) + quad(&hc, control, true)
            }
            Dependence::JointArms { covariance } => {
                let mut h = ht.clone();
                h.extend(hc.iter().map(|v| -v));
                let d = 2 * k;
                let mut value = 0.0;
                let mut magnitude = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        let term = h[i] * covariance[i * d + j] * h[j];
                        value += term;
                        magnitude += term.abs();
                    }
                }
                // A near-zero residual after subtracting much larger covariance terms is
                // numerically unreliable. Withhold the interval rather than clip a negative
                // rounding error to zero and present spurious precision.
                if !value.is_finite()
                    || !magnitude.is_finite()
                    || value < 0.0
                    || (magnitude > 0.0 && value <= 64.0 * f64::EPSILON * magnitude)
                {
                    return None;
                }
                value
            }
            Dependence::PointOnly => return None,
            Dependence::MarginalsOnly { treated, control } => {
                quad(&ht, treated, false) + quad(&hc, control, false)
            }
        };
        let se = var.max(0.0).sqrt();
        let est = vt - vc;
        let critical =
            if matches!(evidence.design.assignment_design.as_str(), "matched_pairs" | "switchback")
            {
                antecedent_stats::student_t_ppf(0.975, (evidence.design.treated_units - 1) as f64)
            } else {
                Z95
            };
        Some(SamplingInterval {
            estimate: est,
            standard_error: se,
            lower: est - critical * se,
            upper: est + critical * se,
        })
    };

    // Components read by the expression at all.
    let mut read: Vec<usize> = slot.iter().flatten().copied().collect();
    read.sort_unstable();
    read.dedup();
    let marginals_only_combination =
        matches!(evidence.dependence, Dependence::MarginalsOnly { .. }) && read.len() > 1;

    let (mt, mc) = apply(None);
    let point = match (expr::value(&quantity.expr, &mt), expr::value(&quantity.expr, &mc)) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => {
            return Err(EvidenceRefusal::Invalid(
                "the expression is not finite at the evidence".into(),
            ));
        }
    };
    let mut unresolved = Vec::new();
    let few_units = evidence.design.treated_units.min(evidence.design.control_units)
        < super::MIN_UNITS_PER_ARM_FOR_INTERVAL;
    if few_units {
        unresolved.push(UnresolvedReason::TooFewIndependentUnits {
            treated: evidence.design.treated_units,
            control: evidence.design.control_units,
            required: super::MIN_UNITS_PER_ARM_FOR_INTERVAL,
        });
    }
    let (sampling, vertices) = if few_units {
        (None, Vec::new())
    } else if matches!(evidence.dependence, Dependence::PointOnly) || marginals_only_combination {
        unresolved.push(UnresolvedReason::JointLawMissing);
        (None, Vec::new())
    } else if expr::has_kink(&quantity.expr) {
        unresolved.push(UnresolvedReason::SamplingUnavailable);
        (None, Vec::new())
    } else {
        let mid = sampling_at(&mt, &mc);
        let mut vs = Vec::new();
        if !dims.is_empty() {
            for mask in 0..(1usize << dims.len()) {
                let (t, c) = apply(Some(mask));
                if let Some(s) = sampling_at(&t, &c) {
                    vs.push(s);
                }
            }
        }
        if mid.is_none() || (!dims.is_empty() && vs.len() != (1usize << dims.len())) {
            unresolved.push(UnresolvedReason::SamplingUnavailable);
            (None, Vec::new())
        } else {
            (mid, vs)
        }
    };

    let range = if dims.is_empty() {
        None
    } else {
        let read_once_per_component = {
            let mut seen = std::collections::BTreeSet::new();
            slot.iter().flatten().all(|c| seen.insert(*c))
        };
        let exact = expr::is_multilinear(&quantity.expr) && read_once_per_component;
        let (lo, hi) = if exact {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for mask in 0..(1usize << dims.len()) {
                let (t, c) = apply(Some(mask));
                let v = expr::value(&quantity.expr, &t)
                    .zip(expr::value(&quantity.expr, &c))
                    .map(|(a, b)| a - b)
                    .ok_or_else(|| {
                        EvidenceRefusal::Invalid(
                            "expression is not finite on the assumption set".into(),
                        )
                    })?;
                lo = lo.min(v);
                hi = hi.max(v);
            }
            (lo, hi)
        } else {
            let (mut lt, mut ut, mut lc, mut uc) =
                (base_t.clone(), base_t.clone(), base_c.clone(), base_c.clone());
            for dim in &dims {
                match dim.kind {
                    DimKind::Shared(k) => {
                        (lt[k], ut[k], lc[k], uc[k]) = (dim.lower, dim.upper, dim.lower, dim.upper);
                    }
                    DimKind::Treated(k) => (lt[k], ut[k]) = (dim.lower, dim.upper),
                    DimKind::Control(k) => (lc[k], uc[k]) = (dim.lower, dim.upper),
                    DimKind::Spill(comp) | DimKind::ControlSpill(comp) => {
                        for (k, s) in slot.iter().enumerate() {
                            if *s == Some(comp) {
                                if matches!(dim.kind, DimKind::Spill(_)) {
                                    lt[k] += dim.lower;
                                    ut[k] += dim.upper;
                                } else {
                                    lc[k] += dim.lower;
                                    uc[k] += dim.upper;
                                }
                            }
                        }
                    }
                }
            }
            let (a, b) = expr::enclosure(&quantity.expr, &lt, &ut)
                .zip(expr::enclosure(&quantity.expr, &lc, &uc))
                .map(|(t, c)| ((t.0, t.1), (c.0, c.1)))
                .ok_or_else(|| {
                    EvidenceRefusal::Invalid(
                        "expression is not finite on the assumption set".into(),
                    )
                })?;
            (a.0 - b.1, a.1 - b.0)
        };
        Some(AssumedRange {
            lower: lo,
            upper: hi,
            exact,
            assumption_ids: assumption_ids.clone(),
            dimensions: dims.len(),
        })
    };
    Ok(finish(unresolved, point, sampling, vertices, range))
}

/// The two actions a decision compares and the rule that separates them.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionFrame {
    /// Action implemented by the treated regime.
    pub treated_action: String,
    /// Action implemented by the control regime.
    pub control_action: String,
    /// The treated action is preferred when the contrast exceeds this threshold.
    pub threshold: f64,
}

impl DecisionFrame {
    /// Identity of the decision rule (not of the evidence).
    #[must_use]
    pub fn rule_identity(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.decision_evidence.decision_rule.v1");
        w.text(&self.treated_action).text(&self.control_action).real(self.threshold);
        w.finish_hex()
    }
}

/// Which action a decision favors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arm {
    /// The treated action.
    Treated,
    /// The control action.
    Control,
}

/// Classification of a decision.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// One action is preferred under sampling uncertainty and under every declared assumption.
    Robust {
        /// Preferred action name.
        action: String,
    },
    /// The sampled interval favors different actions at vertices of the declared range.
    /// Intermediate assumption values may still leave the decision unresolved.
    AssumptionDependent {
        /// Assumptions the preference depends on.
        assumption_ids: Vec<String>,
        /// Preferred action at the lowest contrast.
        at_lower: String,
        /// Preferred action at the highest contrast.
        at_upper: String,
    },
    /// The decision cannot be stated; the reasons say what is missing.
    Unresolved {
        /// Reasons.
        reasons: Vec<UnresolvedReason>,
    },
}

/// A decision and the derived quantity it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResult {
    /// The frame.
    pub frame: DecisionFrame,
    /// The evidence it rests on.
    pub derived: DerivedResult,
    /// The classification.
    pub verdict: Verdict,
    /// Statement that a classification is not a probability.
    pub statement: &'static str,
}

/// Classify a decision over a derived quantity.
///
/// # Errors
///
/// Those of [`derive`].
pub fn decide(
    evidence: &JointExperimentEvidence,
    request: &Request,
    quantity: &DerivedQuantity,
    frame: &DecisionFrame,
) -> Result<DecisionResult, EvidenceRefusal> {
    if frame.treated_action == frame.control_action || !frame.threshold.is_finite() {
        return Err(EvidenceRefusal::FrameShape);
    }
    let derived = derive(evidence, request, quantity)?;
    let t = frame.threshold;
    let side = |lo: f64, hi: f64| -> Option<bool> {
        if lo > t {
            Some(true)
        } else if hi < t {
            Some(false)
        } else {
            None
        }
    };
    let name = |treated: bool| {
        if treated { frame.treated_action.clone() } else { frame.control_action.clone() }
    };
    let verdict = if !derived.unresolved.is_empty() {
        Verdict::Unresolved { reasons: derived.unresolved.clone() }
    } else {
        let intervals: Vec<&SamplingInterval> = if derived.sampling_at_vertices.is_empty() {
            derived.sampling.iter().collect()
        } else {
            derived.sampling_at_vertices.iter().collect()
        };
        let sides: Vec<Option<bool>> = intervals.iter().map(|s| side(s.lower, s.upper)).collect();
        if derived.assumption_range.as_ref().is_some_and(|r| !r.exact)
            || sides.is_empty()
            || sides.iter().any(Option::is_none)
        {
            Verdict::Unresolved { reasons: vec![UnresolvedReason::SamplingStraddlesThreshold] }
        } else {
            let all_true = sides.iter().all(|s| *s == Some(true));
            let all_false = sides.iter().all(|s| *s == Some(false));
            match &derived.assumption_range {
                Some(r) if !(all_true || all_false) || side(r.lower, r.upper).is_none() => {
                    Verdict::AssumptionDependent {
                        assumption_ids: r.assumption_ids.clone(),
                        at_lower: name(r.lower > t),
                        at_upper: name(r.upper > t),
                    }
                }
                _ => Verdict::Robust { action: name(all_true) },
            }
        }
    };
    Ok(DecisionResult {
        frame: frame.clone(),
        derived,
        verdict,
        statement: "a classification over declared assumptions; it is not a probability and no assumption range was turned into one",
    })
}

/// Describe the exact decision-evidence interval construction for calibration measurement.
/// This is a candidate basis only: it does not support or attest an interval.
/// The scope count is the number of independently randomized units, not outcome rows.
///
/// # Errors
/// Returns a refusal when no sampling interval is produced, or when an assumption range
/// would require a different calibration target.
pub fn decision_interval_calibration_basis(
    evidence: &JointExperimentEvidence,
    request: &Request,
    quantity: &DerivedQuantity,
) -> Result<antecedent_core::CalibrationBasis, EvidenceRefusal> {
    let result = derive(evidence, request, quantity)?;
    if result.sampling.is_none()
        || result.assumption_range.is_some()
        || !matches!(
            (&*evidence.design.assignment_design, &request.target),
            ("cluster" | "complete" | "matched_pairs" | "switchback", Target::ExperimentalContrast)
                | ("temporal_excursion", Target::Excursion { .. })
                | ("temporal_regime", Target::Regime { .. })
                | ("adaptive_regime", Target::AdaptiveRegime { .. })
        )
        || !matches!(evidence.exposure, Exposure::NoBetweenUnitInterference { .. })
        || !matches!(
            (evidence.design.assignment_design.as_str(), &evidence.dependence),
            ("cluster" | "complete", Dependence::Joint { .. })
                | (
                    "matched_pairs"
                        | "switchback"
                        | "temporal_excursion"
                        | "temporal_regime"
                        | "adaptive_regime",
                    Dependence::JointArms { .. }
                )
        )
        || quantity.inputs.iter().any(|b| !matches!(b, InputBinding::Measured(_)))
    {
        return Err(EvidenceRefusal::Invalid(
            "calibration basis requires a sampling interval without an assumption range".into(),
        ));
    }
    let key = [
        "DecisionEvidence".to_owned(),
        if matches!(
            evidence.design.assignment_design.as_str(),
            "temporal_regime" | "adaptive_regime"
        ) {
            "Observational"
        } else {
            "RandomizedTrial"
        }
        .to_owned(),
        "design".to_owned(),
        "tabular".to_owned(),
        "Frequentist".to_owned(),
        format!("joint_{}_{}", evidence.design.assignment_design, evidence.adjustment_identity()),
        if matches!(evidence.design.assignment_design.as_str(), "matched_pairs" | "switchback") {
            "delta_method_student_95"
        } else {
            "delta_method_normal_95"
        }
        .to_owned(),
        match evidence.design.assignment_design.as_str() {
            "matched_pairs" => "matched_pair_delta",
            "switchback" => "switchback_sequence_delta",
            "temporal_excursion" => "temporal_cluster_ratio_delta",
            "temporal_regime" => "temporal_regime_bootstrap_delta",
            "adaptive_regime" => "adaptive_regime_jackknife_delta",
            _ => "joint_neyman_delta",
        }
        .to_owned(),
        if matches!(
            evidence.design.assignment_design.as_str(),
            "temporal_regime" | "adaptive_regime"
        ) {
            "independent_subjects"
        } else {
            "independent_assignment_units"
        }
        .to_owned(),
        String::new(),
        match &request.target {
            Target::Excursion { lag, assignment_law_id } => {
                let mut w = CanonicalWriter::new(
                    "antecedent.decision_evidence.excursion_calibration_target.v1",
                );
                w.text(&result.quantity_identity).word(u64::from(*lag)).text(assignment_law_id);
                format!("excursion_{}", w.finish_hex())
            }
            Target::AdaptiveRegime { periods, exchangeability_assumption_id } => {
                let mut w = CanonicalWriter::new(
                    "antecedent.decision_evidence.adaptive_regime_calibration_target.v1",
                );
                w.text(&result.quantity_identity)
                    .word(u64::from(*periods))
                    .text(exchangeability_assumption_id);
                format!("adaptive_regime_{}", w.finish_hex())
            }
            Target::Regime { periods, exchangeability_assumption_id } => {
                let Some(super::DesignDetails::TemporalRegime {
                    method,
                    bootstrap_replicates: Some(replicates),
                    ..
                }) = &evidence.design_details
                else {
                    return Err(EvidenceRefusal::Invalid(
                        "regime calibration requires aligned bootstrap metadata".into(),
                    ));
                };
                let mut w = CanonicalWriter::new(
                    "antecedent.decision_evidence.regime_calibration_target.v1",
                );
                w.text(&result.quantity_identity)
                    .word(u64::from(*periods))
                    .text(exchangeability_assumption_id)
                    .text(method)
                    .word(*replicates as u64);
                format!("regime_{}", w.finish_hex())
            }
            _ => format!("derived_{}", result.quantity_identity),
        },
    ]
    .map(std::sync::Arc::<str>::from);
    Ok(antecedent_core::CalibrationBasis::new(
        key,
        0.95,
        "point".into(),
        (if matches!(
            evidence.design.assignment_design.as_str(),
            "matched_pairs"
                | "switchback"
                | "temporal_excursion"
                | "temporal_regime"
                | "adaptive_regime"
        ) {
            evidence.design.treated_units
        } else {
            evidence.design.treated_units + evidence.design.control_units
        }) as u64,
        None,
        None,
        0.0,
    )
    .with_validation("none"))
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::*;
    use super::*;

    pub(in super::super) fn evidence(
        t_cov_scale: f64,
        exposure: Exposure,
    ) -> JointExperimentEvidence {
        let comps = ["frequency", "value", "cost"]
            .iter()
            .map(|n| OutcomeComponent {
                name: (*n).into(),
                variable_id: format!("v.{n}"),
                units: "u".into(),
            })
            .collect();
        // Arm means; frequency and value are negatively correlated in the treated arm.
        let treated = vec![
            0.0004 * t_cov_scale,
            -0.0001 * t_cov_scale,
            0.0,
            -0.0001 * t_cov_scale,
            0.04 * t_cov_scale,
            0.0,
            0.0,
            0.0,
            0.01 * t_cov_scale,
        ];
        let control = vec![
            0.0003 * t_cov_scale,
            0.0,
            0.0,
            0.0,
            0.03 * t_cov_scale,
            0.0,
            0.0,
            0.0,
            0.01 * t_cov_scale,
        ];
        JointExperimentEvidence::new(
            "exp-1",
            "pop",
            ("ship", "hold"),
            4,
            comps,
            (vec![2.2, 12.0, 6.0], vec![2.0, 11.0, 6.0]),
            Dependence::Joint { treated, control },
            DesignRecord {
                assignment_design: "cluster".into(),
                treated_units: 30,
                control_units: 30,
                outcome_rows: 480,
            },
            vec![],
            exposure,
        )
        .unwrap()
    }

    pub(in super::super) fn request() -> Request {
        Request {
            population_id: "pop".into(),
            treated_regime_id: "ship".into(),
            control_regime_id: "hold".into(),
            horizon: 4,
            target: Target::ExperimentalContrast,
        }
    }

    pub(in super::super) fn margin(cost: InputBinding) -> DerivedQuantity {
        DerivedQuantity {
            name: "margin".into(),
            units: "currency".into(),
            expr: UtilityExpr::product(
                UtilityExpr::Input(0),
                UtilityExpr::difference(UtilityExpr::Input(1), UtilityExpr::Input(2)),
            ),
            inputs: vec![
                InputBinding::Measured("frequency".into()),
                InputBinding::Measured("value".into()),
                cost,
            ],
        }
    }

    pub(in super::super) fn no_interference() -> Exposure {
        Exposure::NoBetweenUnitInterference { assumption_id: "a.sutva".into() }
    }

    #[test]
    fn derived_margin_is_a_delta_method_function_of_means() {
        let r = derive(
            &evidence(1.0, no_interference()),
            &request(),
            &margin(InputBinding::Measured("cost".into())),
        )
        .unwrap();
        // treated 2.2*(12-6)=13.2 ; control 2.0*(11-6)=10.0
        assert!((r.contrast.unwrap() - 3.2).abs() < 1e-12);
        let s = r.sampling.unwrap();
        assert!(s.standard_error > 0.0 && s.lower < 3.2 && s.upper > 3.2);
        assert_eq!(r.standing, Standing::Sampled);
        assert_eq!((r.support, r.calibration), ("none", "not_measured"));
    }

    #[test]
    fn joint_covariance_enters_the_standard_error() {
        let ev = evidence(1.0, no_interference());
        let mut marginal_cov = ev.clone();
        if let Dependence::Joint { treated, .. } = &mut marginal_cov.dependence {
            treated[1] = 0.0;
            treated[3] = 0.0;
        }
        let q = margin(InputBinding::Measured("cost".into()));
        let with = derive(&ev, &request(), &q).unwrap().sampling.unwrap().standard_error;
        let without =
            derive(&marginal_cov, &request(), &q).unwrap().sampling.unwrap().standard_error;
        assert!((with - without).abs() > 1e-6, "dropping the covariance must change the answer");
    }

    #[test]
    fn marginals_only_refuses_a_cross_component_function_but_not_a_single_component() {
        let mut ev = evidence(1.0, no_interference());
        ev.dependence = Dependence::MarginalsOnly {
            treated: vec![0.0004, 0.04, 0.01],
            control: vec![0.0003, 0.03, 0.01],
        };
        let r = derive(&ev, &request(), &margin(InputBinding::Measured("cost".into()))).unwrap();
        assert_eq!(r.unresolved, vec![UnresolvedReason::JointLawMissing]);
        assert!(r.sampling.is_none() && r.contrast.is_some());
        let single = DerivedQuantity {
            name: "frequency_gain".into(),
            units: "u".into(),
            expr: UtilityExpr::Input(0),
            inputs: vec![InputBinding::Measured("frequency".into())],
        };
        assert!(derive(&ev, &request(), &single).unwrap().sampling.is_some());
    }

    #[test]
    fn assumed_cost_gives_a_separate_range_and_never_a_sampling_interval() {
        let cost = InputBinding::Assumed {
            lower: 4.0,
            upper: 11.0,
            assumption_id: "a.cost".into(),
            shared_across_arms: true,
        };
        let r = derive(&evidence(1.0, no_interference()), &request(), &margin(cost)).unwrap();
        let range = r.assumption_range.clone().unwrap();
        assert!(
            range.exact
                && range.dimensions == 1
                && (range.lower - 2.2).abs() < 1e-9
                && (range.upper - 3.6).abs() < 1e-9
        );
        assert_eq!(r.standing, Standing::AssumptionDependent);
        assert_eq!(r.sampling_at_vertices.len(), 2);
    }

    #[test]
    fn missing_input_is_unresolved_not_defaulted() {
        let r = derive(
            &evidence(1.0, no_interference()),
            &request(),
            &margin(InputBinding::Missing { what: "unit cost".into() }),
        )
        .unwrap();
        assert_eq!(r.standing, Standing::Unresolved);
        assert!(r.contrast.is_none() && r.sampling.is_none() && r.assumption_range.is_none());
    }

    #[test]
    fn requests_outside_the_evidence_are_refused_with_a_next_attempt() {
        let ev = evidence(1.0, no_interference());
        let q = margin(InputBinding::Measured("cost".into()));
        let mut r = request();
        r.horizon = 8;
        assert_eq!(
            derive(&ev, &r, &q).unwrap_err().reason_code(),
            antecedent_core::reason_details::decision_evidence::INCOMPATIBLE_HORIZON
        );
        let mut r = request();
        r.target = Target::Deployment { bridge_assumption_id: None };
        let e = derive(&ev, &r, &q).unwrap_err();
        assert_eq!(e, EvidenceRefusal::DeploymentBridgeMissing);
        assert!(!e.next_attempt().is_empty());
        let e = derive(&evidence(1.0, Exposure::Unspecified), &request(), &q).unwrap_err();
        assert_eq!(e, EvidenceRefusal::InterferenceUnspecified);
        let mut r = request();
        r.target = Target::Deployment { bridge_assumption_id: Some("a.bridge".into()) };
        let ok = derive(&ev, &r, &q).unwrap();
        assert!(ok.sampling.is_some());
    }

    #[test]
    fn spillover_bound_widens_into_an_assumption_range() {
        let ev = evidence(
            1.0,
            Exposure::BoundedSpillover {
                assumption_id: "a.spill".into(),
                bias_bounds: vec![0.1, 0.5, 0.0],
            },
        );
        assert!(derive(&ev, &request(), &margin(InputBinding::Measured("cost".into()))).is_err());
        let linear = DerivedQuantity {
            name: "frequency".into(),
            units: "u".into(),
            expr: UtilityExpr::Input(0),
            inputs: vec![InputBinding::Measured("frequency".into())],
        };
        let range = derive(&ev, &request(), &linear).unwrap().assumption_range.unwrap();
        assert!(range.lower < 0.2 && range.upper > 0.2);
    }

    #[test]
    fn decision_is_robust_assumption_dependent_or_unresolved() {
        let frame = DecisionFrame {
            treated_action: "ship".into(),
            control_action: "hold".into(),
            threshold: 0.0,
        };
        let measured = margin(InputBinding::Measured("cost".into()));
        let r = decide(&evidence(0.05, no_interference()), &request(), &measured, &frame).unwrap();
        assert_eq!(r.verdict, Verdict::Robust { action: "ship".into() });

        let cost = InputBinding::Assumed {
            lower: 4.0,
            upper: 11.0,
            assumption_id: "a.cost".into(),
            shared_across_arms: true,
        };
        let bar = DecisionFrame { threshold: 2.5, ..frame.clone() };
        let r = decide(&evidence(0.05, no_interference()), &request(), &margin(cost.clone()), &bar)
            .unwrap();
        assert!(
            matches!(r.verdict, Verdict::AssumptionDependent { .. }),
            "{:?} {:?}",
            r.verdict,
            r.derived.sampling_at_vertices
        );

        let r = decide(&evidence(400.0, no_interference()), &request(), &measured, &frame).unwrap();
        assert_eq!(
            r.verdict,
            Verdict::Unresolved { reasons: vec![UnresolvedReason::SamplingStraddlesThreshold] }
        );

        let r = decide(
            &evidence(0.05, no_interference()),
            &request(),
            &margin(InputBinding::Missing { what: "cost".into() }),
            &frame,
        )
        .unwrap();
        assert!(matches!(r.verdict, Verdict::Unresolved { .. }));
    }

    #[test]
    fn rejects_indefinite_joint_covariance() {
        let ev = evidence(1.0, no_interference());
        let invalid = vec![1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0];
        assert!(
            JointExperimentEvidence::new(
                ev.evidence_id,
                ev.population_id,
                (ev.treated_regime_id, ev.control_regime_id),
                ev.horizon,
                ev.components,
                (ev.treated_means, ev.control_means),
                Dependence::Joint { treated: invalid.clone(), control: invalid },
                ev.design,
                ev.adjustments,
                ev.exposure,
            )
            .is_err()
        );
    }

    #[test]
    fn arm_spillover_bounds_support_nonlinear_margin() {
        let ev = evidence(
            0.05,
            Exposure::BoundedArmSpillover {
                assumption_id: "arm-bias".into(),
                treated_bias_bounds: vec![0.1, 0.5, 0.0],
                control_bias_bounds: vec![0.1, 0.5, 0.0],
            },
        );
        let result =
            derive(&ev, &request(), &margin(InputBinding::Measured("cost".into()))).unwrap();
        let r = result.assumption_range.unwrap();
        assert!(r.lower < 3.2 && r.upper > 3.2 && r.exact);
        assert_eq!(r.assumption_ids, vec!["arm-bias"]);
    }

    #[test]
    fn centered_adjustment_allows_nonlinear_margin() {
        let mut ev = evidence(0.05, no_interference());
        ev.adjustments.push(ComponentAdjustment {
            component: "frequency".into(),
            kind: AdjustmentKind::CenteredFixedCuped { covariate: "pre".into(), coefficient: 1.0 },
        });
        let result =
            derive(&ev, &request(), &margin(InputBinding::Measured("cost".into()))).unwrap();
        assert!(result.sampling.is_some());
    }

    #[test]
    fn nonlinear_bound_and_adjustment_do_not_support_decisions() {
        let mut ev = evidence(0.05, no_interference());
        ev.adjustments.push(ComponentAdjustment {
            component: "frequency".into(),
            kind: AdjustmentKind::FixedCuped { covariate: "pre".into(), coefficient: 1.0 },
        });
        assert!(derive(&ev, &request(), &margin(InputBinding::Measured("cost".into()))).is_err());
    }

    #[test]
    fn repeated_assumed_input_cannot_use_vertices_to_claim_robustness() {
        let ev = evidence(0.05, no_interference());
        let q = DerivedQuantity {
            name: "square".into(),
            units: "u".into(),
            expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
            inputs: vec![InputBinding::Assumed {
                lower: -1.0,
                upper: 1.0,
                assumption_id: "a".into(),
                shared_across_arms: false,
            }],
        };
        let f = DecisionFrame {
            treated_action: "t".into(),
            control_action: "c".into(),
            threshold: 0.1,
        };
        assert!(matches!(
            decide(&ev, &request(), &q, &f).unwrap().verdict,
            Verdict::Unresolved { .. }
        ));
    }

    #[test]
    fn a_threshold_change_changes_only_the_rule_identity() {
        let frame = DecisionFrame {
            treated_action: "ship".into(),
            control_action: "hold".into(),
            threshold: 0.0,
        };
        let other = DecisionFrame { threshold: 1.0, ..frame.clone() };
        assert_ne!(frame.rule_identity(), other.rule_identity());
        let q = margin(InputBinding::Measured("cost".into()));
        let a = decide(&evidence(1.0, no_interference()), &request(), &q, &frame).unwrap();
        let b = decide(&evidence(1.0, no_interference()), &request(), &q, &other).unwrap();
        assert_eq!(a.derived.quantity_identity, b.derived.quantity_identity);
    }
}
