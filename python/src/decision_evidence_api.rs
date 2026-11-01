//! Python bridge for experimental evidence about decision-relevant outcomes.
//!
//! Python declares joint evidence, a derived quantity and a decision rule as JSON; the
//! covariance propagation, assumption enumeration, classification, planning, identities and
//! every refusal are Rust's. A refusal comes back as structured JSON for the Python layer to
//! raise as a typed exception.
//!
//! Nothing here widens an interval or turns an assumption range into a probability.

use antecedent::decision_evidence::{
    AdjustmentKind, Candidate, ComponentAdjustment, DecisionFrame, DecisionResult, Dependence,
    DerivedQuantity, DerivedResult, DesignDetails, DesignRecord, EvidenceRefusal, Exposure,
    InputBinding, JointExperimentEvidence, JointExperimentSpec, JointSwitchbackDesign,
    MatchedPairDesign, OutcomeComponent, PlanValue, ProxyBridge, ProxyMapping, Request,
    SamplingInterval, StageDigests, Standing, Target, UnresolvedReason, UtilityExpr, Verdict,
    compile_joint_experiment, compile_joint_switchback, compile_matched_pair_design,
    complete_cluster_joint_contrast, decide, decide_with_proxy, derive,
    evidence_from_joint_switchback, evidence_from_matched_pairs, plan_evidence,
};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    list, member, object, opt_text, parse, present, real, reals, text,
};

type Refused = (Option<String>, Option<String>);

fn refusal_json(refusal: &EvidenceRefusal) -> String {
    json!({
        "code": refusal.registered_code(),
        "stage": "decision_evidence",
        "detail": refusal.reason_code(),
        "message": refusal.to_string(),
        "offending": Json::Null,
        "remedy": refusal.next_attempt(),
    })
    .to_string()
}

fn finish(result: Result<Json, EvidenceRefusal>) -> Refused {
    match result {
        Ok(value) => (Some(value.to_string()), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

fn invalid(error: impl std::fmt::Display) -> EvidenceRefusal {
    EvidenceRefusal::Invalid(error.to_string())
}

fn count(map: &Map<String, Json>, key: &str) -> PyResult<usize> {
    let value = member(map, key)?
        .as_u64()
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a non-negative integer")))?;
    usize::try_from(value).map_err(|_| crate::value_err(format!("field `{key}` is too large")))
}

fn component(value: &Json) -> PyResult<OutcomeComponent> {
    let map = object(value, "component")?;
    Ok(OutcomeComponent {
        name: text(map, "name")?,
        variable_id: text(map, "variable_id")?,
        units: text(map, "units")?,
    })
}

fn components(map: &Map<String, Json>, key: &str) -> PyResult<Vec<OutcomeComponent>> {
    list(map, key)?.iter().map(component).collect()
}

fn adjustments(map: &Map<String, Json>) -> PyResult<Vec<ComponentAdjustment>> {
    let Some(items) = present(map, "adjustments") else { return Ok(Vec::new()) };
    let items = items.as_array().ok_or_else(|| crate::value_err("`adjustments` must be a list"))?;
    items
        .iter()
        .map(|item| {
            let m = object(item, "adjustment")?;
            let kind = match text(m, "kind")?.as_str() {
                "none" => AdjustmentKind::None,
                "centered_fixed_cuped" => AdjustmentKind::CenteredFixedCuped {
                    covariate: text(m, "covariate")?,
                    coefficient: real(m, "coefficient")?,
                },
                "fixed_cuped" => AdjustmentKind::FixedCuped {
                    covariate: text(m, "covariate")?,
                    coefficient: real(m, "coefficient")?,
                },
                other => {
                    return Err(crate::value_err(format!("unknown adjustment kind `{other}`")));
                }
            };
            Ok(ComponentAdjustment { component: text(m, "component")?, kind })
        })
        .collect()
}

fn exposure(map: &Map<String, Json>) -> PyResult<Exposure> {
    let m = object(member(map, "exposure")?, "exposure")?;
    Ok(match text(m, "kind")?.as_str() {
        "no_between_unit_interference" => {
            Exposure::NoBetweenUnitInterference { assumption_id: text(m, "assumption_id")? }
        }
        "bounded_spillover" => Exposure::BoundedSpillover {
            assumption_id: text(m, "assumption_id")?,
            bias_bounds: reals(m, "bias_bounds")?,
        },
        "bounded_arm_spillover" => Exposure::BoundedArmSpillover {
            assumption_id: text(m, "assumption_id")?,
            treated_bias_bounds: reals(m, "treated_bias_bounds")?,
            control_bias_bounds: reals(m, "control_bias_bounds")?,
        },
        "unspecified" => Exposure::Unspecified,
        other => return Err(crate::value_err(format!("unknown exposure kind `{other}`"))),
    })
}

fn decoded<T: serde::de::DeserializeOwned>(map: &Map<String, Json>, key: &str) -> PyResult<T> {
    serde_json::from_value(member(map, key)?.clone())
        .map_err(|e| crate::value_err(format!("invalid `{key}`: {e}")))
}
fn paired_design(m: &Map<String, Json>) -> PyResult<MatchedPairDesign> {
    Ok(MatchedPairDesign {
        cluster_ids: decoded(m, "cluster_ids")?,
        pairs: decoded(m, "pairs")?,
        matching_variables: decoded(m, "matching_variables")?,
        treatment_probabilities: decoded(m, "treatment_probabilities")?,
        cluster_sizes: decoded(m, "cluster_sizes")?,
        preassignment_assumption_id: text(m, "preassignment_assumption_id")?,
        independent_pairs_assumption_id: text(m, "independent_pairs_assumption_id")?,
    })
}
fn switch_design(m: &Map<String, Json>) -> PyResult<JointSwitchbackDesign> {
    Ok(JointSwitchbackDesign {
        sequence_ids: decoded(m, "sequence_ids")?,
        period_ids: decoded(m, "period_ids")?,
        treatment_probabilities: decoded(m, "treatment_probabilities")?,
        no_carryover_assumption_id: text(m, "no_carryover_assumption_id")?,
        washout_rule_id: text(m, "washout_rule_id")?,
        washout_periods: count(m, "washout_periods")?,
        independent_sequences_assumption_id: text(m, "independent_sequences_assumption_id")?,
    })
}

fn evidence(value: &Json) -> PyResult<Result<JointExperimentEvidence, EvidenceRefusal>> {
    let map = object(value, "evidence")?;
    let dep = object(member(map, "dependence")?, "dependence")?;
    let dependence = match text(dep, "kind")?.as_str() {
        "joint" => {
            Dependence::Joint { treated: reals(dep, "treated")?, control: reals(dep, "control")? }
        }
        "joint_arms" => Dependence::JointArms { covariance: reals(dep, "covariance")? },
        "marginals_only" => Dependence::MarginalsOnly {
            treated: reals(dep, "treated")?,
            control: reals(dep, "control")?,
        },
        other => return Err(crate::value_err(format!("unknown dependence kind `{other}`"))),
    };
    let design = object(member(map, "design")?, "design")?;
    let licenses = match present(map, "marginal_licenses") {
        Some(Json::Array(items)) => {
            items.iter().map(|v| v.as_str().map(str::to_owned)).collect::<Vec<_>>()
        }
        _ => Vec::new(),
    };
    let built = JointExperimentEvidence::new(
        text(map, "evidence_id")?,
        text(map, "population_id")?,
        (text(map, "treated_regime_id")?, text(map, "control_regime_id")?),
        u32::try_from(count(map, "horizon")?).map_err(|_| crate::value_err("horizon too large"))?,
        components(map, "components")?,
        (reals(map, "treated_means")?, reals(map, "control_means")?),
        dependence,
        DesignRecord {
            assignment_design: text(design, "assignment_design")?,
            treated_units: count(design, "treated_units")?,
            control_units: count(design, "control_units")?,
            outcome_rows: count(design, "outcome_rows")?,
        },
        adjustments(map)?,
        exposure(map)?,
    );
    let details =
        present(map, "design_details").map(|v| object(v, "design_details")).transpose()?;
    let design_contract_id = opt_text(map, "design_contract_id")?;
    let decoded_details = match details {
        Some(m) => match text(m, "kind")?.as_str() {
            "matched_pairs" => {
                let d = paired_design(m)?;
                let realized: Vec<bool> = decoded(m, "realized_assignment")?;
                let compiled = match compile_matched_pair_design(d.clone()) {
                    Ok(c) => c,
                    Err(e) => return Ok(Err(e)),
                };
                if realized.len() != d.cluster_ids.len()
                    || d.pairs.iter().any(|(a, b)| realized[*a] == realized[*b])
                    || design_contract_id.as_deref() != Some(compiled.identity.as_str())
                    || text(design, "assignment_design")? != "matched_pairs"
                    || count(design, "treated_units")? != d.pairs.len()
                    || count(design, "control_units")? != d.pairs.len()
                    || Some(count(design, "outcome_rows")?)
                        != d.cluster_sizes.iter().try_fold(0usize, |a, b| a.checked_add(*b))
                {
                    return Ok(Err(EvidenceRefusal::Invalid(
                        "matched evidence does not match its declared design".into(),
                    )));
                }
                Some(DesignDetails::MatchedPairs {
                    cluster_ids: d.cluster_ids,
                    pairs: d.pairs,
                    matching_variables: d.matching_variables,
                    treatment_probabilities: d.treatment_probabilities,
                    cluster_sizes: d.cluster_sizes,
                    realized_assignment: realized,
                    preassignment_assumption_id: d.preassignment_assumption_id,
                    independent_pairs_assumption_id: d.independent_pairs_assumption_id,
                })
            }
            "switchback" => {
                let d = switch_design(m)?;
                let realized: Vec<bool> = decoded(m, "realized_assignment")?;
                let compiled = match compile_joint_switchback(d.clone()) {
                    Ok(c) => c,
                    Err(e) => return Ok(Err(e)),
                };
                let units = d.sequence_ids.iter().collect::<std::collections::BTreeSet<_>>().len();
                if realized.len() != d.sequence_ids.len()
                    || !realized.contains(&true)
                    || !realized.contains(&false)
                    || design_contract_id.as_deref() != Some(compiled.identity.as_str())
                    || text(design, "assignment_design")? != "switchback"
                    || count(design, "treated_units")? != units
                    || count(design, "control_units")? != units
                    || count(design, "outcome_rows")? != d.sequence_ids.len()
                {
                    return Ok(Err(EvidenceRefusal::Invalid(
                        "switchback evidence does not match its declared design".into(),
                    )));
                }
                Some(DesignDetails::Switchback {
                    sequence_ids: d.sequence_ids,
                    period_ids: d.period_ids,
                    realized_assignment: realized,
                    treatment_probabilities: d.treatment_probabilities,
                    no_carryover_assumption_id: d.no_carryover_assumption_id,
                    washout_rule_id: d.washout_rule_id,
                    washout_periods: d.washout_periods,
                    independent_sequences_assumption_id: d.independent_sequences_assumption_id,
                })
            }
            other => return Err(crate::value_err(format!("unknown design `{other}`"))),
        },
        None => None,
    };
    Ok(built.map(|mut e| {
        e.design_contract_id = design_contract_id;
        e.design_details = decoded_details;
        if licenses.is_empty() { e } else { e.with_marginal_licenses(licenses) }
    }))
}

fn request(value: &Json) -> PyResult<Request> {
    let map = object(value, "request")?;
    let target = match present(map, "target") {
        None => Target::ExperimentalContrast,
        Some(t) => {
            let t = object(t, "target")?;
            match text(t, "kind")?.as_str() {
                "experimental_contrast" => Target::ExperimentalContrast,
                "deployment" => Target::Deployment {
                    bridge_assumption_id: opt_text(t, "bridge_assumption_id")?,
                },
                other => return Err(crate::value_err(format!("unknown target kind `{other}`"))),
            }
        }
    };
    Ok(Request {
        population_id: text(map, "population_id")?,
        treated_regime_id: text(map, "treated_regime_id")?,
        control_regime_id: text(map, "control_regime_id")?,
        horizon: u32::try_from(count(map, "horizon")?)
            .map_err(|_| crate::value_err("horizon too large"))?,
        target,
    })
}

fn expr(value: &Json) -> PyResult<UtilityExpr> {
    let map = object(value, "expression")?;
    if let Some(c) = map.get("const") {
        return Ok(UtilityExpr::Const(
            c.as_f64().ok_or_else(|| crate::value_err("`const` must be a number"))?,
        ));
    }
    if let Some(k) = map.get("input") {
        let k = k.as_u64().ok_or_else(|| crate::value_err("`input` must be an index"))?;
        return Ok(UtilityExpr::Input(
            usize::try_from(k).map_err(|_| crate::value_err("input index too large"))?,
        ));
    }
    let op = text(map, "op")?;
    let args = list(map, "args")?;
    let arg = |i: usize| -> PyResult<UtilityExpr> {
        expr(
            args.get(i)
                .ok_or_else(|| crate::value_err(format!("`{op}` is missing an argument")))?,
        )
    };
    let binary = |args: &Vec<Json>| -> PyResult<()> {
        (args.len() == 2)
            .then_some(())
            .ok_or_else(|| crate::value_err(format!("`{op}` takes two arguments")))
    };
    Ok(match op.as_str() {
        "neg" => {
            (args.len() == 1)
                .then_some(())
                .ok_or_else(|| crate::value_err("`neg` takes one argument"))?;
            UtilityExpr::Neg(Box::new(arg(0)?))
        }
        "add" => {
            binary(args)?;
            UtilityExpr::sum(arg(0)?, arg(1)?)
        }
        "sub" => {
            binary(args)?;
            UtilityExpr::difference(arg(0)?, arg(1)?)
        }
        "mul" => {
            binary(args)?;
            UtilityExpr::product(arg(0)?, arg(1)?)
        }
        "min" => {
            binary(args)?;
            UtilityExpr::minimum(arg(0)?, arg(1)?)
        }
        "max" => {
            binary(args)?;
            UtilityExpr::maximum(arg(0)?, arg(1)?)
        }
        other => return Err(crate::value_err(format!("unknown operator `{other}`"))),
    })
}

fn quantity(value: &Json) -> PyResult<DerivedQuantity> {
    let map = object(value, "quantity")?;
    let inputs = list(map, "inputs")?
        .iter()
        .map(|item| {
            let m = object(item, "input binding")?;
            Ok(match text(m, "kind")?.as_str() {
                "measured" => InputBinding::Measured(text(m, "component")?),
                "fixed" => InputBinding::Fixed(real(m, "value")?),
                "assumed" => InputBinding::Assumed {
                    lower: real(m, "lower")?,
                    upper: real(m, "upper")?,
                    assumption_id: text(m, "assumption_id")?,
                    shared_across_arms: m
                        .get("shared_across_arms")
                        .and_then(Json::as_bool)
                        .unwrap_or(true),
                },
                "missing" => InputBinding::Missing { what: text(m, "what")? },
                other => return Err(crate::value_err(format!("unknown binding kind `{other}`"))),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(DerivedQuantity {
        name: text(map, "name")?,
        units: text(map, "units")?,
        expr: expr(member(map, "expr")?)?,
        inputs,
    })
}

fn frame(value: &Json) -> PyResult<DecisionFrame> {
    let map = object(value, "frame")?;
    Ok(DecisionFrame {
        treated_action: text(map, "treated_action")?,
        control_action: text(map, "control_action")?,
        threshold: real(map, "threshold")?,
    })
}

fn candidates(map: &Map<String, Json>) -> PyResult<Vec<Candidate>> {
    let Some(items) = present(map, "candidates") else { return Ok(Vec::new()) };
    let items = items.as_array().ok_or_else(|| crate::value_err("`candidates` must be a list"))?;
    items
        .iter()
        .map(|item| {
            let m = object(item, "candidate")?;
            let cost = match present(m, "cost") {
                None => None,
                Some(v) => {
                    Some(v.as_f64().ok_or_else(|| crate::value_err("`cost` must be a number"))?)
                }
            };
            Ok(match text(m, "kind")?.as_str() {
                "more_units" => Candidate::MoreUnits {
                    id: text(m, "id")?,
                    additional_units_per_arm: member(m, "additional_units_per_arm")?
                        .as_u64()
                        .ok_or_else(|| {
                            crate::value_err("`additional_units_per_arm` must be an integer")
                        })?,
                    cost,
                },
                "resolve_assumption" => Candidate::ResolveAssumption {
                    id: text(m, "id")?,
                    assumption_id: text(m, "assumption_id")?,
                    narrowed: (real(m, "narrowed_lower")?, real(m, "narrowed_upper")?),
                    cost,
                },
                other => return Err(crate::value_err(format!("unknown candidate kind `{other}`"))),
            })
        })
        .collect()
}

fn reason_json(reason: &UnresolvedReason) -> Json {
    match reason {
        UnresolvedReason::MissingInput { what } => json!({"code": "missing_input", "what": what}),
        UnresolvedReason::JointLawMissing => json!({"code": "joint_law_missing"}),
        UnresolvedReason::TooFewIndependentUnits { treated, control, required } => json!({
            "code": "too_few_independent_units",
            "treated": treated, "control": control, "required": required,
        }),
        UnresolvedReason::SamplingUnavailable => json!({"code": "sampling_unavailable"}),
        UnresolvedReason::SamplingStraddlesThreshold => {
            json!({"code": "sampling_straddles_threshold"})
        }
    }
}

fn sampling_json(s: &SamplingInterval) -> Json {
    json!({
        "estimate": s.estimate, "standard_error": s.standard_error,
        "lower": s.lower, "upper": s.upper,
    })
}

fn derived_json(r: &DerivedResult) -> Json {
    json!({
        "name": r.name,
        "units": r.units,
        "treated_value": r.treated_value,
        "control_value": r.control_value,
        "contrast": r.contrast,
        "sampling": r.sampling.as_ref().map(sampling_json),
        "sampling_at_vertices": r.sampling_at_vertices.iter().map(sampling_json).collect::<Vec<_>>(),
        "assumption_range": r.assumption_range.as_ref().map(|a| json!({
            "lower": a.lower, "upper": a.upper, "exact": a.exact,
            "assumption_ids": a.assumption_ids, "dimensions": a.dimensions,
        })),
        "unresolved": r.unresolved.iter().map(reason_json).collect::<Vec<_>>(),
        "standing": match r.standing {
            Standing::Sampled => "sampled",
            Standing::AssumptionDependent => "assumption_dependent",
            Standing::Unresolved => "unresolved",
        },
        "license": r.license,
        "calibration": r.calibration,
        "interval_statement": r.interval_statement,
        "separation": r.separation,
        "quantity_identity": r.quantity_identity,
    })
}

fn verdict_json(v: &Verdict) -> Json {
    match v {
        Verdict::Robust { action } => json!({"class": "robust", "action": action}),
        Verdict::AssumptionDependent { assumption_ids, at_lower, at_upper } => json!({
            "class": "assumption_dependent", "assumption_ids": assumption_ids,
            "at_lower": at_lower, "at_upper": at_upper,
        }),
        Verdict::Unresolved { reasons } => json!({
            "class": "unresolved", "reasons": reasons.iter().map(reason_json).collect::<Vec<_>>(),
        }),
    }
}

fn decision_json(d: &DecisionResult) -> Json {
    json!({
        "derived": derived_json(&d.derived),
        "verdict": verdict_json(&d.verdict),
        "rule_identity": d.frame.rule_identity(),
        "statement": d.statement,
    })
}

fn digests_json(d: &StageDigests) -> Json {
    use antecedent::decision_evidence::Stage;
    let mut m = Map::new();
    for s in Stage::ALL {
        m.insert(s.name().to_owned(), Json::String(d.digest(s).to_owned()));
    }
    Json::Object(m)
}

fn plan_json(p: &antecedent::decision_evidence::EvidencePlan) -> Json {
    let row = |r: &antecedent::decision_evidence::PlanRow| {
        let value = match &r.value {
            PlanValue::Evsi { gross, net } => json!({"gross_evsi": gross, "net_value": net}),
            PlanValue::Resolution { verdict, resolves } => {
                json!({"verdict": verdict_json(verdict), "resolves": resolves})
            }
            PlanValue::Refused(message) => json!({"refused": message}),
        };
        json!({"id": r.id, "obligation": r.kind.as_str(), "value": value})
    };
    json!({
        "evsi_rows": p.evsi_rows.iter().map(row).collect::<Vec<_>>(),
        "resolution_rows": p.resolution_rows.iter().map(row).collect::<Vec<_>>(),
        "model": p.model.statement,
    })
}

/// Compile a joint experiment before any fit. Returns the compiled description or a refusal.
#[pyfunction]
fn decision_evidence_compile(spec_json: &str) -> PyResult<Refused> {
    let value = parse(spec_json, "joint experiment")?;
    let map = object(&value, "joint experiment")?;
    let spec = JointExperimentSpec {
        outcomes: components(map, "outcomes")?,
        assignment_design: text(map, "assignment_design")?,
        population_id: text(map, "population_id")?,
        treated_regime_id: text(map, "treated_regime_id")?,
        control_regime_id: text(map, "control_regime_id")?,
        horizon: u32::try_from(count(map, "horizon")?)
            .map_err(|_| crate::value_err("horizon too large"))?,
        adjustments: adjustments(map)?,
        exposure: exposure(map)?,
    };
    Ok(finish(compile_joint_experiment(spec).map(|c| {
        json!({
            "unit_of_analysis": c.unit_of_analysis,
            "scientific_identity": c.scientific_identity,
            "adjustment_identity": c.adjustment_identity,
            "adjustment_verdicts": c.adjustment_verdicts.iter().map(|v| json!({
                "strategy": v.strategy,
                "coefficient": v.coefficient,
                "admissible": v.admissible.is_ok(),
                "reason": v.admissible.unwrap_or_else(|e| e),
            })).collect::<Vec<_>>(),
            "obligations": c.obligations.iter().map(|o| json!({
                "id": o.id, "kind": o.kind.as_str(), "statement": o.statement,
            })).collect::<Vec<_>>(),
            "statistical_requirements": c.statistical_requirements,
        })
    })))
}

/// Estimate several outcomes of a cluster-randomized experiment jointly.
///
/// `totals[k][c]` is outcome `k` summed over the rows of cluster `c`. Returns the arm means
/// and the covariance of their estimators, or `None` when the totals cannot support one.
#[pyfunction]
fn decision_evidence_cluster_contrast(
    totals: Vec<Vec<f64>>,
    assigned: Vec<bool>,
    outcome_rows: usize,
) -> Option<String> {
    let slices: Vec<&[f64]> = totals.iter().map(Vec::as_slice).collect();
    complete_cluster_joint_contrast(&slices, &assigned, outcome_rows).map(|c| {
        json!({
            "outcomes": c.outcomes,
            "treated_clusters": c.treated_clusters,
            "control_clusters": c.control_clusters,
            "treated_means": c.treated_means,
            "control_means": c.control_means,
            "treated_covariance": c.treated_covariance,
            "control_covariance": c.control_covariance,
            "effects": c.effects,
            "effect_covariance": c.effect_covariance,
            "interval_supported": c.interval_supported,
        })
        .to_string()
    })
}

/// Prepare a design or execute joint estimation against a declared design.
#[pyfunction]
fn decision_evidence_experimental(payload_json: &str, fit: bool) -> PyResult<Refused> {
    let payload = parse(payload_json, "experimental evidence")?;
    let map = object(&payload, "experimental evidence")?;
    let dm = object(member(map, "design")?, "design")?;
    let kind = text(dm, "kind")?;
    let (identity, obligations, paired, switchback) = match kind.as_str() {
        "matched_pairs" => match compile_matched_pair_design(paired_design(dm)?) {
            Ok(d) => (d.identity.clone(), d.obligations.clone(), Some(d), None),
            Err(e) => return Ok(finish(Err(e))),
        },
        "switchback" => match compile_joint_switchback(switch_design(dm)?) {
            Ok(d) => (d.identity.clone(), d.obligations.clone(), None, Some(d)),
            Err(e) => return Ok(finish(Err(e))),
        },
        _ => return Ok(finish(Err(EvidenceRefusal::UnsupportedAssignmentDesign(kind)))),
    };
    if !fit {
        return Ok((
            Some(json!({"identity":identity,"obligations":obligations,"design":dm}).to_string()),
            None,
        ));
    }
    let raw: Vec<Vec<f64>> = decoded(map, "outcomes")?;
    let slices: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
    let assigned: Vec<bool> = decoded(map, "assigned")?;
    let horizon =
        u32::try_from(count(map, "horizon")?).map_err(|_| crate::value_err("horizon too large"))?;
    let comps = components(map, "components")?;
    let evidence_id = text(map, "evidence_id")?;
    let population = text(map, "population_id")?;
    let treated = text(map, "treated_regime_id")?;
    let control = text(map, "control_regime_id")?;
    let exp = exposure(map)?;
    let result = if let Some(d) = paired {
        evidence_from_matched_pairs(
            &d,
            &slices,
            &assigned,
            &evidence_id,
            &population,
            (&treated, &control),
            horizon,
            comps,
            exp,
        )
    } else {
        evidence_from_joint_switchback(
            &switchback.expect("compiled design"),
            &slices,
            &assigned,
            &evidence_id,
            &population,
            (&treated, &control),
            horizon,
            comps,
            exp,
        )
    };
    Ok(finish(result.map(|ev| {
        let mut retained_design = dm.clone();
        retained_design.insert("realized_assignment".into(), json!(assigned));
        let Dependence::JointArms { covariance }=ev.dependence else { unreachable!() };
        json!({"evidence_id":evidence_id,"population_id":population,"treated_regime_id":treated,"control_regime_id":control,"horizon":horizon,
          "components":map["components"],"treated_means":ev.treated_means,"control_means":ev.control_means,
          "dependence":{"kind":"joint_arms","covariance":covariance},
          "design":{"assignment_design":ev.design.assignment_design,"treated_units":ev.design.treated_units,"control_units":ev.design.control_units,"outcome_rows":ev.design.outcome_rows},
          "design_contract_id":identity,"design_details":retained_design,"adjustments":[],"exposure":map["exposure"]})
    })))
}

/// Apply an explicitly scoped causal proxy-effect bridge.
#[pyfunction]
fn decision_evidence_proxy(payload_json: &str) -> PyResult<Refused> {
    let value = parse(payload_json, "proxy evidence")?;
    let map = object(&value, "proxy evidence")?;
    let ev = match evidence(member(map, "evidence")?)? {
        Ok(e) => e,
        Err(e) => return Ok(finish(Err(e))),
    };
    let req = request(member(map, "request")?)?;
    let f = frame(member(map, "frame")?)?;
    let b = object(member(map, "bridge")?, "bridge")?;
    let m = object(member(b, "mapping")?, "mapping")?;
    let mapping = match text(m, "kind")?.as_str() {
        "predictive_association" => ProxyMapping::PredictiveAssociation,
        "causal_effect_bounds" => ProxyMapping::CausalEffectBounds {
            multiplier: decoded(m, "multiplier")?,
            residual: decoded(m, "residual")?,
            assumption_id: text(m, "assumption_id")?,
            mapping_evidence_id: text(m, "mapping_evidence_id")?,
        },
        other => return Err(crate::value_err(format!("unknown proxy mapping `{other}`"))),
    };
    let bridge = ProxyBridge {
        proxy_component: text(b, "proxy_component")?,
        target_variable_id: text(b, "target_variable_id")?,
        target_units: text(b, "target_units")?,
        population_id: text(b, "population_id")?,
        treated_regime_id: text(b, "treated_regime_id")?,
        control_regime_id: text(b, "control_regime_id")?,
        horizon: u32::try_from(count(b, "horizon")?)
            .map_err(|_| crate::value_err("horizon too large"))?,
        mapping,
    };
    Ok(finish(decide_with_proxy(&ev,&req,&bridge,&f).map(|r|json!({"identity":r.identity,"assumption_range":r.assumption_range,"sampling_envelope":r.sampling_envelope,"verdict":verdict_json(&r.verdict),"evidence_ids":r.evidence_ids,"assumption_id":r.assumption_id,"target_variable_id":r.target_variable_id,"target_units":r.target_units}))))
}

/// Derive a quantity and, when a frame is given, classify the decision; with candidates, plan
/// the next evidence. Returns the result JSON or a refusal.
#[pyfunction]
fn decision_evidence_evaluate(payload_json: &str) -> PyResult<Refused> {
    let value = parse(payload_json, "decision evidence")?;
    let map = object(&value, "decision evidence")?;
    let ev = match evidence(member(map, "evidence")?)? {
        Ok(ev) => ev,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let req = request(member(map, "request")?)?;
    let q = quantity(member(map, "quantity")?)?;
    let frame = present(map, "frame").map(frame).transpose()?;
    let cands = candidates(map)?;
    let run = || -> Result<Json, EvidenceRefusal> {
        let mut out = Map::new();
        match &frame {
            Some(f) => {
                let d = decide(&ev, &req, &q, f)?;
                out.insert("decision".into(), decision_json(&d));
                out.insert("stage_digests".into(), digests_json(&StageDigests::of(&ev, &q, f)));
                if !cands.is_empty() {
                    out.insert("plan".into(), plan_json(&plan_evidence(&ev, &req, &q, f, &cands)?));
                }
            }
            None => {
                if !cands.is_empty() {
                    return Err(invalid("planning needs a decision frame"));
                }
                out.insert("derived".into(), derived_json(&derive(&ev, &req, &q)?));
            }
        }
        out.insert("estimate_identity".into(), Json::String(ev.estimate_identity()));
        Ok(Json::Object(out))
    };
    Ok(finish(run()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decision_evidence_compile, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_cluster_contrast, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_experimental, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_proxy, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_evaluate, module)?)?;
    Ok(())
}
