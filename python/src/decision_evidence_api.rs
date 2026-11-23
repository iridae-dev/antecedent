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
    InputBinding, JointExcursionDesign, JointExperimentEvidence, JointExperimentSpec,
    JointStaticRegimeDesign, JointSwitchbackDesign, MatchedPairDesign, OutcomeComponent, PlanValue,
    ProxyBridge, ProxyMapping, Request, SamplingInterval, StageDigests, Standing, Target,
    TemporalEffectBridge, TemporalEstimand, UnresolvedReason, UtilityExpr, Verdict,
    compile_and_evidence_from_adaptive_regime, compile_and_evidence_from_cluster,
    compile_and_evidence_from_joint_excursion, compile_and_evidence_from_joint_static_regime,
    compile_and_evidence_from_joint_static_regime_bootstrap,
    compile_and_evidence_from_joint_switchback, compile_and_evidence_from_matched_pairs,
    compile_joint_excursion, compile_joint_experiment, compile_joint_static_regime,
    compile_joint_switchback, compile_matched_pair_design, compiled_decision_refusal, decide,
    decide_with_proxy, decide_with_temporal_bridge, derive, evidence_from_compiled_adaptive_regime,
    evidence_from_compiled_cluster, evidence_from_compiled_excursion,
    evidence_from_compiled_experiment, evidence_from_compiled_static_regime,
    evidence_from_joint_excursion, evidence_from_joint_static_regime,
    evidence_from_joint_switchback, evidence_from_matched_pairs, plan_evidence,
};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    list, member, object, opt_text, parse, present, real, reals, text,
};

/// JSON can contain an encoded native artifact; bound its expansion before parsing.
pub(crate) const MAX_DECISION_SOURCE_JSON: usize = 4 * 16 * 1024 * 1024 + 8 * 1024 * 1024;

fn parse_decision(input: &str, label: &str) -> PyResult<Json> {
    crate::bridge::check_declared_size(input, MAX_DECISION_SOURCE_JSON, label)?;
    parse(input, label)
}

type Refused = (Option<String>, Option<String>);

fn refusal_json(refusal: &EvidenceRefusal) -> String {
    crate::bridge::compact_refusal_json(
        refusal.registered_code(),
        "decision_evidence",
        refusal.reason_code(),
        &refusal.to_string(),
        Some(refusal.next_attempt()),
    )
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
        consistency_assumption_id: text(m, "consistency_assumption_id")?,
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
        consistency_assumption_id: text(m, "consistency_assumption_id")?,
    })
}

fn excursion_design(m: &Map<String, Json>) -> PyResult<JointExcursionDesign> {
    Ok(JointExcursionDesign {
        unit_ids: decoded(m, "unit_ids")?,
        periods: decoded(m, "periods")?,
        treatment_probabilities: decoded(m, "treatment_probabilities")?,
        lag: u32::try_from(count(m, "lag")?).map_err(|_| crate::value_err("lag too large"))?,
        min_probability: real(m, "min_probability")?,
        assignment_law_id: text(m, "assignment_law_id")?,
        independent_units_assumption_id: text(m, "independent_units_assumption_id")?,
        administrative_followup_assumption_id: text(m, "administrative_followup_assumption_id")?,
        consistency_assumption_id: text(m, "consistency_assumption_id")?,
    })
}

fn regime_design(m: &Map<String, Json>) -> PyResult<JointStaticRegimeDesign> {
    Ok(JointStaticRegimeDesign {
        unit_ids: decoded(m, "unit_ids")?,
        periods: decoded(m, "periods")?,
        realized_assignment: decoded(m, "realized_assignment")?,
        confounder_ids: decoded(m, "confounder_ids")?,
        active_path: decoded(m, "active_path")?,
        reference_path: decoded(m, "reference_path")?,
        method: text(m, "method")?,
        min_probability: real(m, "min_probability")?,
        positivity_assumption_id: text(m, "positivity_assumption_id")?,
        exchangeability_assumption_id: text(m, "exchangeability_assumption_id")?,
        consistency_assumption_id: text(m, "consistency_assumption_id")?,
        followup_assumption_id: text(m, "followup_assumption_id")?,
        model_assumption_id: text(m, "model_assumption_id")?,
    })
}

struct ParsedEvidence {
    value: JointExperimentEvidence,
    owner: Option<antecedent::analysis::joint_panel::designs::NativeJointDesignResult>,
}
impl std::ops::Deref for ParsedEvidence {
    type Target = JointExperimentEvidence;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
fn decode_experimental_design_details(
    details: Option<&Map<String, Json>>,
    map: &Map<String, Json>,
    design: &Map<String, Json>,
) -> PyResult<Result<Option<DesignDetails>, EvidenceRefusal>> {
    let decoded_details = match details {
        Some(m) => match text(m, "kind")?.as_str() {
            "matched_pairs" => {
                let d = paired_design(m)?;
                let realized: Vec<bool> = decoded(m, "realized_assignment")?;
                let _compiled = match compile_matched_pair_design(d.clone()) {
                    Ok(c) => c,
                    Err(e) => return Ok(Err(e)),
                };
                if realized.len() != d.cluster_ids.len()
                    || d.pairs.iter().any(|(a, b)| realized[*a] == realized[*b])
                    || present(map, "compiled_source_artifact").is_none()
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
                    consistency_assumption_id: d.consistency_assumption_id,
                })
            }
            "adaptive_regime" => Some(DesignDetails::AdaptiveRegime {
                subject_ids: decoded(m, "subject_ids")?,
                periods: u32::try_from(count(m, "periods")?)
                    .map_err(|_| crate::value_err("adaptive horizon too large"))?,
                policy_prespecification_assumption_id: text(
                    m,
                    "policy_prespecification_assumption_id",
                )?,
                positivity_assumption_id: text(m, "positivity_assumption_id")?,
                exchangeability_assumption_id: text(m, "exchangeability_assumption_id")?,
                consistency_assumption_id: text(m, "consistency_assumption_id")?,
                followup_assumption_id: text(m, "followup_assumption_id")?,
                failures: decoded(m, "failures")?,
            }),
            "temporal_regime" => {
                let d = regime_design(m)?;
                let compiled = match compile_joint_static_regime(d.clone()) {
                    Ok(c) => c,
                    Err(e) => return Ok(Err(e)),
                };
                if present(map, "compiled_source_artifact").is_none()
                    || text(design, "assignment_design")? != "temporal_regime"
                    || count(design, "treated_units")? != compiled.units
                    || count(design, "control_units")? != compiled.units
                    || count(design, "outcome_rows")? != d.unit_ids.len()
                {
                    return Ok(Err(EvidenceRefusal::Invalid(
                        "regime evidence does not match its declared design".into(),
                    )));
                }
                Some(DesignDetails::TemporalRegime {
                    unit_ids: d.unit_ids,
                    periods: d.periods,
                    realized_assignment: d.realized_assignment,
                    observed_minimum_probabilities: decoded(m, "observed_minimum_probabilities")?,
                    confounder_ids: d.confounder_ids,
                    active_path: d.active_path,
                    reference_path: d.reference_path,
                    method: d.method,
                    min_probability: d.min_probability,
                    positivity_assumption_id: d.positivity_assumption_id,
                    exchangeability_assumption_id: d.exchangeability_assumption_id,
                    consistency_assumption_id: d.consistency_assumption_id,
                    followup_assumption_id: d.followup_assumption_id,
                    model_assumption_id: d.model_assumption_id,
                    bootstrap_replicates: present(m, "bootstrap_replicates")
                        .map(|_| count(m, "bootstrap_replicates"))
                        .transpose()?,
                })
            }
            "temporal_excursion" => {
                let d = excursion_design(m)?;
                let realized: Vec<bool> = decoded(m, "realized_assignment")?;
                let compiled = match compile_joint_excursion(d.clone()) {
                    Ok(c) => c,
                    Err(e) => return Ok(Err(e)),
                };
                if realized.len() != d.unit_ids.len()
                    || present(map, "compiled_source_artifact").is_none()
                    || text(design, "assignment_design")? != "temporal_excursion"
                    || count(design, "treated_units")? != compiled.independent_units
                    || count(design, "control_units")? != compiled.independent_units
                    || count(design, "outcome_rows")? != d.unit_ids.len()
                    || count(m, "paired_decisions")? != compiled.paired_decisions
                {
                    return Ok(Err(EvidenceRefusal::Invalid(
                        "excursion evidence does not match its declared design".into(),
                    )));
                }
                Some(DesignDetails::TemporalExcursion {
                    unit_ids: d.unit_ids,
                    periods: d.periods,
                    treatment_probabilities: d.treatment_probabilities,
                    realized_assignment: realized,
                    lag: d.lag,
                    min_probability: d.min_probability,
                    assignment_law_id: d.assignment_law_id,
                    independent_units_assumption_id: d.independent_units_assumption_id,
                    administrative_followup_assumption_id: d.administrative_followup_assumption_id,
                    consistency_assumption_id: d.consistency_assumption_id,
                    paired_decisions: compiled.paired_decisions,
                })
            }
            "switchback" => {
                let d = switch_design(m)?;
                let realized: Vec<bool> = decoded(m, "realized_assignment")?;
                let _compiled = match compile_joint_switchback(d.clone()) {
                    Ok(c) => c,
                    Err(e) => return Ok(Err(e)),
                };
                let units = d.sequence_ids.iter().collect::<std::collections::BTreeSet<_>>().len();
                if realized.len() != d.sequence_ids.len()
                    || !realized.contains(&true)
                    || !realized.contains(&false)
                    || present(map, "compiled_source_artifact").is_none()
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
                    consistency_assumption_id: d.consistency_assumption_id,
                })
            }
            other => return Err(crate::value_err(format!("unknown design `{other}`"))),
        },
        None => None,
    };
    Ok(Ok(decoded_details))
}

fn replay_compiled_evidence(
    source: &antecedent::analysis::joint_panel::designs::NativeJointDesignResult,
    value: &JointExperimentEvidence,
) -> Result<JointExperimentEvidence, EvidenceRefusal> {
    let regimes = (&*value.treated_regime_id, &*value.control_regime_id);
    match value.design.assignment_design.as_str() {
        "cluster" => evidence_from_compiled_cluster(
            source.clone(),
            &value.evidence_id,
            regimes,
            value.horizon,
        )
        .map(|bound| bound.numerical_evidence().clone()),
        "adaptive_regime" => {
            evidence_from_compiled_adaptive_regime(source.clone(), &value.evidence_id)
                .map(|bound| bound.numerical_evidence().clone())
        }
        "temporal_excursion" => evidence_from_compiled_excursion(
            source.clone(),
            &value.evidence_id,
            regimes,
            value.horizon,
        )
        .map(|bound| bound.numerical_evidence().clone()),
        "temporal_regime" => {
            evidence_from_compiled_static_regime(source.clone(), &value.evidence_id, regimes)
                .map(|bound| bound.numerical_evidence().clone())
        }
        _ => evidence_from_compiled_experiment(
            source.clone(),
            &value.evidence_id,
            regimes,
            value.horizon,
        )
        .map(|bound| bound.numerical_evidence().clone()),
    }
}

fn evidence_with_context(
    value: &Json,
    ctx: &antecedent::prelude::ExecutionContext,
) -> PyResult<Result<ParsedEvidence, EvidenceRefusal>> {
    let map = object(value, "evidence")?;
    let dep = object(member(map, "dependence")?, "dependence")?;
    let dependence = match text(dep, "kind")?.as_str() {
        "joint" => {
            Dependence::Joint { treated: reals(dep, "treated")?, control: reals(dep, "control")? }
        }
        "point_only" => Dependence::PointOnly,
        "joint_arms" => Dependence::JointArms { covariance: reals(dep, "covariance")? },
        "marginals_only" => Dependence::MarginalsOnly {
            treated: reals(dep, "treated")?,
            control: reals(dep, "control")?,
        },
        other => return Err(crate::value_err(format!("unknown dependence kind `{other}`"))),
    };
    let design = object(member(map, "design")?, "design")?;
    let supports = match present(map, "marginal_supports") {
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
    let decoded_details = match decode_experimental_design_details(details, map, design)? {
        Ok(details) => details,
        Err(error) => return Ok(Err(error)),
    };
    let mut value = match built {
        Ok(e) => e,
        Err(error) => return Ok(Err(error)),
    };
    value.design_contract_id = design_contract_id;
    value.design_details = decoded_details;
    if !supports.is_empty() {
        value = value.with_marginal_supports(supports);
    }
    let owner = if matches!(
        value.design.assignment_design.as_str(),
        "temporal_excursion"
            | "temporal_regime"
            | "adaptive_regime"
            | "matched_pairs"
            | "switchback"
            | "cluster"
    ) {
        use antecedent::analysis::joint_panel::designs::{
            DesignArtifactLimits, replay_native_joint_design_execution,
        };
        let artifact: Vec<u8> = decoded(map, "compiled_source_artifact")?;
        let source = match replay_native_joint_design_execution(
            &artifact,
            ctx,
            DesignArtifactLimits::default(),
        ) {
            Ok(source) => source,
            Err(error) => return Ok(Err(compiled_decision_refusal(error))),
        };
        if value.design.assignment_design == "temporal_regime" {
            use antecedent::estimate::joint_design::compiled::JointDesignInference;
            let declared = object(member(map, "design_details")?, "regime details")?;
            match source.inference() {
                JointDesignInference::StaticRegimeBootstrap { replicates, seed } => {
                    if count(declared, "bootstrap_replicates")? != *replicates
                        || u64::try_from(count(declared, "bootstrap_seed")?).ok() != Some(*seed)
                    {
                        return Ok(Err(EvidenceRefusal::Invalid(
                            "bootstrap settings differ from the compiled source".into(),
                        )));
                    }
                }
                JointDesignInference::Frequentist => {
                    if present(declared, "bootstrap_replicates").is_some()
                        || present(declared, "bootstrap_seed").is_some()
                    {
                        return Ok(Err(EvidenceRefusal::Invalid(
                            "point-only source has no bootstrap settings".into(),
                        )));
                    }
                }
                _ => {
                    return Ok(Err(EvidenceRefusal::Invalid(
                        "unsupported regime inference".into(),
                    )));
                }
            }
        }
        let actual = match replay_compiled_evidence(&source, &value) {
            Ok(actual) => actual,
            Err(error) => return Ok(Err(error)),
        };
        let mut compatible = actual == value;
        if !compatible
            && let (Dependence::Joint { treated, control }, Dependence::MarginalsOnly { .. }) =
                (&actual.dependence, &value.dependence)
        {
            let k = actual.components.len();
            let mut downgraded = actual.clone();
            downgraded.dependence = Dependence::MarginalsOnly {
                treated: (0..k).map(|i| treated[i * k + i]).collect(),
                control: (0..k).map(|i| control[i * k + i]).collect(),
            };
            compatible = downgraded == value;
        }
        if !compatible {
            return Ok(Err(EvidenceRefusal::Invalid(
                "decision evidence differs from its compiled source".into(),
            )));
        }
        Some(source)
    } else {
        None
    };
    Ok(Ok(ParsedEvidence { value, owner }))
}

pub(crate) fn compiled_excursion_from_json(
    payload: &str,
    ctx: &antecedent::prelude::ExecutionContext,
) -> PyResult<antecedent::decision_evidence::CompiledExcursionEvidence> {
    let raw = parse_decision(payload, "excursion evidence")?;
    let parsed = evidence_with_context(&raw, ctx)?.map_err(|e| crate::value_err(e.to_string()))?;
    if parsed.value.design.assignment_design != "temporal_excursion" {
        return Err(crate::value_err("a retained excursion is required"));
    }
    let source = parsed.owner.ok_or_else(|| crate::value_err("missing compiled source"))?;
    evidence_from_compiled_excursion(
        source,
        &parsed.value.evidence_id,
        (&parsed.value.treated_regime_id, &parsed.value.control_regime_id),
        parsed.value.horizon,
    )
    .map_err(|e| crate::value_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature=(payload_json,component,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_excursion_quantity(
    payload_json: &str,
    component: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let source = compiled_excursion_from_json(payload_json, &ctx)?;
        Ok(finish(
            antecedent::decision_evidence::compiled_excursion_quantity(&source, component)
                .map(|q| json!(antecedent::io::quantity_wire::ScientificQuantityWire::from(&q))),
        ))
    })
}

fn request(value: &Json) -> PyResult<Request> {
    let map = object(value, "request")?;
    let target = match present(map, "target") {
        None => Target::ExperimentalContrast,
        Some(t) => {
            let t = object(t, "target")?;
            match text(t, "kind")?.as_str() {
                "experimental_contrast" => Target::ExperimentalContrast,
                "adaptive_regime" => Target::AdaptiveRegime {
                    periods: u32::try_from(count(t, "periods")?)
                        .map_err(|_| crate::value_err("adaptive horizon too large"))?,
                    exchangeability_assumption_id: text(t, "exchangeability_assumption_id")?,
                },
                "regime" => Target::Regime {
                    periods: u32::try_from(count(t, "periods")?)
                        .map_err(|_| crate::value_err("regime horizon too large"))?,
                    exchangeability_assumption_id: text(t, "exchangeability_assumption_id")?,
                },
                "excursion" => Target::Excursion {
                    lag: u32::try_from(count(t, "lag")?)
                        .map_err(|_| crate::value_err("lag too large"))?,
                    assignment_law_id: text(t, "assignment_law_id")?,
                },
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

pub(crate) fn expr(value: &Json) -> PyResult<UtilityExpr> {
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
                "more_independent_units" => Candidate::MoreIndependentUnits {
                    id: text(m, "id")?,
                    additional_units: member(m, "additional_units")?
                        .as_u64()
                        .ok_or_else(|| crate::value_err("`additional_units` must be an integer"))?,
                    same_design_assumption_id: text(m, "same_design_assumption_id")?,
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
        "support": r.support,
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
        json!({"id": r.id, "obligation": r.kind.as_str(), "planning_assumption_id": r.planning_assumption_id, "value": value})
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
    crate::catch_ffi(|| {
        let value = parse_decision(spec_json, "joint experiment")?;
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
    })
}

/// Compile complete-cluster outcomes and declared preassignment adjustment.
#[pyfunction]
#[pyo3(signature=(payload_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_compiled_cluster(
    payload_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let v = parse_decision(payload_json, "compiled cluster evidence")?;
        let m = object(&v, "compiled cluster evidence")?;
        let recipe_wire = object(member(m, "recipe")?, "cluster recipe")?;
        let source_exposure = object(member(recipe_wire, "exposure")?, "cluster exposure")?;
        if text(source_exposure, "kind")? == "unspecified" {
            return Ok(finish(Err(EvidenceRefusal::InterferenceUnspecified)));
        }
        let recipe: antecedent::estimate::joint_design::cluster_decision::JointClusterRecipe =
            decoded(m, "recipe")?;
        let evidence_id = text(m, "evidence_id")?;
        let bound = match compile_and_evidence_from_cluster(recipe, &evidence_id, &ctx) {
            Ok(value) => value,
            Err(error) => return Ok(finish(Err(error))),
        };
        let artifact = match bound.source().export_artifact_with_context(
            &ctx,
            antecedent::analysis::joint_panel::DesignArtifactLimits::default(),
        ) {
            Ok(value) => value,
            Err(error) => return Ok(finish(Err(compiled_decision_refusal(error)))),
        };
        let ev = bound.numerical_evidence();
        let Dependence::Joint { treated, control } = &ev.dependence else { unreachable!() };
        Ok(finish(Ok(json!({
            "evidence_id":ev.evidence_id, "population_id":ev.population_id,
            "treated_regime_id":ev.treated_regime_id, "control_regime_id":ev.control_regime_id,
            "horizon":ev.horizon, "components":m["components"], "treated_means":ev.treated_means,
            "control_means":ev.control_means, "dependence":{"kind":"joint", "treated":treated, "control":control},
            "design":{"assignment_design":"cluster", "treated_units":ev.design.treated_units,
                "control_units":ev.design.control_units, "outcome_rows":ev.design.outcome_rows},
            "design_contract_id":ev.design_contract_id, "compiled_source_artifact":artifact,
            "adjustments":m["adjustments"], "exposure":m["exposure"]
        }))))
    })
}

/// Compile a bounded adaptive policy comparison and retain its native source.
#[pyfunction]
#[pyo3(signature=(payload_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_adaptive(
    payload_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let v = parse_decision(payload_json, "adaptive regime evidence")?;
        let m = object(&v, "adaptive regime evidence")?;
        let recipe: antecedent::estimate::joint_design::adaptive_regime::JointAdaptiveRegimeRecipe =
            decoded(m, "recipe")?;
        let bound =
            match compile_and_evidence_from_adaptive_regime(recipe, &text(m, "evidence_id")?, &ctx)
            {
                Ok(v) => v,
                Err(e) => return Ok(finish(Err(e))),
            };
        let artifact = match bound.source().export_artifact_with_context(
            &ctx,
            antecedent::analysis::joint_panel::DesignArtifactLimits::default(),
        ) {
            Ok(v) => v,
            Err(e) => return Ok(finish(Err(compiled_decision_refusal(e)))),
        };
        let ev = bound.numerical_evidence();
        let Some(DesignDetails::AdaptiveRegime {
            subject_ids,
            periods,
            policy_prespecification_assumption_id,
            positivity_assumption_id,
            exchangeability_assumption_id,
            consistency_assumption_id,
            followup_assumption_id,
            failures,
        }) = &ev.design_details
        else {
            unreachable!()
        };
        let dependence = match &ev.dependence {
            Dependence::JointArms { covariance } => {
                json!({"kind":"joint_arms", "covariance":covariance})
            }
            Dependence::PointOnly => json!({"kind":"point_only"}),
            _ => unreachable!(),
        };
        let components: Vec<_> = ev
            .components
            .iter()
            .map(|c| json!({"name":c.name, "variable_id":c.variable_id, "units":c.units}))
            .collect();
        let Exposure::NoBetweenUnitInterference { assumption_id } = &ev.exposure else {
            unreachable!()
        };
        Ok(finish(Ok(json!({
            "evidence_id":ev.evidence_id, "population_id":ev.population_id,
            "treated_regime_id":ev.treated_regime_id, "control_regime_id":ev.control_regime_id,
            "horizon":ev.horizon, "components":components, "treated_means":ev.treated_means, "control_means":ev.control_means,
            "dependence":dependence, "design":{"assignment_design":"adaptive_regime", "treated_units":ev.design.treated_units,"control_units":ev.design.control_units,"outcome_rows":ev.design.outcome_rows},
            "design_contract_id":ev.design_contract_id, "compiled_source_artifact":artifact,
            "design_details":{"kind":"adaptive_regime", "subject_ids":subject_ids,"periods":periods,"policy_prespecification_assumption_id":policy_prespecification_assumption_id,"positivity_assumption_id":positivity_assumption_id,"exchangeability_assumption_id":exchangeability_assumption_id,"consistency_assumption_id":consistency_assumption_id,"followup_assumption_id":followup_assumption_id,"failures":failures},
            "adjustments":[], "exposure":{"kind":"no_between_unit_interference","assumption_id":assumption_id}
        }))))
    })
}

struct ExperimentalProjection<'a> {
    map: &'a Map<String, Json>,
    design: &'a Map<String, Json>,
    assigned: &'a [bool],
    evidence_id: &'a str,
    population: &'a str,
    treated: &'a str,
    control: &'a str,
    horizon: u32,
}

#[derive(Clone, Copy)]
enum ExperimentalProjectionKind {
    Regime,
    Excursion,
    Assignment,
}

impl ExperimentalProjection<'_> {
    fn compiled(
        &self,
        ev: &JointExperimentEvidence,
        artifact: Vec<u8>,
        kind: ExperimentalProjectionKind,
    ) -> Json {
        let mut retained_design = self.design.clone();
        retained_design.insert("realized_assignment".into(), json!(self.assigned));
        match (kind, &ev.design_details) {
            (
                ExperimentalProjectionKind::Regime,
                Some(DesignDetails::TemporalRegime { observed_minimum_probabilities, .. }),
            ) => {
                retained_design.insert(
                    "observed_minimum_probabilities".into(),
                    json!(observed_minimum_probabilities),
                );
            }
            (
                ExperimentalProjectionKind::Excursion,
                Some(DesignDetails::TemporalExcursion { paired_decisions, .. }),
            ) => {
                retained_design.insert("paired_decisions".into(), json!(paired_decisions));
            }
            _ => {}
        }
        let dependence = match (&ev.dependence, kind) {
            (Dependence::PointOnly, ExperimentalProjectionKind::Regime) => {
                json!({"kind":"point_only"})
            }
            (Dependence::JointArms { covariance }, _) => {
                json!({"kind":"joint_arms", "covariance":covariance})
            }
            _ => unreachable!(),
        };
        json!({
            "evidence_id": self.evidence_id, "population_id": self.population,
            "treated_regime_id": self.treated, "control_regime_id": self.control, "horizon": self.horizon,
            "components": self.map["components"], "treated_means": ev.treated_means,
            "control_means": ev.control_means, "dependence": dependence,
            "design": {"assignment_design":ev.design.assignment_design,"treated_units":ev.design.treated_units,
                "control_units":ev.design.control_units,"outcome_rows":ev.design.outcome_rows},
            "design_contract_id":ev.design_contract_id, "design_details":retained_design,
            "compiled_source_artifact":artifact, "adjustments":[], "exposure":self.map["exposure"]
        })
    }
}

fn export_experimental_source(
    source: &antecedent::analysis::joint_panel::designs::NativeJointDesignResult,
    ctx: &antecedent::prelude::ExecutionContext,
) -> Result<Vec<u8>, EvidenceRefusal> {
    source
        .export_artifact_with_context(
            ctx,
            antecedent::analysis::joint_panel::DesignArtifactLimits::default(),
        )
        .map_err(compiled_decision_refusal)
}

/// Prepare a design or execute joint estimation against a declared design.
#[pyfunction]
#[pyo3(signature=(payload_json,fit,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_experimental(
    payload_json: &str,
    fit: bool,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let payload = parse_decision(payload_json, "experimental evidence")?;
        let map = object(&payload, "experimental evidence")?;
        let dm = object(member(map, "design")?, "design")?;
        let kind = text(dm, "kind")?;
        let (identity, obligations, paired, switchback, excursion, regime) = match kind.as_str() {
            "matched_pairs" => match compile_matched_pair_design(paired_design(dm)?) {
                Ok(d) => (d.identity.clone(), d.obligations.clone(), Some(d), None, None, None),
                Err(e) => return Ok(finish(Err(e))),
            },
            "switchback" => match compile_joint_switchback(switch_design(dm)?) {
                Ok(d) => (d.identity.clone(), d.obligations.clone(), None, Some(d), None, None),
                Err(e) => return Ok(finish(Err(e))),
            },
            "temporal_excursion" => match compile_joint_excursion(excursion_design(dm)?) {
                Ok(d) => (d.identity.clone(), d.obligations.clone(), None, None, Some(d), None),
                Err(e) => return Ok(finish(Err(e))),
            },
            "temporal_regime" => match compile_joint_static_regime(regime_design(dm)?) {
                Ok(d) => (d.identity.clone(), d.obligations.clone(), None, None, None, Some(d)),
                Err(e) => return Ok(finish(Err(e))),
            },
            _ => return Ok(finish(Err(EvidenceRefusal::UnsupportedAssignmentDesign(kind)))),
        };
        if !fit {
            return Ok((
                Some(
                    json!({"identity":identity,"obligations":obligations,"design":dm}).to_string(),
                ),
                None,
            ));
        }
        let raw: Vec<Vec<f64>> = decoded(map, "outcomes")?;
        let slices: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
        let assigned: Vec<bool> = decoded(map, "assigned")?;
        let horizon = u32::try_from(count(map, "horizon")?)
            .map_err(|_| crate::value_err("horizon too large"))?;
        let comps = components(map, "components")?;
        let evidence_id = text(map, "evidence_id")?;
        let population = text(map, "population_id")?;
        let treated = text(map, "treated_regime_id")?;
        let control = text(map, "control_regime_id")?;
        let exp = exposure(map)?;
        let projection = ExperimentalProjection {
            map,
            design: dm,
            assigned: &assigned,
            evidence_id: &evidence_id,
            population: &population,
            treated: &treated,
            control: &control,
            horizon,
        };
        let confounders: Vec<Vec<f64>> =
            if regime.is_some() { decoded(map, "confounders")? } else { Vec::new() };
        if let Some(d) = &regime
            && (d.design.realized_assignment != assigned
                || horizon as usize != d.design.active_path.len())
        {
            return Ok(finish(Err(EvidenceRefusal::Invalid(
                "regime assignment and horizon must match the compiled histories".into(),
            ))));
        }
        if let Some(d) = regime.as_ref() {
            let fitted = if present(dm, "bootstrap_replicates").is_some() {
                let replicates = count(dm, "bootstrap_replicates")?;
                let seed = u64::try_from(count(dm, "bootstrap_seed")?)
                    .map_err(|_| crate::value_err("bootstrap seed too large"))?;
                compile_and_evidence_from_joint_static_regime_bootstrap(
                    d,
                    &slices,
                    &confounders,
                    &evidence_id,
                    &population,
                    (&treated, &control),
                    comps.clone(),
                    exp.clone(),
                    replicates,
                    seed,
                    &ctx,
                )
            } else {
                if present(dm, "bootstrap_seed").is_some() {
                    return Ok(finish(Err(EvidenceRefusal::Invalid(
                        "bootstrap seed requires replicate count".into(),
                    ))));
                }
                compile_and_evidence_from_joint_static_regime(
                    d,
                    &slices,
                    &confounders,
                    &evidence_id,
                    &population,
                    (&treated, &control),
                    comps.clone(),
                    exp.clone(),
                    &ctx,
                )
            };
            let bound = match fitted {
                Ok(value) => value,
                Err(error) => return Ok(finish(Err(error))),
            };
            let artifact = match export_experimental_source(bound.source(), &ctx) {
                Ok(value) => value,
                Err(error) => return Ok(finish(Err(error))),
            };
            return Ok(finish(Ok(projection.compiled(
                bound.numerical_evidence(),
                artifact,
                ExperimentalProjectionKind::Regime,
            ))));
        }
        if let Some(d) = excursion.as_ref() {
            let bound = match compile_and_evidence_from_joint_excursion(
                d,
                &slices,
                &assigned,
                &evidence_id,
                &population,
                (&treated, &control),
                horizon,
                comps.clone(),
                exp.clone(),
                &ctx,
            ) {
                Ok(value) => value,
                Err(error) => return Ok(finish(Err(error))),
            };
            let artifact = match export_experimental_source(bound.source(), &ctx) {
                Ok(value) => value,
                Err(error) => return Ok(finish(Err(error))),
            };
            return Ok(finish(Ok(projection.compiled(
                bound.numerical_evidence(),
                artifact,
                ExperimentalProjectionKind::Excursion,
            ))));
        }
        if paired.is_some() || switchback.is_some() {
            let result = if let Some(d) = paired.as_ref() {
                compile_and_evidence_from_matched_pairs(
                    d,
                    &slices,
                    &assigned,
                    &evidence_id,
                    &population,
                    (&treated, &control),
                    horizon,
                    comps.clone(),
                    exp.clone(),
                    &ctx,
                )
            } else {
                compile_and_evidence_from_joint_switchback(
                    switchback.as_ref().expect("compiled switchback"),
                    &slices,
                    &assigned,
                    &evidence_id,
                    &population,
                    (&treated, &control),
                    horizon,
                    comps.clone(),
                    exp.clone(),
                    &ctx,
                )
            };
            let bound = match result {
                Ok(value) => value,
                Err(error) => return Ok(finish(Err(error))),
            };
            let artifact = match export_experimental_source(bound.source(), &ctx) {
                Ok(value) => value,
                Err(error) => return Ok(finish(Err(error))),
            };
            return Ok(finish(Ok(projection.compiled(
                bound.numerical_evidence(),
                artifact,
                ExperimentalProjectionKind::Assignment,
            ))));
        }
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
        } else if let Some(d) = regime {
            evidence_from_joint_static_regime(
                &d,
                &slices,
                &confounders,
                &evidence_id,
                &population,
                (&treated, &control),
                comps,
                exp,
            )
        } else if let Some(d) = excursion {
            evidence_from_joint_excursion(
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
        if let Some(DesignDetails::TemporalRegime { observed_minimum_probabilities, .. }) = &ev.design_details {
            retained_design.insert("observed_minimum_probabilities".into(), json!(observed_minimum_probabilities));
        }
        if let Some(DesignDetails::TemporalExcursion { paired_decisions, .. }) = &ev.design_details {
            retained_design.insert("paired_decisions".into(), json!(paired_decisions));
        }
        let dependence = match ev.dependence {Dependence::JointArms {covariance}=>json!({"kind":"joint_arms","covariance":covariance}),Dependence::PointOnly=>json!({"kind":"point_only"}),_=>unreachable!()};
        json!({"evidence_id":evidence_id,"population_id":population,"treated_regime_id":treated,"control_regime_id":control,"horizon":horizon,
          "components":map["components"],"treated_means":ev.treated_means,"control_means":ev.control_means,
          "dependence":dependence,
          "design":{"assignment_design":ev.design.assignment_design,"treated_units":ev.design.treated_units,"control_units":ev.design.control_units,"outcome_rows":ev.design.outcome_rows},
          "design_contract_id":identity,"design_details":retained_design,"adjustments":[],"exposure":map["exposure"]})
    })))
    })
}

/// Apply an explicitly scoped causal proxy-effect bridge.
#[pyfunction]
#[pyo3(signature=(payload_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_proxy(
    payload_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let value = parse_decision(payload_json, "proxy evidence")?;
        let map = object(&value, "proxy evidence")?;
        let ev = match evidence_with_context(member(map, "evidence")?, &ctx)? {
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
    })
}

fn temporal_estimand(value: &Json) -> PyResult<TemporalEstimand> {
    let m = object(value, "temporal estimand")?;
    let horizon =
        || u32::try_from(count(m, "horizon")?).map_err(|_| crate::value_err("horizon too large"));
    Ok(match text(m, "kind")?.as_str() {
        "excursion" => TemporalEstimand::Excursion {
            lag: u32::try_from(count(m, "lag")?).map_err(|_| crate::value_err("lag too large"))?,
            assignment_law_id: text(m, "assignment_law_id")?,
        },
        "fixed_schedule" => TemporalEstimand::FixedSchedule {
            active_schedule_id: text(m, "active_schedule_id")?,
            reference_schedule_id: text(m, "reference_schedule_id")?,
            horizon: horizon()?,
        },
        "static_regime" => TemporalEstimand::StaticRegime {
            active_regime_id: text(m, "active_regime_id")?,
            reference_regime_id: text(m, "reference_regime_id")?,
            horizon: horizon()?,
            exchangeability_assumption_id: text(m, "exchangeability_assumption_id")?,
        },
        "adaptive_regime" => TemporalEstimand::AdaptiveRegime {
            active_regime_id: text(m, "active_regime_id")?,
            reference_regime_id: text(m, "reference_regime_id")?,
            horizon: horizon()?,
        },
        "model_response" => TemporalEstimand::ModelResponse {
            model_id: text(m, "model_id")?,
            active_schedule_id: text(m, "active_schedule_id")?,
            reference_schedule_id: text(m, "reference_schedule_id")?,
            horizon: horizon()?,
        },
        other => return Err(crate::value_err(format!("unknown temporal estimand `{other}`"))),
    })
}

/// Evaluate a declared bounded mapping between distinct temporal estimands.
#[pyfunction]
#[pyo3(signature=(payload_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_temporal_bridge(
    payload_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let v = parse_decision(payload_json, "temporal bridge")?;
        let m = object(&v, "temporal bridge")?;
        let ev = match evidence_with_context(member(m, "evidence")?, &ctx)? {
            Ok(e) => e,
            Err(e) => return Ok(finish(Err(e))),
        };
        let req = request(member(m, "request")?)?;
        let frame = frame(member(m, "frame")?)?;
        let b = object(member(m, "bridge")?, "bridge")?;
        let mapping = object(member(b, "mapping")?, "mapping")?;
        let mapping = match text(mapping, "kind")?.as_str() {
            "predictive_association" => ProxyMapping::PredictiveAssociation,
            "causal_effect_bounds" => ProxyMapping::CausalEffectBounds {
                multiplier: decoded(mapping, "multiplier")?,
                residual: decoded(mapping, "residual")?,
                assumption_id: text(mapping, "assumption_id")?,
                mapping_evidence_id: text(mapping, "mapping_evidence_id")?,
            },
            other => return Err(crate::value_err(format!("unknown temporal mapping `{other}`"))),
        };
        let bridge = TemporalEffectBridge {
            source: temporal_estimand(member(b, "source")?)?,
            target: temporal_estimand(member(b, "target")?)?,
            source_component: text(b, "source_component")?,
            target_variable_id: text(b, "target_variable_id")?,
            target_units: text(b, "target_units")?,
            mapping,
        };
        Ok(finish(decide_with_temporal_bridge(&ev,&req,&bridge,&frame).map(|r|json!({"identity":r.identity,"source":b["source"],"target":b["target"],"assumption_range":r.decision.assumption_range,"sampling_envelope":r.decision.sampling_envelope,"verdict":verdict_json(&r.decision.verdict),"evidence_ids":r.decision.evidence_ids,"assumption_id":r.decision.assumption_id,"target_variable_id":r.decision.target_variable_id,"target_units":r.decision.target_units}))))
    })
}

/// Derive a quantity and, when a frame is given, classify the decision; with candidates, plan
/// the next evidence. Returns the result JSON or a refusal.
#[pyfunction]
#[pyo3(signature=(payload_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn decision_evidence_evaluate(
    payload_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused> {
    crate::catch_ffi(|| {
        let ctx = crate::joint_panel_api::context(seed, threads, max_bytes, cancel)?;
        let value = parse_decision(payload_json, "decision evidence")?;
        let map = object(&value, "decision evidence")?;
        let ev = match evidence_with_context(member(map, "evidence")?, &ctx)? {
            Ok(ev) => ev,
            Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
        };
        let req = request(member(map, "request")?)?;
        let q = quantity(member(map, "quantity")?)?;
        let frame = present(map, "frame").map(frame).transpose()?;
        let cands = candidates(map)?;
        let run = || -> Result<Json, EvidenceRefusal> {
            let mut out = Map::new();
            if let Some(f) = &frame {
                let d = decide(&ev, &req, &q, f)?;
                out.insert("decision".into(), decision_json(&d));
                out.insert("stage_digests".into(), digests_json(&StageDigests::of(&ev, &q, f)));
                if !cands.is_empty() {
                    out.insert("plan".into(), plan_json(&plan_evidence(&ev, &req, &q, f, &cands)?));
                }
            } else {
                if !cands.is_empty() {
                    return Err(invalid("planning needs a decision frame"));
                }
                out.insert("derived".into(), derived_json(&derive(&ev, &req, &q)?));
            }
            out.insert("estimate_identity".into(), Json::String(ev.estimate_identity()));
            Ok(Json::Object(out))
        };
        Ok(finish(run()))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decision_evidence_compile, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_adaptive, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_compiled_cluster, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_excursion_quantity, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_experimental, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_proxy, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_temporal_bridge, module)?)?;
    module.add_function(wrap_pyfunction!(decision_evidence_evaluate, module)?)?;
    Ok(())
}
