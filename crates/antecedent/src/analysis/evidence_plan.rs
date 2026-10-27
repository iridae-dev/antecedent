//! Evidence plans: from a failed identification to the studies that would close
//! the gap.
//!
//! A study whose effect is not identified holds more than the refusal: the retained
//! product, the structure it ran on and the variables measured today are enough to
//! rebuild every theorem route that a proposed study could repair (back-door,
//! front-door, instrument, hedge and, given a sensitivity finding, the confounder
//! it names). [`PreparedStudy::evidence_plan`] takes the studies a caller declares
//! (cost, timing, recruitment and sample size are the caller's, never defaulted),
//! asks each route's own theorem checker which of them would identify the effect,
//! optionally values them under a declared decision, prior and signal, and returns
//! one [`EvidencePlan`]: the options ordered by the identification gate and then by
//! expected value net of the declared cost map, each stating what it resolves, what
//! it costs and what remains unresolved.
//!
//! [`PreparedStudy::candidate_templates`] is the other half: the unfinished studies
//! the obligations determine, to be completed with declarations and then planned.
//!
//! The plan is hypothetical identification, not an estimate: no data are consumed,
//! no assumption is established by a proposed study, a stopped search is a budget and
//! never an impossibility, the ordering is `point_only` and its calibration
//! `unmeasured`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_design::{
    CandidateGeneration, DurableStudyCandidate, EvidencePlanError, FailureRoutes, PlanInputs,
    PlanValue, RepairArtifactError, RepairError, RepairLimits, RepairObjective, RepairReport,
    RepairReportArtifact, SensitivityFinding, failure_routes, generate_candidates,
    latent_expansion, repair,
};
use antecedent_graph::Admg;
use serde::Deserialize;

use super::contract::{FailureInputs, failure_evidence_needs, failure_inputs, failure_inputs_of};
use super::design_ranking::{
    CandidateDeclWire, CostDeclWire, CostMapDeclWire, DecisionDeclWire, DesignRankingError,
    DesignRankingEvaluation, DesignRankingRequestWire, MonteCarloDeclWire, ProviderDeclWire,
    SignalDeclWire, evaluate,
};
use super::execute::Study;
use super::prepared::PreparedStudy;
use crate::error::CausalError;

pub use antecedent_design::{
    EvidencePlan, EvidencePlanArtifact, EvidencePlanArtifactError, PlanOption, PlanOutcome,
    PlanRoute, PlanValueBasis, UnverifiedRoute,
};

/// The population label a study's data carry; candidates are declared in it.
pub const STUDY_POPULATION: &str = "study";

/// The value of one declared study: the signal it would produce about the decision
/// state, under the declared prior.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyValueWire {
    /// Label of the declared study the value is for (labels are unique).
    pub label: String,
    /// The exact signal request.
    pub signal: SignalDeclWire,
    /// The signal provider.
    pub provider: ProviderDeclWire,
    /// Identities of existing observations the study would reuse.
    #[serde(default)]
    pub reused_observation_ids: Vec<String>,
}

/// A declared decision, prior and signals that value the studies.
///
/// The decision's prior is the caller's belief and is recorded with the ranking;
/// an assumption range from a sensitivity analysis never becomes a probability
/// here. A study with no entry in `studies` is listed unvalued.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceValueRequest {
    /// The decision problem and belief.
    pub decision: DecisionDeclWire,
    /// The valued studies, by label.
    pub studies: Vec<StudyValueWire>,
    /// Cost-to-utility mapping; absent means value and cost are reported separately.
    #[serde(default)]
    pub cost_map: Option<CostMapDeclWire>,
    /// Refuse instead of reporting separately when no mapping is supplied.
    #[serde(default)]
    pub require_net_value: bool,
    /// Observation identities the prior already summarizes.
    #[serde(default)]
    pub prior_observation_ids: Vec<String>,
    /// Digests of the source distributions behind the prior and any external laws.
    #[serde(default)]
    pub source_digests: Vec<String>,
    /// RNG seed of the evaluation.
    pub rng_seed: u64,
    /// Monte Carlo configuration (default: the engine's).
    #[serde(default)]
    pub monte_carlo: Option<MonteCarloDeclWire>,
}

/// What a plan is asked for.
#[derive(Clone, Debug)]
pub struct EvidencePlanRequest {
    /// The declared candidate studies, in the study's population
    /// ([`STUDY_POPULATION`]).
    pub candidates: Vec<DurableStudyCandidate>,
    /// How sufficient subsets are ranked inside each route's search.
    pub objective: RepairObjective,
    /// The declared search bounds.
    pub limits: RepairLimits,
    /// A non-robust sensitivity finding, adding the route that measures the
    /// confounder it names.
    pub sensitivity: Option<SensitivityFinding>,
    /// A declared decision that values the studies.
    pub value: Option<EvidenceValueRequest>,
}

/// Why a plan could not be produced.
#[derive(Debug)]
#[non_exhaustive]
pub enum EvidencePlanFacadeError {
    /// The study retains no not-identified product on a directed or bidirected
    /// graph, so there is no gap to plan for.
    NoFailure,
    /// The failed identification has no route a study could repair.
    NoRoute,
    /// A repair search or route refused.
    Repair(RepairError),
    /// The study did not compile, so no retained failure could be read.
    Compile(CausalError),
    /// The plan refused (unequal cost units, no studies, ...).
    Plan(EvidencePlanError),
    /// The value evaluation refused.
    Value(DesignRankingError),
    /// A study named by the value request is not declared, or is named twice.
    UnknownValueStudy(String),
}

impl EvidencePlanFacadeError {
    /// Registered runtime-refusal reason code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoFailure | Self::NoRoute | Self::UnknownValueStudy(_) | Self::Compile(_) => {
                antecedent_core::reason_code!("invalid_argument")
            }
            Self::Repair(error) => error.code,
            Self::Plan(error) => error.code,
            Self::Value(error) => error
                .refusal()
                .map_or(antecedent_core::reason_code!("invalid_argument"), |r| r.code),
        }
    }

    /// Stable detail.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            Self::NoFailure => "evidence_plan.no_failure_to_plan".to_owned(),
            Self::NoRoute => "evidence_plan.no_route".to_owned(),
            Self::UnknownValueStudy(_) => "evidence_plan.invalid_request".to_owned(),
            Self::Compile(_) => "evidence_plan.compile_refused".to_owned(),
            Self::Repair(error) => error.detail.to_owned(),
            Self::Plan(error) => error.detail.to_owned(),
            Self::Value(error) => error
                .refusal()
                .map_or_else(|| "design_ranking.invalid_request".to_owned(), |r| r.detail),
        }
    }
}

impl std::fmt::Display for EvidencePlanFacadeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFailure => write!(
                f,
                "evidence_plan.no_failure_to_plan: the study retains no not-identified product"
            ),
            Self::NoRoute => write!(
                f,
                "evidence_plan.no_route: the failed identification has no route a study could \
                 repair"
            ),
            Self::UnknownValueStudy(label) => write!(
                f,
                "evidence_plan.invalid_request: the value request names study `{label}` that is \
                 not declared exactly once"
            ),
            Self::Compile(error) => write!(
                f,
                "evidence_plan.compile_refused: the study did not compile, so its retained \
                 failure cannot be read: {error}"
            ),
            Self::Repair(error) => write!(f, "{error}"),
            Self::Plan(error) => write!(f, "{error}"),
            Self::Value(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EvidencePlanFacadeError {}

impl From<RepairError> for EvidencePlanFacadeError {
    fn from(error: RepairError) -> Self {
        Self::Repair(error)
    }
}

impl From<EvidencePlanError> for EvidencePlanFacadeError {
    fn from(error: EvidencePlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<DesignRankingError> for EvidencePlanFacadeError {
    fn from(error: DesignRankingError) -> Self {
        Self::Value(error)
    }
}

/// A produced plan with everything it was derived from.
#[derive(Debug)]
pub struct EvidencePlanResult {
    /// The ordered options.
    pub plan: EvidencePlan,
    /// The unfinished studies the failure's obligations determine.
    pub generation: CandidateGeneration,
    /// One repair report per route, in route order.
    pub reports: Vec<RepairReport>,
    /// The value evaluation, when one was requested.
    pub value: Option<DesignRankingEvaluation>,
    /// Variable names by id: the study's variables, then the unmeasured common
    /// causes of the graph's bidirected edges and the sensitivity confounder.
    pub names: Vec<String>,
    artifact: Result<EvidencePlanArtifact, EvidencePlanArtifactError>,
}

impl EvidencePlanResult {
    /// The name of `variable`.
    #[must_use]
    pub fn name_of(&self, variable: VariableId) -> String {
        self.names
            .get(variable.as_usize())
            .cloned()
            .unwrap_or_else(|| format!("v{}", variable.raw()))
    }

    /// The sealed, replayable artifact, or why it cannot be sealed (a graph the
    /// repair artifact cannot store, a cancelled search).
    ///
    /// # Errors
    /// The artifact refusal.
    pub fn artifact(&self) -> Result<&EvidencePlanArtifact, &EvidencePlanArtifactError> {
        self.artifact.as_ref()
    }

    /// Export the checksummed `evidence_plan_v1` container.
    ///
    /// # Errors
    /// The artifact could not be sealed, a blank id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, EvidencePlanArtifactError> {
        self.artifact.as_ref().map_err(|error| (*error).clone())?.to_bytes(artifact_id)
    }
}

/// The variable names a plan is spelled in, by id: the study's variables, then the
/// unmeasured common cause of each bidirected edge (`common_cause_of_<a>_<b>`) and,
/// for a sensitivity route, `unmeasured_confounder` (suffixed to stay distinct).
#[must_use]
pub fn variable_names(names: &[Arc<str>], routes: &FailureRoutes, graph: &Admg) -> Vec<String> {
    graph_variable_names(names, graph, routes.confounder)
}

/// The variable names of a failure on `graph`: the study's variables, then the unmeasured
/// common cause of each bidirected edge and, when a sensitivity route names one, its confounder.
#[must_use]
pub(crate) fn graph_variable_names(
    names: &[Arc<str>],
    graph: &Admg,
    confounder: Option<VariableId>,
) -> Vec<String> {
    let mut out: Vec<String> = names.iter().map(ToString::to_string).collect();
    let latents = latent_expansion(graph).map(|(_, l)| l).unwrap_or_default();
    let name_of = |variable: VariableId, out: &[String]| {
        out.get(variable.as_usize()).cloned().unwrap_or_else(|| format!("v{}", variable.raw()))
    };
    let mut extra: Vec<(VariableId, String)> = Vec::new();
    for latent in &latents {
        let label = format!(
            "common_cause_of_{}_{}",
            name_of(latent.between.0, &out),
            name_of(latent.between.1, &out)
        );
        extra.push((latent.variable, label));
    }
    if let Some(confounder) = confounder {
        let mut label = "unmeasured_confounder".to_owned();
        while out.contains(&label) || extra.iter().any(|(_, l)| *l == label) {
            label.push('_');
        }
        extra.push((confounder, label));
    }
    for (variable, label) in extra {
        let index = variable.as_usize();
        while out.len() <= index {
            out.push(format!("v{}", out.len()));
        }
        out[index] = label;
    }
    out
}

fn value_request(
    value: &EvidenceValueRequest,
    candidates: &[DurableStudyCandidate],
) -> Result<DesignRankingRequestWire, EvidencePlanFacadeError> {
    let mut declared = Vec::with_capacity(value.studies.len());
    for study in &value.studies {
        let mut matches = candidates.iter().filter(|c| c.label.as_ref() == study.label);
        let (Some(candidate), None) = (matches.next(), matches.next()) else {
            return Err(EvidencePlanFacadeError::UnknownValueStudy(study.label.clone()));
        };
        declared.push(CandidateDeclWire {
            semantic_id: candidate.semantic_id().to_string(),
            sample_size: candidate.sample_size,
            cost: CostDeclWire {
                amount: candidate.cost.units as f64,
                unit: candidate.cost.unit_label.to_string(),
            },
            signal: study.signal.clone(),
            provider: study.provider.clone(),
            reused_observation_ids: study.reused_observation_ids.clone(),
        });
    }
    Ok(DesignRankingRequestWire {
        decision: value.decision.clone(),
        candidates: declared,
        cost_map: value.cost_map.clone(),
        require_net_value: value.require_net_value,
        prior_observation_ids: value.prior_observation_ids.clone(),
        source_digests: value.source_digests.clone(),
        rng_seed: value.rng_seed,
        mc_error_tolerance: 1e-3,
        tie_tolerance: 1e-12,
        max_candidates: antecedent_design::design_ranking_artifact::MAX_RANKING_CANDIDATES,
        monte_carlo: value.monte_carlo.clone(),
    })
}

/// Plan over the routes of an already-read failure.
///
/// # Errors
/// [`EvidencePlanFacadeError::NoRoute`] when no route remains, any refusal of a
/// repair search, of the value evaluation, or of the plan (unequal cost units, no
/// declared studies).
pub fn plan_over_routes(
    routes: &FailureRoutes,
    names: Vec<String>,
    request: &EvidencePlanRequest,
    ctx: &ExecutionContext,
) -> Result<EvidencePlanResult, EvidencePlanFacadeError> {
    if routes.routes.is_empty() {
        return Err(EvidencePlanFacadeError::NoRoute);
    }
    let generation = generate_candidates(&routes.obligations());
    let mut reports = Vec::with_capacity(routes.routes.len());
    let mut repair_artifacts: Vec<Result<RepairReportArtifact, RepairArtifactError>> = Vec::new();
    for route in &routes.routes {
        let report =
            repair(route.family(), &request.candidates, request.objective, request.limits, ctx)?;
        repair_artifacts.push(RepairReportArtifact::build(
            route.reference(),
            &request.candidates,
            &report,
        ));
        reports.push(report);
    }
    let evaluation = match &request.value {
        Some(value) => Some(evaluate(value_request(value, &request.candidates)?)?),
        None => None,
    };
    let value = evaluation
        .as_ref()
        .map(|e| PlanValue::from_ranking(e.ranking(), e.identity()))
        .transpose()?;
    let plan = EvidencePlan::build(
        PlanInputs { reports: &reports, candidates: &request.candidates, value: value.as_ref() },
        routes.unverified(),
    )?;
    let artifact = repair_artifacts
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(EvidencePlanArtifactError::from)
        .and_then(|repairs| {
            EvidencePlanArtifact::build(
                &repairs,
                evaluation.as_ref().map(DesignRankingEvaluation::artifact),
                &plan,
            )
        });
    Ok(EvidencePlanResult { plan, generation, reports, value: evaluation, names, artifact })
}

impl PreparedStudy {
    fn failure(
        &self,
        sensitivity: Option<&SensitivityFinding>,
    ) -> Result<(FailureInputs, FailureRoutes), EvidencePlanFacadeError> {
        let inputs = failure_inputs(self.study()).ok_or(EvidencePlanFacadeError::NoFailure)?;
        let routes = failure_routes(
            &inputs.result,
            &inputs.graph,
            &inputs.observed,
            STUDY_POPULATION,
            sensitivity,
        )?;
        Ok((inputs, routes))
    }

    /// The unfinished studies this study's retained not-identified product
    /// determines: what to collect, never at what cost.
    ///
    /// # Errors
    /// [`EvidencePlanFacadeError::NoFailure`] when the study retains no
    /// not-identified product, or a route the identifiers cannot read.
    pub fn candidate_templates(
        &self,
        sensitivity: Option<&SensitivityFinding>,
    ) -> Result<CandidateGeneration, EvidencePlanFacadeError> {
        let (_, routes) = self.failure(sensitivity)?;
        Ok(generate_candidates(&routes.obligations()))
    }

    /// The options that would close this study's identification gap, verified by
    /// each route's theorem checker and ordered by the gate and then by value net of
    /// the declared cost map.
    ///
    /// # Errors
    /// As [`plan_over_routes`], and [`EvidencePlanFacadeError::NoFailure`] when the
    /// study retains no not-identified product.
    pub fn evidence_plan(
        &self,
        request: &EvidencePlanRequest,
        ctx: &ExecutionContext,
    ) -> Result<EvidencePlanResult, EvidencePlanFacadeError> {
        let (inputs, routes) = self.failure(request.sensitivity.as_ref())?;
        let names = variable_names(&inputs.names, &routes, &inputs.graph);
        plan_over_routes(&routes, names, request, ctx)
    }
}

impl Study {
    /// The failure this study's compiled program retained: the not-identified product the
    /// identifier computed, the structure it ran on and the variables measured today.
    ///
    /// A study whose effect is not identified cannot be prepared: preparation raises the
    /// refusal. Compile does not raise it; the lowered program defers and keeps the product,
    /// so the evidence that would resolve the refusal is read from the same study.
    fn retained_failure(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<FailureInputs, EvidencePlanFacadeError> {
        let plan = self.compile(ctx).map_err(EvidencePlanFacadeError::Compile)?;
        let result = plan
            .lowered()
            .and_then(|lowered| lowered.refused_identification())
            .cloned()
            .ok_or(EvidencePlanFacadeError::NoFailure)?;
        failure_inputs_of(self, result).ok_or(EvidencePlanFacadeError::NoFailure)
    }

    /// The verified evidence obligations and evidence-gathering next actions of a study
    /// whose effect is not identified, read from the not-identified product its compiled
    /// program retained (`measure_variable:<name>`, `run_experiment:<name>`,
    /// `find_instrument`, `collect_panel`).
    ///
    /// The refusal itself is unchanged: [`Study::prepare`] and [`Study::run`] still raise it.
    ///
    /// # Errors
    /// [`EvidencePlanFacadeError::NoFailure`] when the study retains no not-identified
    /// product (it identifies, or its structure is not a directed or bidirected graph), and
    /// [`EvidencePlanFacadeError::Compile`] when it does not compile.
    pub fn evidence_obligations(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        (Vec<antecedent_contract::EvidenceObligation>, Vec<antecedent_contract::NextAction>),
        EvidencePlanFacadeError,
    > {
        let inputs = self.retained_failure(ctx)?;
        let (obligations, actions) = failure_evidence_needs(self, &inputs.result);
        Ok((obligations.to_vec(), actions.to_vec()))
    }

    /// The capability report of a study whose effect is not identified: the cheap
    /// inspection report with the evidence-gathering next actions (`measure_variable`,
    /// `run_experiment`, `find_instrument`, `collect_panel`), one licensed neighbor per
    /// repairing route family and one unmet obligation per evidence obligation appended.
    /// An assumption obligation is listed as unmet and `uncheckable`; no study satisfies it.
    ///
    /// A prepared study cannot be refused, so this is the report a host reads beside the
    /// unchanged refusal.
    ///
    /// # Errors
    /// As [`Study::evidence_obligations`], and canonical-encoding failures of the base report.
    pub fn failure_capability(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_contract::OperationReport, EvidencePlanFacadeError> {
        let inputs = self.retained_failure(ctx)?;
        let (obligations, _) = failure_evidence_needs(self, &inputs.result);
        let routes =
            failure_routes(&inputs.result, &inputs.graph, &inputs.observed, STUDY_POPULATION, None)
                .ok();
        let names = routes.as_ref().map_or_else(
            || inputs.names.iter().map(ToString::to_string).collect(),
            |routes| variable_names(&inputs.names, routes, &inputs.graph),
        );
        let base = self.capability().map_err(EvidencePlanFacadeError::Compile)?;
        Ok(super::fallback::failure_capability(&base, &obligations, &inputs.observed, &names))
    }

    /// The unfinished studies the retained not-identified product determines, as
    /// [`PreparedStudy::candidate_templates`] but for a study that cannot be prepared.
    ///
    /// # Errors
    /// As [`Study::evidence_obligations`], and a route the identifiers cannot read.
    pub fn candidate_templates(
        &self,
        sensitivity: Option<&SensitivityFinding>,
        ctx: &ExecutionContext,
    ) -> Result<CandidateGeneration, EvidencePlanFacadeError> {
        let inputs = self.retained_failure(ctx)?;
        let routes = failure_routes(
            &inputs.result,
            &inputs.graph,
            &inputs.observed,
            STUDY_POPULATION,
            sensitivity,
        )?;
        Ok(generate_candidates(&routes.obligations()))
    }

    /// The ranked evidence plan over the declared studies for a study that cannot be
    /// prepared, as [`PreparedStudy::evidence_plan`].
    ///
    /// # Errors
    /// As [`Study::evidence_obligations`] and [`plan_over_routes`].
    pub fn evidence_plan(
        &self,
        request: &EvidencePlanRequest,
        ctx: &ExecutionContext,
    ) -> Result<EvidencePlanResult, EvidencePlanFacadeError> {
        let inputs = self.retained_failure(ctx)?;
        let routes = failure_routes(
            &inputs.result,
            &inputs.graph,
            &inputs.observed,
            STUDY_POPULATION,
            request.sensitivity.as_ref(),
        )?;
        let names = variable_names(&inputs.names, &routes, &inputs.graph);
        plan_over_routes(&routes, names, request, ctx)
    }
}

/// Plan over the routes of a failed identification read from an explicit graph
/// (no prepared study): the same plan [`PreparedStudy::evidence_plan`] produces.
///
/// # Errors
/// As [`failure_routes`] and [`plan_over_routes`].
#[cfg(test)]
fn plan_for_failure(
    result: &antecedent_identify::IdentificationResult,
    graph: &Admg,
    names: &[Arc<str>],
    observed: &[VariableId],
    request: &EvidencePlanRequest,
    ctx: &ExecutionContext,
) -> Result<EvidencePlanResult, EvidencePlanFacadeError> {
    let routes =
        failure_routes(result, graph, observed, STUDY_POPULATION, request.sensitivity.as_ref())?;
    let labels = variable_names(names, &routes, graph);
    plan_over_routes(&routes, labels, request, ctx)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalQuery, DistributionAvailability, SearchLimits,
    };
    use antecedent_design::{
        ExpectedEvidence, RepairClassification, StudyCostDeclaration, StudyKind, UnitRules,
    };
    use antecedent_graph::DenseNodeId;
    use antecedent_identify::{
        DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    };

    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn failed(treatment: u32, outcome: u32) -> IdentificationResult {
        IdentificationResult::not_identified(
            CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(treatment), v(outcome))),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    fn graph(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
        let mut graph = Admg::with_variables(nodes);
        for (a, b) in directed {
            graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        for (a, b) in bidirected {
            graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        graph
    }

    fn names(raw: &[&str]) -> Vec<Arc<str>> {
        raw.iter().map(|n| Arc::from(*n)).collect()
    }

    fn study(
        label: &str,
        kind: StudyKind,
        interventions: &[u32],
        measured: &[u32],
        cost: u64,
        unit: &str,
    ) -> DurableStudyCandidate {
        let ids = |raw: &[u32]| raw.iter().copied().map(v).collect::<Arc<[VariableId]>>();
        DurableStudyCandidate {
            label: Arc::from(label),
            kind,
            population: Arc::from(STUDY_POPULATION),
            interventions: ids(interventions),
            measured: ids(measured),
            joint_measurement: true,
            sample_size: 100,
            recruitment: Arc::from("consecutive recruitment"),
            timing: Arc::from("baseline"),
            unit_rules: UnitRules {
                unit: Arc::from("patient"),
                cluster: None,
                whole_cluster_sampling: false,
            },
            cost: StudyCostDeclaration {
                units: cost,
                unit_label: Arc::from(unit),
                sample_budget: 100,
            },
            feasible: true,
            feasibility_notes: Arc::from([]),
            expected_evidence: Arc::from([ExpectedEvidence {
                population: Arc::from(STUDY_POPULATION),
                interventions: ids(interventions),
                intervention_values: Arc::from([]),
                conditioned_on: Arc::from([]),
                measured: ids(measured),
                distribution: DistributionAvailability::Joint,
            }]),
            external_provider: None,
        }
    }

    fn request(candidates: Vec<DurableStudyCandidate>) -> EvidencePlanRequest {
        EvidencePlanRequest {
            candidates,
            objective: RepairObjective::MinimizeCost,
            limits: RepairLimits {
                search: SearchLimits { operations: 1000, depth: 2 },
                memory_limit_bytes: 16 * 1024 * 1024,
            },
            sensitivity: None,
            value: None,
        }
    }

    /// T=0 -> W=1 -> Y=2 with T <-> Y and W <-> Y, all three observed: a hedge.
    #[test]
    fn a_hedge_plan_orders_the_experiment_first_and_names_the_latent_causes() {
        let hedge = graph(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)]);
        let observed = [v(0), v(1), v(2)];
        let trial = study("randomized trial", StudyKind::Experiment, &[0], &[2], 9, "usd");
        let survey = study("bigger survey", StudyKind::Observation, &[], &[0, 1, 2], 1, "usd");
        let ctx = ExecutionContext::for_tests(1);
        let result = plan_for_failure(
            &failed(0, 2),
            &hedge,
            &names(&["t", "w", "y"]),
            &observed,
            &request(vec![trial, survey]),
            &ctx,
        )
        .unwrap();
        let best = &result.plan.options[0];
        assert_eq!(best.labels[0].as_ref(), "randomized trial");
        assert_eq!(best.gate, RepairClassification::VerifiedSufficient);
        assert_eq!(result.plan.outcome, PlanOutcome::Closing);
        // The cheap survey cannot answer a hedge, whatever it costs.
        assert!(result.plan.options.iter().any(|o| {
            o.labels[0].as_ref() == "bigger survey"
                && o.gate != RepairClassification::VerifiedSufficient
        }));
        assert_eq!(result.name_of(v(3)), "common_cause_of_t_y");
        // Templates for the same obligations are offered, undeclared.
        assert!(result.generation.templates.iter().any(|t| t.kind == StudyKind::Experiment));
        // The artifact is sealed and replays.
        let bytes = result.export("plan").unwrap();
        let artifact = EvidencePlanArtifact::from_bytes(&bytes).unwrap();
        let replayed =
            artifact.consume(antecedent_design::RepairConsumeLimits::default(), &ctx).unwrap();
        assert_eq!(replayed.identity(), result.plan.identity());
    }

    #[test]
    fn unequal_cost_units_and_no_route_are_refused() {
        let hedge = graph(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)]);
        let observed = [v(0), v(1), v(2)];
        let usd = study("trial", StudyKind::Experiment, &[0], &[2], 9, "usd");
        let eur = study("survey", StudyKind::Observation, &[], &[0, 1, 2], 1, "eur");
        let ctx = ExecutionContext::for_tests(1);
        let refused = plan_for_failure(
            &failed(0, 2),
            &hedge,
            &names(&["t", "w", "y"]),
            &observed,
            &request(vec![usd, eur]),
            &ctx,
        )
        .unwrap_err();
        assert_eq!(refused.code(), "design_cost_units_mismatch");

        // An identified result owes no route at all.
        let simple = graph(2, &[(0, 1)], &[]);
        let mut identified = failed(0, 1);
        identified.status = antecedent_core::IdentificationStatus::NonparametricallyIdentified;
        let refused = plan_for_failure(
            &identified,
            &simple,
            &names(&["t", "y"]),
            &[v(0), v(1)],
            &request(vec![study("s", StudyKind::Observation, &[], &[0, 1], 1, "usd")]),
            &ctx,
        )
        .unwrap_err();
        assert_eq!(refused.detail(), "identification_repair.not_a_failure");
    }
}
