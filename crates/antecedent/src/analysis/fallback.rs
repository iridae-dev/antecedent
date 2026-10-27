//! The opt-in descriptive fallback: what can be said, and what evidence would help, when the
//! causal question is refused for want of identification.
//!
//! A causal query the declared structure does not identify ends in a refusal. That refusal is
//! the answer to the causal question and stays so. [`Study::analyze_with_fallback`] adds two
//! things beside it, and only when the caller asks for them:
//!
//! * the **associational** description of the same outcome, treatment and adjustment variables,
//!   built through [`Study::compile`] as the compiled associational route (never a bypass),
//!   labelled noncausal, and carrying the original refusal's reason code in its identification
//!   layer. It describes the observed law. It is not the effect of an intervention, it is not
//!   identified, and it ranks no action;
//! * the **evidence needs** the failure determines: the verified evidence obligations, the
//!   evidence-gathering next actions and, when the caller declares candidate studies, the ranked
//!   [`EvidencePlan`](super::evidence_plan::EvidencePlan) of the options that would close the
//!   gap.
//!
//! The refusal is returned unchanged. Each of the other two parts is either present or carries a
//! typed reason it is not ([`FallbackUnavailable`]): no association form exists for the query
//! kind, a variable is not measured, the modality has no associational counterpart, the declared
//! graph is not one the identifiers read, or the graph identifies the effect (so the refusal came
//! from the estimator and no structural gap is owed). A query that succeeds returns the ordinary
//! result and never an [`AnalysisWithFallback`]. A search that stopped at a budget, a
//! cancellation, a resource limit, an invalid argument and every data or estimation error are
//! not identification-class refusals and are returned as the errors they are.
//!
//! The declared selection rule of the adjustment variables is [`fallback_roles`]: the outcome
//! and the treatment of the query; the conditioning set is the measured parents of the treatment
//! in the declared directed or bidirected graph together with the query's declared effect
//! modifiers (the declared modifiers alone when the graph is of another class).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use antecedent_contract::{ClaimEnvelope, EvidenceObligation, NextAction};
use antecedent_core::{
    AssociationQuery, AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, VariableId,
};
use antecedent_design::structure_identification::{GraphIdentFlag, identifies};
use antecedent_design::{CandidateGeneration, failure_routes, generate_candidates};
use antecedent_graph::Admg;
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};

use super::builder::{DataInput, data_schema};
use super::contract::evidence_next_actions;
use super::evidence_plan::{
    EvidencePlanRequest, EvidencePlanResult, STUDY_POPULATION, plan_over_routes, variable_names,
};
use super::execute::Study;
use crate::error::CausalError;
use crate::result::StudyResult;

/// What the descriptive part of a fallback is, in one sentence a report repeats.
pub const NONCAUSAL_NOTICE: &str = "The associational result describes the observed law of the \
columns. It is not an answer to the causal question: it is not the effect of an intervention, it \
is not identified, and it ranks no action.";

/// The part of an [`AnalysisWithFallback`] a [`FallbackUnavailable`] is about.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FallbackPart {
    /// The associational description.
    Associational,
    /// The evidence needs and the evidence plan.
    Evidence,
}

impl FallbackPart {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Associational => "associational",
            Self::Evidence => "evidence_plan",
        }
    }
}

/// A typed reason one part of a fallback is not provided.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FallbackUnavailable {
    /// The part that is missing.
    pub part: FallbackPart,
    /// Registered reason code (a runtime-refusal code).
    pub code: String,
    /// Stable `fallback.*` detail, or the detail of the refusal behind it.
    pub detail: String,
    /// What is missing and the next attempt that would provide it.
    pub message: String,
}

impl FallbackUnavailable {
    fn new(
        part: FallbackPart,
        code: impl Into<String>,
        detail: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self { part, code: code.into(), detail: detail.into(), message: message.into() }
    }
}

impl fmt::Display for FallbackUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for FallbackUnavailable {}

/// The roles the declared selection rule reads off the refused causal query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FallbackRoles {
    /// The outcome column.
    pub outcome: VariableId,
    /// The treatment column, the one predictor of the description.
    pub treatment: VariableId,
    /// The conditioning columns, ascending: the measured parents of the treatment in the
    /// declared graph and the query's declared effect modifiers.
    pub adjustment: Vec<VariableId>,
    /// Name of the rule that chose the conditioning columns.
    pub rule: &'static str,
}

/// What a fallback is asked for beyond the refusal itself.
#[derive(Default)]
pub struct FallbackRequest {
    /// Studies the caller declares (cost, timing, recruitment and sample size are the caller's,
    /// never defaulted). With candidates the evidence part carries the ranked plan; without,
    /// the obligations, the next actions and the candidate templates.
    pub evidence: Option<EvidencePlanRequest>,
}

/// The associational part: the compiled description and its noncausal claim.
#[derive(Debug)]
pub struct AssociationalFallback {
    /// The roles the selection rule read off the refused query.
    pub roles: FallbackRoles,
    /// The result of the compiled associational route.
    pub result: StudyResult,
    /// Its claim: kind `associational`, identification not claimed.
    pub claim: ClaimEnvelope,
}

/// The ranked plan of a fallback.
#[derive(Debug)]
pub enum RankedPlan {
    /// No candidate studies were declared, so no plan was asked for.
    NotRequested,
    /// The options that close the gap, verified and ordered.
    Ranked(Box<EvidencePlanResult>),
    /// Candidates were declared and the plan refused; the typed reason.
    Unavailable(FallbackUnavailable),
}

/// The evidence that would resolve the identification failure.
#[derive(Debug)]
pub struct EvidenceNeeds {
    /// Variable names by id: the study's variables, then the unmeasured common causes of the
    /// graph's bidirected edges.
    pub names: Vec<String>,
    /// The verified obligations, each with its proof step. An assumption obligation is listed
    /// and satisfiable by no study.
    pub obligations: Vec<EvidenceObligation>,
    /// The evidence-gathering next actions the obligations imply.
    pub next_actions: Vec<NextAction>,
    /// The unfinished studies the obligations determine.
    pub candidate_templates: CandidateGeneration,
    /// The ranked plan over the declared candidates.
    pub ranking: RankedPlan,
}

/// The unchanged refusal of an identification-class failure and, beside it, a noncausal
/// description and the evidence that would resolve the failure.
#[derive(Debug)]
pub struct AnalysisWithFallback {
    /// The original refusal, exactly as the causal query raised it.
    pub refusal: CausalError,
    /// The compiled associational description, or why there is none.
    pub associational: Result<AssociationalFallback, FallbackUnavailable>,
    /// The obligations, next actions and plan, or why there are none.
    pub evidence: Result<EvidenceNeeds, FallbackUnavailable>,
}

impl AnalysisWithFallback {
    /// Whether the associational part ranks actions: never.
    #[must_use]
    pub const fn ranks_actions(&self) -> bool {
        false
    }

    /// Whether any part answers the causal question: never. The refusal is its answer.
    #[must_use]
    pub const fn answers_causal_question(&self) -> bool {
        false
    }

    /// Plain statement of what was returned and what it is not.
    #[must_use]
    pub fn explain(&self) -> String {
        let mut text =
            format!("The causal question was refused: {}. {NONCAUSAL_NOTICE}", self.refusal);
        match &self.associational {
            Ok(part) => {
                let payload = part.result.association();
                let analysis =
                    payload.map_or("association", |p| p.estimate.selection.analysis.as_str());
                text.push_str(&format!(
                    " Associational description: {analysis} of the outcome on the treatment \
                     holding {} column(s) fixed (rule {}).",
                    part.roles.adjustment.len(),
                    part.roles.rule
                ));
            }
            Err(reason) => text.push_str(&format!(" No associational description: {reason}")),
        }
        match &self.evidence {
            Ok(needs) => {
                text.push_str(&format!(
                    " Evidence that would resolve the refusal: {} obligation(s), {} next \
                     action(s).",
                    needs.obligations.len(),
                    needs.next_actions.len()
                ));
                match &needs.ranking {
                    RankedPlan::Ranked(plan) => text.push_str(&format!(
                        " Ranked plan over the declared studies: {} option(s).",
                        plan.plan.options.len()
                    )),
                    RankedPlan::Unavailable(reason) => {
                        text.push_str(&format!(" No ranked plan: {reason}"));
                    }
                    RankedPlan::NotRequested => {
                        text.push_str(" Declare candidate studies to rank the options.");
                    }
                }
            }
            Err(reason) => text.push_str(&format!(" No evidence plan: {reason}")),
        }
        text
    }
}

/// What [`Study::analyze_with_fallback`] returns.
#[derive(Debug)]
#[non_exhaustive]
pub enum FallbackOutcome {
    /// The causal query succeeded: the ordinary result, never a fallback.
    Causal(Box<StudyResult>),
    /// The causal query was refused for identification; the refusal beside its description.
    Described(Box<AnalysisWithFallback>),
}

fn no_form(query: &CausalQuery) -> FallbackUnavailable {
    FallbackUnavailable::new(
        FallbackPart::Associational,
        "route_not_supported",
        "fallback.no_association_form",
        format!(
            "no association form exists for a {:?} query: the description relates one outcome to \
             one treatment, and this query has no such reading. Next attempt: state the \
             associational question directly with an AssociationQuery",
            query.kind()
        ),
    )
}

fn unmeasured(part: FallbackPart, what: &str) -> FallbackUnavailable {
    FallbackUnavailable::new(
        part,
        "invalid_argument",
        "fallback.variable_not_measured",
        format!(
            "{what} is not a measured column of the study's data, so there is nothing to \
             describe or to plan over. Next attempt: measure it, or state the query over the \
             measured columns"
        ),
    )
}

/// The directed or bidirected graph the structure-level evidence reads.
fn failure_graph(study: &Study) -> Option<Admg> {
    study
        .graph
        .as_admg()
        .cloned()
        .or_else(|| study.graph.as_dag().and_then(|dag| Admg::from_dag(dag).ok()))
}

/// The measured parents of `treatment` in `graph`.
fn measured_parents(graph: &Admg, treatment: VariableId, width: usize) -> Vec<VariableId> {
    let Some(node) = graph.dense_of_static(treatment) else {
        return Vec::new();
    };
    graph
        .parents(node)
        .iter()
        .filter_map(|parent| graph.static_variable(*parent))
        .filter(|variable| variable.as_usize() < width)
        .collect()
}

/// The roles the declared selection rule reads off a refused query.
///
/// An average or conditional effect has an association form: the outcome against the treatment,
/// holding fixed the measured parents of the treatment (the covariates the declared graph says
/// move it) and the declared effect modifiers. Every other query kind has none.
///
/// # Errors
///
/// [`FallbackUnavailable`] when the query kind has no association form or a variable it names is
/// not a measured column.
pub fn fallback_roles(study: &Study) -> Result<FallbackRoles, FallbackUnavailable> {
    let (treatment, outcome, modifiers) = match &study.query {
        CausalQuery::AverageEffect(query) => {
            (query.treatment, query.outcome, query.effect_modifiers.to_vec())
        }
        CausalQuery::ConditionalEffect(query) => {
            (query.inner.treatment, query.inner.outcome, query.inner.effect_modifiers.to_vec())
        }
        other => return Err(no_form(other)),
    };
    let width = data_schema(&study.data).variables().len();
    for (what, variable) in [("the treatment", treatment), ("the outcome", outcome)]
        .into_iter()
        .chain(modifiers.iter().map(|modifier| ("an effect modifier", *modifier)))
    {
        if variable.as_usize() >= width {
            return Err(unmeasured(FallbackPart::Associational, what));
        }
    }
    let graph = failure_graph(study);
    let mut adjustment: BTreeSet<VariableId> = modifiers.into_iter().collect();
    let rule = if let Some(graph) = &graph {
        adjustment.extend(measured_parents(graph, treatment, width));
        "measured_parents_of_treatment_and_declared_modifiers"
    } else {
        "declared_modifiers_only"
    };
    adjustment.remove(&treatment);
    adjustment.remove(&outcome);
    Ok(FallbackRoles { outcome, treatment, adjustment: adjustment.into_iter().collect(), rule })
}

/// The evidence-gathering next actions of `obligations`: `measure_variable:<name>` for a
/// variable the obligations name that `observed` does not hold, `run_experiment:<name>` for a
/// variable an obligation sets, `find_instrument` and `collect_panel`. An assumption obligation
/// yields no action, because no study satisfies it. `names` spells variables by id.
#[must_use]
pub fn failure_next_actions(
    obligations: &[EvidenceObligation],
    observed: &[VariableId],
    names: &[String],
) -> Vec<NextAction> {
    evidence_next_actions(obligations, observed, &|variable| {
        names.get(variable.as_usize()).cloned().unwrap_or_else(|| format!("v{}", variable.raw()))
    })
}

/// What a failed identification adds to a capability report: the evidence-gathering next
/// actions, one licensed neighbor per route family that could repair it, and one unresolved
/// obligation per evidence obligation (an assumption obligation is `uncheckable`: no study
/// establishes it).
#[derive(Clone, Debug)]
pub struct FailureCapabilityParts {
    /// `measure_variable`, `run_experiment`, `find_instrument`, `collect_panel` actions.
    pub next_actions: Vec<NextAction>,
    /// One neighbor per route family: the evidence regime that would replace the observed law.
    pub neighbors: Vec<antecedent_contract::LicensedNeighbor>,
    /// One unresolved obligation per evidence obligation, carrying its proof step as the
    /// required check.
    pub unmet_obligations: Vec<antecedent_contract::ObligationRecord>,
}

/// The capability-report parts of the verified `obligations` of a failed identification.
#[must_use]
pub fn failure_capability_parts(
    obligations: &[EvidenceObligation],
    observed: &[VariableId],
    names: &[String],
) -> FailureCapabilityParts {
    use antecedent_contract::{LicensedNeighbor, ObligationKind, ObligationRecord, PremiseChange};
    use antecedent_core::{AssumptionSource, AssumptionStatus};

    let next_actions = failure_next_actions(obligations, observed, names);
    let mut families: Vec<&str> =
        obligations.iter().map(|o| o.provenance.family.as_ref()).collect();
    families.sort_unstable();
    families.dedup();
    let neighbors = families
        .into_iter()
        .map(|family| {
            let own: Vec<EvidenceObligation> = obligations
                .iter()
                .filter(|o| o.provenance.family.as_ref() == family)
                .cloned()
                .collect();
            let action = failure_next_actions(&own, observed, names)
                .first()
                .map_or_else(|| "establish_assumption".to_owned(), |a| a.id.to_string());
            LicensedNeighbor::new(
                format!("identified_by:{family}"),
                vec![PremiseChange::new("evidence", "observational_base_law", family)],
                action,
            )
        })
        .collect();
    let unmet_obligations = obligations
        .iter()
        .map(|o| {
            let assumption = !o.satisfiable_by_study();
            let record = ObligationRecord::new(
                format!("evidence.{}", o.id),
                o.scope,
                AssumptionSource::Derived { from: Arc::from("identification_failure") },
                if assumption { ObligationKind::Uncheckable } else { ObligationKind::CheckNotRun },
                if assumption { AssumptionStatus::Untestable } else { AssumptionStatus::Declared },
                Arc::clone(&o.reason),
            );
            match o.provenance.proof_step.as_deref() {
                Some(step) => record.with_required_check(step),
                None => record,
            }
        })
        .collect();
    FailureCapabilityParts { next_actions, neighbors, unmet_obligations }
}

/// `base` with the failure's evidence needs appended: the report a refused study's host
/// record carries (actions, neighbors and unmet obligations), the base entries first.
#[must_use]
pub(super) fn failure_capability(
    base: &antecedent_contract::OperationReport,
    obligations: &[EvidenceObligation],
    observed: &[VariableId],
    names: &[String],
) -> antecedent_contract::OperationReport {
    let parts = failure_capability_parts(obligations, observed, names);
    let mut report = base.clone();
    let mut actions = base.next_actions.to_vec();
    actions.extend(parts.next_actions);
    let mut neighbors = base.neighbors.to_vec();
    neighbors.extend(parts.neighbors);
    let mut unmet = base.unmet_obligations.to_vec();
    unmet.extend(parts.unmet_obligations);
    report.next_actions = Arc::from(actions);
    report.neighbors = Arc::from(neighbors);
    report.unmet_obligations = Arc::from(unmet);
    report
}

/// Cancellation and resource limits are not scientific: they are returned, never described.
fn is_stop(error: &CausalError) -> bool {
    matches!(error.peeled(), CausalError::Cancelled { .. } | CausalError::Resource { .. })
}

fn stopped(ctx: &ExecutionContext) -> Result<(), CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    Ok(())
}

type Provided<T> = Result<Result<T, FallbackUnavailable>, CausalError>;

impl Study {
    /// Run the causal query; when it is refused for identification, return the unchanged
    /// refusal beside a noncausal description and the evidence that would resolve it.
    ///
    /// A query that succeeds returns [`FallbackOutcome::Causal`] with the ordinary result.
    /// Any other failure (a budget stop, a cancellation, an invalid argument, a data or
    /// estimation error, a search that stopped at a cap) is returned as the error it is.
    ///
    /// # Errors
    ///
    /// The query's own non-identification failures, and a cancellation or resource limit that
    /// stops the description or the plan.
    pub fn analyze_with_fallback(
        &self,
        request: &FallbackRequest,
        ctx: &ExecutionContext,
    ) -> Result<FallbackOutcome, CausalError> {
        match self.run(ctx) {
            Ok(result) => Ok(FallbackOutcome::Causal(Box::new(result))),
            Err(refusal) if refusal.is_identification_refusal() => {
                let associational = self.describe_refused(&refusal, ctx)?;
                stopped(ctx)?;
                let evidence = self.evidence_needs(request, ctx)?;
                Ok(FallbackOutcome::Described(Box::new(AnalysisWithFallback {
                    refusal,
                    associational,
                    evidence,
                })))
            }
            Err(other) => Err(other),
        }
    }

    /// The study over the same data and declarations whose query is the association `query`,
    /// or why the data modality has none.
    fn associational_counterpart(
        &self,
        query: AssociationQuery,
    ) -> Result<Study, FallbackUnavailable> {
        let builder = match &self.data {
            DataInput::Tabular(data) => Study::tabular(data.clone()),
            DataInput::Temporal(data) => Study::series(data.clone()),
            _ => {
                return Err(FallbackUnavailable::new(
                    FallbackPart::Associational,
                    "route_not_supported",
                    "fallback.modality_not_described",
                    "the associational route describes the rows of one table or the points of \
                     one series; this study holds panel, event or multi-environment data. Next \
                     attempt: describe one table or one series with an AssociationQuery",
                ));
            }
        };
        let mut builder = builder
            .query(CausalQuery::Association(query))
            .inference(self.inference.clone())
            .bootstrap_replicates(self.bootstrap_replicates);
        if let Some(design) = self.sampling_design {
            builder = builder.sampling_design(design);
        }
        if let Some(missing) = &self.missing {
            builder = builder.missing(missing.clone());
        }
        if let Some(participants) = &self.participants {
            builder = builder.participants(participants.clone());
        }
        builder.build().map_err(|error| refused_association(&error))
    }

    /// The compiled associational description of the refused query.
    fn describe_refused(
        &self,
        refusal: &CausalError,
        ctx: &ExecutionContext,
    ) -> Provided<AssociationalFallback> {
        let roles = match fallback_roles(self) {
            Ok(roles) => roles,
            Err(unavailable) => return Ok(Err(unavailable)),
        };
        let reason = refusal.reason_code().unwrap_or("effect_not_identified");
        let query = AssociationQuery::new(roles.outcome, [roles.treatment])
            .with_conditioning(roles.adjustment.clone())
            .following_refusal(reason);
        let study = match self.associational_counterpart(query) {
            Ok(study) => study,
            Err(unavailable) => return Ok(Err(unavailable)),
        };
        let run = || -> Result<(StudyResult, ClaimEnvelope), CausalError> {
            let prepared = study.prepare(ctx)?;
            let result = prepared.estimate_retained(ctx)?;
            let contract = prepared.contract()?;
            let claim = result.claim(&contract, ctx)?;
            Ok((result, claim))
        };
        match run() {
            Ok((result, claim)) => Ok(Ok(AssociationalFallback { roles, result, claim })),
            Err(error) if is_stop(&error) => Err(error),
            Err(error) => Ok(Err(refused_association(&error))),
        }
    }

    /// The treatment, outcome, variable names and graph the evidence is read from, when the
    /// refusal is one the declared structure explains.
    fn failure_setting(&self) -> Result<FailureSetting, FallbackUnavailable> {
        let (treatment, outcome) = match &self.query {
            CausalQuery::AverageEffect(query) => (query.treatment, query.outcome),
            CausalQuery::ConditionalEffect(query) => (query.inner.treatment, query.inner.outcome),
            other => {
                return Err(FallbackUnavailable::new(
                    FallbackPart::Evidence,
                    "route_not_supported",
                    "fallback.no_association_form",
                    format!(
                        "evidence obligations are read off the identification of one treatment \
                         and one outcome, and a {:?} query has neither. Next attempt: ask the \
                         average effect of the treatment on the outcome",
                        other.kind()
                    ),
                ));
            }
        };
        let names: Vec<Arc<str>> = data_schema(&self.data)
            .variables()
            .iter()
            .map(|variable| Arc::clone(&variable.name))
            .collect();
        if treatment.as_usize() >= names.len() || outcome.as_usize() >= names.len() {
            return Err(unmeasured(FallbackPart::Evidence, "the treatment or the outcome"));
        }
        let Some(graph) = failure_graph(self) else {
            return Err(FallbackUnavailable::new(
                FallbackPart::Evidence,
                "route_not_supported",
                "fallback.graph_class_not_read",
                "the evidence routes (back-door, front-door, instrument, hedge) are read from a \
                 declared directed or bidirected graph, and this study declares another graph \
                 class. Next attempt: declare a Dag or an Admg",
            ));
        };
        match identifies(&graph, treatment, outcome) {
            Ok(GraphIdentFlag::Unidentified) => {}
            Ok(GraphIdentFlag::Identified) => {
                return Err(FallbackUnavailable::new(
                    FallbackPart::Evidence,
                    "invalid_argument",
                    "identification_repair.not_a_failure",
                    "the declared graph identifies the effect, so the refusal came from the \
                     estimator or the data and no structural evidence gap is owed. Next \
                     attempt: read the refusal's remedy",
                ));
            }
            Err(error) => {
                return Err(FallbackUnavailable::new(
                    FallbackPart::Evidence,
                    error.code,
                    error.detail,
                    error.message,
                ));
            }
        }
        Ok(FailureSetting { treatment, outcome, names, graph })
    }

    /// The obligations, next actions and (with declared candidates) the ranked plan.
    fn evidence_needs(
        &self,
        request: &FallbackRequest,
        ctx: &ExecutionContext,
    ) -> Provided<EvidenceNeeds> {
        let FailureSetting { treatment, outcome, names, graph } = match self.failure_setting() {
            Ok(setting) => setting,
            Err(unavailable) => return Ok(Err(unavailable)),
        };
        stopped(ctx)?;
        let observed: Vec<VariableId> = (0..names.len())
            .filter_map(|raw| u32::try_from(raw).ok().map(VariableId::from_raw))
            .filter(|variable| graph.dense_of_static(*variable).is_some())
            .collect();
        // The product the study's own compiled program retained when its identification
        // refused; the structure-level product only when the route retained none.
        let retained = self
            .compile(ctx)
            .ok()
            .and_then(|plan| plan.lowered().and_then(|l| l.refused_identification()).cloned())
            .filter(|result| result.status == antecedent_core::IdentificationStatus::NotIdentified);
        let failed = retained.unwrap_or_else(|| {
            IdentificationResult::not_identified(
                CausalQuery::average_effect(AverageEffectQuery::binary_ate(treatment, outcome)),
                DerivationTrace::default(),
                AssumptionSet::new(),
                IdentificationPerformanceRecord::default(),
            )
        });
        let routes = match failure_routes(&failed, &graph, &observed, STUDY_POPULATION, None) {
            Ok(routes) => routes,
            Err(error) => {
                return Ok(Err(FallbackUnavailable::new(
                    FallbackPart::Evidence,
                    error.code,
                    error.detail,
                    error.message,
                )));
            }
        };
        let labels = variable_names(&names, &routes, &graph);
        let obligations = routes.obligations();
        let next_actions = failure_next_actions(&obligations, &observed, &labels);
        let candidate_templates = generate_candidates(&obligations);
        let ranking = match &request.evidence {
            None => RankedPlan::NotRequested,
            Some(plan_request) => {
                match plan_over_routes(&routes, labels.clone(), plan_request, ctx) {
                    Ok(plan) => RankedPlan::Ranked(Box::new(plan)),
                    Err(error) => {
                        stopped(ctx)?;
                        RankedPlan::Unavailable(FallbackUnavailable::new(
                            FallbackPart::Evidence,
                            error.code(),
                            error.detail(),
                            error.to_string(),
                        ))
                    }
                }
            }
        };
        Ok(Ok(EvidenceNeeds {
            names: labels,
            obligations,
            next_actions,
            candidate_templates,
            ranking,
        }))
    }
}

/// What the evidence is read from: the refused query's treatment and outcome, the study's
/// variable names and the declared graph the complete algorithm does not identify them on.
struct FailureSetting {
    treatment: VariableId,
    outcome: VariableId,
    names: Vec<Arc<str>>,
    graph: Admg,
}

/// The typed reason the compiled associational route could not describe the data.
fn refused_association(error: &CausalError) -> FallbackUnavailable {
    FallbackUnavailable::new(
        FallbackPart::Associational,
        error.reason_code().unwrap_or("route_not_supported"),
        "fallback.association_refused",
        error.to_string(),
    )
}
