//! Anomaly and change attribution over the members of a structure class.
//!
//! The attribution engine reconstructs the target from the noise of its ancestors, so it needs
//! one structural model. A structure that stands for several (a CPDAG, a PAG and their temporal
//! forms) is therefore repeated over its members, and a mixed graph is attributed to the noise
//! blocks its latent common causes tie together:
//!
//! - a CPDAG member is a DAG of its Markov class, and a temporal CPDAG member a temporal DAG;
//!   each runs the DAG attribution unchanged;
//! - a PAG member is a MAG completion read as its most-confounded mixed graph (an invisible
//!   directed edge may also carry a latent common cause), and a temporal PAG member the same
//!   template; a member without latent confounding among the ancestors of the targets runs the
//!   DAG attribution, one with it runs the noise-block attribution;
//! - an ADMG (and a temporal ADMG, unrolled over the window the longest lag spans, with the
//!   causes that precede the window latent) is attributed to the c-components of the ancestral
//!   sub-graph when every ancestor's mechanism is identified from the observed law, and refused
//!   with the hedge and the two-SCM witness otherwise (see `antecedent_attribution::latent_blocks`).
//!
//! Members that identify the same attribution keep it. Members that disagree leave it partly
//! identified: the result publishes the enumeration-weighted mean of the contributions, the
//! spread across members as the graph sensitivity, and the per-component range as an identified
//! set. Member weights are enumeration weights, not posterior probabilities; a refused member is
//! unidentified mass, and a member whose fit failed is unevaluable mass.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_data::TableView;

use antecedent_attribution::{
    AnomalyScores, CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget,
    MechanismHedge, NoiseBlockIdentification, NoiseBlockPlan, plan_noise_blocks,
};
use antecedent_core::{ComponentId, Lag, ResponseEnvelope};
use antecedent_graph::{Endpoint, NodeRef, Pag, TemporalAdmg, TemporalPag};
use antecedent_prob::PosteriorQuantityKind;

use super::temporal_attribution::LagAlignedScm;
use super::{
    AcceptedGraph, Admg, Arc, CausalError, CausalPosterior, CausalQuery,
    CheckedAttributionOperation, DenseNodeId, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, ExecutionContext, GraphRef, IdentificationStatus, OverlapPolicy,
    PhysicalExecutionPlan, ResponseValue, StudyResult, TabularData, TemporalDag, TemporalIndexer,
    TimeSeriesData, VariableId, temporal_admg_projection,
};
use crate::result::{
    ResultPayload, StructuralResponseAtom, StructuralResponseMixture, StructuralWeightBasis,
};

/// Members retained when the study sets no `max_completions`: the identifier's default.
const DEFAULT_CLASS_MEMBERS: usize = 32;

/// Stable code of the diagnostic that accounts for the members of the class.
pub(crate) const CLASS_ATTRIBUTION_ENVELOPE: &str = "attribution.class.envelope";
/// Stable code of the diagnostic that publishes the identified set of the contributions.
pub(crate) const CLASS_ATTRIBUTION_IDENTIFIED_SET: &str = "attribution.class.identified_set";
/// Stable code of the diagnostic that names a member refused for a hedge.
pub(crate) const CLASS_ATTRIBUTION_HEDGE: &str = "attribution.class.noise_block_hedge";
/// Stable code of the diagnostic that states the noise blocks a member was attributed to.
pub(crate) const CLASS_ATTRIBUTION_BLOCKS: &str = "attribution.noise_blocks";

/// One member of the structure class an attribution is repeated over.
struct Member {
    /// Enumeration weight (not a posterior probability).
    weight: f64,
    kind: MemberKind,
}

enum MemberKind {
    /// No latent confounding among the ancestors: the DAG attribution on this graph.
    Directed(AcceptedGraph),
    /// Latent confounding whose attribution is identified: the noise-block attribution.
    Blocks { plan: NoiseBlockPlan, unfolding: Option<LagAlignedScm> },
    /// Latent confounding that breaks the identification of a mechanism.
    Hedged(MechanismHedge),
}

/// The members of a class, and whether they cover it.
struct ClassMembers {
    members: Vec<Member>,
    /// Whether the enumeration stopped at the retention cap or its search budget, so the
    /// members are a subset of the class.
    capped: bool,
    /// The structure class, for the disclosures.
    class: &'static str,
}

type MemberRun = Result<StudyResult, CausalError>;

impl super::Study {
    /// Anomaly or change attribution over a CPDAG, a PAG or a mixed graph, on tabular data.
    pub(super) fn execute_class_attribution(
        &self,
        data: &TabularData,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let class = self.static_class_members()?;
        let cache = self.class_attribution_identification(physical, ctx)?;
        let outcomes = self.run_class_members(&class, ctx, |member| match &member.kind {
            MemberKind::Directed(graph) => {
                let atom = self.member_study(graph);
                CheckedAttributionOperation::checked(&atom, physical, &cache)?.execute(data, ctx)
            }
            MemberKind::Blocks { plan, .. } => {
                let mut result = CheckedAttributionOperation::checked_blocks(
                    self,
                    physical,
                    &cache,
                    (plan.directed.clone(), None),
                    plan.blocks.clone(),
                )?
                .execute(data, ctx)?;
                result.diagnostics.push(blocks_diagnostic(plan));
                Ok(result)
            }
            MemberKind::Hedged(hedge) => Err(hedge_refusal(hedge, false)),
        })?;
        self.finish_class_attribution(&class, outcomes)
    }

    /// Anomaly or change attribution over a temporal CPDAG, a temporal PAG or a temporal ADMG,
    /// on a series.
    pub(super) fn execute_temporal_class_attribution(
        &self,
        series: &TimeSeriesData,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let class = self.temporal_class_members(series)?;
        let cache = self.class_attribution_identification(physical, ctx)?;
        let outcomes = self.run_class_members(&class, ctx, |member| match &member.kind {
            MemberKind::Directed(graph) => {
                let atom = self.member_study(graph);
                CheckedAttributionOperation::checked(&atom, physical, &cache)?
                    .execute_series(series, ctx)
            }
            MemberKind::Blocks { plan, unfolding } => {
                let mut result = CheckedAttributionOperation::checked_blocks(
                    self,
                    physical,
                    &cache,
                    (plan.directed.clone(), unfolding.clone()),
                    plan.blocks.clone(),
                )?
                .execute_series(series, ctx)?;
                result.diagnostics.push(blocks_diagnostic(plan));
                Ok(result)
            }
            MemberKind::Hedged(hedge) => Err(hedge_refusal(hedge, false)),
        })?;
        self.finish_class_attribution(&class, outcomes)
    }

    /// The study the checked DAG attribution runs a member under: this one on the member's
    /// graph.
    fn member_study(&self, graph: &AcceptedGraph) -> Self {
        let mut member = self.clone();
        member.graph = Arc::new(graph.clone());
        member
    }

    /// The identification every member shares: the parametric SCM claim of the query, which
    /// names no graph.
    fn class_attribution_identification(
        &self,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Arc<crate::analysis::prepared::CachedStaticIdentification>, CausalError> {
        if let Some((cache, _)) = self.reuse_static_identification(physical) {
            return Ok(cache);
        }
        super::report_identify_compute(ctx);
        Ok(Arc::new(self.prepare_static_identification(physical)?.ok_or_else(|| {
            CausalError::Compile {
                message: "class attribution identified no structural product".into(),
            }
        })?))
    }

    /// The variables the attribution reconstructs.
    fn class_attribution_targets(&self) -> Result<Vec<VariableId>, CausalError> {
        match &self.query {
            CausalQuery::AnomalyAttribution(query) if !query.targets.is_empty() => {
                Ok(query.targets.iter().copied().collect())
            }
            CausalQuery::ChangeAttribution(query) => Ok(vec![query.outcome]),
            _ => Err(CausalError::Compile {
                message: "class attribution needs an anomaly query with a target or a change query"
                    .into(),
            }),
        }
    }

    /// Run `run` on every member, in order. A member's refusal or failure is its outcome; only a
    /// cancellation ends the run.
    fn run_class_members(
        &self,
        class: &ClassMembers,
        ctx: &ExecutionContext,
        mut run: impl FnMut(&Member) -> MemberRun,
    ) -> Result<Vec<MemberRun>, CausalError> {
        let mut outcomes = Vec::with_capacity(class.members.len());
        for member in &class.members {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let outcome = run(member);
            if let Err(error) = &outcome {
                if matches!(error, CausalError::Cancelled { .. }) {
                    return Err(error.clone());
                }
            }
            outcomes.push(outcome);
        }
        Ok(outcomes)
    }

    /// The members of a static structure class.
    fn static_class_members(&self) -> Result<ClassMembers, CausalError> {
        let targets = self.class_attribution_targets()?;
        let cap = self.max_completions.unwrap_or(DEFAULT_CLASS_MEMBERS);
        match self.graph.graph_ref() {
            GraphRef::Admg(admg) => Ok(ClassMembers {
                members: vec![Member { weight: 1.0, kind: mixed_member(admg, &targets)? }],
                capped: false,
                class: "admg",
            }),
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                Ok(ClassMembers {
                    members: vec![Member { weight: 1.0, kind: mixed_member(&admg, &targets)? }],
                    capped: false,
                    class: "temporal_admg",
                })
            }
            GraphRef::Cpdag(cpdag) => {
                let mut sampler = antecedent_graph::CpdagCompletionSampler::new(cpdag.clone(), cap)
                    .map_err(CausalError::rejected)?;
                let members = sampler
                    .by_ref()
                    .map(|completion| Member {
                        weight: 1.0,
                        kind: MemberKind::Directed(AcceptedGraph::from(completion.graph)),
                    })
                    .collect();
                Ok(ClassMembers { members, capped: sampler.hit_cap(), class: "cpdag" })
            }
            GraphRef::Pag(pag) => {
                let mut sampler = antecedent_graph::CompletionSampler::new(pag.clone(), cap)
                    .map_err(CausalError::rejected)?;
                let mut members = Vec::new();
                for completion in sampler.by_ref() {
                    let admg = labelled_confounded_admg(&completion.graph)?;
                    members.push(Member { weight: 1.0, kind: mixed_member(&admg, &targets)? });
                }
                Ok(ClassMembers {
                    members,
                    capped: sampler.hit_cap() || sampler.class_audit_incomplete(),
                    class: "pag",
                })
            }
            _ => Err(CausalError::Compile {
                message: "class attribution requires a Cpdag, a Pag, an Admg or a TemporalAdmg"
                    .into(),
            }),
        }
    }

    /// The members of a temporal structure class over `series`.
    fn temporal_class_members(&self, series: &TimeSeriesData) -> Result<ClassMembers, CausalError> {
        let targets = self.class_attribution_targets()?;
        let cap = self.max_completions.unwrap_or(DEFAULT_CLASS_MEMBERS);
        match self.graph.graph_ref() {
            GraphRef::TemporalAdmg(template) => Ok(ClassMembers {
                members: vec![Member {
                    weight: 1.0,
                    kind: temporal_mixed_member(template, series, &targets)?,
                }],
                capped: false,
                class: "temporal_admg",
            }),
            GraphRef::TemporalCpdag(cpdag) => {
                let mut sampler =
                    antecedent_graph::TemporalCpdagCompletionSampler::new(cpdag.clone(), cap)
                        .map_err(CausalError::rejected)?;
                let members = sampler
                    .by_ref()
                    .map(|completion| Member {
                        weight: 1.0,
                        kind: MemberKind::Directed(AcceptedGraph::from(completion.graph)),
                    })
                    .collect();
                Ok(ClassMembers { members, capped: sampler.hit_cap(), class: "temporal_cpdag" })
            }
            GraphRef::TemporalPag(pag) => {
                let mut sampler =
                    antecedent_graph::TemporalPagCompletionSampler::new(pag.clone(), cap)
                        .map_err(CausalError::rejected)?;
                let mut members = Vec::new();
                for completion in sampler.by_ref() {
                    let template = temporal_mag_template(&completion.graph, series)?;
                    members.push(Member {
                        weight: 1.0,
                        kind: temporal_mixed_member(&template, series, &targets)?,
                    });
                }
                Ok(ClassMembers {
                    members,
                    capped: sampler.hit_cap() || sampler.class_audit_incomplete(),
                    class: "temporal_pag",
                })
            }
            _ => Err(CausalError::Compile {
                message: "temporal class attribution requires a TemporalCpdag, a TemporalPag or \
                          a TemporalAdmg"
                    .into(),
            }),
        }
    }

    /// Publish what the members attributed: the one result when they agree, the weighted mean
    /// with the identified set across them when they disagree.
    #[allow(
        clippy::too_many_lines,
        reason = "one pass: account the members, merge the payload and the posterior, publish the set"
    )]
    fn finish_class_attribution(
        &self,
        class: &ClassMembers,
        outcomes: Vec<MemberRun>,
    ) -> Result<StudyResult, CausalError> {
        let mut identified: Vec<(usize, StudyResult)> = Vec::new();
        let mut hedged: Vec<(usize, MechanismHedge)> = Vec::new();
        let mut failed: Vec<(usize, String)> = Vec::new();
        for (index, (member, outcome)) in class.members.iter().zip(outcomes).enumerate() {
            match (&member.kind, outcome) {
                (MemberKind::Hedged(hedge), _) => hedged.push((index, hedge.clone())),
                (_, Ok(result)) => identified.push((index, result)),
                (_, Err(error)) => failed.push((index, error.to_string())),
            }
        }
        if identified.is_empty() {
            return Err(match (hedged.first(), failed.first()) {
                (Some((_, hedge)), _) => hedge_refusal(hedge, class.capped),
                (None, Some((_, message))) => CausalError::Compile {
                    message: format!(
                        "attribution failed on every member of the {} class: {message}",
                        class.class
                    ),
                },
                (None, None) => CausalError::Compile {
                    message: format!("the {} class has no completion to attribute", class.class),
                },
            });
        }

        let total: f64 = class.members.iter().map(|member| member.weight).sum();
        let mass = |indices: &mut dyn Iterator<Item = usize>| -> f64 {
            indices.map(|index| class.members[index].weight).sum::<f64>() / total
        };
        let identified_mass = mass(&mut identified.iter().map(|(index, _)| *index));
        let unidentified_mass = mass(&mut hedged.iter().map(|(index, _)| *index));
        let unevaluable_mass = mass(&mut failed.iter().map(|(index, _)| *index));

        // One member and nothing left out: the member's own result.
        if identified.len() == 1 && hedged.is_empty() && failed.is_empty() && !class.capped {
            let (_, mut result) = identified.remove(0);
            result.diagnostics.push(envelope_diagnostic(
                class,
                (identified_mass, unidentified_mass, unevaluable_mass),
                1,
            ));
            return Ok(result);
        }

        let weights: Vec<f64> =
            identified.iter().map(|(index, _)| class.members[*index].weight).collect();
        let weights = normalized(&weights);
        let hedge_components: Vec<ComponentId> = hedged
            .iter()
            .map(|(_, hedge)| ComponentId::from_variable(hedge.node))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let merged = merge_payload(&identified, &weights, &hedge_components)?;
        let posterior = shared_posterior(&identified, unidentified_mass);
        let envelope_note =
            if posterior.is_none() { posterior_envelope_diagnostic(&identified) } else { None };

        let mut atoms = Vec::with_capacity(class.members.len());
        for (index, member) in class.members.iter().enumerate() {
            let status = if matches!(member.kind, MemberKind::Hedged(_)) {
                IdentificationStatus::NotIdentified
            } else {
                IdentificationStatus::IdentifiedUnderParametricRestrictions
            };
            let value_position = identified.iter().position(|(held, _)| *held == index);
            let value = value_position.map(|position| {
                ResponseValue::Vector(Arc::from(merged.per_member[position].clone()))
            });
            let posterior =
                value_position.and_then(|position| identified[position].1.posterior.clone());
            atoms.push(StructuralResponseAtom {
                graph_key: u64::try_from(index).unwrap_or(u64::MAX),
                weight: member.weight,
                status,
                value,
                posterior,
                response: None,
            });
        }
        let envelope = ResponseEnvelope {
            grid: Arc::from(
                merged.components.iter().map(|c| f64::from(c.raw())).collect::<Vec<_>>(),
            ),
            dimension: 1,
            lower: Arc::from(merged.lower.clone()),
            upper: Arc::from(merged.upper.clone()),
        };
        let mixture = StructuralResponseMixture {
            weight_basis: StructuralWeightBasis::CompletionEnumeration,
            atoms,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
            subsampled_out_mass: 0.0,
            identified_set: Some(envelope),
            identified_set_interval: None,
            conditional_on_identified: Some(ResponseValue::Vector(Arc::from(merged.mean.clone()))),
            full_mass_scope: !class.capped,
            truncated_atoms: usize::from(class.capped),
        };

        let set_diagnostic = identified_set_diagnostic(&merged);
        let (_, mut result) = identified.remove(0);
        result.payload = merged.payload;
        if let Some(posterior) = posterior {
            if let (ResultPayload::ChangeAttribution(_), Some(index)) =
                (&result.payload, posterior.effect_column())
            {
                if let Some(estimate) = result.estimate.as_effect_mut() {
                    estimate.ate = posterior.summaries.mean[index];
                    estimate.uncertainty.analytic.se = posterior.summaries.sd[index];
                }
            }
            result.posterior = Some(posterior);
        } else {
            // The members' posteriors are conditional on their own structures, and enumeration
            // weights are not posterior mass: the class publishes no mixture of them. Each
            // member's posterior stays on its atom, and the envelope of their credible
            // intervals is the interval for the identified set.
            result.posterior = None;
            if let (ResultPayload::ChangeAttribution(change), Some(estimate)) =
                (&result.payload, result.estimate.as_effect_mut())
            {
                let assumptions = estimate.provenance.assumptions.clone();
                *estimate = EffectEstimate::new(
                    change.total_change,
                    f64::NAN,
                    assumptions,
                    OverlapPolicy::ExplicitOverride,
                );
            }
        }
        if unidentified_mass > 0.0 || unevaluable_mass > 0.0 || class.capped {
            result.identification.status = IdentificationStatus::PartiallyIdentified;
            if let Some(posterior) = result.posterior.as_mut() {
                posterior.identification = IdentificationStatus::PartiallyIdentified;
            }
        }
        result.structural_response = Some(mixture);
        result.diagnostics.push(envelope_diagnostic(
            class,
            (identified_mass, unidentified_mass, unevaluable_mass),
            weights.len(),
        ));
        result.diagnostics.push(set_diagnostic);
        result.diagnostics.extend(envelope_note);
        for (index, hedge) in &hedged {
            result.diagnostics.push(
                Diagnostic::new(
                    CLASS_ATTRIBUTION_HEDGE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    format!("member {index} of the class is refused: {}", hedge.describe()),
                )
                .with_fields([
                    ("member", index.to_string()),
                    ("node", hedge.node.raw().to_string()),
                    ("witness", antecedent_attribution::HEDGE_WITNESS_ID.to_owned()),
                ]),
            );
        }
        for (index, message) in &failed {
            result.diagnostics.push(Diagnostic::new(
                "attribution.class.member_failed",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Warning,
                format!("member {index} of the class could not be attributed: {message}"),
            ));
        }
        Ok(result)
    }
}

/// The typed refusal of a hedged mechanism.
fn hedge_refusal(hedge: &MechanismHedge, search_capped: bool) -> CausalError {
    CausalError::not_identified(
        IdentificationStatus::NotIdentified,
        search_capped,
        &hedge.describe(),
    )
}

/// A member of a mixed graph: the DAG attribution when no latent confounding reaches the
/// ancestors of the targets, the noise-block attribution when it does and every mechanism is
/// identified, the hedge otherwise.
fn mixed_member(admg: &Admg, targets: &[VariableId]) -> Result<MemberKind, CausalError> {
    if !admg.has_bidirected() {
        return Ok(MemberKind::Directed(AcceptedGraph::from(
            admg.directed_part().map_err(CausalError::rejected)?,
        )));
    }
    Ok(match plan_noise_blocks(admg, targets).map_err(CausalError::rejected)? {
        NoiseBlockIdentification::Identified(plan)
            if plan.blocks.iter().all(|block| block.len() == 1) =>
        {
            MemberKind::Directed(AcceptedGraph::from(plan.directed))
        }
        NoiseBlockIdentification::Identified(plan) => MemberKind::Blocks { plan, unfolding: None },
        NoiseBlockIdentification::Hedged(hedge) => MemberKind::Hedged(hedge),
    })
}

/// A MAG completion as the most-confounded mixed graph it allows, labelled as the PAG is.
fn labelled_confounded_admg(pag: &Pag) -> Result<Admg, CausalError> {
    let confounded =
        antecedent_identify::mag_to_confounded_admg(pag).ok_or_else(|| CausalError::Compile {
            message: "a PAG completion carries a mark that no mixed graph reads".into(),
        })?;
    let mut labelled = Admg::empty();
    for node in pag.nodes() {
        labelled.add_node(*node).map_err(CausalError::rejected)?;
    }
    for from in confounded.node_ids() {
        for &to in confounded.children(from) {
            labelled.insert_directed(from, to).map_err(CausalError::rejected)?;
        }
        for &other in confounded.bidirected_neighbors(from) {
            if from.raw() < other.raw() {
                labelled.insert_bidirected(from, other).map_err(CausalError::rejected)?;
            }
        }
    }
    Ok(labelled)
}

/// The longest lag of a template.
fn template_history(nodes: &[NodeRef]) -> u32 {
    nodes
        .iter()
        .filter_map(|node| match node {
            NodeRef::Lagged { lag, .. } => Some(lag.raw()),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// A member of a temporal ADMG: the temporal DAG attribution when the template carries no
/// bidirected edge, otherwise the noise-block attribution on the template unrolled over the
/// periods the longest lag spans (the causes before the window are latent common causes of their
/// in-window effects).
fn temporal_mixed_member(
    template: &TemporalAdmg,
    series: &TimeSeriesData,
    targets: &[VariableId],
) -> Result<MemberKind, CausalError> {
    if !template.has_bidirected() {
        let mut dag = TemporalDag::empty();
        for node in template.nodes() {
            dag.add_node(*node).map_err(CausalError::rejected)?;
        }
        for index in 0..template.node_count() {
            let from = DenseNodeId::from_index(index);
            for &to in template.children(from) {
                dag.insert_directed(from, to).map_err(CausalError::rejected)?;
            }
        }
        return Ok(MemberKind::Directed(AcceptedGraph::from(dag)));
    }
    let history = template_history(template.nodes());
    let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
        CausalError::Compile { message: "series schema exceeds the variable id range".into() }
    })?;
    let window =
        template.unroll_window_over(history + 1, variable_count).map_err(CausalError::rejected)?;
    let (unfolding, admg) = LagAlignedScm::unfold_mixed(series, history, window.admg())?;
    Ok(match plan_noise_blocks(&admg, targets).map_err(CausalError::rejected)? {
        NoiseBlockIdentification::Identified(plan) => {
            MemberKind::Blocks { plan, unfolding: Some(unfolding) }
        }
        NoiseBlockIdentification::Hedged(hedge) => MemberKind::Hedged(hedge),
    })
}

/// A temporal MAG completion as a temporal ADMG template: its directed and bidirected edges,
/// and a bidirected edge beside every directed edge that is invisible in the unfolded window
/// (an invisible edge may carry a latent common cause, Zhang 2008).
fn temporal_mag_template(
    mag: &TemporalPag,
    series: &TimeSeriesData,
) -> Result<TemporalAdmg, CausalError> {
    let mut template = TemporalAdmg::empty();
    for node in mag.nodes() {
        template.add_node(*node).map_err(CausalError::rejected)?;
    }
    let tolerate = |inserted: Result<(), antecedent_graph::GraphError>| match inserted {
        Ok(()) | Err(antecedent_graph::GraphError::DuplicateEdge { .. }) => Ok(()),
        Err(error) => Err(CausalError::rejected(error)),
    };
    for edge in mag.edges() {
        if let Some((parent, child)) = edge.parent_child() {
            tolerate(template.insert_directed(parent, child))?;
        } else if edge.at_a == Endpoint::Arrow && edge.at_b == Endpoint::Arrow {
            tolerate(template.insert_bidirected(edge.a, edge.b))?;
        } else {
            return Err(CausalError::Compile {
                message: "a temporal PAG completion carries a mark that no mixed graph reads"
                    .into(),
            });
        }
    }
    let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
        CausalError::Compile { message: "series schema exceeds the variable id range".into() }
    })?;
    let history = template_history(mag.nodes());
    let indexer = TemporalIndexer::new(variable_count, history, 1)?;
    let unfolded = mag.unfold(indexer).map_err(CausalError::rejected)?;
    let node_of = |template: &mut TemporalAdmg,
                   variable: VariableId,
                   lag: u32|
     -> Result<DenseNodeId, CausalError> {
        let lag = Lag::from_raw(lag);
        let held = template.nodes().iter().position(|node| {
            matches!(node, NodeRef::Lagged { variable: v, lag: l } if *v == variable && *l == lag)
        });
        match held {
            Some(index) => Ok(DenseNodeId::from_index(index)),
            None => template.add_lagged(variable, lag).map_err(CausalError::rejected),
        }
    };
    for (tail, head) in antecedent_identify::invisible_directed_edges(&unfolded.pag) {
        let tail_key = unfolded.indexer.key_of(tail.raw())?;
        let head_key = unfolded.indexer.key_of(head.raw())?;
        let delay =
            u32::try_from(head_key.offset - tail_key.offset).map_err(|_| CausalError::Compile {
                message: "an invisible directed edge points backwards in time".into(),
            })?;
        let tail_node = node_of(&mut template, tail_key.variable, delay)?;
        let head_node = node_of(&mut template, head_key.variable, 0)?;
        if tail_node != head_node {
            tolerate(template.insert_bidirected(tail_node, head_node))?;
        }
    }
    Ok(template)
}

/// The noise blocks a member was attributed to.
fn blocks_diagnostic(plan: &NoiseBlockPlan) -> Diagnostic {
    let blocks: Vec<String> = plan
        .blocks
        .iter()
        .map(|block| {
            let members: Vec<String> = block.iter().map(|v| format!("v{}", v.raw())).collect();
            format!("{{{}}}", members.join(","))
        })
        .collect();
    Diagnostic::new(
        CLASS_ATTRIBUTION_BLOCKS,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the attribution is to the noise blocks {} of the ancestral sub-graph: the noise \
             terms of the mechanisms a latent common cause ties together are one player. Every \
             mechanism is fitted by regression on its parents, which identifies it because no \
             district member reaches it inside its district; the unit's noise vector is the \
             residual vector of its row",
            blocks.join(" ")
        ),
    )
    .with_fields([("blocks", blocks.len().to_string())])
}

/// The accounting of the members of the class.
fn envelope_diagnostic(
    class: &ClassMembers,
    (identified, unidentified, unevaluable): (f64, f64, f64),
    contributing: usize,
) -> Diagnostic {
    Diagnostic::new(
        CLASS_ATTRIBUTION_ENVELOPE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "{} class: {} member(s) enumerated ({} attributed); identified_mass={identified}, \
             unidentified_mass={unidentified}, unevaluable_mass={unevaluable}; member weights \
             are enumeration weights, not posterior probabilities{}",
            class.class,
            class.members.len(),
            contributing,
            if class.capped {
                "; the enumeration stopped at its cap or budget, so the members are a subset of \
                 the class and no class-wide claim is earned"
            } else {
                ""
            }
        ),
    )
    .with_fields([
        ("identified_mass", identified.to_string()),
        ("unidentified_mass", unidentified.to_string()),
        ("unevaluable_mass", unevaluable.to_string()),
    ])
}

/// The per-component range across the members.
fn identified_set_diagnostic(merged: &Merged) -> Diagnostic {
    let ranges: Vec<String> = merged
        .components
        .iter()
        .zip(merged.lower.iter().zip(&merged.upper))
        .map(|(component, (lower, upper))| format!("v{}: [{lower}, {upper}]", component.raw()))
        .collect();
    Diagnostic::new(
        CLASS_ATTRIBUTION_IDENTIFIED_SET,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "contribution of each component across the members that were attributed (the mean \
             over rows for an anomaly attribution): {}; the published contribution is the \
             enumeration-weighted mean, and its standard deviation across members is the graph \
             sensitivity",
            ranges.join("; ")
        ),
    )
}

/// What merging the members' payloads produced.
struct Merged {
    payload: ResultPayload,
    /// Components of the union of the members, ascending.
    components: Vec<ComponentId>,
    /// The value of each component in each member (zero where a member has no such player).
    per_member: Vec<Vec<f64>>,
    mean: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn normalized(weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// Weighted mean, minimum, maximum and weighted standard deviation across members, per column.
fn spread(per_member: &[Vec<f64>], weights: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let width = per_member.first().map_or(0, Vec::len);
    let mut mean = vec![0.0; width];
    let mut lower = vec![f64::INFINITY; width];
    let mut upper = vec![f64::NEG_INFINITY; width];
    for (values, weight) in per_member.iter().zip(weights) {
        for (column, value) in values.iter().enumerate() {
            mean[column] += weight * value;
            lower[column] = lower[column].min(*value);
            upper[column] = upper[column].max(*value);
        }
    }
    let mut deviation = vec![0.0; width];
    for (values, weight) in per_member.iter().zip(weights) {
        for (column, value) in values.iter().enumerate() {
            let centred = value - mean[column];
            deviation[column] += weight * centred * centred;
        }
    }
    for value in &mut deviation {
        *value = value.sqrt();
    }
    (mean, lower, upper, deviation)
}

/// Merge the members' anomaly or change payloads.
fn merge_payload(
    identified: &[(usize, StudyResult)],
    weights: &[f64],
    hedge_components: &[ComponentId],
) -> Result<Merged, CausalError> {
    match &identified[0].1.payload {
        ResultPayload::ChangeAttribution(_) => {
            let parts: Vec<&ChangeAttributionResult> = identified
                .iter()
                .filter_map(|(_, result)| match &result.payload {
                    ResultPayload::ChangeAttribution(change) => Some(change),
                    _ => None,
                })
                .collect();
            if parts.len() != identified.len() {
                return Err(class_payload_mismatch());
            }
            Ok(merge_change(&parts, weights, hedge_components))
        }
        ResultPayload::Anomaly(_) => {
            let parts: Vec<&[AnomalyScores]> = identified
                .iter()
                .filter_map(|(_, result)| match &result.payload {
                    ResultPayload::Anomaly(scores) => Some(scores.as_slice()),
                    _ => None,
                })
                .collect();
            if parts.len() != identified.len() {
                return Err(class_payload_mismatch());
            }
            merge_anomaly(&parts, weights)
        }
        _ => Err(class_payload_mismatch()),
    }
}

fn class_payload_mismatch() -> CausalError {
    CausalError::Compile {
        message: "the members of a class attribution published different result kinds".into(),
    }
}

fn merge_change(
    parts: &[&ChangeAttributionResult],
    weights: &[f64],
    hedge_components: &[ComponentId],
) -> Merged {
    let components: Vec<ComponentId> = parts
        .iter()
        .flat_map(|part| part.contributions.iter().map(|c| c.component))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let per_member: Vec<Vec<f64>> = parts
        .iter()
        .map(|part| {
            components
                .iter()
                .map(|component| {
                    part.contributions
                        .iter()
                        .find(|c| c.component == *component)
                        .map_or(0.0, |c| c.contribution)
                })
                .collect()
        })
        .collect();
    let (mean, lower, upper, deviation) = spread(&per_member, weights);
    let total_change: f64 =
        parts.iter().zip(weights).map(|(part, weight)| weight * part.total_change).sum();
    let observed_change = parts
        .iter()
        .zip(weights)
        .map(|(part, weight)| part.observed_change.map(|observed| weight * observed))
        .sum::<Option<f64>>();
    let unidentified: Vec<ComponentId> = parts
        .iter()
        .flat_map(|part| part.unidentified.iter().copied())
        .chain(hedge_components.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let contributions: Vec<ComponentContribution> = components
        .iter()
        .zip(&mean)
        .map(|(component, contribution)| ComponentContribution {
            component: *component,
            contribution: *contribution,
            stderr: None,
            ci_low: None,
            ci_high: None,
        })
        .collect();
    let budget = ComputeBudget {
        evaluations: parts.iter().map(|part| part.budget.evaluations).sum(),
        samples: parts.iter().map(|part| part.budget.samples).sum(),
        exact_coalitions: parts.iter().map(|part| part.budget.exact_coalitions).sum(),
    };
    let result = ChangeAttributionResult {
        outcome: parts[0].outcome,
        total_change,
        observed_change,
        contributions: Arc::from(contributions),
        interactions: Arc::from([]),
        path_breakdown: Arc::from([]),
        unidentified: Arc::from(unidentified),
        graph_sensitivity: Some(Arc::from(deviation)),
        budget,
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: CacheStats::default(),
        fit_uncertainty: None,
    };
    Merged {
        payload: ResultPayload::ChangeAttribution(result),
        components,
        per_member,
        mean,
        lower,
        upper,
    }
}

/// Merge the members' anomaly scores: the factual scores do not depend on the graph, so the
/// rows and scores are the first member's; the contributions are the weighted mean over members
/// on the union of their players. The identified set is read on the first target.
fn merge_anomaly(parts: &[&[AnomalyScores]], weights: &[f64]) -> Result<Merged, CausalError> {
    let targets = parts[0].len();
    if parts.iter().any(|part| part.len() != targets) {
        return Err(class_payload_mismatch());
    }
    let mut merged_scores = Vec::with_capacity(targets);
    let mut first: Option<(Vec<ComponentId>, Vec<Vec<f64>>)> = None;
    for target in 0..targets {
        let base = &parts[0][target];
        let rows = base.rows.len();
        if parts.iter().any(|part| {
            part[target].target != base.target
                || part[target].rows.len() != rows
                || part[target].baseline_scores.len() != rows
        }) {
            return Err(class_payload_mismatch());
        }
        let components: Vec<ComponentId> = parts
            .iter()
            .flat_map(|part| part[target].noise_components.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let width = components.len();
        let mut contributions = vec![0.0; rows * width];
        let mut baselines = vec![0.0; rows];
        let mut per_member = Vec::with_capacity(parts.len());
        for (part, weight) in parts.iter().zip(weights) {
            let scores = &part[target];
            let held = scores.noise_components.len();
            let positions: Vec<usize> = scores
                .noise_components
                .iter()
                .filter_map(|component| components.iter().position(|c| c == component))
                .collect();
            if positions.len() != held || scores.noise_contributions.len() != rows * held {
                return Err(class_payload_mismatch());
            }
            let mut row_means = vec![0.0; width];
            for row in 0..rows {
                baselines[row] += weight * scores.baseline_scores[row];
                for (column, &position) in positions.iter().enumerate() {
                    let value = scores.noise_contributions[row * held + column];
                    contributions[row * width + position] += weight * value;
                    row_means[position] += value / rows as f64;
                }
            }
            per_member.push(row_means);
        }
        let absolute: Vec<f64> = (0..rows)
            .map(|row| contributions[row * width..(row + 1) * width].iter().map(|v| v.abs()).sum())
            .collect();
        merged_scores.push(AnomalyScores {
            target: base.target,
            rows: Arc::clone(&base.rows),
            scores: Arc::clone(&base.scores),
            baseline_scores: Arc::from(baselines),
            abs_contribution_sum: Arc::from(absolute),
            noise_components: Arc::from(components.clone()),
            noise_contributions: Arc::from(contributions),
        });
        if first.is_none() {
            first = Some((components, per_member));
        }
    }
    let (components, per_member) = first.ok_or_else(class_payload_mismatch)?;
    let (mean, lower, upper, _) = spread(&per_member, weights);
    Ok(Merged {
        payload: ResultPayload::Anomaly(merged_scores),
        components,
        per_member,
        mean,
        lower,
        upper,
    })
}

/// The posterior of the class, when it has one.
///
/// A member's posterior is conditional on its own structure and the enumeration weights are not
/// posterior mass, so members' posteriors are never mixed. A posterior is published only when
/// every member published the very same draws, as a quantity that does not depend on the
/// structure does (the mean anomaly score): it is then each member's own posterior.
fn shared_posterior(
    identified: &[(usize, StudyResult)],
    unidentified_mass: f64,
) -> Option<CausalPosterior> {
    let first = identified[0].1.posterior.as_ref()?;
    for (_, result) in &identified[1..] {
        if result.posterior.as_ref()?.draws != first.draws {
            return None;
        }
    }
    let mut shared = first.clone();
    shared.unidentified_mass = unidentified_mass;
    Some(shared)
}

/// The envelope of the members' 95% credible intervals, per quantity: the interval for the
/// identified set of a quantity the members' posteriors disagree on. A member without the
/// quantity (a component that is not one of its players) contributes the point zero.
fn posterior_envelope_diagnostic(identified: &[(usize, StudyResult)]) -> Option<Diagnostic> {
    let posteriors: Vec<&CausalPosterior> =
        identified.iter().map(|(_, result)| result.posterior.as_ref()).collect::<Option<_>>()?;
    if posteriors.len() < 2 {
        return None;
    }
    let name_of = |kind: &PosteriorQuantityKind| match kind {
        PosteriorQuantityKind::Effect { name } | PosteriorQuantityKind::Scalar { name } => {
            Some(Arc::clone(name))
        }
        _ => None,
    };
    let mut names: Vec<Arc<str>> = Vec::new();
    for posterior in &posteriors {
        for name in posterior.draws.schema.quantities.iter().filter_map(name_of) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    let mut ranges = Vec::with_capacity(names.len());
    for name in &names {
        let (mut lower, mut upper) = (f64::INFINITY, f64::NEG_INFINITY);
        for posterior in &posteriors {
            let position = posterior
                .draws
                .schema
                .quantities
                .iter()
                .position(|kind| name_of(kind).as_ref() == Some(name));
            let (low, high) = position.map_or((0.0, 0.0), |index| {
                (posterior.summaries.q025[index], posterior.summaries.q975[index])
            });
            lower = lower.min(low);
            upper = upper.max(high);
        }
        ranges.push(format!("{name}: [{lower}, {upper}]"));
    }
    Some(Diagnostic::new(
        "attribution.class.posterior_envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the members' posteriors differ, so the class publishes no mixture of them \
             (enumeration weights are not posterior mass; each posterior stays on its atom): the \
             envelope of their 95% credible intervals is the interval for the identified set: {}",
            ranges.join("; ")
        ),
    ))
}
