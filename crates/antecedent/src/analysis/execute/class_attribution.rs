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
use antecedent_discovery::adjacency_mask::{
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask,
    pag_from_adjacency_mask,
};
use antecedent_discovery::{AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, LagMask};
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
    /// The graph posterior the members were drawn from, when the structures come from one.
    posterior: Option<PosteriorBasis>,
}

/// The graph posterior a set of members was drawn from. A member's weight is then the
/// posterior probability of its atom (a class atom shares its probability equally among its
/// completions), not an enumeration weight.
struct PosteriorBasis {
    /// Graph key of the atom each member stands for, aligned with the members.
    keys: Vec<u64>,
    /// Atoms that name no structure of the posterior's class (a cyclic mask, an unreadable
    /// mark), with their posterior probability: unidentified mass.
    outside: Vec<(u64, f64)>,
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
        let class = self.static_class_members(ctx)?;
        let cache = self.class_attribution_identification(physical, ctx)?;
        let outcomes = Self::run_class_members(&class, ctx, |member| {
            self.attribute_static_member(member, data, physical, &cache, ctx)
        })?;
        Self::finish_class_attribution(&class, outcomes)
    }

    /// Attribute one member of a static structure class: the DAG attribution on a directed
    /// member, the noise-block attribution on a confounded one, the typed refusal on a hedged
    /// one. `self` is the study the member is attributed under, so it must be a supplied or
    /// accepted structure.
    fn attribute_static_member(
        &self,
        member: &Member,
        data: &TabularData,
        physical: &Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        ctx: &ExecutionContext,
    ) -> MemberRun {
        match &member.kind {
            MemberKind::Directed(graph) => {
                let atom = self.member_study(graph);
                CheckedAttributionOperation::checked(&atom, physical, cache)?.execute(data, ctx)
            }
            MemberKind::Blocks { plan, .. } => {
                let mut result = CheckedAttributionOperation::checked_blocks(
                    self,
                    physical,
                    cache,
                    (plan.directed.clone(), None),
                    plan.blocks.clone(),
                )?
                .execute(data, ctx)?;
                result.diagnostics.push(blocks_diagnostic(plan));
                Ok(result)
            }
            MemberKind::Hedged(hedge) => Err(hedge_refusal(hedge, false)),
        }
    }

    /// Anomaly or change attribution over a temporal CPDAG, a temporal PAG or a temporal ADMG,
    /// on a series.
    pub(super) fn execute_temporal_class_attribution(
        &self,
        series: &TimeSeriesData,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let class = self.temporal_class_members(series, ctx)?;
        let cache = self.class_attribution_identification(physical, ctx)?;
        let outcomes = Self::run_class_members(&class, ctx, |member| {
            self.attribute_series_member(member, series, physical, &cache, ctx)
        })?;
        Self::finish_class_attribution(&class, outcomes)
    }

    /// Attribute one member of a temporal structure class on a series, as
    /// [`Self::attribute_static_member`] does on a table.
    fn attribute_series_member(
        &self,
        member: &Member,
        series: &TimeSeriesData,
        physical: &Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        ctx: &ExecutionContext,
    ) -> MemberRun {
        match &member.kind {
            MemberKind::Directed(graph) => {
                let atom = self.member_study(graph);
                CheckedAttributionOperation::checked(&atom, physical, cache)?
                    .execute_series(series, ctx)
            }
            MemberKind::Blocks { plan, unfolding } => {
                let mut result = CheckedAttributionOperation::checked_blocks(
                    self,
                    physical,
                    cache,
                    (plan.directed.clone(), unfolding.clone()),
                    plan.blocks.clone(),
                )?
                .execute_series(series, ctx)?;
                result.diagnostics.push(blocks_diagnostic(plan));
                Ok(result)
            }
            MemberKind::Hedged(hedge) => Err(hedge_refusal(hedge, false)),
        }
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
                if matches!(error, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
                    return Err(error.clone());
                }
            }
            outcomes.push(outcome);
        }
        Ok(outcomes)
    }

    /// The members of a static structure class.
    fn static_class_members(&self, ctx: &ExecutionContext) -> Result<ClassMembers, CausalError> {
        let targets = self.class_attribution_targets()?;
        let cap = self.max_completions.unwrap_or(DEFAULT_CLASS_MEMBERS);
        match self.graph.graph_ref() {
            GraphRef::Admg(admg) => Ok(ClassMembers {
                members: vec![Member { weight: 1.0, kind: mixed_member(admg, &targets)? }],
                capped: false,
                class: "admg",
                posterior: None,
            }),
            GraphRef::TemporalAdmg(temporal) => {
                let admg = temporal_admg_projection(temporal)?;
                Ok(ClassMembers {
                    members: vec![Member { weight: 1.0, kind: mixed_member(&admg, &targets)? }],
                    capped: false,
                    class: "temporal_admg",
                    posterior: None,
                })
            }
            GraphRef::Cpdag(cpdag) => {
                let mut sampler = antecedent_graph::CpdagCompletionSampler::with_budget(
                    cpdag.clone(),
                    cap,
                    class_completion_limits(),
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                let members = sampler
                    .by_ref()
                    .map(|completion| Member {
                        weight: 1.0,
                        kind: MemberKind::Directed(AcceptedGraph::from(completion.graph)),
                    })
                    .collect();
                crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                Ok(ClassMembers {
                    members,
                    capped: sampler.hit_cap(),
                    class: "cpdag",
                    posterior: None,
                })
            }
            GraphRef::Pag(pag) => {
                let mut sampler = antecedent_graph::CompletionSampler::with_budget(
                    pag,
                    cap,
                    class_completion_limits(),
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                let mut members = Vec::new();
                for completion in sampler.by_ref() {
                    let admg = labelled_confounded_admg(&completion.graph)?;
                    members.push(Member { weight: 1.0, kind: mixed_member(&admg, &targets)? });
                }
                crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                Ok(ClassMembers {
                    members,
                    capped: sampler.hit_cap() || sampler.class_audit_incomplete(),
                    class: "pag",
                    posterior: None,
                })
            }
            _ => Err(CausalError::Compile {
                message: "class attribution requires a Cpdag, a Pag, an Admg or a TemporalAdmg"
                    .into(),
            }),
        }
    }

    /// The members of a temporal structure class over `series`.
    fn temporal_class_members(
        &self,
        series: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<ClassMembers, CausalError> {
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
                posterior: None,
            }),
            GraphRef::TemporalCpdag(cpdag) => {
                let mut sampler = antecedent_graph::TemporalCpdagCompletionSampler::with_budget(
                    cpdag.clone(),
                    cap,
                    class_completion_limits(),
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                let members = sampler
                    .by_ref()
                    .map(|completion| Member {
                        weight: 1.0,
                        kind: MemberKind::Directed(AcceptedGraph::from(completion.graph)),
                    })
                    .collect();
                crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                Ok(ClassMembers {
                    members,
                    capped: sampler.hit_cap(),
                    class: "temporal_cpdag",
                    posterior: None,
                })
            }
            GraphRef::TemporalPag(pag) => {
                let mut sampler = antecedent_graph::TemporalPagCompletionSampler::with_budget(
                    pag.clone(),
                    cap,
                    class_completion_limits(),
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                let mut members = Vec::new();
                for completion in sampler.by_ref() {
                    let template = temporal_mag_template(&completion.graph, series)?;
                    members.push(Member {
                        weight: 1.0,
                        kind: temporal_mixed_member(&template, series, &targets)?,
                    });
                }
                crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                Ok(ClassMembers {
                    members,
                    capped: sampler.hit_cap() || sampler.class_audit_incomplete(),
                    class: "temporal_pag",
                    posterior: None,
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

        // Atoms of a graph posterior that name no structure of its class are unidentified mass
        // beside the members.
        let outside_weight: f64 = class
            .posterior
            .as_ref()
            .map_or(0.0, |posterior| posterior.outside.iter().map(|(_, weight)| *weight).sum());
        let total: f64 =
            class.members.iter().map(|member| member.weight).sum::<f64>() + outside_weight;
        let mass = |indices: &mut dyn Iterator<Item = usize>| -> f64 {
            indices.map(|index| class.members[index].weight).sum::<f64>() / total
        };
        let identified_mass = mass(&mut identified.iter().map(|(index, _)| *index));
        let unidentified_mass =
            mass(&mut hedged.iter().map(|(index, _)| *index)) + outside_weight / total;
        let unevaluable_mass = mass(&mut failed.iter().map(|(index, _)| *index));

        // One member and nothing left out: the member's own result.
        if identified.len() == 1
            && hedged.is_empty()
            && failed.is_empty()
            && !class.capped
            && outside_weight <= 0.0
        {
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
            let graph_key = class
                .posterior
                .as_ref()
                .and_then(|posterior| posterior.keys.get(index).copied())
                .unwrap_or_else(|| u64::try_from(index).unwrap_or(u64::MAX));
            atoms.push(StructuralResponseAtom {
                graph_key,
                weight: member.weight,
                status,
                value,
                posterior,
                response: None,
            });
        }
        if let Some(posterior) = class.posterior.as_ref() {
            for (key, weight) in &posterior.outside {
                atoms.push(StructuralResponseAtom {
                    graph_key: *key,
                    weight: *weight,
                    status: IdentificationStatus::NotIdentified,
                    value: None,
                    posterior: None,
                    response: None,
                });
            }
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
            weight_basis: if class.posterior.is_some() {
                StructuralWeightBasis::PosteriorProbability
            } else {
                StructuralWeightBasis::CompletionEnumeration
            },
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
        // Reports the members themselves carry (a member's own refuters) mix by the posterior
        // weight of the atoms that produced them; a whole-procedure validation of the
        // published mixture is run by the functional refuter state on top of this.
        let member_reports = if class.posterior.is_some() {
            let per_member: Vec<(f64, Vec<antecedent_validate::RefutationReport>)> = identified
                .iter()
                .zip(&weights)
                .map(|((_, member), weight)| (*weight, member.refutations.clone()))
                .filter(|(_, reports)| !reports.is_empty())
                .collect();
            (!per_member.is_empty()).then(|| super::mix_atom_refutation_reports(&per_member))
        } else {
            None
        };
        let (_, mut result) = identified.remove(0);
        if let Some((reports, notes)) = member_reports {
            result.refutations = reports;
            result.diagnostics.extend(notes);
        }
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
            // A class leaves the attribution partly identified; a posterior over structures
            // leaves it dependent on the graph.
            let status = if class.posterior.is_some() {
                IdentificationStatus::GraphDependent
            } else {
                IdentificationStatus::PartiallyIdentified
            };
            result.identification.status = status;
            if let Some(posterior) = result.posterior.as_mut() {
                posterior.identification = status;
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

impl super::Study {
    /// Anomaly or change attribution over the atoms of a tabular graph posterior.
    ///
    /// Each atom is attributed as a supplied structure of its kind would be: a DAG atom by the
    /// DAG attribution, a mixed-graph atom by the noise-block attribution (or the typed hedge
    /// refusal), a class atom member by member over its completions. The members are then merged
    /// as the class route merges them, weighted by the posterior probability of their atom: the
    /// published contribution of each component is the posterior-weighted mean over the atoms
    /// that attribute it, the spread across atoms is the graph sensitivity, and the range is
    /// the identified set. An atom the identifier refuses is unidentified posterior mass and an
    /// atom whose mechanisms cannot be fitted is unevaluable mass; neither is redistributed.
    ///
    /// # Errors
    ///
    /// A posterior with DBN lag structure, a validation suite other than `none`, no atom that
    /// attributes, or a cancelled run.
    pub(super) fn execute_attribution_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if gp.lag_masks().is_some() || gp.max_lag().is_some() {
            return Err(CausalError::Unsupported {
                message: "a static attribution over a graph posterior requires static atoms; \
                          this posterior carries DBN lag structure",
            });
        }
        let class = self.posterior_class_members(gp, ctx)?;
        let cache = self.class_attribution_identification(physical, ctx)?;
        // The mechanisms of an atom are fitted on it exactly as on an accepted structure: the
        // posterior is the provenance of the atom's graph, not a different graph.
        let mut base = self.clone();
        base.structure_source = crate::support::StructureSource::Accepted;
        let outcomes = Self::run_class_members(&class, ctx, |member| {
            base.attribute_static_member(member, data, physical, &cache, ctx)
        })?;
        let mut result = Self::finish_class_attribution(&class, outcomes)?;
        result.structure_source = crate::support::StructureSource::GraphPosterior;
        Ok(result)
    }

    /// Anomaly or change attribution over the atoms of a graph posterior on a series: a temporal
    /// DAG atom by the temporal DAG attribution, a lagged temporal-ADMG atom by the noise-block
    /// attribution on its window (or the typed hedge refusal), a temporal CPDAG or PAG atom
    /// member by member over its completions, merged as the static posterior route merges them.
    ///
    /// # Errors
    ///
    /// A posterior without per-atom lag masks and `max_lag`, a validation suite other than
    /// `none`, no atom that attributes, or a cancelled run.
    pub(super) fn execute_series_attribution_graph_posterior(
        &self,
        series: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let class = self.series_posterior_members(series, gp, ctx)?;
        let cache = self.class_attribution_identification(physical, ctx)?;
        let mut base = self.clone();
        base.structure_source = crate::support::StructureSource::Accepted;
        let outcomes = Self::run_class_members(&class, ctx, |member| {
            base.attribute_series_member(member, series, physical, &cache, ctx)
        })?;
        let mut result = Self::finish_class_attribution(&class, outcomes)?;
        result.structure_source = crate::support::StructureSource::GraphPosterior;
        Ok(result)
    }

    /// The members an attribution over a series posterior is repeated over, weighted by the
    /// posterior probability of their atom (a class atom shares its probability equally among
    /// its completions). Two samples that hold the same template are one atom.
    fn series_posterior_members(
        &self,
        series: &TimeSeriesData,
        gp: &GraphPosterior,
        ctx: &ExecutionContext,
    ) -> Result<ClassMembers, CausalError> {
        type Template = (AdjacencyMask, LagMask, AdjacencyMask, LagMask);
        let (Some(lag_masks), Some(max_lag)) = (gp.lag_masks(), gp.max_lag()) else {
            return Err(CausalError::Unsupported {
                message: "a series attribution over a graph posterior needs per-atom lag masks \
                          and max_lag",
            });
        };
        let targets = self.class_attribution_targets()?;
        let variables: Vec<VariableId> =
            series.schema().variables().iter().map(|variable| variable.id).collect();
        let mark_of = |index: usize| gp.mark_atom(index);
        let latent_of = |index: usize| {
            gp.lag_bidirected_masks().map_or(LagMask::EMPTY, |masks| masks[index].clone())
        };
        let mut mass: std::collections::HashMap<Template, f64> = std::collections::HashMap::new();
        let mut firsts: Vec<(usize, Template)> = Vec::new();
        for index in 0..gp.n_graphs() {
            let template = (
                gp.atoms()[index].clone(),
                lag_masks[index].clone(),
                mark_of(index),
                latent_of(index),
            );
            let slot = mass.entry(template.clone()).or_insert_with(|| {
                firsts.push((index, template));
                0.0
            });
            *slot += gp.weights()[index];
        }
        let mut members: Vec<Member> = Vec::new();
        let mut keys: Vec<u64> = Vec::new();
        let mut outside: Vec<(u64, f64)> = Vec::new();
        let mut capped = false;
        for (index, template) in firsts {
            let key = u64::try_from(index).unwrap_or(u64::MAX);
            let weight = mass[&template];
            let mut kinds: Vec<(f64, MemberKind)> = Vec::new();
            let class_graph = match series_posterior_atom(
                series,
                gp,
                index,
                (lag_masks, max_lag),
                &variables,
                &targets,
            )? {
                Some(SeriesAtomGraph::Member(kind)) => {
                    kinds.push((weight, kind));
                    None
                }
                Some(SeriesAtomGraph::Class(graph)) => Some(graph),
                None => {
                    outside.push((key, weight));
                    None
                }
            };
            if let Some(graph) = class_graph {
                let mut atom = self.clone();
                atom.graph = Arc::new(graph);
                let completions = atom.temporal_class_members(series, ctx)?;
                capped |= completions.capped;
                let share = weight / completions.members.len().max(1) as f64;
                kinds.extend(completions.members.into_iter().map(|member| (share, member.kind)));
            }
            for (share, kind) in kinds {
                keys.push(key);
                members.push(Member { weight: share, kind });
            }
        }
        Ok(ClassMembers {
            members,
            capped,
            class: "graph_posterior",
            posterior: Some(PosteriorBasis { keys, outside }),
        })
    }

    /// The members an attribution over `gp` is repeated over: one per DAG or mixed-graph atom,
    /// one per completion of a class atom, weighted by the posterior probability of the atom.
    fn posterior_class_members(
        &self,
        gp: &GraphPosterior,
        ctx: &ExecutionContext,
    ) -> Result<ClassMembers, CausalError> {
        let targets = self.class_attribution_targets()?;
        let cap = self.max_completions.unwrap_or(DEFAULT_CLASS_MEMBERS);
        let n_vars = gp.n_vars();
        let mut mass: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        let mut firsts: Vec<usize> = Vec::new();
        for index in 0..gp.n_graphs() {
            let key = gp.graph_keys()[index];
            let slot = mass.entry(key).or_insert_with(|| {
                firsts.push(index);
                0.0
            });
            *slot += gp.weights()[index];
        }
        let mut members: Vec<Member> = Vec::new();
        let mut keys: Vec<u64> = Vec::new();
        let mut outside: Vec<(u64, f64)> = Vec::new();
        let mut capped = false;
        for index in firsts {
            let key = gp.graph_keys()[index];
            let weight = mass[&key];
            let atom = gp.atoms()[index].clone();
            let mut kinds: Vec<(f64, MemberKind)> = Vec::new();
            match gp.atom_kind() {
                GraphPosteriorAtomKind::Dag => match dag_from_adjacency_mask(atom, n_vars) {
                    Ok(dag) => kinds.push((weight, MemberKind::Directed(AcceptedGraph::from(dag)))),
                    Err(_) => outside.push((key, weight)),
                },
                GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
                    match admg_from_adjacency_mask(atom, n_vars) {
                        Ok(admg) => kinds.push((weight, mixed_member(&admg, &targets)?)),
                        Err(_) => outside.push((key, weight)),
                    }
                }
                GraphPosteriorAtomKind::Cpdag => match cpdag_from_adjacency_mask(atom, n_vars) {
                    Ok(cpdag) => {
                        let mut sampler = antecedent_graph::CpdagCompletionSampler::with_budget(
                            cpdag,
                            cap,
                            class_completion_limits(),
                            ctx,
                        )
                        .map_err(CausalError::rejected)?;
                        let completions: Vec<_> = sampler.by_ref().collect();
                        crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                        capped |= sampler.hit_cap();
                        let share = weight / completions.len().max(1) as f64;
                        for completion in completions {
                            kinds.push((
                                share,
                                MemberKind::Directed(AcceptedGraph::from(completion.graph)),
                            ));
                        }
                    }
                    Err(_) => outside.push((key, weight)),
                },
                GraphPosteriorAtomKind::Pag => {
                    let mark = gp.mark_atom(index);
                    match pag_from_adjacency_mask(atom, mark, n_vars) {
                        Ok(pag) => {
                            let mut sampler = antecedent_graph::CompletionSampler::with_budget(
                                &pag,
                                cap,
                                class_completion_limits(),
                                ctx,
                            )
                            .map_err(CausalError::rejected)?;
                            let mut found = Vec::new();
                            for completion in sampler.by_ref() {
                                let admg = labelled_confounded_admg(&completion.graph)?;
                                found.push(mixed_member(&admg, &targets)?);
                            }
                            crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                            capped |= sampler.hit_cap() || sampler.class_audit_incomplete();
                            let share = weight / found.len().max(1) as f64;
                            kinds.extend(found.into_iter().map(|kind| (share, kind)));
                        }
                        Err(_) => outside.push((key, weight)),
                    }
                }
                _ => {
                    return Err(CausalError::Unsupported {
                        message: "an attribution over a graph posterior reads Dag, Admg, \
                                  TemporalAdmg, Cpdag or Pag atoms",
                    });
                }
            }
            for (share, kind) in kinds {
                keys.push(key);
                members.push(Member { weight: share, kind });
            }
        }
        Ok(ClassMembers {
            members,
            capped,
            class: "graph_posterior",
            posterior: Some(PosteriorBasis { keys, outside }),
        })
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
             unidentified_mass={unidentified}, unevaluable_mass={unevaluable}; {}{}",
            class.class,
            class.members.len(),
            contributing,
            if class.posterior.is_some() {
                "member weights are the posterior probabilities of the graph posterior's atoms (a \
                 class atom shares its probability equally among its completions)"
            } else {
                "member weights are enumeration weights, not posterior probabilities"
            },
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

/// A decoded posterior atom is either one mechanism model or a class to enumerate.
enum SeriesAtomGraph {
    Member(MemberKind),
    Class(AcceptedGraph),
}

fn series_posterior_atom(
    series: &TimeSeriesData,
    gp: &GraphPosterior,
    index: usize,
    lags: (&[LagMask], u32),
    variables: &[VariableId],
    targets: &[VariableId],
) -> Result<Option<SeriesAtomGraph>, CausalError> {
    use antecedent_discovery::dbn_posterior::{
        temporal_cpdag_from_dbn_atoms, temporal_dag_from_dbn_atoms, temporal_pag_from_dbn_atoms,
    };
    let (lag_masks, max_lag) = lags;
    let graph = match gp.atom_kind() {
        GraphPosteriorAtomKind::Dag => {
            let Ok(mut dag) = temporal_dag_from_dbn_atoms(
                gp.atoms()[index].clone(),
                &lag_masks[index],
                gp.n_vars(),
                max_lag,
                variables,
            ) else {
                return Ok(None);
            };
            // Empty templates still carry isolated target mechanisms.
            if dag.nodes().is_empty() {
                for &target in targets {
                    antecedent_graph::unfold::ensure_lagged(
                        &mut dag,
                        target,
                        antecedent_core::Lag::CONTEMPORANEOUS,
                    )
                    .map_err(CausalError::rejected)?;
                }
            }
            SeriesAtomGraph::Member(MemberKind::Directed(AcceptedGraph::temporal_dag(dag)))
        }
        GraphPosteriorAtomKind::TemporalAdmg => {
            let Some(template) = gp
                .temporal_admg_atom(index)
                .and_then(|atom| atom.to_template(gp.n_vars(), max_lag, variables).ok())
            else {
                return Ok(None);
            };
            SeriesAtomGraph::Member(temporal_mixed_member(&template, series, targets)?)
        }
        GraphPosteriorAtomKind::Cpdag => {
            let Some(graph) = temporal_cpdag_from_dbn_atoms(
                gp.atoms()[index].clone(),
                &lag_masks[index],
                gp.n_vars(),
                max_lag,
                variables,
            )
            .ok()
            .and_then(|cpdag| AcceptedGraph::temporal_cpdag(cpdag).ok()) else {
                return Ok(None);
            };
            SeriesAtomGraph::Class(graph)
        }
        GraphPosteriorAtomKind::Pag => {
            let mark = gp.mark_atom(index);
            let Ok(pag) = temporal_pag_from_dbn_atoms(
                gp.atoms()[index].clone(),
                &lag_masks[index],
                mark,
                gp.n_vars(),
                max_lag,
                variables,
            ) else {
                return Ok(None);
            };
            SeriesAtomGraph::Class(AcceptedGraph::temporal_pag(pag))
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "a series attribution over a graph posterior reads Dag, \
                      TemporalAdmg, Cpdag or Pag atoms with lag masks",
            });
        }
    };
    Ok(Some(graph))
}

fn class_completion_limits() -> antecedent_core::SearchLimits {
    antecedent_core::SearchLimits {
        operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
        depth: usize::MAX,
    }
}
