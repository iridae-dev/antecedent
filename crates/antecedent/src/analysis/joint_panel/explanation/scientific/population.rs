//! Population-standardized reference worlds under an explicit selection diagram.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    decompose_scientific_reference, invalid, ExecutionContext, Intervention, JointPanelError,
    PanelDecomposition, ReferenceComponentBindings, ScientificConsumptionScope,
    ScientificDecisionSource, ScientificReferenceDecomposition, ScientificWorldId, SemanticDigest,
    TemporalPolicy, VariableId, WorldAction,
};
use antecedent_graph::{DenseNodeId, NodeRef, SelectionDiagram};

/// Known baseline-stratum distributions and actual identified source interventions.
#[derive(Clone, Debug)]
pub struct PopulationReferenceBindings {
    /// Root, causally exogenous baseline covariate whose distribution may change.
    pub baseline: VariableId,
    /// Distinct finite supported baseline values in source-world order.
    pub strata: Vec<f64>,
    /// One complete hybrid action basis per stratum; population bits are aliases here.
    pub stratum_worlds: Vec<[ScientificWorldId; 16]>,
    /// Known before target probabilities over the declared strata.
    pub before_weights: Vec<f64>,
    /// Known after target probabilities, replacing only the population block.
    pub after_weights: Vec<f64>,
    /// Exposure/saturation/mechanism variable bindings and requested ordering.
    /// Its `hybrids` field must equal the first stratum's declared world basis.
    pub components: ReferenceComponentBindings,
}
/// Derived root-covariate transport theorem, retaining its actual structural premises.
#[derive(Clone, Debug)]
pub struct PopulationReferenceTransportProof {
    diagram: SelectionDiagram,
    identity: SemanticDigest,
}
impl PopulationReferenceTransportProof {
    /// Exact source/target mechanism-invariance diagram consumed by the proof.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagram {
        &self.diagram
    }
    /// Binds the actual graph, selection mechanism, source question and population weights.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        self.identity
    }
    /// Identifying functional; exogenous-root intervention equals baseline conditioning.
    #[must_use]
    pub const fn functional(&self) -> &'static str {
        "sum_l p_target(l) E_source[Y | do(exposure,saturation,mechanism,L=l)]"
    }
}
/// Actual population-transported hybrid evaluation, with the original scientific source.
#[derive(Clone, Debug)]
pub struct PopulationReferenceDecomposition {
    /// Original full-procedure resampling failures, retaining every requested slot.
    pub procedure_failures: Vec<Option<String>>,
    /// All sixteen standardized worlds and complete joint uncertainty realizations.
    pub decomposition: PanelDecomposition,
    /// Full covariance propagated from the actual native source, when available.
    pub covariance: Option<super::super::engine::decomposition::ReferenceDecompositionCovariance>,
    /// Original complete response source and its uncertainty/identification semantics.
    pub source: ScientificDecisionSource,
    /// Derived transport proof, rather than a caller-asserted support flag.
    pub transport: PopulationReferenceTransportProof,
    /// Explicit target distributions and component bindings.
    pub bindings: PopulationReferenceBindings,
    /// Scientific scope checked on the source before numerical access.
    pub scope: ScientificConsumptionScope,
    _lease: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}
fn baseline_supported(
    action: &WorldAction,
    variable: VariableId,
    value: f64,
    horizon: Option<u32>,
) -> bool {
    let is_set = |intervention: &Intervention| matches!(intervention,Intervention::Set{variable:v,value:x} if *v==variable && x.as_f64()==Some(value));
    if let Some(interventions) = action.at_decision_interventions() {
        return interventions
            .iter()
            .filter(|action| action.target_variables().contains(&variable))
            .all(is_set)
            && interventions.iter().any(is_set);
    }
    let Some((intervention, policy)) = action.schedule_parts() else {
        return false;
    };
    let mut intervals = Vec::new();
    let mut add = |intervention: &Intervention, policy: &TemporalPolicy| {
        if !intervention.target_variables().contains(&variable) {
            return true;
        }
        if !is_set(intervention) {
            return false;
        }
        match policy {
            TemporalPolicy::Pulse { at } => intervals.push((*at, *at)),
            TemporalPolicy::Sustained { from, until } => intervals.push((*from, *until)),
            TemporalPolicy::Dynamic { active_at, .. } => {
                intervals.extend(active_at.iter().map(|at| (*at, *at)))
            }
            _ => return false,
        }
        true
    };
    let valid = if let Intervention::Sequence(sequence) = intervention {
        sequence.steps.iter().all(|step| add(&step.intervention, &step.temporal))
    } else {
        add(intervention, policy)
    };
    if !valid {
        return false;
    }
    intervals.sort_unstable();
    let mut covered = 0_i64;
    for (from, until) in intervals {
        if i64::from(from) > covered {
            return false;
        }
        covered = covered.max(i64::from(until) + 1);
    }
    covered >= i64::from(horizon.unwrap_or(0)) && covered > 0
}
fn validate_transport(
    source: &ScientificDecisionSource,
    bindings: &PopulationReferenceBindings,
    diagram: &SelectionDiagram,
) -> Result<PopulationReferenceTransportProof, JointPanelError> {
    let n = bindings.strata.len();
    if n == 0
        || n > 256
        || bindings.stratum_worlds.len() != n
        || bindings.components.hybrids != bindings.stratum_worlds[0]
        || bindings.strata.iter().any(|value| !value.is_finite())
        || bindings.strata.iter().enumerate().any(|(j, value)| bindings.strata[..j].contains(value))
        || bindings
            .components
            .exposure
            .iter()
            .chain(&bindings.components.saturation)
            .chain(&bindings.components.mechanism)
            .any(|variable| *variable == bindings.baseline)
    {
        return Err(invalid(
            "population reference requires distinct supported strata and a separately bound baseline variable",
        ));
    }
    for weights in [&bindings.before_weights, &bindings.after_weights] {
        if weights.len() != n
            || weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(invalid(
                "population reference weights must be finite probabilities on the declared strata",
            ));
        }
    }
    let graph = diagram.causal_graph();
    let index = graph
        .nodes()
        .iter()
        .position(|node| *node == NodeRef::Static(bindings.baseline))
        .ok_or_else(|| invalid("baseline covariate absent from transport graph"))?;
    let dense = DenseNodeId::from_index(index);
    if !graph.parents(dense).is_empty()
        || !graph.bidirected_neighbors(dense).is_empty()
        || diagram.selection_targets() != [bindings.baseline]
    {
        return Err(JointPanelError::Unsupported("baseline population transport requires an exogenous root with no latent common cause and selection confined to its distribution; outcome/mechanism selection needs a different transport theorem".into()));
    }
    let worlds = super::super::contracts::target_worlds(source.target())?;
    for (stratum, ids) in bindings.strata.iter().zip(&bindings.stratum_worlds) {
        for id in ids {
            let world = worlds
                .iter()
                .find(|world| world.id() == *id)
                .ok_or_else(|| invalid("population stratum names an absent source world"))?;
            if !baseline_supported(
                world.action(),
                bindings.baseline,
                *stratum,
                world.time().horizon_steps(),
            ) {
                return Err(invalid(
                    "every stratum response requires the exact constant baseline intervention throughout its supported horizon",
                ));
            }
            if !graph.nodes().contains(&NodeRef::Static(world.binding().outcome().outcome())) {
                return Err(invalid("transport graph omits the source outcome"));
            }
        }
    }
    for variable in bindings
        .components
        .exposure
        .iter()
        .chain(&bindings.components.saturation)
        .chain(&bindings.components.mechanism)
    {
        if !graph.nodes().contains(&NodeRef::Static(*variable)) {
            return Err(invalid("transport graph omits an intervention component"));
        }
    }
    let graph_wire = antecedent_io::admg_to_wire(graph).map_err(JointPanelError::compile)?;
    let bytes = antecedent_io::convert::to_cbor(&graph_wire).map_err(JointPanelError::compile)?;
    let mut writer =
        antecedent_core::CanonicalWriter::new("antecedent.reference.population.root_transport.v1");
    writer
        .bytes(source.question_identity().as_bytes())
        .bytes(&bytes)
        .word(u64::from(bindings.baseline.raw()));
    for (j, stratum) in bindings.strata.iter().enumerate() {
        writer
            .word(stratum.to_bits())
            .word(bindings.before_weights[j].to_bits())
            .word(bindings.after_weights[j].to_bits());
        for id in bindings.stratum_worlds[j] {
            writer.word(u64::from(id.raw()));
        }
    }
    Ok(PopulationReferenceTransportProof {
        diagram: diagram.clone(),
        identity: SemanticDigest::from_bytes(writer.finish()),
    })
}
/// Standardize identified root-stratum responses to separately supplied population laws.
///
/// The selection diagram explicitly asserts invariance of all non-baseline mechanisms.
/// The compiler checks the root/no-latent-confounding condition, selection placement,
/// support of every stratum intervention and every component hybrid. It then executes
/// the transported mixture under the original aligned law. This is an earned special
/// case of population transport; diagrams changing the response mechanism are refused.
/// # Errors
/// Missing transport/identification premises, unsupported strata, invalid components,
/// resources, numerical failure or cancellation.
pub fn decompose_population_reference(
    source: &ScientificDecisionSource,
    bindings: &PopulationReferenceBindings,
    diagram: &SelectionDiagram,
    scope: ScientificConsumptionScope,
    ctx: &ExecutionContext,
) -> Result<PopulationReferenceDecomposition, JointPanelError> {
    source.authorize(scope).map_err(JointPanelError::compile)?;
    let graph = diagram.causal_graph();
    let quote = graph
        .node_count()
        .checked_add(
            graph
                .node_ids()
                .map(|node| graph.parents(node).len() + graph.bidirected_neighbors(node).len())
                .sum(),
        )
        .and_then(|n| n.checked_mul(256))
        .and_then(|n| n.checked_add(bindings.strata.len().checked_mul(256)?))
        .and_then(|n| n.checked_add(65536))
        .ok_or_else(|| invalid("population transport retention overflow"))?;
    let lease = std::sync::Arc::new(
        ctx.acquire_admission(0, quote as u64)
            .map_err(|error| super::super::engine::PanelError::Resource(error.to_string()))?,
    );
    let proof = validate_transport(source, bindings, diagram)?;
    let mut strata = Vec::with_capacity(bindings.strata.len());
    for worlds in &bindings.stratum_worlds {
        let mut components = bindings.components.clone();
        components.hybrids = *worlds;
        components.mechanism.push(bindings.baseline);
        strata.push(decompose_scientific_reference(source, &components, scope, ctx)?);
    }
    let slots = strata[0].decomposition.draws.len();
    let quote = slots
        .checked_mul(std::mem::size_of::<Option<[f64; 16]>>())
        .ok_or_else(|| invalid("population mixture scratch overflow"))?;
    let _scratch = ctx
        .acquire_admission(0, quote as u64)
        .map_err(|error| super::super::engine::PanelError::Resource(error.to_string()))?;
    let mix = |mask: usize, slot: Option<usize>| -> Option<f64> {
        let weights =
            if mask & 4 == 0 { &bindings.before_weights } else { &bindings.after_weights };
        strata.iter().zip(weights).try_fold(0.0, |sum, (stratum, weight)| {
            let values = if let Some(slot) = slot {
                stratum.decomposition.draws.get(slot)?.as_ref()?
            } else {
                &stratum.decomposition.point
            };
            Some(sum + weight * values.hybrids[mask])
        })
    };
    let mut point = [0.0; 16];
    for (mask, value) in point.iter_mut().enumerate() {
        *value = mix(mask, None).ok_or_else(|| invalid("incomplete population stratum point"))?;
    }
    let draws = (0..slots)
        .map(|slot| {
            let mut row = [0.0; 16];
            for (mask, value) in row.iter_mut().enumerate() {
                *value = mix(mask, Some(slot))?;
            }
            Some(row)
        })
        .collect::<Vec<_>>();
    let weights = strata[0].decomposition.weights.as_deref();
    let decomposition = super::super::engine::decomposition::decompose_weighted_hybrid_law(
        point,
        &draws,
        weights,
        bindings.components.order,
        ctx,
    )?;
    Ok(PopulationReferenceDecomposition {
        procedure_failures: Vec::new(),
        decomposition,
        covariance: None,
        source: source.clone(),
        transport: proof,
        bindings: bindings.clone(),
        scope,
        _lease: lease,
    })
}
impl crate::analysis::joint_panel::designs::JointDesignResult {
    /// Transport and decompose native experimental strata, including their joint covariance.
    /// # Errors
    /// The same source, graph, support, covariance and resource refusals as the generic route.
    pub fn decompose_population_reference(
        &self,
        bindings: &PopulationReferenceBindings,
        diagram: &SelectionDiagram,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
    ) -> Result<PopulationReferenceDecomposition, JointPanelError> {
        let source = match self.law() {
            antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(_) => {
                self.posterior_source()?
            }
            antecedent_estimate::joint_design::compiled::JointDesignLaw::Frequentist(_)
            | antecedent_estimate::joint_design::compiled::JointDesignLaw::FrequentistBootstrap {
                ..
            } => self.mean_source()?,
        };
        let mut result = decompose_population_reference(&source, bindings, diagram, scope, ctx)?;
        let worlds = &self.prepared().proof().worlds;
        let mut coefficients = vec![vec![0.0; worlds.len()]; 16];
        for (stratum, ids) in bindings.stratum_worlds.iter().enumerate() {
            for (mask, id) in ids.iter().enumerate() {
                let index = worlds
                    .iter()
                    .position(|world| world.world == id.raw())
                    .ok_or_else(|| invalid("native covariance omits a population stratum"))?;
                coefficients[mask][index] += if mask & 4 == 0 {
                    bindings.before_weights[stratum]
                } else {
                    bindings.after_weights[stratum]
                };
            }
        }
        if let Some(realizations) = self.law().frequentist_realizations() {
            let bytes = realizations
                .len()
                .checked_mul(16 * 8 + 16)
                .ok_or_else(|| invalid("bootstrap reference projection quote overflow"))?;
            let _projection = ctx
                .acquire_admission(0, bytes as u64)
                .map_err(|error| super::super::engine::PanelError::Resource(error.to_string()))?;
            let draws = realizations
                .iter()
                .map(|row| {
                    let row = row.as_ref()?;
                    if row.len() != worlds.len() {
                        return None;
                    }
                    Some(std::array::from_fn(|mask| {
                        coefficients[mask]
                            .iter()
                            .zip(row)
                            .map(|(weight, value)| weight * value)
                            .sum()
                    }))
                })
                .collect::<Vec<_>>();
            result.decomposition = super::super::engine::decomposition::decompose_hybrid_law(
                result.decomposition.point.hybrids,
                &draws,
                bindings.components.order,
                ctx,
            )?;
            result.procedure_failures =
                self.law().frequentist_failures().unwrap_or_default().to_vec();
        }
        let native = self.law().estimate();
        if native.covariance.is_empty() && self.law().frequentist_realizations().is_some() {
            return Ok(result);
        }
        if native.covariance.len() != worlds.len() * worlds.len() {
            return Err(invalid("native joint-design covariance axis mismatch"));
        }
        let covariance = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                (0..worlds.len())
                    .map(|j| {
                        (0..worlds.len())
                            .map(|k| {
                                coefficients[a][j]
                                    * native.covariance[j * worlds.len() + k]
                                    * coefficients[b][k]
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
        });
        result.covariance = Some(super::super::engine::decomposition::propagate_hybrid_covariance(
            &covariance,
            bindings.components.order,
        )?);
        Ok(result)
    }
}

/// Actual next supported intervention contrast after a finer population explanation fails.
#[derive(Clone, Debug)]
pub enum PopulationReferenceAttempt {
    /// The supplied selection diagram earned population-standardized reference worlds.
    Supported(PopulationReferenceDecomposition),
    /// Population attribution lacked transport premises; the explicit coarser world
    /// request was evaluated under its own retained scientific scope.
    Coarser {
        /// Original concrete scientific refusal; numerical errors never enter here.
        refusal: String,
        /// Actual compiled fallback execution, including its total and uncertainty.
        executed: ScientificReferenceDecomposition,
    },
}
impl crate::analysis::joint_panel::designs::JointDesignResult {
    /// Execute population reference worlds or a caller-explicit coarser intervention basis.
    /// The fallback's actual target and scope remain visible; it is not relabeled as
    /// a proof of the failed population explanation.
    /// # Errors
    /// Invalid inputs, numerical/resource/cancellation failure or unsupported fallback.
    pub fn attempt_population_reference(
        &self,
        bindings: &PopulationReferenceBindings,
        diagram: &SelectionDiagram,
        scope: ScientificConsumptionScope,
        fallback: &ReferenceComponentBindings,
        ctx: &ExecutionContext,
    ) -> Result<PopulationReferenceAttempt, JointPanelError> {
        match self.decompose_population_reference(bindings, diagram, scope, ctx) {
            Ok(result) => Ok(PopulationReferenceAttempt::Supported(result)),
            Err(JointPanelError::Unsupported(refusal)) => {
                let executed = self.decompose_reference_components(fallback, scope, ctx)?;
                Ok(PopulationReferenceAttempt::Coarser { refusal, executed })
            }
            Err(error) => Err(error),
        }
    }
}
