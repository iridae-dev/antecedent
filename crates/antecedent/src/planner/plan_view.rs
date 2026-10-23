//! One plan view for every compiled route.
//!
//! A [`PlanView`] reads the lowered program `Study::compile` attached to a plan and
//! states each compiler layer: the scientific problem, the identification product,
//! the statistical program, the evidence and decision program, the physical plan,
//! and the uncertainty and claim the execution must publish, together with the
//! program identity and the support cell. On a prepared handle it also reads the
//! identification preparation installed and names the sealed operation preparation
//! retained.
//!
//! Building a view reads retained values only. It does not identify, fit, scan
//! rows, enumerate graphs, simulate or lease from an admission pool. The work that
//! produced those values happened at compile (identification retained with the
//! program, a quoted design width) or at prepare, and is reported there.
//!
//! Every route a view describes is a [`PlanRoute`]. The route table that registers
//! each `RouteInput` cell to its family lowering also generates [`PlanRoute`] and
//! the cell's [`RouteFamily`], and the view matches every family exhaustively, so a
//! route cell or a family added without a view does not compile.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DecisionRequest, PhysicalExecutionPlanRecord};
use antecedent_core::{
    AssumptionSet, AssumptionStatus, ExecutionContext, KernelPolicy, POSTERIOR_MASS_TOLERANCE,
    PhysicalOperator, SemanticDigest, VariableId,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::{
    CompiledIdentificationProduct, IdentificationClaimClass, IdentificationOutcome,
    IdentificationResult, IdentificationStatus, TransportIdentification,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use serde_json::{Value as Json, json};

pub use crate::analysis::lowering::{GraphPosteriorCell, PlanRoute};

use crate::analysis::IdentificationCache;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, IdentificationLayer,
    LoweredProgram, LoweringDepth, NonGraphicalIdentification, ProblemTarget, ProgramCell,
    ScientificProblem, UncertaintyReading, WidthSource,
};
use crate::planner::statistical::{PosteriorPlan, Realization, ResamplingPlan, StatisticalProgram};
use crate::support::CellStatus;

/// The family a route is lowered by.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RouteFamily {
    /// Static average, conditional, distribution and path-specific effects.
    StaticEffect,
    /// Static response curves, intervention responses and joint responses.
    StaticResponse,
    /// Static mediation and the fitted-mechanism routes: queries, attribution
    /// and simulation.
    Structural,
    /// Temporal effects, responses, mediation and cross-world queries.
    Temporal,
    /// Design-based routes: transport, interference, randomized trials, policy
    /// value, panel, dose, discontinuity, survival and sequential designs.
    DesignBased,
    /// Structure discovery.
    Discovery,
    /// A mixture over the atoms of a supplied graph posterior.
    GraphPosterior,
    /// A cyclic family, prepared through its own entry.
    Cyclic,
    /// A standalone experiment plan.
    Planning,
    /// A chain of transformations carrying uncertainty from declared sources to a
    /// reading.
    PropagationChain,
}

impl RouteFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaticEffect => "static_effect",
            Self::StaticResponse => "static_response",
            Self::Structural => "structural",
            Self::Temporal => "temporal",
            Self::DesignBased => "design_based",
            Self::Discovery => "discovery",
            Self::GraphPosterior => "graph_posterior",
            Self::Cyclic => "cyclic",
            Self::Planning => "planning",
            Self::PropagationChain => "propagation_chain",
        }
    }
}

/// Where a view was taken.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlanStage {
    /// From a compiled plan, before preparation.
    Compiled,
    /// From a prepared handle: the identification preparation installed and the
    /// sealed operation it retained are read as well.
    Prepared,
}

impl PlanStage {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compiled => "compiled",
            Self::Prepared => "prepared",
        }
    }
}

/// One identified estimand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EstimandView {
    method: Arc<str>,
    adjustment_set: Arc<[VariableId]>,
    instruments: Arc<[VariableId]>,
    mediators: Arc<[VariableId]>,
}

impl EstimandView {
    fn of(estimand: &IdentifiedEstimand) -> Self {
        Self {
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            instruments: Arc::clone(&estimand.instruments),
            mediators: Arc::clone(&estimand.mediators),
        }
    }

    /// Estimand method tag.
    #[must_use]
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Adjustment set; empty when the estimand is not an adjustment.
    #[must_use]
    pub fn adjustment_set(&self) -> &[VariableId] {
        &self.adjustment_set
    }

    /// Instruments; empty unless an instrumental-variable estimand.
    #[must_use]
    pub fn instruments(&self) -> &[VariableId] {
        &self.instruments
    }

    /// Mediators; empty unless a front-door or two-stage estimand.
    #[must_use]
    pub fn mediators(&self) -> &[VariableId] {
        &self.mediators
    }

    fn to_json(&self) -> Json {
        json!({
            "method": &*self.method,
            "adjustment_set": ids(&self.adjustment_set),
            "instruments": ids(&self.instruments),
            "mediators": ids(&self.mediators),
        })
    }
}

/// One assumption an identification relies on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssumptionView {
    label: String,
    status: AssumptionStatus,
}

impl AssumptionView {
    /// Stable label of the assumption.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Whether the assumption is declared, supported, contradicted or untestable.
    #[must_use]
    pub const fn status(&self) -> AssumptionStatus {
        self.status
    }

    fn to_json(&self) -> Json {
        json!({"label": self.label, "status": self.status.as_str()})
    }
}

fn assumption_views(set: &AssumptionSet) -> Vec<AssumptionView> {
    let mut views: Vec<AssumptionView> = Vec::with_capacity(set.entries.len());
    for record in &set.entries {
        let view = AssumptionView {
            label: crate::analysis::assumption_label(&record.assumption),
            status: record.status,
        };
        if !views.contains(&view) {
            views.push(view);
        }
    }
    views
}

/// An obligation identification left unresolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObligationView {
    code: Arc<str>,
    detail: Arc<str>,
}

impl ObligationView {
    /// Stable obligation code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// What the obligation asks for.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// The checked identification product bound to the scientific target.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductView {
    identity: SemanticDigest,
    outcome: IdentificationOutcome,
    claim_class: IdentificationClaimClass,
    estimands: Vec<EstimandView>,
    assumptions: Vec<AssumptionView>,
    required_factors: Vec<(Arc<str>, Arc<[VariableId]>)>,
    unresolved: Vec<ObligationView>,
}

impl ProductView {
    fn of(product: &CompiledIdentificationProduct) -> Self {
        Self {
            identity: product.identity(),
            outcome: product.outcome(),
            claim_class: product.claim_class(),
            estimands: product.identified().iter().map(EstimandView::of).collect(),
            assumptions: assumption_views(product.assumptions()),
            required_factors: product
                .required_factors()
                .iter()
                .map(|factor| (Arc::from(factor.name()), Arc::from(factor.variables())))
                .collect(),
            unresolved: product
                .unresolved_obligations()
                .iter()
                .map(|obligation| ObligationView {
                    code: Arc::from(obligation.code()),
                    detail: Arc::from(obligation.detail()),
                })
                .collect(),
        }
    }

    /// Identity of the product.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Identification outcome.
    #[must_use]
    pub const fn outcome(&self) -> IdentificationOutcome {
        self.outcome
    }

    /// What kind of statement the outcome may make.
    #[must_use]
    pub const fn claim_class(&self) -> IdentificationClaimClass {
        self.claim_class
    }

    /// Identified estimands.
    #[must_use]
    pub fn estimands(&self) -> &[EstimandView] {
        &self.estimands
    }

    /// Assumptions the product relies on.
    #[must_use]
    pub fn assumptions(&self) -> &[AssumptionView] {
        &self.assumptions
    }

    /// Factors the identified functional reads, by name and variables.
    #[must_use]
    pub fn required_factors(&self) -> &[(Arc<str>, Arc<[VariableId]>)] {
        &self.required_factors
    }

    /// Obligations left unresolved.
    #[must_use]
    pub fn unresolved_obligations(&self) -> &[ObligationView] {
        &self.unresolved
    }

    fn to_json(&self) -> Json {
        json!({
            "identity": self.identity.to_hex(),
            "outcome": outcome_name(self.outcome),
            "claim_class": claim_class_name(self.claim_class),
            "estimands": self.estimands.iter().map(EstimandView::to_json).collect::<Vec<_>>(),
            "assumptions": self.assumptions.iter().map(AssumptionView::to_json).collect::<Vec<_>>(),
            "required_factors": self
                .required_factors
                .iter()
                .map(|(name, variables)| json!({"name": &**name, "variables": ids(variables)}))
                .collect::<Vec<_>>(),
            "unresolved_obligations": self
                .unresolved
                .iter()
                .map(|obligation| json!({"code": &*obligation.code, "detail": &*obligation.detail}))
                .collect::<Vec<_>>(),
        })
    }
}

/// One identification product retained with the program or installed by preparation.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedProductView {
    kind: &'static str,
    status: Option<IdentificationStatus>,
    estimands: Vec<EstimandView>,
    assumptions: Vec<AssumptionView>,
    horizons: Vec<u32>,
    completions: usize,
    truncated_completions: usize,
    atoms: usize,
    identified_mass: Option<f64>,
    unidentified_mass: Option<f64>,
    transport: Option<&'static str>,
}

impl RetainedProductView {
    fn empty(kind: &'static str) -> Self {
        Self {
            kind,
            status: None,
            estimands: Vec::new(),
            assumptions: Vec::new(),
            horizons: Vec::new(),
            completions: 0,
            truncated_completions: 0,
            atoms: 0,
            identified_mass: None,
            unidentified_mass: None,
            transport: None,
        }
    }

    fn of_result(kind: &'static str, result: &IdentificationResult) -> Self {
        Self {
            status: Some(result.status),
            assumptions: assumption_views(&result.required_assumptions),
            ..Self::empty(kind)
        }
    }

    fn of_samples(kind: &'static str, graphs: &WeightedGraphSamples, atoms: usize) -> Self {
        let identified = graphs.identified_mass();
        let unidentified = graphs.unidentified_mass();
        Self {
            status: Some(mixture_status(identified, unidentified)),
            atoms,
            identified_mass: Some(identified),
            unidentified_mass: Some(unidentified),
            ..Self::empty(kind)
        }
    }

    fn of_cache(cache: &IdentificationCache) -> Self {
        match cache {
            IdentificationCache::Static(fixed) => Self {
                estimands: vec![EstimandView::of(&fixed.estimand)],
                ..Self::of_result(cache.kind(), &fixed.identification)
            },
            IdentificationCache::Pag(pag) => Self {
                status: Some(pag.envelope.status),
                estimands: pag.envelope.invariant.iter().map(EstimandView::of).collect(),
                completions: pag.envelope.cases.len(),
                truncated_completions: pag.envelope.truncated_completions,
                identified_mass: Some(pag.envelope.identified_weight.0),
                unidentified_mass: Some(pag.envelope.unidentified_weight.0),
                ..Self::of_result(cache.kind(), &pag.identification)
            },
            IdentificationCache::Cpdag(cpdag) => Self {
                status: Some(cpdag.envelope.status),
                estimands: cpdag.envelope.invariant.iter().map(EstimandView::of).collect(),
                completions: cpdag.envelope.cases.len(),
                truncated_completions: cpdag.envelope.truncated_completions,
                identified_mass: Some(cpdag.envelope.identified_weight.0),
                unidentified_mass: Some(cpdag.envelope.unidentified_weight.0),
                ..Self::of_result(cache.kind(), &cpdag.identification)
            },
            IdentificationCache::Temporal(temporal) => {
                let mut view = Self::empty(cache.kind());
                for horizon in temporal.by_horizon.iter() {
                    view.horizons.push(horizon.horizon);
                    view.status = Some(weakest(view.status, horizon.identification.status));
                    view.estimands.push(EstimandView::of(&horizon.estimand));
                    for assumption in assumption_views(&horizon.identification.required_assumptions)
                    {
                        if !view.assumptions.contains(&assumption) {
                            view.assumptions.push(assumption);
                        }
                    }
                }
                view
            }
            IdentificationCache::TemporalClass(class) => {
                let envelope = &class.envelope.envelope;
                Self {
                    status: Some(envelope.status),
                    estimands: envelope.invariant.iter().map(EstimandView::of).collect(),
                    horizons: class.by_horizon.iter().map(|(horizon, _)| *horizon).collect(),
                    completions: envelope.cases.len(),
                    truncated_completions: envelope.truncated_completions,
                    identified_mass: Some(envelope.identified_weight.0),
                    unidentified_mass: Some(envelope.unidentified_weight.0),
                    ..Self::empty(cache.kind())
                }
            }
            IdentificationCache::Transport(transport) => {
                let (status, outcome) = match &**transport {
                    TransportIdentification::Transportable { .. } => {
                        (IdentificationStatus::NonparametricallyIdentified, "transportable")
                    }
                    TransportIdentification::NotCertified(_) => {
                        (IdentificationStatus::NotIdentified, "not_certified")
                    }
                    TransportIdentification::MissingEvidence(_) => {
                        (IdentificationStatus::NotIdentified, "missing_evidence")
                    }
                };
                Self { status: Some(status), transport: Some(outcome), ..Self::empty(cache.kind()) }
            }
            IdentificationCache::GraphPosterior(posterior) => Self::of_samples(
                cache.kind(),
                &posterior.graphs,
                posterior.atoms.len() + posterior.class_atoms.len(),
            ),
            IdentificationCache::DbnPosterior(posterior) => {
                Self::of_samples(cache.kind(), &posterior.graphs, posterior.atoms.len())
            }
            IdentificationCache::TemporalClassPosterior(posterior) => {
                Self::of_samples(cache.kind(), &posterior.graphs, posterior.class_atoms.len())
            }
        }
    }

    /// Slot the product fills (`static`, `pag`, `cpdag`, `temporal`,
    /// `temporal_class`, `transport`, `graph_posterior`, `dbn_posterior`,
    /// `temporal_class_posterior`).
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        self.kind
    }

    /// Identification status, when the product states one.
    #[must_use]
    pub const fn status(&self) -> Option<IdentificationStatus> {
        self.status
    }

    /// Selected estimands: the static estimand, one per temporal horizon, or the
    /// estimand every identified completion of a class shares.
    #[must_use]
    pub fn estimands(&self) -> &[EstimandView] {
        &self.estimands
    }

    /// Assumptions the product relies on.
    #[must_use]
    pub fn assumptions(&self) -> &[AssumptionView] {
        &self.assumptions
    }

    /// Horizons a temporal product was identified for.
    #[must_use]
    pub fn horizons(&self) -> &[u32] {
        &self.horizons
    }

    /// Completions an equivalence-class envelope enumerated at compile.
    #[must_use]
    pub const fn completions(&self) -> usize {
        self.completions
    }

    /// Completions whose search was capped.
    #[must_use]
    pub const fn truncated_completions(&self) -> usize {
        self.truncated_completions
    }

    /// Identified posterior atoms a graph-posterior product retained.
    #[must_use]
    pub const fn atoms(&self) -> usize {
        self.atoms
    }

    /// Identified mass of an envelope or a posterior.
    #[must_use]
    pub const fn identified_mass(&self) -> Option<f64> {
        self.identified_mass
    }

    /// Unidentified mass of an envelope or a posterior; it is never dropped.
    #[must_use]
    pub const fn unidentified_mass(&self) -> Option<f64> {
        self.unidentified_mass
    }

    /// Transport certificate outcome (`transportable`, `not_certified`,
    /// `missing_evidence`), for a transport product.
    #[must_use]
    pub const fn transport(&self) -> Option<&'static str> {
        self.transport
    }

    fn to_json(&self) -> Json {
        json!({
            "kind": self.kind,
            "status": self.status.map(IdentificationStatus::as_str),
            "estimands": self.estimands.iter().map(EstimandView::to_json).collect::<Vec<_>>(),
            "assumptions": self.assumptions.iter().map(AssumptionView::to_json).collect::<Vec<_>>(),
            "horizons": self.horizons,
            "completions": self.completions,
            "truncated_completions": self.truncated_completions,
            "atoms": self.atoms,
            "identified_mass": self.identified_mass,
            "unidentified_mass": self.unidentified_mass,
            "transport": self.transport,
        })
    }
}

/// How layer two carries identification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationKind {
    /// Identified at compile and retained with the program.
    Retained,
    /// Refused at compile; preparation identifies and reports the same refusal.
    Deferred,
    /// The route identifies on each execution.
    PerExecution,
    /// A feedback model's identification, which is not a graphical product.
    NonGraphical,
}

impl IdentificationKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Retained => "retained",
            Self::Deferred => "deferred",
            Self::PerExecution => "per_execution",
            Self::NonGraphical => "non_graphical",
        }
    }
}

/// Layer two: the identification product.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentificationView {
    kind: IdentificationKind,
    reason: Option<&'static str>,
    non_graphical: Option<&'static str>,
    unresolved_directions: usize,
    product: Option<ProductView>,
    retained: Vec<RetainedProductView>,
    installed_at_prepare: bool,
    retention_key: Option<[u8; 32]>,
}

impl IdentificationView {
    fn of(layer: &IdentificationLayer, installed: &[IdentificationCache]) -> Self {
        let mut view = Self {
            kind: IdentificationKind::Retained,
            reason: None,
            non_graphical: None,
            unresolved_directions: 0,
            product: None,
            retained: Vec::new(),
            installed_at_prepare: false,
            retention_key: None,
        };
        match layer {
            IdentificationLayer::Retained(retained) => {
                view.product = retained.product.as_ref().map(ProductView::of);
                view.retained =
                    retained.caches().iter().map(RetainedProductView::of_cache).collect();
                view.retention_key = retained.key;
            }
            IdentificationLayer::Deferred { reason } => {
                view.kind = IdentificationKind::Deferred;
                view.reason = Some(*reason);
            }
            IdentificationLayer::PerExecution { reason } => {
                view.kind = IdentificationKind::PerExecution;
                view.reason = Some(*reason);
            }
            IdentificationLayer::NonGraphical(identification) => {
                view.kind = IdentificationKind::NonGraphical;
                view.non_graphical = Some(identification.as_str());
                if let NonGraphicalIdentification::ExperimentalRecovery { unresolved, .. } =
                    &**identification
                {
                    view.unresolved_directions = unresolved.len();
                }
            }
        }
        // What preparation installed replaces what compile retained: it is the
        // product the prepared handle executes, and it is the only product a
        // deferred program has.
        if !installed.is_empty() {
            view.retained = installed.iter().map(RetainedProductView::of_cache).collect();
            view.installed_at_prepare = true;
        }
        view
    }

    /// How the layer carries identification.
    #[must_use]
    pub const fn kind(&self) -> IdentificationKind {
        self.kind
    }

    /// Why compile retained no identification, for a deferred or per-execution layer.
    #[must_use]
    pub const fn reason(&self) -> Option<&'static str> {
        self.reason
    }

    /// The non-graphical identification a feedback program carries
    /// (`specified_model_conditional`, `experimental_recovery`, `cyclic_ioscm`).
    #[must_use]
    pub const fn non_graphical(&self) -> Option<&'static str> {
        self.non_graphical
    }

    /// Directions an experimental recovery leaves unresolved.
    #[must_use]
    pub const fn unresolved_directions(&self) -> usize {
        self.unresolved_directions
    }

    /// The checked product bound to the scientific target, when the route builds one.
    #[must_use]
    pub const fn product(&self) -> Option<&ProductView> {
        self.product.as_ref()
    }

    /// Products compile retained, or preparation installed, in install order.
    #[must_use]
    pub fn retained(&self) -> &[RetainedProductView] {
        &self.retained
    }

    /// Whether [`Self::retained`] lists what preparation installed.
    #[must_use]
    pub const fn installed_at_prepare(&self) -> bool {
        self.installed_at_prepare
    }

    /// Digest of the study inputs a retained product was identified from.
    #[must_use]
    pub const fn retention_key(&self) -> Option<[u8; 32]> {
        self.retention_key
    }

    /// The weakest status the retained products state, when any states one.
    #[must_use]
    pub fn status(&self) -> Option<IdentificationStatus> {
        self.retained
            .iter()
            .filter_map(RetainedProductView::status)
            .fold(None, |weakest_so_far, status| Some(weakest(weakest_so_far, status)))
    }

    /// Assumptions the product and the retained products rely on, without repeats.
    #[must_use]
    pub fn assumptions(&self) -> Vec<AssumptionView> {
        let mut all: Vec<AssumptionView> = Vec::new();
        let from_product = self.product.iter().flat_map(|product| product.assumptions.iter());
        let from_retained = self.retained.iter().flat_map(|retained| retained.assumptions.iter());
        for assumption in from_product.chain(from_retained) {
            if !all.contains(assumption) {
                all.push(assumption.clone());
            }
        }
        all
    }

    /// Obligations the checked product left unresolved.
    #[must_use]
    pub fn unresolved_obligations(&self) -> &[ObligationView] {
        self.product.as_ref().map_or(&[][..], |product| product.unresolved.as_slice())
    }

    fn to_json(&self) -> Json {
        json!({
            "kind": self.kind.as_str(),
            "status": self.status().map(IdentificationStatus::as_str),
            "reason": self.reason,
            "non_graphical": self.non_graphical,
            "unresolved_directions": self.unresolved_directions,
            "product": self.product.as_ref().map(ProductView::to_json),
            "retained": self.retained.iter().map(RetainedProductView::to_json).collect::<Vec<_>>(),
            "installed_at_prepare": self.installed_at_prepare,
            "retention_key": self.retention_key.map(|key| SemanticDigest::from_bytes(key).to_hex()),
        })
    }
}

/// Layer five: the physical plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalView {
    operators: Arc<[PhysicalOperator]>,
    design: DesignShape,
    record: Option<PhysicalExecutionPlanRecord>,
    kernel_policy: Option<KernelPolicy>,
}

impl PhysicalView {
    /// Physical operator contracts of the program.
    #[must_use]
    pub fn operators(&self) -> &[PhysicalOperator] {
        &self.operators
    }

    /// The design the program fits.
    #[must_use]
    pub const fn design(&self) -> DesignShape {
        self.design
    }

    /// Encoded columns of a quoted design, including the intercept.
    #[must_use]
    pub const fn quoted_width(&self) -> Option<u64> {
        self.design.columns()
    }

    /// Why the route quotes no single design, when it is not design based.
    #[must_use]
    pub const fn not_design_based(&self) -> Option<&'static str> {
        match self.design.source() {
            WidthSource::Quoted => None,
            WidthSource::NotDesignBased(reason) => Some(reason),
        }
    }

    /// The physical record the plan was sized to, when the view was taken from a plan.
    #[must_use]
    pub const fn record(&self) -> Option<&PhysicalExecutionPlanRecord> {
        self.record.as_ref()
    }

    /// Kernel policy of the compile context, when the view was taken under one.
    #[must_use]
    pub const fn kernel_policy(&self) -> Option<KernelPolicy> {
        self.kernel_policy
    }

    fn operator_bytes(&self, read: impl Fn(&PhysicalOperator) -> Option<u64>) -> Option<u64> {
        if self.operators.is_empty() {
            return None;
        }
        self.operators.iter().try_fold(0_u64, |total, operator| total.checked_add(read(operator)?))
    }

    /// Scratch bytes the operators declare. `None` when an operator declares none
    /// (an absent estimate is not zero) or the program has no operators.
    #[must_use]
    pub fn scratch_bytes(&self) -> Option<u64> {
        self.operator_bytes(|operator| operator.estimates().scratch())
    }

    /// Retained bytes the operators declare, read as [`Self::scratch_bytes`].
    #[must_use]
    pub fn retained_bytes(&self) -> Option<u64> {
        self.operator_bytes(|operator| operator.estimates().retained())
    }

    /// Output bytes the operators declare, read as [`Self::scratch_bytes`].
    #[must_use]
    pub fn output_bytes(&self) -> Option<u64> {
        self.operator_bytes(|operator| operator.estimates().output())
    }

    fn to_json(&self) -> Json {
        let operators: Vec<Json> = self
            .operators
            .iter()
            .map(|operator| {
                let estimates = operator.estimates();
                json!({
                    "effect": operator.effect().as_str(),
                    "sources": operator.sources().len(),
                    "designs": operator.designs().len(),
                    "scratch_bytes": estimates.scratch(),
                    "retained_bytes": estimates.retained(),
                    "output_bytes": estimates.output(),
                    "capability": operator.capability().to_string(),
                    "device_work": operator.device_work(),
                })
            })
            .collect();
        let record = self.record.as_ref().map(|record| {
            json!({
                "plan_id": &*record.plan_id,
                "materializations": record
                    .materializations
                    .iter()
                    .map(|(buffer, materialization)| {
                        json!({"buffer": &**buffer, "materialization": format!("{materialization:?}")})
                    })
                    .collect::<Vec<_>>(),
                "kernels": record
                    .kernels
                    .iter()
                    .map(|(kernel, selection)| {
                        json!({"kernel": &**kernel, "selection": format!("{selection:?}")})
                    })
                    .collect::<Vec<_>>(),
                "batch_size": record.batch_size,
                "workspace_bytes": record.workspace_bytes,
                "estimated_peak_memory_bytes": record.estimated_peak_memory_bytes,
                "estimated_copy_bytes": record.estimated_copy_bytes,
                "task_schedule": record
                    .task_schedule
                    .iter()
                    .map(|task| json!({"dimension": &*task.dimension, "units": task.units}))
                    .collect::<Vec<_>>(),
                "worker_threads": record.worker_threads,
                "deterministic_reductions": record.deterministic_reductions,
                "expected_python_crossings": record.expected_python_crossings,
            })
        });
        json!({
            "operators": operators,
            "design": design_json(self.design),
            "scratch_bytes": self.scratch_bytes(),
            "retained_bytes": self.retained_bytes(),
            "output_bytes": self.output_bytes(),
            "record": record,
            "kernel_policy": self.kernel_policy.map(|policy| json!({
                "allow_portable_optimized": policy.allow_portable_optimized,
                "allow_arch_simd": policy.allow_arch_simd,
                "force_scalar": policy.force_scalar,
                "arch_simd_effective": policy.arch_simd_effective(),
            })),
        })
    }
}

/// The uncertainty and claim an execution of the program must publish.
#[derive(Clone, Debug, PartialEq)]
pub struct UncertaintyDemand {
    reading: UncertaintyReading,
    claim_kind: Option<ClaimKind>,
    realization: &'static str,
    resampling: Option<ResamplingPlan>,
    posterior: Option<PosteriorPlan>,
    decision_request: Option<DecisionRequest>,
}

impl UncertaintyDemand {
    fn of(program: &LoweredProgram) -> Self {
        let statistical = program.statistical();
        Self {
            reading: program.claim().uncertainty(),
            claim_kind: program.claim().kind(),
            realization: realization_name(statistical.realization()),
            resampling: statistical.resampling(),
            posterior: statistical.posterior().cloned(),
            decision_request: program
                .evidence()
                .decision()
                .and_then(|decision| decision.carrier().evidence_request()),
        }
    }

    /// How the published uncertainty must be read.
    #[must_use]
    pub const fn reading(&self) -> UncertaintyReading {
        self.reading
    }

    /// Claim kind the route publishes, when its lowering states one.
    #[must_use]
    pub const fn claim_kind(&self) -> Option<ClaimKind> {
        self.claim_kind
    }

    /// Realization the uncertainty comes from (`frequentist`, `bayesian`,
    /// `generalized_bayes`, `envelope`, `exact_point`).
    #[must_use]
    pub const fn realization(&self) -> &'static str {
        self.realization
    }

    /// Resampling the program performs, when requested.
    #[must_use]
    pub const fn resampling(&self) -> Option<ResamplingPlan> {
        self.resampling
    }

    /// Posterior computation, when the realization is a posterior.
    #[must_use]
    pub const fn posterior(&self) -> Option<&PosteriorPlan> {
        self.posterior.as_ref()
    }

    /// The evidence-port request a decision puts to the statistical program.
    #[must_use]
    pub const fn decision_request(&self) -> Option<DecisionRequest> {
        self.decision_request
    }

    fn to_json(&self) -> Json {
        json!({
            "reading": self.reading.as_str(),
            "claim_kind": self.claim_kind.map(ClaimKind::as_str),
            "realization": self.realization,
            "resampling": self.resampling.map(resampling_json),
            "posterior": self.posterior.as_ref().map(posterior_json),
            "decision_request": self.decision_request.map(DecisionRequest::as_str),
        })
    }
}

/// The support-matrix cell of the study and its licensing status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupportView {
    coordinate: Option<Arc<str>>,
    licensed: bool,
    status: Option<CellStatus>,
}

impl SupportView {
    /// Support-matrix coordinate, when the query is on the matrix axis.
    #[must_use]
    pub fn coordinate(&self) -> Option<&str> {
        self.coordinate.as_deref()
    }

    /// Whether the coordinate is a licensed cell.
    #[must_use]
    pub const fn licensed(&self) -> bool {
        self.licensed
    }

    /// Evidence status of the cell, when the view was taken with the study.
    #[must_use]
    pub const fn status(&self) -> Option<CellStatus> {
        self.status
    }

    fn to_json(&self) -> Json {
        json!({
            "coordinate": self.coordinate.as_deref(),
            "licensed": self.licensed,
            "status": self.status.map(CellStatus::as_str),
        })
    }
}

/// One structure of a graph-posterior mixture.
#[derive(Clone, Debug, PartialEq)]
pub struct MixtureAtomView {
    key: u64,
    weight: f64,
    samples: usize,
    identified: Option<bool>,
    status: Option<IdentificationStatus>,
    estimand: Option<EstimandView>,
    completions: usize,
    envelope_identified_weight: Option<f64>,
}

impl MixtureAtomView {
    /// Opaque graph key of the atom.
    #[must_use]
    pub const fn key(&self) -> u64 {
        self.key
    }

    /// Posterior weight of the atom, summed over its samples.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Posterior samples that carry this structure.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Whether the atom identifies the target; `None` when the program
    /// identifies on each execution and retained no per-atom product.
    #[must_use]
    pub const fn identified(&self) -> Option<bool> {
        self.identified
    }

    /// Identification status of the atom, when retained.
    #[must_use]
    pub const fn status(&self) -> Option<IdentificationStatus> {
        self.status
    }

    /// The atom's estimand, or the estimand every completion of a class atom shares.
    #[must_use]
    pub const fn estimand(&self) -> Option<&EstimandView> {
        self.estimand.as_ref()
    }

    /// Completions of a class atom's envelope; zero for a single-graph atom.
    #[must_use]
    pub const fn completions(&self) -> usize {
        self.completions
    }

    /// Identified weight inside a class atom's envelope, before posterior mixing.
    #[must_use]
    pub const fn envelope_identified_weight(&self) -> Option<f64> {
        self.envelope_identified_weight
    }

    fn to_json(&self) -> Json {
        json!({
            "key": self.key,
            "weight": self.weight,
            "samples": self.samples,
            "identified": self.identified,
            "status": self.status.map(IdentificationStatus::as_str),
            "estimand": self.estimand.as_ref().map(EstimandView::to_json),
            "completions": self.completions,
            "envelope_identified_weight": self.envelope_identified_weight,
        })
    }
}

/// The single logical plan of a graph-posterior analysis: its atoms, their
/// weights and per-atom identification, and how they mix.
#[derive(Clone, Debug, PartialEq)]
pub struct MixtureView {
    atom_kind: Option<&'static str>,
    atoms: Vec<MixtureAtomView>,
    total_weight: f64,
    identified_mass: Option<f64>,
    unidentified_mass: Option<f64>,
    composition: Composition,
    identification_retained: bool,
}

/// Per-key identification a graph-posterior product retained for one atom.
struct AtomIdentification {
    status: IdentificationStatus,
    estimand: Option<EstimandView>,
    completions: usize,
    envelope_identified_weight: Option<f64>,
}

impl MixtureView {
    #[allow(
        clippy::too_many_lines,
        reason = "indexes the retained atoms of every cache kind, then folds the sampled graphs into one view"
    )]
    fn of(
        program: &LoweredProgram,
        caches: &[IdentificationCache],
        posterior: Option<&antecedent_discovery::GraphPosterior>,
    ) -> Self {
        let mut by_key: HashMap<u64, AtomIdentification> = HashMap::new();
        let mut samples: Option<&WeightedGraphSamples> = None;
        for cache in caches {
            match cache {
                IdentificationCache::GraphPosterior(retained) => {
                    samples = Some(&retained.graphs);
                    for atom in retained.atoms.iter() {
                        by_key.insert(
                            atom.key,
                            AtomIdentification {
                                status: atom.identification.status,
                                estimand: Some(EstimandView::of(&atom.estimand)),
                                completions: 0,
                                envelope_identified_weight: None,
                            },
                        );
                    }
                    for atom in retained.class_atoms.iter() {
                        by_key.insert(
                            atom.key,
                            AtomIdentification {
                                status: atom.identification.status,
                                estimand: atom.invariant.as_ref().map(EstimandView::of),
                                completions: atom.cases.len(),
                                envelope_identified_weight: Some(atom.identified_weight),
                            },
                        );
                    }
                }
                IdentificationCache::DbnPosterior(retained) => {
                    samples = Some(&retained.graphs);
                    for atom in retained.atoms.iter() {
                        by_key.insert(
                            atom.key,
                            AtomIdentification {
                                status: atom.identification.status,
                                estimand: Some(EstimandView::of(&atom.estimand)),
                                completions: 0,
                                envelope_identified_weight: None,
                            },
                        );
                    }
                }
                IdentificationCache::TemporalClassPosterior(retained) => {
                    samples = Some(&retained.graphs);
                    for atom in retained.class_atoms.iter() {
                        by_key.insert(
                            atom.key,
                            AtomIdentification {
                                status: atom.identification.status,
                                estimand: atom.invariant.as_ref().map(EstimandView::of),
                                completions: atom.envelope.envelope.cases.len(),
                                envelope_identified_weight: Some(atom.identified_weight),
                            },
                        );
                    }
                }
                IdentificationCache::Static(_)
                | IdentificationCache::Pag(_)
                | IdentificationCache::Cpdag(_)
                | IdentificationCache::Temporal(_)
                | IdentificationCache::TemporalClass(_)
                | IdentificationCache::Transport(_) => {}
            }
        }
        let mut atoms: Vec<MixtureAtomView> = Vec::new();
        let mut index: HashMap<u64, usize> = HashMap::new();
        let mut push = |key: u64, weight: f64, flag: Option<bool>| {
            if let Some(&slot) = index.get(&key) {
                let atom: &mut MixtureAtomView = &mut atoms[slot];
                atom.weight += weight;
                atom.samples += 1;
                atom.identified = match (atom.identified, flag) {
                    (Some(left), Some(right)) => Some(left || right),
                    (left, right) => left.or(right),
                };
                return;
            }
            index.insert(key, atoms.len());
            let retained = by_key.get(&key);
            atoms.push(MixtureAtomView {
                key,
                weight,
                samples: 1,
                identified: flag,
                status: retained.map(|atom| atom.status),
                estimand: retained.and_then(|atom| atom.estimand.clone()),
                completions: retained.map_or(0, |atom| atom.completions),
                envelope_identified_weight: retained
                    .and_then(|atom| atom.envelope_identified_weight),
            });
        };
        let (identified_mass, unidentified_mass) = if let Some(graphs) = samples {
            for (sample, weight) in graphs.weights.iter().enumerate() {
                let key = graphs.graph_keys.get(sample).copied().unwrap_or(sample as u64);
                let flag =
                    graphs.identified.get(sample).map(|flag| *flag == GraphIdentFlag::Identified);
                push(key, *weight, flag);
            }
            (Some(graphs.identified_mass()), Some(graphs.unidentified_mass()))
        } else {
            if let Some(posterior) = posterior {
                for (sample, weight) in posterior.weights.iter().enumerate() {
                    let key = posterior.graph_keys.get(sample).copied().unwrap_or(sample as u64);
                    push(key, *weight, None);
                }
            }
            (None, None)
        };
        let total_weight = atoms.iter().map(|atom| atom.weight).sum();
        Self {
            atom_kind: posterior.map(|posterior| posterior.atom_kind.as_str()),
            atoms,
            total_weight,
            identified_mass,
            unidentified_mass,
            composition: program.evidence().composition(),
            identification_retained: samples.is_some(),
        }
    }

    /// Atom kind of the posterior (`dag`, `cpdag`, `pag`, `admg`, ...), when the
    /// view was taken with the study.
    #[must_use]
    pub const fn atom_kind(&self) -> Option<&'static str> {
        self.atom_kind
    }

    /// Distinct structures in order of first appearance.
    #[must_use]
    pub fn atoms(&self) -> &[MixtureAtomView] {
        &self.atoms
    }

    /// Total posterior weight.
    #[must_use]
    pub const fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Weight on atoms that identify the target, when identification was retained.
    #[must_use]
    pub const fn identified_mass(&self) -> Option<f64> {
        self.identified_mass
    }

    /// Weight on atoms that do not identify the target; it is reported, never dropped.
    #[must_use]
    pub const fn unidentified_mass(&self) -> Option<f64> {
        self.unidentified_mass
    }

    /// How the atoms combine into one answer.
    #[must_use]
    pub const fn composition(&self) -> Composition {
        self.composition
    }

    /// Whether every atom was identified at compile and retained with the program.
    #[must_use]
    pub const fn identification_retained(&self) -> bool {
        self.identification_retained
    }

    fn to_json(&self) -> Json {
        json!({
            "atom_kind": self.atom_kind,
            "atoms": self.atoms.iter().map(MixtureAtomView::to_json).collect::<Vec<_>>(),
            "total_weight": self.total_weight,
            "identified_mass": self.identified_mass,
            "unidentified_mass": self.unidentified_mass,
            "composition": composition_json(self.composition),
            "identification_retained": self.identification_retained,
        })
    }
}

/// One view over every layer of a compiled route.
///
/// Built from the lowered program before execution: no identification, fit, row
/// scan, graph enumeration, simulation or pool lease happens while it is built.
#[derive(Clone, Debug)]
pub struct PlanView {
    route: PlanRoute,
    stage: PlanStage,
    program: Arc<LoweredProgram>,
    identification: IdentificationView,
    physical: PhysicalView,
    uncertainty: UncertaintyDemand,
    support: SupportView,
    mixture: Option<MixtureView>,
    operation: Option<&'static str>,
}

/// What a view is built from.
pub(crate) struct ViewSource<'a> {
    /// The lowered program.
    pub(crate) program: Arc<LoweredProgram>,
    /// Where the view is taken.
    pub(crate) stage: PlanStage,
    /// The physical record the plan was sized to.
    pub(crate) record: Option<&'a PhysicalExecutionPlanRecord>,
    /// The study the program was compiled for, when the caller holds it.
    pub(crate) study: Option<&'a crate::Study>,
    /// Kernel policy of the compile context.
    pub(crate) kernel_policy: Option<KernelPolicy>,
    /// Stable name of the sealed operation a prepared handle retained.
    pub(crate) operation: Option<&'static str>,
}

impl PlanView {
    /// Build a view from what compile and, for a prepared handle, preparation retained.
    pub(crate) fn build(source: ViewSource<'_>) -> Self {
        let ViewSource { program, stage, record, study, kernel_policy, operation } = source;
        let route = plan_route(program.cell());
        let installed = match (stage, study) {
            (PlanStage::Prepared, Some(study)) => installed_caches(study),
            _ => Vec::new(),
        };
        let identification = IdentificationView::of(program.identification_layer(), &installed);
        let caches: &[IdentificationCache] = if installed.is_empty() {
            program.retained_identification().map_or(&[][..], |retained| retained.caches())
        } else {
            &installed
        };
        // One arm per family: a family added to the route table without a view
        // of its own family-specific layer does not compile.
        let mixture = match route.family() {
            RouteFamily::GraphPosterior => Some(MixtureView::of(
                &program,
                caches,
                study.and_then(|study| study.graph_posterior.as_ref()),
            )),
            RouteFamily::StaticEffect
            | RouteFamily::StaticResponse
            | RouteFamily::Structural
            | RouteFamily::Temporal
            | RouteFamily::DesignBased
            | RouteFamily::Discovery
            | RouteFamily::Cyclic
            | RouteFamily::Planning
            | RouteFamily::PropagationChain => None,
        };
        let scientific = program.scientific();
        let support = SupportView {
            coordinate: scientific.support_coordinate.clone(),
            licensed: scientific.licensed(),
            status: study.and_then(|study| study.support_status),
        };
        let physical = PhysicalView {
            operators: Arc::clone(&program.physical().operators),
            design: program.physical().design(),
            record: record.cloned(),
            kernel_policy,
        };
        let uncertainty = UncertaintyDemand::of(&program);
        Self {
            route,
            stage,
            program,
            identification,
            physical,
            uncertainty,
            support,
            mixture,
            operation,
        }
    }

    /// The view of a lowered program held without a physical plan record: a
    /// cyclic family handle or a standalone experiment plan.
    pub(crate) fn of_program(program: Arc<LoweredProgram>, stage: PlanStage) -> Self {
        Self::build(ViewSource {
            program,
            stage,
            record: None,
            study: None,
            kernel_policy: None,
            operation: None,
        })
    }

    /// The view of `plan`'s lowered program.
    ///
    /// # Errors
    ///
    /// `route_not_lowered` when the plan carries no lowered program.
    pub(crate) fn of_plan(
        plan: &PhysicalExecutionPlan,
        stage: PlanStage,
        study: Option<&crate::Study>,
        kernel_policy: Option<KernelPolicy>,
        operation: Option<&'static str>,
    ) -> Result<Self, CausalError> {
        let program = plan.lowered.as_ref().map(Arc::clone).ok_or_else(|| {
            crate::compile_reason!(
                "route_not_lowered",
                "the physical plan carries no lowered program; compile it with Study::compile"
            )
        })?;
        Ok(Self::build(ViewSource {
            program,
            stage,
            record: Some(&plan.record),
            study,
            kernel_policy,
            operation,
        }))
    }

    /// The route the program was lowered for.
    #[must_use]
    pub const fn route(&self) -> PlanRoute {
        self.route
    }

    /// The family that lowered the route.
    #[must_use]
    pub const fn family(&self) -> RouteFamily {
        self.route.family()
    }

    /// Where the view was taken.
    #[must_use]
    pub const fn stage(&self) -> PlanStage {
        self.stage
    }

    /// Identity of the lowered program; equal to [`LoweredProgram::identity`].
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        self.program.identity()
    }

    /// How far the route's lowering was carried.
    #[must_use]
    pub fn depth(&self) -> LoweringDepth {
        self.program.depth()
    }

    /// The lowered program the view reads.
    #[must_use]
    pub fn program(&self) -> &LoweredProgram {
        &self.program
    }

    /// Layer one: the scientific problem.
    #[must_use]
    pub fn scientific(&self) -> &ScientificProblem {
        self.program.scientific()
    }

    /// Layer two: the identification product.
    #[must_use]
    pub const fn identification(&self) -> &IdentificationView {
        &self.identification
    }

    /// Layer three: the statistical program.
    #[must_use]
    pub fn statistical(&self) -> &StatisticalProgram {
        self.program.statistical()
    }

    /// Layer four: the evidence and decision program.
    #[must_use]
    pub fn evidence(&self) -> &EvidenceDecisionProgram {
        self.program.evidence()
    }

    /// Layer five: the physical plan.
    #[must_use]
    pub const fn physical(&self) -> &PhysicalView {
        &self.physical
    }

    /// Layer six: the claim the execution must publish.
    #[must_use]
    pub fn claim(&self) -> &ClaimDemand {
        self.program.claim()
    }

    /// The uncertainty and claim demand.
    #[must_use]
    pub const fn uncertainty(&self) -> &UncertaintyDemand {
        &self.uncertainty
    }

    /// Support cell and licensing status.
    #[must_use]
    pub const fn support(&self) -> &SupportView {
        &self.support
    }

    /// The mixture of a graph-posterior analysis; `None` for every other family.
    #[must_use]
    pub const fn mixture(&self) -> Option<&MixtureView> {
        self.mixture.as_ref()
    }

    /// Stable name of the sealed operation a prepared handle retained
    /// (`lowered_dispatch` when no specialised operation admitted it); `None` for a compiled view.
    #[must_use]
    pub const fn operation(&self) -> Option<&'static str> {
        self.operation
    }

    /// The whole view as JSON. Variables are written as their dense ids, digests
    /// as lowercase hex.
    #[must_use]
    pub fn to_json(&self) -> Json {
        json!({
            "route": self.route.as_str(),
            "family": self.family().as_str(),
            "cyclic_family": self.program.cyclic_family(),
            "planned_design": self.program.planned_design(),
            "stage": self.stage.as_str(),
            "identity": self.identity().to_hex(),
            "depth": self.depth().as_str(),
            "scientific": scientific_json(self.scientific()),
            "identification": self.identification.to_json(),
            "statistical": statistical_json(self.statistical()),
            "evidence": evidence_json(self.evidence()),
            "physical": self.physical.to_json(),
            "uncertainty": self.uncertainty.to_json(),
            "support": self.support.to_json(),
            "mixture": self.mixture.as_ref().map(MixtureView::to_json),
            "operation": self.operation,
        })
    }
}

impl PhysicalExecutionPlan {
    /// The plan view of this plan's lowered program. It reads what compile
    /// retained and does no identification, fitting, scanning or simulation.
    ///
    /// # Errors
    ///
    /// `route_not_lowered` when the plan was not built by `Study::compile`.
    pub fn plan_view(&self) -> Result<PlanView, CausalError> {
        PlanView::of_plan(self, PlanStage::Compiled, None, None, None)
    }
}

impl crate::Study {
    /// Compile the study and view every layer of its lowered program.
    ///
    /// This is compile: a route whose program retains identification identifies
    /// once (as `Study::compile` does), and a linear adjustment quotes its design
    /// width from the schema and the categorical levels. Nothing is fitted,
    /// sampled, simulated or enumerated beyond what compile does, and no execution
    /// runs. [`Self::inspect`] stays the non-identifying preflight.
    ///
    /// # Errors
    ///
    /// Every refusal of `Study::compile`.
    pub fn plan_view(&self, ctx: &ExecutionContext) -> Result<PlanView, CausalError> {
        let plan = self.compile(ctx)?;
        PlanView::of_plan(&plan, PlanStage::Compiled, Some(self), Some(ctx.kernel_policy), None)
    }
}

/// The plan route of a program cell.
const fn plan_route(cell: ProgramCell) -> PlanRoute {
    match cell {
        ProgramCell::Route(cell) => cell.plan_route(),
        ProgramCell::CyclicFamily(family) => PlanRoute::CyclicFamily(family),
        ProgramCell::Planning(design) => PlanRoute::Planning(design),
        ProgramCell::PropagationChain(reading) => PlanRoute::PropagationChain(reading),
    }
}

/// The identification products preparation installed into `study`, in the
/// order preparation installs them.
fn installed_caches(study: &crate::Study) -> Vec<IdentificationCache> {
    let mut caches = Vec::new();
    if let Some(cache) = &study.identification_cache {
        caches.push(IdentificationCache::Static(Arc::clone(cache)));
    }
    if let Some(cache) = &study.pag_identification_cache {
        caches.push(IdentificationCache::Pag(Arc::clone(cache)));
    }
    if let Some(cache) = &study.cpdag_identification_cache {
        caches.push(IdentificationCache::Cpdag(Arc::clone(cache)));
    }
    if let Some(cache) = &study.temporal_identification_cache {
        caches.push(IdentificationCache::Temporal(Arc::clone(cache)));
    }
    if let Some(cache) = &study.temporal_class_identification_cache {
        caches.push(IdentificationCache::TemporalClass(Arc::clone(cache)));
    }
    if let Some(cache) = &study.transport_identification_cache {
        caches.push(IdentificationCache::Transport(Arc::clone(cache)));
    }
    if let Some(cache) = &study.graph_posterior_identification_cache {
        caches.push(IdentificationCache::GraphPosterior(Arc::clone(cache)));
    }
    if let Some(cache) = &study.dbn_posterior_identification_cache {
        caches.push(IdentificationCache::DbnPosterior(Arc::clone(cache)));
    }
    if let Some(cache) = &study.temporal_class_posterior_identification_cache {
        caches.push(IdentificationCache::TemporalClassPosterior(Arc::clone(cache)));
    }
    caches
}

/// The weaker of an optional status and `status`, by evidential strength.
fn weakest(
    current: Option<IdentificationStatus>,
    status: IdentificationStatus,
) -> IdentificationStatus {
    match current {
        Some(current) if current.strength_rank() <= status.strength_rank() => current,
        _ => status,
    }
}

/// Status of a mixture with the given identified and unidentified mass.
fn mixture_status(identified: f64, unidentified: f64) -> IdentificationStatus {
    if unidentified <= POSTERIOR_MASS_TOLERANCE {
        IdentificationStatus::NonparametricallyIdentified
    } else if identified <= POSTERIOR_MASS_TOLERANCE {
        IdentificationStatus::NotIdentified
    } else {
        IdentificationStatus::GraphDependent
    }
}

const fn outcome_name(outcome: IdentificationOutcome) -> &'static str {
    match outcome {
        IdentificationOutcome::Identified => "identified",
        IdentificationOutcome::Unidentified => "unidentified",
        IdentificationOutcome::Unsupported => "unsupported",
        IdentificationOutcome::TimedOut => "timed_out",
        IdentificationOutcome::NumericallyInfeasible => "numerically_infeasible",
        IdentificationOutcome::Uncalibrated => "uncalibrated",
    }
}

const fn claim_class_name(class: IdentificationClaimClass) -> &'static str {
    match class {
        IdentificationClaimClass::IdentifiedClaim => "identified_claim",
        IdentificationClaimClass::ModelConditional => "model_conditional",
        IdentificationClaimClass::UnsupportedExtrapolation => "unsupported_extrapolation",
    }
}

const fn realization_name(realization: &Realization) -> &'static str {
    match realization {
        Realization::Frequentist(_) => "frequentist",
        Realization::Bayesian(_) => "bayesian",
        Realization::GeneralizedBayes(_) => "generalized_bayes",
        Realization::Envelope(_) => "envelope",
        Realization::ExactPoint { .. } => "exact_point",
    }
}

fn ids(variables: &[VariableId]) -> Vec<u32> {
    variables.iter().map(|variable| variable.raw()).collect()
}

fn hex(digest: &[u8; 32]) -> String {
    SemanticDigest::from_bytes(*digest).to_hex()
}

fn design_json(design: DesignShape) -> Json {
    match design.source() {
        WidthSource::Quoted => json!({
            "source": "quoted",
            "rows": design.rows(),
            "columns": design.columns(),
            "direct": design.direct(),
        }),
        WidthSource::NotDesignBased(reason) => json!({
            "source": "not_design_based",
            "reason": reason,
        }),
    }
}

fn composition_json(composition: Composition) -> Json {
    match composition {
        Composition::Single => json!({"kind": "single"}),
        Composition::AtomMixture { atoms } => json!({"kind": "atom_mixture", "atoms": atoms}),
        Composition::Transported => json!({"kind": "transported"}),
        Composition::Pooled { sources } => json!({"kind": "pooled", "sources": sources}),
        Composition::Chained { stages } => json!({"kind": "chained", "stages": stages}),
    }
}

fn resampling_json(plan: ResamplingPlan) -> Json {
    json!({
        "method": plan.method().as_str(),
        "replicates": plan.replicates(),
        "unit": plan.unit().as_str(),
    })
}

fn posterior_json(plan: &PosteriorPlan) -> Json {
    json!({
        "algorithm": plan.algorithm().as_str(),
        "draws": plan.draws(),
        "prior": plan.prior().name(),
    })
}

fn scientific_json(problem: &ScientificProblem) -> Json {
    let target = match problem.target() {
        ProblemTarget::Target(target) => {
            json!({"form": "target", "identity": target.identity().to_hex()})
        }
        ProblemTarget::Targets(targets) => json!({
            "form": "targets",
            "identities": targets.iter().map(|target| target.identity().to_hex()).collect::<Vec<_>>(),
        }),
        ProblemTarget::QueryFunctional { kind, outcome, query } => json!({
            "form": "query_functional",
            "kind": kind.as_str(),
            "outcome": outcome.map(VariableId::raw),
            "query": hex(query),
        }),
    };
    json!({
        "target": target,
        "plan_id": problem.plan_id(),
        "data": crate::planner::lowered::data_classification_name(problem.data()),
        "graph_class": problem.graph_class(),
        "query_variables": ids(problem.query_variables()),
        "source_digest": problem.source_digest().as_ref().map(hex),
    })
}

fn statistical_json(program: &StatisticalProgram) -> Json {
    json!({
        "identity": program.identity().to_hex(),
        "identifier": program.identifier().as_ref().map(crate::IdentifierId::as_str),
        "estimator": program.estimator().as_ref().map(crate::EstimatorId::as_str),
        "realization": realization_name(program.realization()),
        "models": program
            .models()
            .iter()
            .map(|model| json!({
                "role": model.role().as_str(),
                "family": model.family(),
                "inputs": ids(model.inputs()),
                "output": model.output().map(VariableId::raw),
            }))
            .collect::<Vec<_>>(),
        "folds": program.folds().map(|folds| json!({
            "folds": folds.folds(),
            "unit": folds.unit().as_str(),
            "embargo": folds.embargo(),
        })),
        "resampling": program.resampling().map(resampling_json),
        "posterior": program.posterior().map(posterior_json),
        "design": design_json(program.design()),
        "operators": program.operators().len(),
    })
}

fn evidence_json(program: &EvidenceDecisionProgram) -> Json {
    let decision = program.decision().map(|decision| {
        let evsi = decision.information_value().map(|evsi| {
            let tasks = evsi.monte_carlo();
            json!({
                "candidates": evsi.candidates().iter().map(|candidate| &**candidate).collect::<Vec<_>>(),
                "prior_observations": evsi.prior_observation_ids().len(),
                "rng_seed": evsi.rng_seed(),
                "max_candidates": evsi.max_candidates(),
                "draws_per_candidate": evsi.rank().draws_per_candidate(),
                "monte_carlo_tasks": tasks.tasks(),
                "monte_carlo_bytes": tasks.total_bytes(),
                "cost_map": evsi.cost_map().is_some(),
            })
        });
        json!({
            "carrier": decision.carrier().as_str(),
            "affine_utility": decision.utility().is_affine(),
            "actions": decision.actions(),
            "constraints": decision.constraints(),
            "chance_threshold": decision.chance_threshold(),
            "utility_unit": decision.utility_unit(),
            "contract_identity": decision.contract_identity(),
            "information_value": evsi,
        })
    });
    let planning = program.planning().map(|planning| {
        json!({
            "request_identity": planning.request_identity().to_hex(),
            "simulation": planning.simulation().map(|quote| json!({
                "replicates": quote.replicates(),
                "record_bytes": quote.record_bytes(),
                "scratch_bytes": quote.scratch_bytes(),
                "workers": quote.workers(),
                "total_bytes": quote.total_bytes(),
            })),
        })
    });
    json!({
        "identity": program.identity().to_hex(),
        "sources": program
            .sources()
            .iter()
            .map(|source| json!({
                "role": source.role().as_str(),
                "digest": source.digest().as_ref().map(hex),
            }))
            .collect::<Vec<_>>(),
        "composition": composition_json(program.composition()),
        "decision": decision,
        "planning": planning,
    })
}

/// Shared assertions of the in-crate route coverage tests.
#[cfg(test)]
pub(crate) mod checks {
    use antecedent_core::execution_attempt::{PhysicalEvent, observe_execution};

    use super::{IdentificationKind, PlanView, RouteFamily};
    use crate::planner::PhysicalExecutionPlan;
    use crate::planner::lowered::IdentificationLayer;

    /// The view of `plan` agrees with its lowered program layer by layer, and
    /// building it identifies, fits, scans and samples nothing.
    pub(crate) fn assert_view_agrees(name: &str, plan: &PhysicalExecutionPlan) -> PlanView {
        let program = plan.lowered().unwrap_or_else(|| panic!("{name}: no lowered program"));
        let ((attempt, solves), bayes) = antecedent_prob::fit_counts::count_bayesian_work(|| {
            antecedent_stats::fit_counts::count_least_squares_solves(|| {
                observe_execution(|| plan.plan_view())
            })
        });
        let view = attempt.result.unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(
            attempt.report.operations().all(|(_, counts)| counts.attempted == 0),
            "{name}: building the view ran an instrumented operation"
        );
        for event in [PhysicalEvent::Scan, PhysicalEvent::ModelFit, PhysicalEvent::Copy] {
            assert_eq!(attempt.report.physical(event).unwrap_or(0), 0, "{name}: {event:?}");
        }
        assert_eq!(solves, 0, "{name}: building the view solved a design");
        assert_eq!(bayes.model_fits + bayes.posterior_draws, 0, "{name}: posterior work");
        assert_eq!(view.identity(), program.identity(), "{name}: identity");
        assert_eq!(view.route().as_str(), program.route(), "{name}: route");
        assert_eq!(view.depth(), program.depth(), "{name}: depth");
        assert_eq!(
            view.statistical().identity(),
            program.statistical().identity(),
            "{name}: statistical program"
        );
        assert_eq!(view.evidence(), program.evidence(), "{name}: evidence program");
        assert_eq!(view.claim(), program.claim(), "{name}: claim demand");
        assert_eq!(view.uncertainty().reading(), program.claim().uncertainty(), "{name}");
        assert_eq!(view.physical().design(), program.physical().design(), "{name}: design");
        assert_eq!(view.physical().operators(), program.physical().operators(), "{name}");
        assert_eq!(view.physical().record(), Some(&plan.record), "{name}: physical record");
        assert_eq!(view.support().licensed(), program.scientific().licensed(), "{name}");
        assert_eq!(
            view.support().coordinate(),
            program.scientific().support_coordinate(),
            "{name}: support coordinate"
        );
        let expected = match program.identification_layer() {
            IdentificationLayer::Retained(retained) => {
                assert_eq!(
                    view.identification().retained().len(),
                    retained.caches().len(),
                    "{name}"
                );
                assert_eq!(
                    view.identification().product().map(super::ProductView::identity),
                    retained
                        .product
                        .as_ref()
                        .map(antecedent_identify::CompiledIdentificationProduct::identity),
                    "{name}: product identity"
                );
                IdentificationKind::Retained
            }
            IdentificationLayer::Deferred { .. } => IdentificationKind::Deferred,
            IdentificationLayer::PerExecution { .. } => IdentificationKind::PerExecution,
            IdentificationLayer::NonGraphical(_) => IdentificationKind::NonGraphical,
        };
        assert_eq!(view.identification().kind(), expected, "{name}: identification kind");
        assert_eq!(
            view.mixture().is_some(),
            view.family() == RouteFamily::GraphPosterior,
            "{name}: only a graph posterior carries a mixture"
        );
        let json = view.to_json();
        assert_eq!(json["identity"], program.identity().to_hex(), "{name}: JSON identity");
        assert_eq!(json["route"], program.route(), "{name}: JSON route");
        view
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{PlanRoute, RouteFamily};
    use crate::analysis::lowering::{GraphPosteriorCell, REGISTERED, RouteCell};

    #[test]
    fn every_route_cell_has_a_plan_route_of_its_registered_family() {
        let mut names = BTreeSet::new();
        for (cell, family) in REGISTERED {
            let route = cell.plan_route();
            assert_eq!(route.as_str(), cell.as_str(), "the view names the cell it lowers");
            assert_eq!(route.family().as_str(), *family, "{} family", cell.as_str());
            assert_eq!(route.family(), cell.family(), "{}", cell.as_str());
            assert!(names.insert(route.as_str()), "{} has one plan route", cell.as_str());
        }
        for cell in GraphPosteriorCell::ALL {
            let route = RouteCell::GraphPosterior(cell).plan_route();
            assert_eq!(route, PlanRoute::GraphPosterior(cell));
            assert_eq!(route.family(), RouteFamily::GraphPosterior);
            assert!(names.insert(route.as_str()), "{} has one plan route", cell.as_str());
        }
        assert_eq!(PlanRoute::CyclicFamily("system").family(), RouteFamily::Cyclic);
        assert_eq!(PlanRoute::CyclicFamily("system").as_str(), "CyclicFamily");
        assert_eq!(PlanRoute::Planning("bernoulli").family(), RouteFamily::Planning);
        assert_eq!(PlanRoute::Planning("bernoulli").as_str(), "ExperimentPlanning");
    }
}
