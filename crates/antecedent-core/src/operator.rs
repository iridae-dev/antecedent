//! Typed operator edges over a scientific target.
//!
//! An [`Operator`] records one effect, the graph that edge belongs to, the data
//! roles it consumes and produces, and whether frozen-fit reuse is allowed.
//! Roles are population, outcome, world or schedule, conditioning, fit handle,
//! prior, and transported law. They are data, not estimators.
//!
//! Three graphs stay distinct. [`OperatorGraph::Causal`] is mechanisms and
//! interventions. [`OperatorGraph::Provenance`] is evidence dependence and
//! selection. [`OperatorGraph::Execution`] is computation. Causal feedback is
//! only an explicit [`OperatorEffect::Solve`] or [`OperatorEffect::Scan`], and
//! only on the causal graph. A provenance cycle is not causal feedback. An
//! execution iteration is not causal feedback.
//!
//! A legal operator does not silently change population, history, or the
//! scientific obligations on a [`crate::ScientificTarget`]. Selection,
//! conditioning, intervention, transport, and priors are explicit effects.
//! Nothing here schedules kernels, solves a system, or lowers an identified
//! functional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

/// Graph an operator edge belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OperatorGraph {
    /// Mechanisms and interventions. Causal feedback lives only here.
    Causal,
    /// Evidence dependence and selection. A cycle here is not causal feedback.
    Provenance,
    /// Computation. An iteration here is not causal feedback.
    Execution,
}

impl OperatorGraph {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Causal => "causal",
            Self::Provenance => "provenance",
            Self::Execution => "execution",
        }
    }
}

impl fmt::Display for OperatorGraph {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Effect an operator is allowed to have.
///
/// [`Self::Solve`] and [`Self::Scan`] are the only effects that may carry
/// causal feedback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OperatorEffect {
    /// Read recorded observations.
    Read,
    /// Select units. This is the explicit population change.
    Select,
    /// Fit a nuisance model. The product is a fit handle, not an estimator.
    Fit,
    /// Use a prior, including a data-derived prior declared as this effect.
    Prior,
    /// Condition on declared variables.
    Condition,
    /// Intervene, naming a world or schedule.
    Intervene,
    /// Transport a law. The target population is an input, not a new population.
    Transport,
    /// Reuse a frozen fit, with no new observation read and no new nuisance fit.
    ReuseFrozenFit,
    /// Solve a simultaneous block. May carry causal feedback on the causal graph.
    Solve,
    /// Scan a temporal recurrence. May carry causal feedback on the causal graph.
    Scan,
}

impl OperatorEffect {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Select => "select",
            Self::Fit => "fit",
            Self::Prior => "prior",
            Self::Condition => "condition",
            Self::Intervene => "intervene",
            Self::Transport => "transport",
            Self::ReuseFrozenFit => "reuse_frozen_fit",
            Self::Solve => "solve",
            Self::Scan => "scan",
        }
    }

    /// Whether this effect is allowed to carry causal feedback.
    #[must_use]
    pub const fn may_carry_causal_feedback(self) -> bool {
        matches!(self, Self::Solve | Self::Scan)
    }

    /// Whether a declaration of this effect may set [`FrozenFitReuse::Allowed`].
    ///
    /// Only an effect that produces or consumes a fit, or transports a law from a
    /// fit, has a frozen fit to reuse. Every other effect has none.
    #[must_use]
    pub const fn may_allow_frozen_fit_reuse(self) -> bool {
        matches!(self, Self::Fit | Self::Transport | Self::ReuseFrozenFit)
    }

    /// Whether `role` is a product of this effect.
    ///
    /// Any other output would change population, history, or a scientific
    /// obligation without the effect that owns that change.
    #[must_use]
    pub const fn owns_output(self, role: OperatorRole) -> bool {
        matches!(
            (self, role),
            (Self::Read, OperatorRole::Outcome)
                | (Self::Select, OperatorRole::Population)
                | (Self::Fit | Self::ReuseFrozenFit, OperatorRole::FitHandle)
                | (Self::Prior, OperatorRole::Prior)
                | (Self::Condition, OperatorRole::Conditioning)
                | (Self::Intervene | Self::Solve | Self::Scan, OperatorRole::WorldOrSchedule)
                | (Self::Transport, OperatorRole::TransportedLaw)
        )
    }
}

impl fmt::Display for OperatorEffect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Data role on an operator port.
///
/// These are values an operator reads or writes. None of them is an estimator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OperatorRole {
    /// Population the operator reads, or the population a selection writes.
    Population,
    /// Outcome quantity, including observations of that outcome.
    Outcome,
    /// Intervention world or temporal schedule.
    WorldOrSchedule,
    /// Conditioning set.
    Conditioning,
    /// Handle to a fit. The handle is data; the estimator is not.
    FitHandle,
    /// Prior, including a data-derived prior declared as data.
    Prior,
    /// Law produced by transport.
    TransportedLaw,
}

impl OperatorRole {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Population => "population",
            Self::Outcome => "outcome",
            Self::WorldOrSchedule => "world_or_schedule",
            Self::Conditioning => "conditioning",
            Self::FitHandle => "fit_handle",
            Self::Prior => "prior",
            Self::TransportedLaw => "transported_law",
        }
    }
}

impl fmt::Display for OperatorRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Whether this operator may reuse a fit that was frozen earlier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FrozenFitReuse {
    /// A previously frozen fit may be reused.
    Allowed,
    /// Frozen-fit reuse is not part of this operator.
    Forbidden,
}

impl FrozenFitReuse {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Forbidden => "forbidden",
        }
    }
}

impl fmt::Display for FrozenFitReuse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Extra evidence-use claims beside the operator's one effect.
///
/// [`OperatorEffect::Read`] may claim a new observation read.
/// [`OperatorEffect::Fit`] may claim a new nuisance fit. Any other effect that
/// carries either claim is refused. [`OperatorEffect::ReuseFrozenFit`] is
/// refused for both, including when an input role is [`OperatorRole::Outcome`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct EvidenceClaims {
    new_observation_read: bool,
    new_nuisance_fit: bool,
}

impl EvidenceClaims {
    /// Neither a new observation read nor a new nuisance fit.
    #[must_use]
    pub const fn none() -> Self {
        Self { new_observation_read: false, new_nuisance_fit: false }
    }

    /// A new observation read, and no new nuisance fit.
    #[must_use]
    pub const fn observation_read() -> Self {
        Self { new_observation_read: true, new_nuisance_fit: false }
    }

    /// A new nuisance fit, and no new observation read.
    #[must_use]
    pub const fn nuisance_fit() -> Self {
        Self { new_observation_read: false, new_nuisance_fit: true }
    }

    /// Both a new observation read and a new nuisance fit.
    #[must_use]
    pub const fn observation_read_and_nuisance_fit() -> Self {
        Self { new_observation_read: true, new_nuisance_fit: true }
    }

    /// Whether the declaration claims a new observation read.
    #[must_use]
    pub const fn new_observation_read(self) -> bool {
        self.new_observation_read
    }

    /// Whether the declaration claims a new nuisance fit.
    #[must_use]
    pub const fn new_nuisance_fit(self) -> bool {
        self.new_nuisance_fit
    }
}

/// Input and output roles of one operator.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct OperatorPorts {
    inputs: Arc<[OperatorRole]>,
    outputs: Arc<[OperatorRole]>,
}

impl OperatorPorts {
    /// Ports in declaration order.
    #[must_use]
    pub fn new(
        inputs: impl IntoIterator<Item = OperatorRole>,
        outputs: impl IntoIterator<Item = OperatorRole>,
    ) -> Self {
        Self { inputs: inputs.into_iter().collect(), outputs: outputs.into_iter().collect() }
    }

    /// Roles the operator consumes.
    #[must_use]
    pub fn inputs(&self) -> &[OperatorRole] {
        &self.inputs
    }

    /// Roles the operator produces.
    #[must_use]
    pub fn outputs(&self) -> &[OperatorRole] {
        &self.outputs
    }
}

/// Declaration submitted to [`Operator::new`].
///
/// Causal feedback starts off, frozen-fit reuse starts forbidden, and
/// [`EvidenceClaims::none`] is the evidence claim. Each `with_` method replaces
/// that default.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorDeclaration {
    effect: OperatorEffect,
    graph: OperatorGraph,
    inputs: Arc<[OperatorRole]>,
    outputs: Arc<[OperatorRole]>,
    causal_feedback: bool,
    frozen_fit_reuse: FrozenFitReuse,
    claims: EvidenceClaims,
}

impl OperatorDeclaration {
    /// Declaration of `effect` on `graph` with `ports`.
    #[must_use]
    pub fn new(effect: OperatorEffect, graph: OperatorGraph, ports: OperatorPorts) -> Self {
        Self {
            effect,
            graph,
            inputs: ports.inputs,
            outputs: ports.outputs,
            causal_feedback: false,
            frozen_fit_reuse: FrozenFitReuse::Forbidden,
            claims: EvidenceClaims::none(),
        }
    }

    /// Set whether the edge carries causal feedback.
    #[must_use]
    pub const fn with_causal_feedback(mut self, causal_feedback: bool) -> Self {
        self.causal_feedback = causal_feedback;
        self
    }

    /// Set whether frozen-fit reuse is allowed.
    #[must_use]
    pub const fn with_frozen_fit_reuse(mut self, frozen_fit_reuse: FrozenFitReuse) -> Self {
        self.frozen_fit_reuse = frozen_fit_reuse;
        self
    }

    /// Replace the extra evidence-use claims.
    #[must_use]
    pub const fn with_evidence_claims(mut self, claims: EvidenceClaims) -> Self {
        self.claims = claims;
        self
    }
}

/// Refusal from [`Operator::new`].
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum OperatorError {
    /// Solve or Scan was placed on the provenance or execution graph.
    #[error("{effect} belongs on the causal graph, not {graph}")]
    FeedbackOperatorOffCausalGraph {
        /// Solve or Scan.
        effect: OperatorEffect,
        /// Provenance or execution.
        graph: OperatorGraph,
    },
    /// An effect other than Solve or Scan claimed causal feedback.
    #[error("{effect} cannot carry causal feedback")]
    CausalFeedbackNotPermitted {
        /// Effect that claimed feedback.
        effect: OperatorEffect,
    },
    /// Frozen-fit reuse also claimed a new observation read or a new nuisance fit.
    #[error("frozen-fit reuse cannot also claim a new observation read or a new nuisance fit")]
    ReuseClaimsNewReadOrFit {
        /// Claimed a new observation read, or consumed an outcome.
        new_observation_read: bool,
        /// Claimed a new nuisance fit.
        new_nuisance_fit: bool,
    },
    /// An effect claimed a read or a fit that is not its own effect.
    #[error("{effect} cannot claim a new observation read or a new nuisance fit")]
    UndeclaredReadOrFit {
        /// Effect that made the extra claim.
        effect: OperatorEffect,
        /// Claimed a new observation read.
        new_observation_read: bool,
        /// Claimed a new nuisance fit.
        new_nuisance_fit: bool,
    },
    /// Frozen-fit reuse was declared with reuse forbidden.
    #[error("frozen-fit reuse must declare that frozen-fit reuse is allowed")]
    ReuseRequiresFrozenFit,
    /// A fit declared no new nuisance fit.
    #[error("a fit must claim a new nuisance fit; declaring no evidence claim hides the fit")]
    FitMustClaimNuisanceFit,
    /// Frozen-fit reuse was allowed on an effect that has no frozen fit.
    #[error("{effect} has no frozen fit, so it cannot allow frozen-fit reuse")]
    FrozenFitReuseNotApplicable {
        /// Effect that declared reuse allowed.
        effect: OperatorEffect,
    },
    /// An output role would change population, history, or a scientific obligation.
    #[error(
        "{effect} cannot output {role}; that would change population, history, or a scientific obligation"
    )]
    SilentObligationChange {
        /// Effect that does not own `role`.
        effect: OperatorEffect,
        /// Output role the effect does not own.
        role: OperatorRole,
    },
}

/// One typed operator edge.
///
/// The effect, the graph, the input and output roles, and whether frozen-fit
/// reuse is allowed are all part of the value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Operator {
    effect: OperatorEffect,
    graph: OperatorGraph,
    inputs: Arc<[OperatorRole]>,
    outputs: Arc<[OperatorRole]>,
    causal_feedback: bool,
    frozen_fit_reuse: FrozenFitReuse,
}

impl Operator {
    /// Construct an operator from a checked declaration.
    ///
    /// # Errors
    ///
    /// * [`OperatorError::FeedbackOperatorOffCausalGraph`] when the effect is
    ///   Solve or Scan and the graph is provenance or execution.
    /// * [`OperatorError::CausalFeedbackNotPermitted`] when any other effect
    ///   claims causal feedback.
    /// * [`OperatorError::ReuseClaimsNewReadOrFit`] when frozen-fit reuse claims
    ///   a new observation read, consumes an outcome, or claims a new nuisance fit.
    /// * [`OperatorError::UndeclaredReadOrFit`] when some other effect claims a
    ///   read or a fit that is not that effect.
    /// * [`OperatorError::ReuseRequiresFrozenFit`] when the effect is frozen-fit
    ///   reuse and reuse is forbidden.
    /// * [`OperatorError::SilentObligationChange`] when an output role is not
    ///   owned by the effect.
    /// * [`OperatorError::FitMustClaimNuisanceFit`] when a fit declares
    ///   [`EvidenceClaims::none`].
    /// * [`OperatorError::FrozenFitReuseNotApplicable`] when reuse is allowed on an
    ///   effect that has no frozen fit.
    pub fn new(declaration: OperatorDeclaration) -> Result<Self, OperatorError> {
        let effect = declaration.effect;
        let graph = declaration.graph;
        if effect.may_carry_causal_feedback() && graph != OperatorGraph::Causal {
            return Err(OperatorError::FeedbackOperatorOffCausalGraph { effect, graph });
        }
        if !effect.may_carry_causal_feedback() && declaration.causal_feedback {
            return Err(OperatorError::CausalFeedbackNotPermitted { effect });
        }
        let new_nuisance_fit = declaration.claims.new_nuisance_fit();
        let new_observation_read = declaration.claims.new_observation_read()
            || (effect == OperatorEffect::ReuseFrozenFit
                && declaration.inputs.contains(&OperatorRole::Outcome));
        if (new_observation_read && effect != OperatorEffect::Read)
            || (new_nuisance_fit && effect != OperatorEffect::Fit)
        {
            if effect == OperatorEffect::ReuseFrozenFit {
                return Err(OperatorError::ReuseClaimsNewReadOrFit {
                    new_observation_read,
                    new_nuisance_fit,
                });
            }
            return Err(OperatorError::UndeclaredReadOrFit {
                effect,
                new_observation_read,
                new_nuisance_fit,
            });
        }
        if effect == OperatorEffect::ReuseFrozenFit
            && declaration.frozen_fit_reuse != FrozenFitReuse::Allowed
        {
            return Err(OperatorError::ReuseRequiresFrozenFit);
        }
        for role in declaration.outputs.iter() {
            if !effect.owns_output(*role) {
                return Err(OperatorError::SilentObligationChange { effect, role: *role });
            }
        }
        if effect == OperatorEffect::Fit && !new_nuisance_fit {
            return Err(OperatorError::FitMustClaimNuisanceFit);
        }
        if declaration.frozen_fit_reuse == FrozenFitReuse::Allowed
            && !effect.may_allow_frozen_fit_reuse()
        {
            return Err(OperatorError::FrozenFitReuseNotApplicable { effect });
        }
        Ok(Self {
            effect,
            graph,
            inputs: declaration.inputs,
            outputs: declaration.outputs,
            causal_feedback: declaration.causal_feedback,
            frozen_fit_reuse: declaration.frozen_fit_reuse,
        })
    }

    /// Effect of this edge.
    #[must_use]
    pub const fn effect(&self) -> OperatorEffect {
        self.effect
    }

    /// Graph this edge belongs to.
    #[must_use]
    pub const fn graph(&self) -> OperatorGraph {
        self.graph
    }

    /// Roles consumed, in declaration order.
    #[must_use]
    pub fn inputs(&self) -> &[OperatorRole] {
        &self.inputs
    }

    /// Roles produced, in declaration order.
    #[must_use]
    pub fn outputs(&self) -> &[OperatorRole] {
        &self.outputs
    }

    /// Whether this edge carries causal feedback.
    #[must_use]
    pub const fn causal_feedback(&self) -> bool {
        self.causal_feedback
    }

    /// Whether frozen-fit reuse is allowed.
    #[must_use]
    pub const fn frozen_fit_reuse(&self) -> FrozenFitReuse {
        self.frozen_fit_reuse
    }

    /// Whether [`Self::frozen_fit_reuse`] is [`FrozenFitReuse::Allowed`].
    #[must_use]
    pub const fn allows_frozen_fit_reuse(&self) -> bool {
        matches!(self.frozen_fit_reuse, FrozenFitReuse::Allowed)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EvidenceClaims, FrozenFitReuse, Operator, OperatorDeclaration, OperatorEffect,
        OperatorError, OperatorGraph, OperatorPorts, OperatorRole,
    };

    fn role_for(effect: OperatorEffect) -> OperatorRole {
        match effect {
            OperatorEffect::Read => OperatorRole::Outcome,
            OperatorEffect::Select => OperatorRole::Population,
            OperatorEffect::Fit | OperatorEffect::ReuseFrozenFit => OperatorRole::FitHandle,
            OperatorEffect::Prior => OperatorRole::Prior,
            OperatorEffect::Condition => OperatorRole::Conditioning,
            OperatorEffect::Intervene | OperatorEffect::Solve | OperatorEffect::Scan => {
                OperatorRole::WorldOrSchedule
            }
            OperatorEffect::Transport => OperatorRole::TransportedLaw,
        }
    }

    fn declaration(effect: OperatorEffect, graph: OperatorGraph) -> OperatorDeclaration {
        let role = role_for(effect);
        OperatorDeclaration::new(effect, graph, OperatorPorts::new([role], [role]))
    }

    #[test]
    fn acyclic_effects_declare_roles_without_causal_feedback() {
        let cases = [
            (OperatorEffect::Read, OperatorGraph::Provenance, FrozenFitReuse::Forbidden),
            (OperatorEffect::Select, OperatorGraph::Provenance, FrozenFitReuse::Forbidden),
            (OperatorEffect::Fit, OperatorGraph::Execution, FrozenFitReuse::Allowed),
            (OperatorEffect::Prior, OperatorGraph::Provenance, FrozenFitReuse::Forbidden),
            (OperatorEffect::Condition, OperatorGraph::Causal, FrozenFitReuse::Forbidden),
            (OperatorEffect::Intervene, OperatorGraph::Causal, FrozenFitReuse::Forbidden),
            (OperatorEffect::Transport, OperatorGraph::Causal, FrozenFitReuse::Allowed),
            (OperatorEffect::ReuseFrozenFit, OperatorGraph::Execution, FrozenFitReuse::Allowed),
        ];
        for (effect, graph, reuse) in cases {
            let built = declaration(effect, graph).with_frozen_fit_reuse(reuse);
            let built = match effect {
                OperatorEffect::Read => {
                    built.with_evidence_claims(EvidenceClaims::observation_read())
                }
                OperatorEffect::Fit => built.with_evidence_claims(EvidenceClaims::nuisance_fit()),
                _ => built,
            };
            let operator = Operator::new(built).unwrap();
            assert_eq!(operator.effect(), effect);
            assert_eq!(operator.graph(), graph);
            assert!(!operator.causal_feedback());
            assert_eq!(operator.frozen_fit_reuse(), reuse);
            assert_eq!(operator.allows_frozen_fit_reuse(), reuse == FrozenFitReuse::Allowed);
            assert_eq!(operator.inputs(), &[role_for(effect)]);
            assert_eq!(operator.outputs(), &[role_for(effect)]);
            assert!(!effect.may_carry_causal_feedback());
        }
    }

    #[test]
    fn solve_and_scan_may_carry_causal_feedback_on_the_causal_graph() {
        for effect in [OperatorEffect::Solve, OperatorEffect::Scan] {
            assert!(effect.may_carry_causal_feedback());
            for feedback in [false, true] {
                let operator = Operator::new(
                    declaration(effect, OperatorGraph::Causal).with_causal_feedback(feedback),
                )
                .unwrap();
                assert_eq!(operator.causal_feedback(), feedback);
                assert_eq!(operator.graph(), OperatorGraph::Causal);
                assert_eq!(operator.outputs(), &[OperatorRole::WorldOrSchedule]);
                assert!(!operator.allows_frozen_fit_reuse());
            }
        }
    }

    #[test]
    fn solve_and_scan_refuse_provenance_and_execution_graphs() {
        for effect in [OperatorEffect::Solve, OperatorEffect::Scan] {
            for graph in [OperatorGraph::Provenance, OperatorGraph::Execution] {
                for feedback in [false, true] {
                    let err =
                        Operator::new(declaration(effect, graph).with_causal_feedback(feedback))
                            .unwrap_err();
                    assert_eq!(
                        err,
                        OperatorError::FeedbackOperatorOffCausalGraph { effect, graph }
                    );
                }
            }
        }
    }

    #[test]
    fn effects_other_than_solve_and_scan_refuse_causal_feedback() {
        let effects = [
            OperatorEffect::Read,
            OperatorEffect::Select,
            OperatorEffect::Fit,
            OperatorEffect::Prior,
            OperatorEffect::Condition,
            OperatorEffect::Intervene,
            OperatorEffect::Transport,
            OperatorEffect::ReuseFrozenFit,
        ];
        for effect in effects {
            let graphs =
                [OperatorGraph::Causal, OperatorGraph::Provenance, OperatorGraph::Execution];
            for graph in graphs {
                let err = Operator::new(
                    declaration(effect, graph)
                        .with_causal_feedback(true)
                        .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
                )
                .unwrap_err();
                assert_eq!(err, OperatorError::CausalFeedbackNotPermitted { effect });
            }
        }
    }

    #[test]
    fn reuse_frozen_fit_refuses_a_new_observation_read_or_nuisance_fit() {
        let claims = [
            EvidenceClaims::observation_read(),
            EvidenceClaims::nuisance_fit(),
            EvidenceClaims::observation_read_and_nuisance_fit(),
        ];
        for claim in claims {
            let err = Operator::new(
                declaration(OperatorEffect::ReuseFrozenFit, OperatorGraph::Execution)
                    .with_frozen_fit_reuse(FrozenFitReuse::Allowed)
                    .with_evidence_claims(claim),
            )
            .unwrap_err();
            assert_eq!(
                err,
                OperatorError::ReuseClaimsNewReadOrFit {
                    new_observation_read: claim.new_observation_read(),
                    new_nuisance_fit: claim.new_nuisance_fit(),
                }
            );
        }
    }

    #[test]
    fn reuse_frozen_fit_refuses_an_outcome_input() {
        let ports = OperatorPorts::new(
            [OperatorRole::FitHandle, OperatorRole::Outcome],
            [OperatorRole::FitHandle],
        );
        let err = Operator::new(
            OperatorDeclaration::new(
                OperatorEffect::ReuseFrozenFit,
                OperatorGraph::Provenance,
                ports,
            )
            .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
        )
        .unwrap_err();
        assert_eq!(
            err,
            OperatorError::ReuseClaimsNewReadOrFit {
                new_observation_read: true,
                new_nuisance_fit: false,
            }
        );
    }

    #[test]
    fn reuse_frozen_fit_must_allow_frozen_fit_reuse() {
        let err =
            Operator::new(declaration(OperatorEffect::ReuseFrozenFit, OperatorGraph::Execution))
                .unwrap_err();
        assert_eq!(err, OperatorError::ReuseRequiresFrozenFit);
    }

    #[test]
    fn other_effects_cannot_smuggle_a_read_or_a_fit() {
        let err = Operator::new(
            declaration(OperatorEffect::Select, OperatorGraph::Provenance)
                .with_evidence_claims(EvidenceClaims::observation_read()),
        )
        .unwrap_err();
        assert_eq!(
            err,
            OperatorError::UndeclaredReadOrFit {
                effect: OperatorEffect::Select,
                new_observation_read: true,
                new_nuisance_fit: false,
            }
        );
        let err = Operator::new(
            declaration(OperatorEffect::Read, OperatorGraph::Provenance)
                .with_evidence_claims(EvidenceClaims::nuisance_fit()),
        )
        .unwrap_err();
        assert_eq!(
            err,
            OperatorError::UndeclaredReadOrFit {
                effect: OperatorEffect::Read,
                new_observation_read: false,
                new_nuisance_fit: true,
            }
        );
        assert_eq!(EvidenceClaims::none(), EvidenceClaims::default());
    }

    #[test]
    fn outputs_cannot_silently_change_population_history_or_obligations() {
        let cases = [
            (OperatorEffect::Read, OperatorRole::Population),
            (OperatorEffect::Read, OperatorRole::WorldOrSchedule),
            (OperatorEffect::Read, OperatorRole::Conditioning),
            (OperatorEffect::Fit, OperatorRole::Population),
            (OperatorEffect::Transport, OperatorRole::Population),
            (OperatorEffect::Select, OperatorRole::Outcome),
            (OperatorEffect::Condition, OperatorRole::Prior),
            (OperatorEffect::Intervene, OperatorRole::Population),
            (OperatorEffect::Solve, OperatorRole::Outcome),
            (OperatorEffect::Scan, OperatorRole::Conditioning),
            (OperatorEffect::ReuseFrozenFit, OperatorRole::Outcome),
            (OperatorEffect::Prior, OperatorRole::FitHandle),
        ];
        for (effect, role) in cases {
            let ports = OperatorPorts::new([role_for(effect)], [role]);
            let err = Operator::new(
                OperatorDeclaration::new(effect, OperatorGraph::Causal, ports)
                    .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
            )
            .unwrap_err();
            assert_eq!(err, OperatorError::SilentObligationChange { effect, role });
        }
    }

    #[test]
    fn only_the_owning_effect_may_produce_a_role() {
        assert!(OperatorEffect::Read.owns_output(OperatorRole::Outcome));
        assert!(OperatorEffect::Select.owns_output(OperatorRole::Population));
        assert!(OperatorEffect::Fit.owns_output(OperatorRole::FitHandle));
        assert!(OperatorEffect::ReuseFrozenFit.owns_output(OperatorRole::FitHandle));
        assert!(OperatorEffect::Prior.owns_output(OperatorRole::Prior));
        assert!(OperatorEffect::Condition.owns_output(OperatorRole::Conditioning));
        assert!(OperatorEffect::Intervene.owns_output(OperatorRole::WorldOrSchedule));
        assert!(OperatorEffect::Solve.owns_output(OperatorRole::WorldOrSchedule));
        assert!(OperatorEffect::Scan.owns_output(OperatorRole::WorldOrSchedule));
        assert!(OperatorEffect::Transport.owns_output(OperatorRole::TransportedLaw));
        assert!(!OperatorEffect::Transport.owns_output(OperatorRole::Population));
        assert!(!OperatorEffect::Solve.owns_output(OperatorRole::Outcome));
        assert!(!OperatorEffect::Fit.owns_output(OperatorRole::Outcome));
    }

    #[test]
    fn fit_may_read_population_and_conditioning_without_emitting_them() {
        let ports = OperatorPorts::new(
            [
                OperatorRole::Population,
                OperatorRole::Outcome,
                OperatorRole::Conditioning,
                OperatorRole::WorldOrSchedule,
            ],
            [OperatorRole::FitHandle],
        );
        assert_eq!(ports.inputs().len(), 4);
        assert_eq!(ports.outputs(), &[OperatorRole::FitHandle]);
        let operator = Operator::new(
            OperatorDeclaration::new(OperatorEffect::Fit, OperatorGraph::Execution, ports)
                .with_frozen_fit_reuse(FrozenFitReuse::Allowed)
                .with_evidence_claims(EvidenceClaims::nuisance_fit()),
        )
        .unwrap();
        assert_eq!(
            operator.inputs(),
            &[
                OperatorRole::Population,
                OperatorRole::Outcome,
                OperatorRole::Conditioning,
                OperatorRole::WorldOrSchedule,
            ]
        );
        assert_eq!(operator.outputs(), &[OperatorRole::FitHandle]);
        assert!(operator.allows_frozen_fit_reuse());
        assert!(!operator.causal_feedback());
        assert_eq!(operator.effect().as_str(), "fit");
        assert_eq!(operator.graph().as_str(), "execution");
    }

    #[test]
    fn a_fit_must_claim_a_nuisance_fit() {
        let err =
            Operator::new(declaration(OperatorEffect::Fit, OperatorGraph::Execution)).unwrap_err();
        assert_eq!(err, OperatorError::FitMustClaimNuisanceFit);
        let err = Operator::new(
            declaration(OperatorEffect::Fit, OperatorGraph::Execution)
                .with_evidence_claims(EvidenceClaims::observation_read()),
        )
        .unwrap_err();
        assert!(matches!(err, OperatorError::UndeclaredReadOrFit { .. }));
    }

    #[test]
    fn frozen_fit_reuse_is_refused_on_effects_with_no_frozen_fit() {
        let cases = [
            (OperatorEffect::Read, OperatorGraph::Provenance),
            (OperatorEffect::Select, OperatorGraph::Provenance),
            (OperatorEffect::Prior, OperatorGraph::Provenance),
            (OperatorEffect::Condition, OperatorGraph::Causal),
            (OperatorEffect::Intervene, OperatorGraph::Causal),
            (OperatorEffect::Solve, OperatorGraph::Causal),
            (OperatorEffect::Scan, OperatorGraph::Causal),
        ];
        for (effect, graph) in cases {
            assert!(!effect.may_allow_frozen_fit_reuse());
            let mut built =
                declaration(effect, graph).with_frozen_fit_reuse(FrozenFitReuse::Allowed);
            if effect == OperatorEffect::Read {
                built = built.with_evidence_claims(EvidenceClaims::observation_read());
            }
            assert_eq!(
                Operator::new(built).unwrap_err(),
                OperatorError::FrozenFitReuseNotApplicable { effect }
            );
        }
        for effect in
            [OperatorEffect::Fit, OperatorEffect::Transport, OperatorEffect::ReuseFrozenFit]
        {
            assert!(effect.may_allow_frozen_fit_reuse());
        }
    }

    #[test]
    fn names_match_display() {
        assert_eq!(OperatorEffect::ReuseFrozenFit.as_str(), "reuse_frozen_fit");
        assert_eq!(OperatorEffect::Scan.to_string(), "scan");
        assert_eq!(OperatorGraph::Provenance.to_string(), "provenance");
        assert_eq!(OperatorRole::WorldOrSchedule.to_string(), "world_or_schedule");
        assert_eq!(OperatorRole::TransportedLaw.to_string(), "transported_law");
        assert_eq!(FrozenFitReuse::Allowed.to_string(), "allowed");
        assert_eq!(OperatorRole::FitHandle.as_str(), "fit_handle");
    }
}
