//! Prepared exact binary observation recovery (2.2B X10): graph-licensed
//! recovery, not MAR/IPCW.
//!
//! Preparation decides once (class check, checked recovery formula or verified
//! nonrecoverability witness, and the downstream effect by ordinary target ID,
//! all under one budget) and validates the named observed pattern law.
//! Estimation evaluates the retained derivation and never decides again. Refresh
//! replaces the observed law only for the snapshot the frozen catalog binds.
//! The route is point-only: exact laws carry no sampling uncertainty.
use super::StudyBuilder;
use antecedent_core::{EvidenceCatalog, ExecutionContext};
use antecedent_estimate::{RecoveredLaw, evaluate_exact_recovery, evaluate_recovered_effect};
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits};
use antecedent_graph::Admg;
use antecedent_identify::{
    ObservationRecoveryQuery, RecoveredEffectQuery, RecoveryDecision, RecoveryDerivation,
    RecoveryDetail, RecoveryError, RecoveryLimits, decide_observation_recovery,
};
use antecedent_io::IoError;
use antecedent_io::recovery_artifact::{
    RecoveryArtifactInput, RecoveryArtifactWire, RecoveryConsumeLimits, recovery_io_error,
};

/// A prepared recovery: frozen derivation, catalog, observed law and requests.
#[derive(Clone, Debug)]
pub struct PreparedObservationRecovery {
    graph: Admg,
    catalog: EvidenceCatalog,
    effect: Option<RecoveredEffectQuery>,
    derivation: RecoveryDerivation,
    observed: ExactDiscreteLaw,
    requests: Vec<Assignment>,
    limits: ExactEvaluationLimits,
}

/// One execution: the recovered law and the effect point of every request.
#[derive(Clone, Debug)]
pub struct ObservationRecoveryResult {
    recovered: RecoveredLaw,
    effects: Vec<ExactDistribution>,
}

impl ObservationRecoveryResult {
    /// The recovered full law `P(X(1), O)` with its provenance.
    #[must_use]
    pub const fn recovered(&self) -> &RecoveredLaw {
        &self.recovered
    }

    /// The effect point of every request, in request order.
    #[must_use]
    pub fn effects(&self) -> &[ExactDistribution] {
        &self.effects
    }

    /// Export this result with its derivation, observed law and requests for an
    /// independent consumer.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn export(&self, prepared: &PreparedObservationRecovery) -> Result<Vec<u8>, IoError> {
        self.export_named(prepared, &[])
    }

    /// [`Self::export`], binding the name of every m-graph node into the verified
    /// identity so a consumer can refuse relabelled names.
    ///
    /// # Errors
    /// As [`Self::export`], or names that do not cover the m-graph.
    pub fn export_named(
        &self,
        prepared: &PreparedObservationRecovery,
        variable_names: &[String],
    ) -> Result<Vec<u8>, IoError> {
        RecoveryArtifactWire::checked(&RecoveryArtifactInput {
            graph: &prepared.graph,
            effect: prepared.effect.as_ref(),
            derivation: &prepared.derivation,
            catalog: &prepared.catalog,
            observed: &prepared.observed,
            recovered: &self.recovered,
            requests: &prepared.requests,
            effects: &self.effects,
            limits: prepared.limits,
            variable_names,
        })?
        .export()
    }
}

/// Independently consume a recovery artifact: re-decide under the stored limits,
/// re-check the formula, recompute the recovered law and every effect point bit
/// for bit, without fetching data.
///
/// # Errors
/// Any reconstruction or replay failure, with the X10 reason code.
pub fn consume_observation_recovery_artifact(
    bytes: &[u8],
    limits: RecoveryConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ObservationRecoveryResult, IoError> {
    let consumed = RecoveryArtifactWire::consume_with_limits(bytes, limits, ctx)?;
    Ok(ObservationRecoveryResult { recovered: consumed.recovered, effects: consumed.effects })
}

fn refused(error: &RecoveryError) -> IoError {
    recovery_io_error(error)
}

impl StudyBuilder {
    /// Decide recovery once and prepare exact evaluation of the recovered law and,
    /// when `effect` is given, of its effect at every request.
    ///
    /// # Errors
    /// Any decision refusal (a nonrecoverable decision refuses with
    /// `transport_proven_non_transportable`), requests without an effect, or an
    /// observed law that is not the named one.
    #[allow(clippy::too_many_arguments)]
    pub fn observation_recovery(
        graph: Admg,
        query: &ObservationRecoveryQuery,
        catalog: EvidenceCatalog,
        effect: Option<RecoveredEffectQuery>,
        decision: RecoveryLimits,
        observed: ExactDiscreteLaw,
        requests: Vec<Assignment>,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedObservationRecovery, IoError> {
        if effect.is_none() && !requests.is_empty() {
            return Err(refused(&RecoveryError::new(
                RecoveryDetail::InvalidQuery,
                "effect requests need a downstream effect query",
            )));
        }
        let derivation = match decide_observation_recovery(
            &graph,
            query,
            &catalog,
            effect.as_ref(),
            decision,
            ctx,
        )
        .map_err(|e| refused(&e))?
        {
            RecoveryDecision::Recovered(derivation) => *derivation,
            RecoveryDecision::NonRecoverable(witness) => {
                return Err(refused(&RecoveryError::new(
                    RecoveryDetail::NonrecoverableWitness,
                    format!(
                        "self-censoring edge {:?} -> {:?}: a verified witness shows the target is not recoverable",
                        witness.edge.0, witness.edge.1
                    ),
                )));
            }
        };
        evaluate_exact_recovery(&derivation, &observed, ctx).map_err(|e| refused(&e))?;
        Ok(PreparedObservationRecovery {
            graph,
            catalog,
            effect,
            derivation,
            observed,
            requests,
            limits,
        })
    }
}

impl PreparedObservationRecovery {
    /// Evaluate the retained derivation: the recovered law, then every effect
    /// request. Never decides again. Exact laws are point-only.
    ///
    /// # Errors
    /// Cancellation or an evaluation refusal.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<ObservationRecoveryResult, IoError> {
        let recovered = evaluate_exact_recovery(&self.derivation, &self.observed, ctx)
            .map_err(|e| refused(&e))?;
        let effects = self
            .requests
            .iter()
            .map(|request| {
                evaluate_recovered_effect(
                    &self.derivation,
                    &recovered,
                    request.clone(),
                    self.limits,
                    ctx,
                )
                .map_err(|e| refused(&e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ObservationRecoveryResult { recovered, effects })
    }

    /// Replace the observed pattern law under the snapshot the frozen catalog
    /// binds; the derivation and requests are unchanged. A law of another
    /// population, regime or snapshot is refused and needs a new preparation.
    ///
    /// # Errors
    /// An observed law outside the frozen contract.
    pub fn refresh(
        &self,
        observed: ExactDiscreteLaw,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        evaluate_exact_recovery(&self.derivation, &observed, ctx).map_err(|e| refused(&e))?;
        Ok(Self { observed, ..self.clone() })
    }

    /// The frozen derivation.
    #[must_use]
    pub const fn derivation(&self) -> &RecoveryDerivation {
        &self.derivation
    }

    /// The retained observed pattern law.
    #[must_use]
    pub const fn observed(&self) -> &ExactDiscreteLaw {
        &self.observed
    }

    /// The retained effect requests.
    #[must_use]
    pub fn requests(&self) -> &[Assignment] {
        &self.requests
    }
}
