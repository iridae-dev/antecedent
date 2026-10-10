//! 2.3 B2 (X10 second row): ordered-response recovery of a two-variable binary
//! m-graph, and its portable, independently consumable artifact.
//!
//! The row is exactly the graphs the 2.2 observation-recovery route refuses because
//! of one response chain `R_h -> R_t`. [`RecoveryChain::decide`] decides the graph:
//! a checked plan, or a self-censoring edge with an exactly verified witness (two
//! models that agree on the nine observed pattern cells and differ on the target).
//! A recovered plan is then evaluated on an exact nine-cell pattern law with
//! [`RecoveryChain::recover`]. Edges outside the row are `route_not_supported`, never
//! nonrecoverable; a missing cell is `transport_support_failure`. Exact laws only: a
//! sampled provider composed on this row is calibrated separately and is unmeasured
//! and closed.
//!
//! [`RecoveryChain::consume`] re-decides, re-evaluates and re-verifies the witness
//! from the exported artifact and refuses any resealed change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
pub use antecedent_estimate::{ChainPatternLaw, ChainRecoveredLaw, evaluate_chain_recovery};
use antecedent_graph::Admg;
pub use antecedent_identify::{
    CHAIN_RECOVERY_LIMITS, CHAIN_RECOVERY_RULE_VERSION, ChainPartial, ChainRecoveryDecision,
    ChainRecoveryDetail, ChainRecoveryError, ChainRecoveryPlan, ChainRecoveryQuery,
    ChainRecoveryWitness, ChainWitnessCheck, ChainWitnessMechanism, decide_chain_recovery,
    verify_chain_witness,
};
pub use antecedent_io::recovery_chain_artifact::{
    ChainGraphWire, ChainLawWire, ChainMechanismWire, ChainPlanWire, ChainQueryWire,
    ChainRecoveredWire, ChainWitnessCheckWire, ChainWitnessWire, ConsumedRecoveryChain,
    RECOVERY_CHAIN_ARTIFACT_FEATURE, RECOVERY_CHAIN_ARTIFACT_VERSION,
    RECOVERY_CHAIN_NONRECOVERABLE, RECOVERY_CHAIN_RECOVERED, RecoveryChainArtifactError,
    RecoveryChainArtifactWire,
};

use crate::error::CausalError;

/// A decided (and, when recovered, evaluated) ordered-response recovery.
#[derive(Clone, Debug)]
pub struct RecoveryChain {
    graph: Admg,
    query: ChainRecoveryQuery,
    variable_names: Vec<String>,
    decision: ChainRecoveryDecision,
    observed: Option<ChainPatternLaw>,
    recovered: Option<ChainRecoveredLaw>,
    check: Option<ChainWitnessCheck>,
}

impl RecoveryChain {
    /// Decide whether `P(X1, X2)` is recoverable on the m-graph. A nonrecoverable
    /// decision carries a witness that is verified here before it is returned.
    ///
    /// # Errors
    /// A typed [`ChainRecoveryError`]: invalid query, unsupported mechanism, an
    /// unverifiable witness or a budget stop with its receipt.
    pub fn decide(
        graph: &Admg,
        query: &ChainRecoveryQuery,
        variable_names: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Self, ChainRecoveryError> {
        let decision = decide_chain_recovery(graph, query, ctx)?;
        let check = match &decision {
            ChainRecoveryDecision::NonRecoverable(witness) => {
                Some(verify_chain_witness(graph, query, witness, ctx)?)
            }
            ChainRecoveryDecision::Recovered(_) => None,
        };
        Ok(Self {
            graph: graph.clone(),
            query: *query,
            variable_names: variable_names.to_vec(),
            decision,
            observed: None,
            recovered: None,
            check,
        })
    }

    /// Evaluate the checked recovery formula on an exact observed pattern law.
    ///
    /// # Errors
    /// `recovery_chain.nonrecoverable_witness` when the decision is nonrecoverable;
    /// `recovery_chain.positivity` when a required cell or denominator has no mass.
    pub fn recover(mut self, observed: ChainPatternLaw) -> Result<Self, ChainRecoveryError> {
        let ChainRecoveryDecision::Recovered(plan) = &self.decision else {
            return Err(ChainRecoveryError::new(
                ChainRecoveryDetail::NonrecoverableWitness,
                "a self-censoring edge is not recoverable for every model Markov to the m-graph; see the witness",
            ));
        };
        let recovered = evaluate_chain_recovery(plan, &observed)?;
        self.observed = Some(observed);
        self.recovered = Some(recovered);
        Ok(self)
    }

    /// Consume an exported artifact by re-deciding, re-evaluating and re-verifying.
    ///
    /// # Errors
    /// Corruption, an unknown version, a changed digest, or a stored decision or law
    /// that does not replay.
    pub fn consume(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<Self, RecoveryChainArtifactError> {
        let consumed = RecoveryChainArtifactWire::consume_typed(bytes, ctx)?;
        let graph = consumed
            .wire
            .graph
            .to_graph()
            .map_err(|error| RecoveryChainArtifactError::Undecodable(error.to_string()))?;
        let observed = consumed.wire.observed.as_ref().map(ChainLawWire::to_law).transpose()?;
        Ok(Self {
            graph,
            query: consumed.wire.query.to_query(),
            variable_names: consumed.wire.variable_names.clone(),
            decision: consumed.decision,
            observed,
            recovered: consumed.recovered,
            check: consumed.check,
        })
    }

    /// Export the decision as an artifact.
    ///
    /// # Errors
    /// A recovered decision that has not been evaluated with [`Self::recover`], or an
    /// encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, CausalError> {
        let wire = match (&self.decision, &self.observed, &self.recovered, &self.check) {
            (ChainRecoveryDecision::Recovered(plan), Some(observed), Some(recovered), _) => {
                RecoveryChainArtifactWire::from_recovered(
                    &self.graph,
                    &self.query,
                    &self.variable_names,
                    plan,
                    observed,
                    recovered,
                )
            }
            (ChainRecoveryDecision::NonRecoverable(witness), _, _, Some(check)) => {
                RecoveryChainArtifactWire::from_nonrecoverable(
                    &self.graph,
                    &self.query,
                    &self.variable_names,
                    witness,
                    check,
                )
            }
            _ => {
                return Err(CausalError::Compile {
                    message: format!(
                        "{}{}: recovery_chain.not_evaluated: evaluate the recovered decision on \
                         an observed pattern law before exporting",
                        crate::error::REASON_PREFIX,
                        antecedent_core::reason_code!("not_executed")
                    ),
                });
            }
        }
        .map_err(CausalError::Serialization)?;
        wire.export().map_err(CausalError::Serialization)
    }

    /// The decision.
    #[must_use]
    pub fn decision(&self) -> &ChainRecoveryDecision {
        &self.decision
    }

    /// The observed pattern law, once evaluated or consumed.
    #[must_use]
    pub fn observed(&self) -> Option<&ChainPatternLaw> {
        self.observed.as_ref()
    }

    /// The recovered law `P(X1, X2)`, once evaluated or consumed.
    #[must_use]
    pub fn recovered(&self) -> Option<&ChainRecoveredLaw> {
        self.recovered.as_ref()
    }

    /// The verified witness check of a nonrecoverable decision.
    #[must_use]
    pub fn witness_check(&self) -> Option<&ChainWitnessCheck> {
        self.check.as_ref()
    }

    /// The query roles.
    #[must_use]
    pub fn query(&self) -> &ChainRecoveryQuery {
        &self.query
    }

    /// Variable name of every graph node in dense order, or empty.
    #[must_use]
    pub fn variable_names(&self) -> &[String] {
        &self.variable_names
    }
}

impl From<ChainRecoveryError> for CausalError {
    fn from(error: ChainRecoveryError) -> Self {
        CausalError::Compile {
            message: format!("{}{}: {}", crate::error::REASON_PREFIX, error.reason_code(), error),
        }
    }
}

impl From<RecoveryChainArtifactError> for CausalError {
    fn from(error: RecoveryChainArtifactError) -> Self {
        let (code, detail) = error.refusal();
        CausalError::Compile {
            message: format!("{}{code}: {detail}: {error}", crate::error::REASON_PREFIX),
        }
    }
}
