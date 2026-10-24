//! Checked scientific standing carried from a compiled response into a decision.
//!
//! Numerical mean/law engines remain reusable components. Native compiled workflows use
//! this wrapper so current causal consumption cannot discard model-conditional standing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decision_contract::DecisionContract;
use crate::decision_eval::{
    DecisionEvalError, DecisionResult, MeanSource, evaluate_contract, evaluate_contract_on_means,
};
use crate::program_search::{
    FiniteProgramSearchResult, PreparedFiniteProgramSearch, ProgramSearchError, ProgramSearchSource,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CanonicalWriter, ScientificTarget, ScientificWorldId, SemanticDigest, TargetForm, VariableId,
};
use antecedent_identify::CompiledIdentificationProduct;
use antecedent_io_core::distribution_artifact::DistributionArtifact;
use std::sync::Arc;

/// Scientific authorization requested by a downstream consumer, separate from numerical risk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScientificConsumptionScope {
    /// Evaluate only inside the explicitly retained model and its assumptions.
    ModelConditional,
    /// Require an identified target and every retained world/assembled obligation.
    IdentifiedCausal,
}
/// Explicit numerical carrier; creating one does not issue causal authorization.
#[derive(Clone, Debug)]
pub enum ScientificSourcePayload {
    /// Finite means of the source's scientific coordinates.
    Means(MeanSource),
    /// One aligned parameter/function law with its original uncertainty meaning.
    Law(Box<DistributionArtifact>),
}
/// Why a response cannot authorize the requested scientific consumption.
#[derive(Debug, thiserror::Error)]
pub enum ScientificSourceError {
    /// Scientific binding or causal authorization failed before numerical consumption.
    #[error("scientific source refused: {0}")]
    Authorization(String),
    /// Original typed numerical decision refusal.
    #[error("decision evaluation refused: {0:?}")]
    Decision(Box<DecisionEvalError>),
    /// Original search refusal, including resource exhaustion and cancellation.
    #[error(transparent)]
    Search(#[from] ProgramSearchError),
}
fn refused(error: impl std::fmt::Debug) -> ScientificSourceError {
    ScientificSourceError::Authorization(format!("{error:?}"))
}

/// One mean coordinate of an actual world in the retained scientific basis.
#[derive(Clone, Debug)]
pub struct ScientificProjectionBasis {
    /// Exact world key in the source target.
    pub world: ScientificWorldId,
    /// Outcome variable represented by this coordinate.
    pub variable: VariableId,
    /// Portable numerical coordinate, bound to the same world and time.
    pub quantity: ScientificQuantity,
}
/// A sparse linear output over the retained world-mean basis.
#[derive(Clone, Debug)]
pub struct ScientificProjectionOutput {
    /// Exact output coordinate in the numerical carrier.
    pub quantity: ScientificQuantity,
    /// Unique basis indices and finite nonzero coefficients, in canonical index order.
    pub terms: Vec<(usize, f64)>,
}
/// Typed projection lineage: intermediate times remain actual prefix worlds, not terminal proofs.
#[derive(Clone, Debug)]
pub struct ScientificSourceProjection {
    /// Each world appears exactly once, in numerical basis order.
    pub basis: Vec<ScientificProjectionBasis>,
    /// Sparse output rows in payload coordinate order.
    pub outputs: Vec<ScientificProjectionOutput>,
}
impl ScientificSourceProjection {
    /// Complete question identity, including actual target, aliases and sparse coefficients.
    #[must_use]
    pub fn identity(&self, target: &ScientificTarget) -> SemanticDigest {
        let mut writer = CanonicalWriter::new("antecedent.scientific_source.projection.v1");
        writer.bytes(target.identity().as_bytes()).word(self.basis.len() as u64);
        for basis in &self.basis {
            writer.word(u64::from(basis.world.raw())).word(u64::from(basis.variable.raw()));
            basis.quantity.write_canonical(&mut writer);
        }
        writer.word(self.outputs.len() as u64);
        for output in &self.outputs {
            output.quantity.write_canonical(&mut writer);
            writer.word(output.terms.len() as u64);
            for &(index, coefficient) in &output.terms {
                writer.word(index as u64).word(coefficient.to_bits());
            }
        }
        SemanticDigest::from_bytes(writer.finish())
    }
    fn validate(
        &self,
        target: &ScientificTarget,
        payload: &ScientificSourcePayload,
    ) -> Result<(), ScientificSourceError> {
        let worlds = match target.form() {
            TargetForm::World(world) => std::slice::from_ref(world),
            TargetForm::Worlds(worlds) => worlds.worlds(),
            TargetForm::Contrast(contrast) => contrast.worlds().worlds(),
            TargetForm::DoseTime(surface) => surface.worlds().worlds(),
            _ => return Err(refused("projection needs explicit world basis")),
        };
        if worlds.len() != self.basis.len() || self.outputs.is_empty() {
            return Err(refused("projection must cover the exact target world basis"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for basis in &self.basis {
            let world = worlds
                .iter()
                .find(|world| world.id() == basis.world)
                .ok_or_else(|| refused("foreign projection basis world"))?;
            let outcome = world.binding().outcome();
            let quantity = &basis.quantity;
            if !seen.insert(basis.world)
                || basis.variable != outcome.outcome()
                || quantity.role != QuantityRole::Outcome
                || quantity.functional_id != "mean"
                || quantity.units != outcome.units().name()
                || quantity.horizon != world.time().horizon_steps().unwrap_or(0)
                || quantity.regime_id
                    != format!("compiled_world:{}:{}", target.identity(), basis.world.raw())
            {
                return Err(refused(
                    "projection basis differs from actual world outcome, horizon or units",
                ));
            }
        }
        let quantities = match payload {
            ScientificSourcePayload::Means(means) => means.coordinates.clone(),
            ScientificSourcePayload::Law(law) => law
                .quantities()
                .iter()
                .cloned()
                .map(ScientificQuantity::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(refused)?,
        };
        if quantities.len() != self.outputs.len() {
            return Err(refused("projection output width differs from payload"));
        }
        for (output, quantity) in self.outputs.iter().zip(&quantities) {
            if output.quantity != *quantity || output.terms.is_empty() {
                return Err(refused("projection output differs from payload coordinate"));
            }
            let mut previous = None;
            for &(index, coefficient) in &output.terms {
                if index >= self.basis.len()
                    || previous.is_some_and(|previous| previous >= index)
                    || !coefficient.is_finite()
                    || coefficient == 0.0
                {
                    return Err(refused("invalid sparse projection coefficient or basis index"));
                }
                let input = &self.basis[index].quantity;
                if quantity.role != input.role
                    || quantity.variable_id != input.variable_id
                    || quantity.units != input.units
                    || quantity.population_id != input.population_id
                    || quantity.conditioning != input.conditioning
                    || quantity.transform_id != input.transform_id
                {
                    return Err(refused(
                        "projected output changes outcome, population, units or transformation",
                    ));
                }
                previous = Some(index);
            }
            let horizon = output
                .terms
                .iter()
                .map(|(index, _)| self.basis[*index].quantity.horizon)
                .max()
                .unwrap_or(0);
            if quantity.horizon != horizon {
                return Err(refused("projected output horizon differs from its actual support"));
            }
        }
        Ok(())
    }
}

/// A numerical source inseparable from its target, retained proof and dependency identity.
#[derive(Clone, Debug)]
pub struct ScientificDecisionSource {
    target: Arc<ScientificTarget>,
    identification: Arc<CompiledIdentificationProduct>,
    dependency: SemanticDigest,
    payload: Arc<ScientificSourcePayload>,
    projection: Option<Arc<ScientificSourceProjection>>,
    retention: Option<Arc<[Arc<antecedent_core::execution::AdmissionLease>]>>,
}
/// An actual decision result with the scientific scope that authorized its consumption.
#[derive(Clone, Debug)]
pub struct ScopedDecisionResult {
    /// Actual evaluated decision, without an additional calibration license.
    pub result: DecisionResult,
    /// Explicit requested scientific standing; conditional use never becomes causal use.
    pub scope: ScientificConsumptionScope,
    /// Full scientific question identity.
    pub target_identity: SemanticDigest,
    /// Actual retained identification/support product identity.
    pub identification_identity: SemanticDigest,
    /// Fitting/source dependency identity, retained through downstream utility changes.
    pub dependency_identity: SemanticDigest,
    /// Exact derived question, including any temporal projection.
    pub question_identity: SemanticDigest,
    /// Sparse scientific projection retained through downstream consumption.
    pub projection: Option<Arc<ScientificSourceProjection>>,
    _retention: Option<Arc<[Arc<antecedent_core::execution::AdmissionLease>]>>,
}
/// An actual finite search result carrying the scientific scope of its response source.
#[derive(Clone, Debug)]
pub struct ScopedProgramSearchResult {
    /// Exhaustion is scoped to the finite candidate set and retained numerical source.
    pub result: FiniteProgramSearchResult,
    /// Explicit scientific standing, separate from selection reliability.
    pub scope: ScientificConsumptionScope,
    /// Complete source question identity.
    pub target_identity: SemanticDigest,
    /// Source identification and support identity.
    pub identification_identity: SemanticDigest,
    /// Actual fitting/source dependency identity.
    pub dependency_identity: SemanticDigest,
    /// Exact derived question, including any temporal projection.
    pub question_identity: SemanticDigest,
    /// Sparse scientific projection retained through downstream consumption.
    pub projection: Option<Arc<ScientificSourceProjection>>,
    _retention: Option<Arc<[Arc<antecedent_core::execution::AdmissionLease>]>>,
}
impl ScientificDecisionSource {
    /// Bind an original numerical carrier to its checked compiler product.
    ///
    /// # Errors
    /// A changed or unbound scientific target. Posterior meaning does not establish identification.
    pub fn new(
        target: ScientificTarget,
        identification: CompiledIdentificationProduct,
        dependency: SemanticDigest,
        payload: ScientificSourcePayload,
    ) -> Result<Self, ScientificSourceError> {
        Self::new_shared(Arc::new(target), Arc::new(identification), dependency, payload)
    }
    /// Share an existing compiled scientific owner without copying target/proof metadata.
    /// The caller retains the admissions covering shared metadata and the owned payload.
    /// # Errors
    /// The retained proof does not bind the supplied target.
    pub fn new_shared(
        target: Arc<ScientificTarget>,
        identification: Arc<CompiledIdentificationProduct>,
        dependency: SemanticDigest,
        payload: ScientificSourcePayload,
    ) -> Result<Self, ScientificSourceError> {
        identification.validate_target(&target).map_err(refused)?;
        Ok(Self {
            target,
            identification,
            dependency,
            payload: Arc::new(payload),
            projection: None,
            retention: None,
        })
    }
    /// Bind a derived numerical carrier to its exact world basis and sparse functional.
    /// # Errors
    /// Changed proof, basis, units, horizons, coefficient structure or payload coordinates.
    pub fn new_projected(
        target: ScientificTarget,
        identification: CompiledIdentificationProduct,
        dependency: SemanticDigest,
        payload: ScientificSourcePayload,
        projection: ScientificSourceProjection,
    ) -> Result<Self, ScientificSourceError> {
        projection.validate(&target, &payload)?;
        let mut source = Self::new(target, identification, dependency, payload)?;
        source.projection = Some(Arc::new(projection));
        Ok(source)
    }
    /// Retain the admission covering the source's owned projection and numerical allocations.
    #[must_use]
    pub fn retain_admission(
        mut self,
        lease: Arc<antecedent_core::execution::AdmissionLease>,
    ) -> Self {
        let leases = self
            .retention
            .iter()
            .flat_map(|leases| leases.iter())
            .cloned()
            .chain(std::iter::once(lease))
            .collect::<Vec<_>>();
        self.retention = Some(leases.into());
        self
    }
    /// Exact sparse lineage for a derived source, when present.
    #[must_use]
    pub fn projection(&self) -> Option<&ScientificSourceProjection> {
        self.projection.as_deref()
    }
    /// Identity of the actual derived question rather than only its world basis.
    #[must_use]
    pub fn question_identity(&self) -> SemanticDigest {
        self.projection
            .as_ref()
            .map_or_else(|| self.target.identity(), |projection| projection.identity(&self.target))
    }
    /// The original target; inference selection remains separate from scientific equality.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Original world/assembled identification and support obligations.
    pub fn identification(&self) -> &CompiledIdentificationProduct {
        &self.identification
    }
    /// Actual source and fitting dependency identity.
    #[must_use]
    pub const fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// Explicitly extract the numerical component, carrying no current causal authorization.
    #[must_use]
    pub fn as_numerical_source(&self) -> &ScientificSourcePayload {
        &self.payload
    }
    /// Check current scientific use before any decision or search is evaluated.
    ///
    /// # Errors
    /// A causal request with unsupported/model-conditional worlds or an invalid target.
    pub fn authorize(
        &self,
        scope: ScientificConsumptionScope,
    ) -> Result<(), ScientificSourceError> {
        if let Some(leases) = &self.retention {
            for lease in leases.iter() {
                lease.observe_current();
            }
        }
        self.identification.validate_target(&self.target).map_err(refused)?;
        if scope == ScientificConsumptionScope::IdentifiedCausal
            && !self.identification.is_identified_claim()
        {
            return Err(refused(
                "the compiled design does not identify this target; explicitly conditional consumption grants no causal permission",
            ));
        }
        Ok(())
    }
    /// Delegate the original numerical decision engine while preserving scientific standing.
    ///
    /// # Errors
    /// Scientific authorization or original quantity, source or decision-contract refusal.
    pub fn evaluate(
        &self,
        contract: &DecisionContract,
        scope: ScientificConsumptionScope,
    ) -> Result<ScopedDecisionResult, ScientificSourceError> {
        self.authorize(scope)?;
        let result = match self.payload.as_ref() {
            ScientificSourcePayload::Means(source) => evaluate_contract_on_means(contract, source),
            ScientificSourcePayload::Law(source) => evaluate_contract(contract, source),
        }
        .map_err(|error| ScientificSourceError::Decision(Box::new(error)))?;
        Ok(self.scoped(result, scope))
    }
    /// [`Self::evaluate`] through the demand executor: a stored law is read in batches and
    /// its per-action utilities are held under `ctx`'s admission pool, and a mean source
    /// answers an affine demand without a draw. Cancellation of `ctx` stops the draws.
    ///
    /// # Errors
    /// As [`Self::evaluate`], and a refusal naming the memory budget when the pool cannot
    /// hold the evaluation.
    pub fn evaluate_admitted(
        &self,
        contract: &DecisionContract,
        scope: ScientificConsumptionScope,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<ScopedDecisionResult, ScientificSourceError> {
        self.authorize(scope)?;
        let result = match self.payload.as_ref() {
            ScientificSourcePayload::Means(source) => {
                crate::decision_evaluation::evaluate_mean_source_admitted(contract, source, ctx)
            }
            ScientificSourcePayload::Law(source) => {
                crate::decision_evaluation::evaluate_artifact_admitted(contract, source, ctx)
            }
        }
        .map_err(|error| ScientificSourceError::Decision(Box::new(error)))?;
        Ok(self.scoped(result, scope))
    }
    fn scoped(
        &self,
        mut result: DecisionResult,
        scope: ScientificConsumptionScope,
    ) -> ScopedDecisionResult {
        result.assumptions.push(format!(
            "scientific_scope={scope:?}; target={}; identification={}; dependency={}",
            self.target.identity(),
            self.identification.identity(),
            self.dependency
        ));
        ScopedDecisionResult {
            result,
            scope,
            target_identity: self.target.identity(),
            identification_identity: self.identification.identity(),
            dependency_identity: self.dependency,
            question_identity: self.question_identity(),
            projection: self.projection.clone(),
            _retention: self.retention.clone(),
        }
    }
    /// Execute the existing finite-search owner under an explicit scientific scope.
    ///
    /// # Errors
    /// Scientific authorization, source demand, operational, resource or search failure.
    pub fn search(
        &self,
        search: &PreparedFiniteProgramSearch,
        scope: ScientificConsumptionScope,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<ScopedProgramSearchResult, ScientificSourceError> {
        self.authorize(scope)?;
        let result = match self.payload.as_ref() {
            ScientificSourcePayload::Means(source) => {
                search.execute(ProgramSearchSource::Means(source), ctx)
            }
            ScientificSourcePayload::Law(source) => {
                search.execute(ProgramSearchSource::Law(source), ctx)
            }
        }
        .map_err(ScientificSourceError::Search)?;
        Ok(ScopedProgramSearchResult {
            result,
            scope,
            target_identity: self.target.identity(),
            identification_identity: self.identification.identity(),
            dependency_identity: self.dependency,
            question_identity: self.question_identity(),
            projection: self.projection.clone(),
            _retention: self.retention.clone(),
        })
    }
}
