//! Exact reuse of an independently retained static common-adjustment proof.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    JointPanelDesignAttempt, JointPanelDesignOutcome, JointPanelError, JointPanelState,
    JointPanelWorldSupport, contracts, source,
};
use crate::{PreparedJointResponse, PreparedStudy};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    CanonicalWriter, ExecutionContext, GraphFamily, IdentificationStatus, ModelAssumptions,
    ScientificTarget, SemanticDigest, TargetForm,
};
use antecedent_data::{TableView, TabularData};
use antecedent_identify::compiled_product::IdentificationObligation;
use antecedent_identify::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome,
};
use std::sync::Arc;

impl PreparedStudy<JointPanelState> {
    /// Bind an existing independent nonparametric static joint-adjustment proof.
    ///
    /// The proof's original additive Gaussian numerical restriction is never transferred.
    /// Its underlying graph derivation must stand independently. Exact training rows,
    /// supported interventions and the empirical reference law are checked before the
    /// row-to-balanced-group reindexing is accepted. No builder, identification or fit runs.
    /// Nonlinear numerical fitting and uncertainty retain their own conditional/unmeasured meaning.
    ///
    /// # Errors
    /// Refuses temporal targets, changed source data, unsupported worlds, parametric-only
    /// identification, incomplete rows and a different group/covariate reference law.
    pub fn with_static_source_proof(
        &self,
        proof: PreparedJointResponse,
        proof_data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        source::check_cancelled(ctx)?;
        crate::analysis::prepared::joint_target_admission(&self.request().target)
            .map_err(JointPanelError::compile)?;
        if self.request().lag_order != 0 || !proof.identification().is_identified_claim() {
            return Err(JointPanelError::Unsupported(
                "static panel binding requires L=0 and independently supported static worlds"
                    .into(),
            ));
        }
        validate_static_source(self, &proof, proof_data)?;
        let original = proof.source_identification();
        let current_worlds = contracts::target_worlds(&self.request().target)?;
        let graph = match proof.source_graph().graph_ref() {
            crate::accepted::GraphRef::Dag(graph) => (
                GraphFamily::Dag,
                antecedent_io::to_cbor(
                    &antecedent_io::dag_to_wire(graph).map_err(JointPanelError::compile)?,
                )
                .map_err(JointPanelError::compile)?,
            ),
            crate::accepted::GraphRef::Admg(graph) => (
                GraphFamily::Admg,
                antecedent_io::to_cbor(
                    &antecedent_io::admg_to_wire(graph).map_err(JointPanelError::compile)?,
                )
                .map_err(JointPanelError::compile)?,
            ),
            _ => {
                return Err(JointPanelError::Unsupported(
                    "static source has no accepted point graph".into(),
                ));
            }
        };
        let mut writer = CanonicalWriter::new("antecedent.panel.static_source_binding.v1");
        writer
            .bytes(&graph.1)
            .bytes(&proof.source_digest())
            .bytes(proof.identification().identity().as_bytes());
        let binding = SemanticDigest::from_bytes(writer.finish());
        let mut assumptions = self.request().target.assumptions().records().clone();
        assumptions.extend_unique(&original.required_assumptions.entries);
        assumptions.push(AssumptionRecord{assumption:Assumption::Custom{id:"panel.static_empirical_source_identity".into(),description:binding.to_hex().into()},source:AssumptionSource::Derived{from:"retained static common-adjustment proof with exact balanced empirical row/group law".into()},scope:AssumptionScope::Identification,status:AssumptionStatus::Supported});
        let target = ScientificTarget::new(
            self.request().target.form().clone(),
            ModelAssumptions::new(graph.0, assumptions.clone()),
            self.request().target.inference(),
        );
        let mut retained = original.clone();
        retained.required_assumptions = assumptions;
        retained.derivation.push("compiler.empirical_law_identity","exact observed rows, balanced groups, fixed group-specific covariate contexts and supported static interventions bind the existing nonparametric proof to the same empirical target");
        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            IdentificationClaimClass::IdentifiedClaim,
        );
        parts.required_factors = proof.identification().required_factors().to_vec();
        parts.world_obligations = current_worlds
            .iter()
            .map(|world| {
                IdentificationObligation::for_world(
                    world.id(),
                    IdentificationOutcome::Identified,
                    IdentificationClaimClass::IdentifiedClaim,
                    vec![],
                    None,
                )
            })
            .collect();
        if matches!(target.form(), TargetForm::Contrast(_) | TargetForm::DoseTime(_)) {
            parts.assembled = Some(IdentificationObligation::assembled(
                IdentificationOutcome::Identified,
                IdentificationClaimClass::IdentifiedClaim,
                vec![],
                None,
            ));
        }
        let identification =
            CompiledIdentificationProduct::retain_for_target(&retained, parts, &target)
                .map_err(JointPanelError::compile)?;
        let mut next = self.clone();
        let mut request = self.request().clone();
        request.target = target;
        next.state.request = Arc::new(request);
        next.state.identification = Arc::new(identification);
        let mut attempts = self.state.attempts.to_vec();
        attempts.insert(1,JointPanelDesignAttempt{id:"joint_adjustment.static_empirical_source".into(),outcome:JointPanelDesignOutcome::Identified,detail:format!("independent nonparametric source proof {}, exact graph/data/empirical-law binding {binding}; no Gaussian numerical restriction or calibration transferred",proof.identification().identity()),next_attempt:None,target_identity:next.request().target.identity(),worlds:next.request().worlds.iter().map(|world|JointPanelWorldSupport{world:world.world,matched_groups:next.model_data().groups.iter().map(|group|group.id.clone()).collect(),identified:true,reason:"exact supported static intervention and empirical reference law".into()}).collect(),proof:None});
        for attempt in &mut attempts {
            if attempt.id == "joint_adjustment.observational" {
                attempt.detail = "temporal joint adjustment was not needed: with no carryover the retained static source proof identifies the worlds; the attempt names no certificate of its own".into();
                attempt.next_attempt = None;
            }
        }
        next.state.attempts = attempts.into();
        next.state.static_source = Some(Arc::new(proof));
        Ok(next)
    }
    /// Original independently retained compiler source used by static proof binding, if any.
    #[must_use]
    pub fn static_source_proof(&self) -> Option<&PreparedJointResponse> {
        self.state.static_source.as_deref()
    }
}
fn validate_static_source(
    prepared: &PreparedStudy<JointPanelState>,
    proof: &PreparedJointResponse,
    proof_data: &TabularData,
) -> Result<(), JointPanelError> {
    let original = proof.source_identification();
    if original.status != IdentificationStatus::NonparametricallyIdentified
        || original.required_assumptions.entries.iter().any(|record| {
            matches!(
                record.assumption,
                Assumption::ParametricRestriction(_) | Assumption::PriorRestriction(_)
            )
        })
    {
        return Err(JointPanelError::Unsupported("the retained source derivation does not establish identification independently of parametric/prior restrictions".into()));
    }
    if proof_data.schema() != prepared.source_data().schema()
        || proof.source_schema() != proof_data.schema()
        || proof_data.storage().content_digest() != proof.source_digest()
    {
        return Err(JointPanelError::compile(
            "static proof schema or immutable source digest differs from the actual panel",
        ));
    }
    let query = proof.source_query();
    if query.target_population != antecedent_core::TargetPopulation::AllObserved
        || query.temporal.is_some()
        || !query.outcome_functional.is_mean()
        || proof
            .numerical_program()
            .problem()
            .design
            .row_selection
            .iter()
            .copied()
            .ne(0..proof_data.row_count())
    {
        return Err(JointPanelError::Unsupported(
            "static proof does not cover the complete equally weighted observed row law".into(),
        ));
    }
    let allowed = [prepared.request().columns.confounder, prepared.request().columns.season];
    if original
        .estimands
        .iter()
        .any(|estimand| estimand.adjustment_set.iter().any(|variable| !allowed.contains(variable)))
    {
        return Err(JointPanelError::Unsupported(
            "static proof requires covariates absent from this panel mechanism".into(),
        ));
    }
    let current_worlds = contracts::target_worlds(&prepared.request().target)?;
    let source_worlds = contracts::target_worlds(proof.target())?;
    if current_worlds.len() != source_worlds.len()
        || current_worlds.iter().zip(source_worlds).any(|(current, source)| {
            current.id() != source.id()
                || current.action() != source.action()
                || current.is_reference() != source.is_reference()
                || current.time() != source.time()
                || current.binding().outcome() != source.binding().outcome()
        })
    {
        return Err(JointPanelError::Unsupported(
            "static proof concerns different world actions, outcomes, units or horizons".into(),
        ));
    }
    exact_rows_and_context(prepared, proof_data)?;
    Ok(())
}
fn exact_rows_and_context(
    prepared: &PreparedStudy<JointPanelState>,
    proof_data: &TabularData,
) -> Result<(), JointPanelError> {
    let groups = prepared.model_data();
    let Some(first) = groups.groups.first() else {
        return Err(JointPanelError::compile("empty panel"));
    };
    let rows_per_group = first.outcome.len();
    if groups.groups.iter().any(|group| group.outcome.len() != rows_per_group)
        || proof_data.row_count()
            != groups
                .groups
                .len()
                .checked_mul(rows_per_group)
                .ok_or_else(|| JointPanelError::compile("source row count overflow"))?
    {
        return Err(JointPanelError::Unsupported(
            "unequal group weighting differs from the static empirical row law".into(),
        ));
    }
    if proof_data.storage().weights().is_some() || proof_data.storage().analysis_mask().is_some() {
        return Err(JointPanelError::Unsupported(
            "weighted or masked static proof needs its own exact empirical-law translation".into(),
        ));
    }
    for variable in proof_data.schema().variables() {
        let expected = proof_data.float64_cow(variable.id).map_err(JointPanelError::compile)?;
        let mut offset = 0;
        for unit in prepared.source_data().units() {
            let actual = unit.series.float64_cow(variable.id).map_err(JointPanelError::compile)?;
            if actual
                .iter()
                .zip(&expected[offset..offset + actual.len()])
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(JointPanelError::compile(
                    "static proof training rows differ from the actual panel source",
                ));
            }
            offset += actual.len();
        }
    }
    for world in &prepared.request().worlds {
        for group in &groups.groups {
            let scenario = world
                .program
                .scenarios
                .iter()
                .find(|scenario| scenario.group == group.id)
                .ok_or_else(|| JointPanelError::compile("world omits a group"))?;
            if world.horizon() != 1
                || scenario.confounder.len() != 1
                || scenario.season.len() != 1
                || group
                    .confounder
                    .iter()
                    .any(|value| value.to_bits() != scenario.confounder[0].to_bits())
                || group.season.iter().any(|value| *value != scenario.season[0])
            {
                return Err(JointPanelError::Unsupported("future group-specific context law differs from the independently identified fixed empirical row law".into()));
            }
        }
    }
    Ok(())
}
