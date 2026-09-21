//! Learner-backed trial modality of the common prepared-study lifecycle.
use super::{PreparedStudy, StudyBuilder};
use antecedent_core::{ExecutionContext, TransportQuery};
use antecedent_estimate::{TrialAipwInput, TrialAipwOptions};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::{IoError, learned_trial_wire::LearnedTrialWire};
use std::sync::Arc;

use super::transport_common::err;

/// Checked structural request and immutable trial/target snapshot.
#[derive(Clone, Debug)]
pub struct LearnedTrialState {
    diagram: SelectionDiagram,
    query: TransportQuery,
    identification: TransportIdentification,
    input: TrialAipwInput,
    options: TrialAipwOptions,
    seed: u64,
}
/// Retained score evidence and its scientific execution identity.
#[derive(Clone, Debug)]
pub struct LearnedTrialResult {
    wire: LearnedTrialWire,
    identity: String,
}
impl LearnedTrialResult {
    /// Target nonparticipant contrast and nominal uncertainty.
    #[must_use]
    pub fn estimate(&self) -> &antecedent_estimate::TrialAipwEstimate {
        &self.wire.result
    }
    /// Frozen scientific execution identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Export complete score evidence without fitting.
    /// # Errors
    /// Inconsistent claim or encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.wire.export()
    }
    /// Core compiler reasoning; structural premises and estimation assumptions are distinct.
    #[must_use]
    pub fn reasoning(&self) -> antecedent_core::ReasoningView {
        trial_reasoning(self.wire.result.uncertainty_reason.as_deref())
    }
    /// Verified portable consumption without model fitting or resampling.
    /// # Errors
    /// Invalid artifact or score evidence.
    pub fn consume(bytes: &[u8]) -> Result<Self, IoError> {
        let wire = LearnedTrialWire::consume(bytes)?;
        let identity = wire.identity()?;
        Ok(Self { wire, identity })
    }
}
fn trial_reasoning(unavailable: Option<&str>) -> antecedent_core::ReasoningView {
    use antecedent_core::*;
    ReasoningView::new(
        SlotAvailability::Available(IdentificationSlot::identified_singleton(
            IdentificationStatus::NonparametricallyIdentified,
        )),
        SlotAvailability::Available(SupportSlot::new(
            "stage_contract",
            Some(Arc::from("transport.learned_trial_aipw")),
            SlotAvailability::Available(Arc::from("trial_and_target_overlap_required")),
        )),
        unavailable.map_or_else(
            || {
                SlotAvailability::Available(UncertaintySlot::new(vec![UncertaintyComponent::new(
                    UncertaintySource::Sampling,
                    "nominal_pointwise_joint_outer_bootstrap_uncalibrated",
                    false,
                )]))
            },
            SlotAvailability::unavailable,
        ),
        SlotAvailability::Available(AssumptionSlot::new(vec![ObligationRecord::new(
            "transport.trial_aipw",
            ObligationScope::Program,
            AssumptionSource::UserDeclared,
            ObligationKind::UserAssertion,
            AssumptionStatus::Declared,
            "Randomized source treatment with supplied assignment probabilities; representative IID target nonparticipants; declared sampling design; baseline S-admissibility and overlap. Nuisance consistency and bootstrap regularity are separate from graphical identification. No calibration claim.",
        )])),
    )
}
impl StudyBuilder {
    /// Prepare the certified learner-backed binary trial-to-target contrast.
    /// # Errors
    /// Unsupported target, certificate, sampling contract, or learner configuration.
    pub fn learned_trial_transport(
        diagram: SelectionDiagram,
        query: TransportQuery,
        input: TrialAipwInput,
        options: TrialAipwOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<LearnedTrialState>, IoError> {
        if ctx.cancellation.is_cancelled() {
            return Err(err("trial transport cancelled"));
        }
        antecedent_estimate::validate_trial_query(&query).map_err(err)?;
        let identification = TransportIdentifier::new().identify(&diagram, &query).map_err(err)?;
        antecedent_estimate::validate_trial_aipw(&identification, &input, &options).map_err(err)?;
        Ok(PreparedStudy {
            state: LearnedTrialState {
                diagram,
                query,
                identification,
                input,
                options,
                seed: ctx.rng.master_seed(),
            },
        })
    }
}
impl PreparedStudy<LearnedTrialState> {
    /// Preview invalidations without fitting or publishing replacement state.
    /// # Errors
    /// Scientific identity encoding failure.
    pub fn preview_transform(
        &self,
        intent: antecedent_core::TransformIntent,
    ) -> Result<antecedent_core::TransformationReport, IoError> {
        use antecedent_core::{IdentityDomain, IdentityRef, TransformIntent, TransformationReport};
        let identity = antecedent_io::identity::digest_wire(
            IdentityDomain::Execution,
            &(
                antecedent_io::admg_to_wire(self.state.diagram.causal_graph())?,
                self.state.diagram.selection_targets().iter().map(|v| v.raw()).collect::<Vec<_>>(),
                antecedent_io::query_wire::transport_query_to_wire(&self.state.query)?,
                &self.state.input,
                &self.state.options,
                self.state.seed,
            ),
        )?;
        let report = TransformationReport::new(
            intent,
            vec![IdentityRef::new(IdentityDomain::Execution, identity)],
            antecedent_core::intent_effects(intent).iter().cloned(),
            vec![],
        );
        Ok(
            if matches!(
                intent,
                TransformIntent::DisplayPrecision
                    | TransformIntent::FilterDisplay
                    | TransformIntent::CompatibleDataReplace
            ) {
                report
            } else {
                report.refused_on_handle(
                    "transport.reprepare_required: structural or evidence contract changed",
                )
            },
        )
    }
    /// Metadata-only compiler reasoning.
    #[must_use]
    pub fn inspect(&self) -> antecedent_core::ReasoningView {
        trial_reasoning(Some("not_executed"))
    }
    /// Frozen feature schema.
    #[must_use]
    pub fn features(&self) -> &[u32] {
        &self.state.input.features
    }
    /// Estimate on the retained snapshot and frozen seed.
    /// # Errors
    /// Fit, support, budget, or cancellation failure.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<LearnedTrialResult, IoError> {
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.state.seed);
        let result = antecedent_estimate::estimate_trial_aipw(
            &self.state.identification,
            &self.state.input,
            &self.state.options,
            &ctx,
        )
        .map_err(err)?;
        let wire = LearnedTrialWire {
            version: 1,
            graph: antecedent_io::admg_to_wire(self.state.diagram.causal_graph())?,
            selections: self.state.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: antecedent_io::query_wire::transport_query_to_wire(&self.state.query)?,
            input: self.state.input.clone(),
            options: self.state.options,
            seed: self.state.seed,
            result,
        };
        let identity = wire.identity()?;
        Ok(LearnedTrialResult { wire, identity })
    }
    /// Replace a compatible snapshot after validation, without retaining old results.
    /// # Errors
    /// Changed schema/sampling contract, cancellation, or invalid data.
    pub fn replace_snapshot(
        &mut self,
        input: TrialAipwInput,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        if ctx.cancellation.is_cancelled() {
            return Err(err("trial transport cancelled"));
        }
        if input.features != self.state.input.features
            || input.sampling != self.state.input.sampling
        {
            return Err(err("transport.reprepare_required"));
        }
        antecedent_estimate::validate_trial_aipw(
            &self.state.identification,
            &input,
            &self.state.options,
        )
        .map_err(err)?;
        self.state.input = input;
        Ok(())
    }
    /// Atomically replace the compatible snapshot and execute it.
    /// # Errors
    /// Schema/sampling change or failed candidate execution; old state remains intact.
    pub fn refresh(
        &mut self,
        input: TrialAipwInput,
        ctx: &ExecutionContext,
    ) -> Result<LearnedTrialResult, IoError> {
        if input.features != self.state.input.features
            || input.sampling != self.state.input.sampling
        {
            return Err(err("transport.reprepare_required"));
        }
        antecedent_estimate::validate_trial_aipw(
            &self.state.identification,
            &input,
            &self.state.options,
        )
        .map_err(err)?;
        let mut candidate = self.clone();
        candidate.state.input = input;
        let result = candidate.estimate(ctx)?;
        *self = candidate;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, VariableId,
    };
    fn fixture() -> (SelectionDiagram, TransportQuery, TrialAipwInput, TrialAipwOptions) {
        let mut graph = antecedent_graph::Admg::with_variables(2);
        graph
            .insert_directed(
                antecedent_graph::DenseNodeId::from_raw(0),
                antecedent_graph::DenseNodeId::from_raw(1),
            )
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, vec![]).unwrap();
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0., 1.])),
                ),
            }),
            "trial",
            "target",
            [VariableId::from_raw(0)],
        );
        let input = TrialAipwInput {
            features: vec![],
            covariates: vec![],
            outcome: (0..200)
                .map(|i| if i < 120 { 1. + 2. * (i % 2) as f64 } else { 0. })
                .collect(),
            treatment: (0..200).map(|i| i % 2 == 1).collect(),
            source: (0..200).map(|i| i < 120).collect(),
            randomization: vec![0.5; 200],
            sampling: antecedent_estimate::TrialSampling::IndependentSamples,
        };
        let options = TrialAipwOptions {
            outcome: antecedent_estimate::LearnerSpec::Linear(Default::default()),
            folds: 3,
            bootstrap: 5,
            ..Default::default()
        };
        (diagram, query, input, options)
    }
    #[test]
    fn trial_result_is_replayed_and_refresh_is_atomic() {
        let (diagram, query, input, options) = fixture();
        let ctx = ExecutionContext::for_tests(8);
        let mut study =
            StudyBuilder::learned_trial_transport(diagram, query, input.clone(), options, &ctx)
                .unwrap();
        let result = study.estimate(&ctx).unwrap();
        assert!((result.estimate().estimate - 2.).abs() < 1e-10);
        let loaded = LearnedTrialResult::consume(&result.export().unwrap()).unwrap();
        assert_eq!(loaded.identity(), result.identity());
        assert_eq!(loaded.estimate().replicates.len(), 5);
        let mut bad = input;
        bad.randomization.fill(0.);
        assert!(study.refresh(bad, &ctx).is_err());
        assert_eq!(study.estimate(&ctx).unwrap().identity(), result.identity());
        let mut bad = result.wire.clone();
        bad.result.estimate += 1.;
        assert!(bad.verify().is_err());
        let mut bad = result.wire.clone();
        bad.result.replicates[0].0 = 5;
        assert!(bad.verify().is_err());
        let mut bad = result.wire.clone();
        bad.version = 2;
        assert!(bad.verify().is_err());
    }
}
