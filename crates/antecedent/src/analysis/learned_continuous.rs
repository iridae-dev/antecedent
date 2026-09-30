//! Prepared learned continuous-outcome trial transport (2.2A cell X4).
//!
//! Preparation checks the binary-contrast query, derives the transport certificate once
//! and validates the request before any nuisance is fitted. Estimation cross-fits every
//! nuisance on the retained rows under the frozen seed and never identifies again;
//! refresh replaces the rows of the same schema and sampling design and keeps the
//! certificate. The result is the point estimate with its provenance, diagnostics and
//! interval status.
//!
//! The interval route is registered closed (`cell_not_licensed`) until its coverage
//! records are measured: an estimate reports the interval withheld, and
//! [`PreparedLearnedContinuous::interval`] refuses. The internal estimator the
//! calibration harness measures is
//! `antecedent_estimate::learned_continuous_interval_internal`, compiled only under the
//! `calibration-internal` feature that the facade's dev-dependencies enable.
use super::StudyBuilder;
use super::transport_common::{err, estimate_err};
use antecedent_core::{ExecutionContext, TransportQuery};
use antecedent_estimate::{
    EstimatorMenu, LearnedContinuousEstimate, LearnedContinuousOptions, TrialAipwInput,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::IoError;
use antecedent_io::learned_continuous_artifact::{
    LearnedContinuousArtifactInput, LearnedContinuousArtifactWire, LearnedContinuousConsumeLimits,
};

/// A prepared learned continuous transport: frozen certificate, rows, options and seed.
#[derive(Clone, Debug)]
pub struct PreparedLearnedContinuous {
    diagram: SelectionDiagram,
    query: TransportQuery,
    identification: TransportIdentification,
    input: TrialAipwInput,
    options: LearnedContinuousOptions,
    variable_names: Vec<String>,
    seed: u64,
}

/// One execution: the verified artifact wire that carries the point and its evidence.
#[derive(Clone, Debug)]
pub struct LearnedContinuousResult {
    wire: LearnedContinuousArtifactWire,
    identity: String,
}

impl LearnedContinuousResult {
    /// The point estimate with provenance, diagnostics and the interval status.
    #[must_use]
    pub const fn estimate(&self) -> &LearnedContinuousEstimate {
        &self.wire.result
    }

    /// Frozen scientific execution identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The verified artifact wire.
    #[must_use]
    pub const fn wire(&self) -> &LearnedContinuousArtifactWire {
        &self.wire
    }

    /// Export the point with its certificate, rows and evidence; no fitting is stored
    /// and none is needed to consume it.
    ///
    /// # Errors
    /// Verification or encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.wire.verify(LearnedContinuousConsumeLimits::default())?;
        self.wire.export()
    }
}

/// Independently consume an artifact: re-derive the certificate, recompute the folds,
/// replay the point and diagnostics bit for bit and re-check the interval status,
/// without fitting or resampling.
///
/// # Errors
/// Any decoding or verification failure.
pub fn consume_learned_continuous_artifact(
    bytes: &[u8],
    limits: LearnedContinuousConsumeLimits,
) -> Result<LearnedContinuousResult, IoError> {
    let consumed = LearnedContinuousArtifactWire::consume(bytes, limits)?;
    let identity = consumed.wire.identity()?;
    Ok(LearnedContinuousResult { wire: consumed.wire, identity })
}

impl StudyBuilder {
    /// Prepare the learned continuous-outcome trial-to-target mean contrast.
    ///
    /// `variable_names` names every graph coordinate (binding them into the artifact's
    /// verified identity), or is empty.
    ///
    /// # Errors
    /// An unsupported query, an uncertified graph, a request outside the frozen bounds,
    /// insufficient randomization overlap, invalid rows or cancellation.
    pub fn learned_continuous_transport(
        diagram: SelectionDiagram,
        query: TransportQuery,
        input: TrialAipwInput,
        options: LearnedContinuousOptions,
        variable_names: Vec<String>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedLearnedContinuous, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "learned continuous transport prepare")
            .map_err(estimate_err)?;
        antecedent_estimate::validate_trial_query(&query).map_err(err)?;
        if !variable_names.is_empty()
            && variable_names.len() != diagram.causal_graph().nodes().len()
        {
            return Err(err("variable names must cover every graph coordinate"));
        }
        let identification = TransportIdentifier::new().identify(&diagram, &query)?;
        antecedent_estimate::validate_learned_continuous(&identification, &input, &options)
            .map_err(estimate_err)?;
        Ok(PreparedLearnedContinuous {
            diagram,
            query,
            identification,
            input,
            options,
            variable_names,
            seed: ctx.rng.master_seed(),
        })
    }

    /// The estimator menu for this graph, query and learners: inspection only.
    ///
    /// # Errors
    /// An identification failure that is not itself a menu outcome.
    pub fn learned_continuous_menu(
        diagram: &SelectionDiagram,
        query: &TransportQuery,
        learners: Option<(antecedent_estimate::LearnerSpec, antecedent_estimate::LearnerSpec)>,
    ) -> Result<EstimatorMenu, IoError> {
        let identification = TransportIdentifier::new().identify(diagram, query)?;
        Ok(antecedent_estimate::transport_estimator_menu(&identification, query, learners))
    }
}

impl PreparedLearnedContinuous {
    /// Estimate on the retained rows and frozen seed. The certificate is not re-derived.
    ///
    /// # Errors
    /// Fit, support (`learned_transport.membership_overlap`), budget or cancellation failure.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<LearnedContinuousResult, IoError> {
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let result = antecedent_estimate::estimate_learned_continuous(
            &self.identification,
            &self.input,
            &self.options,
            &ctx,
        )
        .map_err(estimate_err)?;
        let wire = LearnedContinuousArtifactWire::checked(LearnedContinuousArtifactInput {
            graph: self.diagram.causal_graph(),
            selections: self.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: &self.query,
            variable_names: &self.variable_names,
            identification: &self.identification,
            input: &self.input,
            options: &self.options,
            seed: self.seed,
            result: &result,
        })?;
        let identity = wire.identity()?;
        Ok(LearnedContinuousResult { wire, identity })
    }

    /// The closed interval route: the point is retained by [`Self::estimate`] and the
    /// interval is withheld until this cell's coverage records are measured.
    ///
    /// # Errors
    /// Always `cell_not_licensed` (`learned_transport.interval_withheld`), or cancellation.
    pub fn interval(&self, ctx: &ExecutionContext) -> Result<LearnedContinuousResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "learned continuous interval")
            .map_err(estimate_err)?;
        Err(estimate_err(antecedent_estimate::refuse_learned_continuous_interval()))
    }

    /// Replace the rows with a compatible snapshot after validation; the certificate,
    /// options and seed are unchanged.
    ///
    /// # Errors
    /// A changed feature schema or sampling design (`transport.reprepare_required`),
    /// invalid rows, insufficient randomization overlap or cancellation.
    pub fn refresh(&self, input: TrialAipwInput, ctx: &ExecutionContext) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "learned continuous refresh")
            .map_err(estimate_err)?;
        if input.features != self.input.features || input.sampling != self.input.sampling {
            return Err(err("transport.reprepare_required"));
        }
        antecedent_estimate::validate_learned_continuous(
            &self.identification,
            &input,
            &self.options,
        )
        .map_err(estimate_err)?;
        Ok(Self { input, ..self.clone() })
    }

    /// The estimator menu for this prepared graph, query and learners.
    #[must_use]
    pub fn estimator_menu(&self) -> EstimatorMenu {
        antecedent_estimate::transport_estimator_menu_with(
            &self.identification,
            &self.query,
            &antecedent_estimate::MenuContext {
                options: Some(self.options),
                learners: None,
                sampling: Some(self.input.sampling),
            },
        )
    }

    /// Frozen feature schema.
    #[must_use]
    pub fn features(&self) -> &[u32] {
        &self.input.features
    }

    /// The frozen request options.
    #[must_use]
    pub const fn options(&self) -> &LearnedContinuousOptions {
        &self.options
    }

    /// The certificate derived at preparation.
    #[must_use]
    pub const fn identification(&self) -> &TransportIdentification {
        &self.identification
    }
}
