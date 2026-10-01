//! Prepared smoothed dose-response transport grid (2.2B cell X4).
//!
//! Preparation checks the smoothed-dose query, derives the transport certificate once
//! and validates the request (bounds, known density, dose support, local support)
//! before any nuisance is fitted. Estimation cross-fits the outcome and membership
//! nuisances on the retained rows under the frozen seed and never identifies again;
//! refresh replaces the rows of the same schema and sampling design and keeps the
//! certificate. The result is the grid of points with their quadrature, smoothing-bias
//! and support records and the interval status.
//!
//! The interval route is registered closed (`cell_not_licensed`) until its coverage
//! records are measured: an estimate reports the interval withheld, and
//! [`PreparedSmoothedDose::interval`] refuses. The internal estimator the calibration
//! harness measures is `antecedent_estimate::smoothed_dose_interval_internal`, compiled
//! only under the `calibration-internal` feature that the facade's dev-dependencies
//! enable.
use super::StudyBuilder;
use super::transport_common::{err, estimate_err};
use antecedent_core::{ExecutionContext, SmoothedDoseTransportQuery};
use antecedent_estimate::{
    EstimatorMenu, SmoothedDoseEstimate, SmoothedDoseInput, SmoothedDoseOptions,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::IoError;
use antecedent_io::smoothed_dose_artifact::{
    SmoothedDoseArtifactInput, SmoothedDoseArtifactWire, SmoothedDoseConsumeLimits,
};

/// A prepared smoothed dose transport: frozen certificate, query, rows, options and seed.
#[derive(Clone, Debug)]
pub struct PreparedSmoothedDose {
    diagram: SelectionDiagram,
    query: SmoothedDoseTransportQuery,
    identification: TransportIdentification,
    input: SmoothedDoseInput,
    options: SmoothedDoseOptions,
    variable_names: Vec<String>,
    seed: u64,
}

/// One execution: the verified artifact wire that carries the grid and its evidence.
#[derive(Clone, Debug)]
pub struct SmoothedDoseResult {
    wire: SmoothedDoseArtifactWire,
    identity: String,
}

impl SmoothedDoseResult {
    /// The grid with its fold models, diagnostics and the interval status.
    #[must_use]
    pub const fn estimate(&self) -> &SmoothedDoseEstimate {
        &self.wire.result
    }

    /// Frozen scientific execution identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The verified artifact wire.
    #[must_use]
    pub const fn wire(&self) -> &SmoothedDoseArtifactWire {
        &self.wire
    }

    /// Export the grid with its certificate, rows, fold models and evidence; a consumer
    /// replays it without fitting.
    ///
    /// # Errors
    /// Verification or encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.wire.verify(
            SmoothedDoseConsumeLimits::default(),
            &ExecutionContext::production_default(self.wire.seed),
        )?;
        self.wire.export()
    }
}

/// Independently consume an artifact: re-derive the certificate, recompute the folds,
/// re-predict every nuisance from the stored fold models and replay the grid bit for
/// bit, without fitting or resampling. The replay runs under `limits` (rows, covariates
/// and a memory limit below the mandatory cap) and `ctx`'s cancellation token and hard
/// memory limit.
///
/// # Errors
/// Any decoding or verification failure; a replay workspace above the memory limit;
/// cancellation (`transport_budget_cancel`).
pub fn consume_smoothed_dose_artifact(
    bytes: &[u8],
    limits: SmoothedDoseConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<SmoothedDoseResult, IoError> {
    let consumed = SmoothedDoseArtifactWire::consume(bytes, limits, ctx)?;
    let identity = consumed.wire.identity()?;
    Ok(SmoothedDoseResult { wire: consumed.wire, identity })
}

impl StudyBuilder {
    /// Prepare the smoothed dose-response transport grid.
    ///
    /// `variable_names` names every graph coordinate (binding them into the artifact's
    /// verified identity), or is empty.
    ///
    /// # Errors
    /// A malformed query, an uncertified graph, a request outside the frozen bounds, an
    /// estimated density, invalid or thin dose support, invalid rows or cancellation.
    pub fn smoothed_dose_transport(
        diagram: SelectionDiagram,
        query: SmoothedDoseTransportQuery,
        input: SmoothedDoseInput,
        options: SmoothedDoseOptions,
        variable_names: Vec<String>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedSmoothedDose, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "smoothed dose transport prepare")
            .map_err(estimate_err)?;
        query.validate().map_err(err)?;
        if !variable_names.is_empty()
            && variable_names.len() != diagram.causal_graph().nodes().len()
        {
            return Err(err("variable names must cover every graph coordinate"));
        }
        let identification =
            TransportIdentifier::new().identify(&diagram, &query.transport_query())?;
        antecedent_estimate::validate_smoothed_dose(&identification, &query, &input, &options)
            .map_err(estimate_err)?;
        Ok(PreparedSmoothedDose {
            diagram,
            query,
            identification,
            input,
            options,
            variable_names,
            seed: ctx.rng.master_seed(),
        })
    }

    /// The estimator menu for this graph and smoothed-dose query under the release
    /// default options: inspection only.
    ///
    /// # Errors
    /// A malformed query or an identification failure that is not itself a menu outcome.
    pub fn smoothed_dose_menu(
        diagram: &SelectionDiagram,
        query: &SmoothedDoseTransportQuery,
        options: Option<&SmoothedDoseOptions>,
    ) -> Result<EstimatorMenu, IoError> {
        query.validate().map_err(err)?;
        let identification =
            TransportIdentifier::new().identify(diagram, &query.transport_query())?;
        Ok(antecedent_estimate::smoothed_dose_estimator_menu(&identification, query, options, None))
    }
}

impl PreparedSmoothedDose {
    /// Estimate on the retained rows and frozen seed. The certificate is not re-derived.
    ///
    /// # Errors
    /// Fit, support (`dose_response.membership_overlap`), quadrature
    /// (`dose_response.quadrature_tolerance`), a workspace estimate above the mandatory
    /// memory cap or the context's lower hard limit, or cancellation.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<SmoothedDoseResult, IoError> {
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let result = antecedent_estimate::estimate_smoothed_dose(
            &self.identification,
            &self.query,
            &self.input,
            &self.options,
            &ctx,
        )
        .map_err(estimate_err)?;
        let wire = SmoothedDoseArtifactWire::checked(SmoothedDoseArtifactInput {
            graph: self.diagram.causal_graph(),
            selections: self.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: &self.query,
            variable_names: &self.variable_names,
            identification: &self.identification,
            input: &self.input,
            options: &self.options,
            seed: self.seed,
            result: &result,
            ctx: &ctx,
        })?;
        let identity = wire.identity()?;
        Ok(SmoothedDoseResult { wire, identity })
    }

    /// The closed interval route: the point is retained by [`Self::estimate`] and the
    /// interval is withheld until this cell's coverage records are measured.
    ///
    /// # Errors
    /// Always `cell_not_licensed` (`dose_response.interval_withheld`), or cancellation.
    pub fn interval(&self, ctx: &ExecutionContext) -> Result<SmoothedDoseResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "smoothed dose interval")
            .map_err(estimate_err)?;
        Err(estimate_err(antecedent_estimate::smoothed_dose::refuse_smoothed_dose_interval()))
    }

    /// Replace the rows with a compatible snapshot after validation; the certificate,
    /// query, options and seed are unchanged.
    ///
    /// # Errors
    /// A changed feature schema or sampling design (`transport.reprepare_required`),
    /// invalid rows, an invalid or thin dose support, or cancellation.
    pub fn refresh(
        &self,
        input: SmoothedDoseInput,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "smoothed dose refresh")
            .map_err(estimate_err)?;
        if input.features != self.input.features || input.sampling != self.input.sampling {
            return Err(err("transport.reprepare_required"));
        }
        antecedent_estimate::validate_smoothed_dose(
            &self.identification,
            &self.query,
            &input,
            &self.options,
        )
        .map_err(estimate_err)?;
        Ok(Self { input, ..self.clone() })
    }

    /// The estimator menu for this prepared graph, query and options.
    #[must_use]
    pub fn estimator_menu(&self) -> EstimatorMenu {
        antecedent_estimate::smoothed_dose_estimator_menu(
            &self.identification,
            &self.query,
            Some(&self.options),
            Some(self.input.sampling),
        )
    }

    /// The frozen smoothed-dose query.
    #[must_use]
    pub const fn query(&self) -> &SmoothedDoseTransportQuery {
        &self.query
    }

    /// The frozen request options.
    #[must_use]
    pub const fn options(&self) -> &SmoothedDoseOptions {
        &self.options
    }

    /// The certificate derived at preparation.
    #[must_use]
    pub const fn identification(&self) -> &TransportIdentification {
        &self.identification
    }
}
