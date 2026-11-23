//! Prepared learned continuous-outcome trial transport.
//!
//! Preparation checks the binary-contrast query, derives the transport certificate once
//! and validates the request before any nuisance is fitted. Estimation cross-fits every
//! nuisance on the retained rows under the frozen seed and never identifies again;
//! refresh replaces the rows of the same schema and sampling design and keeps the
//! certificate. Estimation returns the point with provenance and diagnostics;
//! [`PreparedLearnedContinuous::interval`] returns the pointwise analytic influence
//! interval under the declared IID sampling design (`bootstrap = 0`);
//! [`PreparedLearnedContinuous::bootstrap_interval`] returns the joint outer-refit
//! percentile-bootstrap interval for a request of two or more replicates, with the
//! calibration slot of the coverage records (`measured_below_nominal` where measured).
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
    input: std::sync::Arc<TrialAipwInput>,
    options: LearnedContinuousOptions,
    variable_names: Vec<String>,
    seed: u64,
}

/// The joint outer-refit percentile-bootstrap interval of the mean contrast with its calibration.
#[derive(Clone, Debug)]
pub struct LearnedContinuousBootstrapInterval {
    run: antecedent_estimate::learned_trial::TrialAipwEstimate,
    calibration: antecedent_io_core::contract_section::CalibrationSlotWire,
}

impl LearnedContinuousBootstrapInterval {
    /// The point run the interval is built around.
    #[must_use]
    pub const fn point(&self) -> &antecedent_estimate::learned_trial::TrialAipwEstimate {
        &self.run
    }

    /// The percentile interval; `None` when a replicate failed (see [`Self::uncertainty_reason`]).
    #[must_use]
    pub const fn interval(&self) -> Option<(f64, f64)> {
        self.run.interval
    }

    /// Why no interval was produced, when none was.
    #[must_use]
    pub fn uncertainty_reason(&self) -> Option<&str> {
        self.run.uncertainty_reason.as_deref()
    }

    /// Successful replicate estimates, in replicate order.
    #[must_use]
    pub fn replicates(&self) -> &[(u32, f64)] {
        &self.run.replicates
    }

    /// Calibration of the reported interval.
    #[must_use]
    pub const fn calibration(&self) -> &antecedent_io_core::contract_section::CalibrationSlotWire {
        &self.calibration
    }
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
        let identification =
            TransportIdentifier::new().identify_with_context(&diagram, &query, ctx)?;
        antecedent_estimate::validate_learned_continuous(&identification, &input, &options)
            .map_err(estimate_err)?;
        Ok(PreparedLearnedContinuous {
            diagram,
            query,
            identification,
            input: std::sync::Arc::new(input),
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

    /// Whether the options request a bootstrap interval (two or more replicates are formed by
    /// [`Self::bootstrap_interval`]; the analytic [`Self::interval`] takes none).
    #[must_use]
    pub const fn bootstrap_requested(&self) -> bool {
        self.options.bootstrap != 0
    }

    /// The joint outer-refit percentile-bootstrap interval on the frozen seed, with its
    /// calibration slot. The slot annotates; it never gates the interval.
    ///
    /// # Errors
    /// The refusals of the estimate, a bootstrap below the replicate floor
    /// (`learned_transport.bootstrap_below_floor`), a workspace above the memory cap, or
    /// cancellation.
    pub fn bootstrap_interval(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<LearnedContinuousBootstrapInterval, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "learned continuous bootstrap interval")
            .map_err(estimate_err)?;
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let run = antecedent_estimate::learned_continuous_interval_internal(
            &self.identification,
            &self.input,
            &self.options,
            &ctx,
        )
        .map_err(estimate_err)?;
        let calibration = if run.interval.is_some() {
            use antecedent_io_core::calibration::{
                CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
            };
            calibration_slot(&CalibrationBasisWire {
                key: CalibrationKeyWire {
                    validation: "not_applicable".into(),
                    query: "ClassicalTransport".into(),
                    graph_class: "Admg".into(),
                    structure: "fixed".into(),
                    modality: "tabular".into(),
                    inference: "Frequentist".into(),
                    estimator: "transport.learned_trial_aipw".into(),
                    interval_method: "percentile_bootstrap".into(),
                    se_kind: "percentile".into(),
                    dependence: "iid".into(),
                    posterior: String::new(),
                    functional: "target_mean_contrast".into(),
                    level: self.options.coverage_level,
                    identification: "point".into(),
                },
                scope: CalibrationScopeWire {
                    row_count: self.input.outcome.len() as u64,
                    replicates_ok: Some(u32::try_from(run.replicates.len()).unwrap_or(u32::MAX)),
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
            })
        } else {
            antecedent_io_core::contract_section::CalibrationSlotWire::not_measured(
                antecedent_io_core::calibration::NO_INTERVAL_REPORTED,
            )
        };
        Ok(LearnedContinuousBootstrapInterval { run, calibration })
    }

    /// Compute the design-specific analytic influence interval and exportable result.
    ///
    /// # Errors
    /// Fit, support, budget or cancellation failure. A non-zero bootstrap request refuses
    /// with `estimator_inference_mismatch`: this route is the analytic interval, and the
    /// bootstrap interval is [`Self::bootstrap_interval`].
    pub fn interval(&self, ctx: &ExecutionContext) -> Result<LearnedContinuousResult, IoError> {
        let mut ctx = ctx.clone();
        ctx.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let result =
            antecedent_estimate::learned_continuous::estimate_learned_continuous_analytic_interval(
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
        Ok(Self { input: std::sync::Arc::new(input), ..self.clone() })
    }

    /// Reconfigure the execution stream while sharing the immutable input and checked certificate.
    /// # Errors
    /// Cancellation or physical workspace refusal under the caller's policy.
    pub fn with_seed(&self, seed: u64, ctx: &ExecutionContext) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "transport execution reconfiguration")
            .map_err(estimate_err)?;
        antecedent_estimate::learned_trial::check_trial_aipw_resources(
            &self.input,
            &self.options.trial_options(0),
            ctx,
        )
        .map_err(estimate_err)?;
        let mut branch = self.clone();
        branch.seed = seed;
        Ok(branch)
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
