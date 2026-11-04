//! Prepared execution of a multi-source limited-experiment (`TR^mz`) formula.
//!
//! Preparation freezes one checked, catalog-bound derivation and compiles one
//! evaluation plan per request. Estimation evaluates those plans and never
//! searches again; request 0 is the baseline of every point contrast. Refresh
//! replaces laws only for the snapshots the frozen catalog binds, so the proof
//! is unchanged; evidence under another snapshot, regime or catalog needs a new
//! decision and a new preparation.
//!
//! An estimate on counted laws returns the point and the nominal pointwise joint
//! percentile-bootstrap interval of `antecedent_estimate::mz_transport_bootstrap_interval`
//! (seeded from the execution, so a consumer recomputes it bit for bit), with the
//! calibration slot of the multi-source coverage records. Calibration annotates the
//! interval and never withholds it; an interval is withheld only for a data or
//! dependence reason (undeclared or unsupported sampling dependence).
use super::StudyBuilder;
use super::transport_common::{compile_plans, err, estimate_err, eval_err, evaluate_plans};
use antecedent_core::{ExecutionContext, SearchLimits, TheoremScope};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
};
use antecedent_identify::BoundMzTransportFunctional;
use antecedent_io::IoError;
use antecedent_io::mz_transport_artifact::{
    MzContrastWire, MzTransportArtifactInput, MzTransportArtifactWire, MzTransportConsumeLimits,
    MzUncertaintyWire, mz_point_contrasts,
};
use antecedent_io_core::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, NO_INTERVAL_REPORTED,
    calibration_slot,
};
use antecedent_io_core::contract_section::CalibrationSlotWire;

/// Bootstrap replicates of the multi-source interval (the measured independent-study design).
pub const MZ_INTERVAL_REPLICATES: u32 = 999;
/// Nominal level of the multi-source interval.
pub const MZ_INTERVAL_LEVEL: f64 = 0.95;

/// Calibration slot of the interval of `uncertainty` over `data` under `functional`.
fn interval_calibration(
    functional: &BoundMzTransportFunctional,
    data: &ExactTransportData,
    uncertainty: &MzUncertaintyWire,
) -> CalibrationSlotWire {
    if !uncertainty.available() {
        return CalibrationSlotWire::not_measured(NO_INTERVAL_REPORTED);
    }
    let shared = {
        let cited = functional.cited_regimes();
        let catalog = functional.catalog();
        cited.iter().enumerate().any(|(i, a)| {
            cited[i + 1..]
                .iter()
                .any(|b| catalog.shared_data(*a, *b) == antecedent_core::SharedData::SameDataset)
        })
    };
    let rows = data
        .laws()
        .iter()
        .filter_map(|law| law.empirical_counts().map(|counts| counts.iter().sum::<u64>()))
        .min()
        .unwrap_or(0);
    calibration_slot(&CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "not_applicable".into(),
            query: "ClassicalTransport".into(),
            graph_class: "Admg".into(),
            structure: "fixed".into(),
            modality: "tabular".into(),
            inference: "Frequentist".into(),
            estimator: "transport.mz_joint_bootstrap".into(),
            interval_method: "percentile_bootstrap".into(),
            se_kind: "percentile".into(),
            dependence: if shared { "shared_units" } else { "iid" }.into(),
            posterior: String::new(),
            functional: "target_interventional_mean".into(),
            level: uncertainty.coverage_target.unwrap_or(MZ_INTERVAL_LEVEL),
            identification: "point".into(),
        },
        scope: CalibrationScopeWire {
            row_count: rows,
            replicates_ok: Some(uncertainty.replicates_ok),
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    })
}

/// A prepared mz formula: frozen proof, bound catalog, laws and compiled plans.
#[derive(Clone, Debug)]
pub struct PreparedMzTransport {
    graph: antecedent_graph::Admg,
    functional: BoundMzTransportFunctional,
    search: SearchLimits,
    data: ExactTransportData,
    requests: Vec<Assignment>,
    limits: ExactEvaluationLimits,
    /// One compiled plan per request, in request order.
    plans: Vec<ExactEvaluationPlan>,
    /// Every retained law carries empirical counts; a refresh keeps that contract.
    empirical: bool,
}

/// One execution: the point of every request, the point contrasts against
/// request 0, and the interval bookkeeping.
#[derive(Clone, Debug)]
pub struct MzTransportResult {
    distributions: Vec<ExactDistribution>,
    contrasts: Vec<MzContrastWire>,
    uncertainty: MzUncertaintyWire,
    calibration: CalibrationSlotWire,
}

impl MzTransportResult {
    /// The point distribution of request 0.
    #[must_use]
    pub fn distribution(&self) -> &ExactDistribution {
        &self.distributions[0]
    }

    /// The point distribution of every request, in request order.
    #[must_use]
    pub fn distributions(&self) -> &[ExactDistribution] {
        &self.distributions
    }

    /// Point contrasts of every outcome mean, each request against request 0.
    #[must_use]
    pub fn contrasts(&self) -> &[MzContrastWire] {
        &self.contrasts
    }

    /// Interval bookkeeping: the nominal pointwise intervals of counted laws, or the
    /// stable reason one is withheld (a data or dependence reason).
    #[must_use]
    pub const fn uncertainty(&self) -> &MzUncertaintyWire {
        &self.uncertainty
    }

    /// Calibration of the reported interval: `calibrated` inside a current coverage
    /// record's measured scope, otherwise the status that names why not
    /// (`not_measured` when no interval is reported).
    #[must_use]
    pub const fn calibration(&self) -> &CalibrationSlotWire {
        &self.calibration
    }

    /// Export this result with its checked proof, bindings, laws and bookkeeping.
    ///
    /// # Errors
    /// The premises do not encode or the bookkeeping does not check.
    pub fn export(
        &self,
        prepared: &PreparedMzTransport,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, IoError> {
        self.export_named(prepared, &[], ctx)
    }

    /// [`Self::export`], binding the name of every graph coordinate into the
    /// artifact's verified identity, so a consumer can refuse relabelled names.
    ///
    /// # Errors
    /// As [`Self::export`], or names that do not cover the graph.
    pub fn export_named(
        &self,
        prepared: &PreparedMzTransport,
        variable_names: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, IoError> {
        MzTransportArtifactWire::checked(
            MzTransportArtifactInput {
                graph: &prepared.graph,
                functional: &prepared.functional,
                search: prepared.search,
                data: &prepared.data,
                requests: &prepared.requests,
                limits: prepared.limits,
                variable_names,
                results: &self.distributions,
                uncertainty: self.uncertainty.clone(),
            },
            ctx,
        )?
        .export()
    }
}

/// Independently consume an mz artifact: re-decide, re-bind, recompute every
/// point and contrast and recheck the bookkeeping (a carried interval is recomputed
/// bit for bit from its seed), without fetching data or providers.
///
/// # Errors
/// Any reconstruction or replay failure.
pub fn consume_mz_transport_artifact(
    bytes: &[u8],
    limits: MzTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<MzTransportResult, IoError> {
    let consumed = MzTransportArtifactWire::consume_with_limits(bytes, limits, ctx)?;
    let calibration =
        interval_calibration(&consumed.functional, &consumed.data, &consumed.wire.uncertainty);
    Ok(MzTransportResult {
        distributions: consumed.distributions,
        contrasts: consumed.contrasts,
        uncertainty: consumed.wire.uncertainty,
        calibration,
    })
}

impl StudyBuilder {
    /// Prepare exact-law evaluation of a checked mz functional for `requests`
    /// (request 0 is the contrast baseline). `search` records the limits it was
    /// decided under, so a consumer can refuse larger ones.
    ///
    /// # Errors
    /// No request, a functional that was not decided on `graph`, or laws,
    /// requests or resources that do not compile.
    pub fn mz_transport(
        graph: antecedent_graph::Admg,
        functional: BoundMzTransportFunctional,
        search: SearchLimits,
        data: ExactTransportData,
        requests: Vec<Assignment>,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMzTransport, IoError> {
        if requests.is_empty() {
            return Err(err("prepare at least one target request"));
        }
        let rebound = antecedent_identify::bind_mz_transport_catalog(
            &graph,
            functional.derivation(),
            functional.catalog(),
        )
        .map_err(|error| match error {
            // The functional is already a checked derivation: it fails to rebind
            // only against a graph it was not decided on.
            antecedent_identify::IdentificationError::InvalidDerivation { .. } => graph_mismatch(),
            other => antecedent_io::mz_transport_artifact::mz_identification_error(other),
        })?;
        if rebound.root() != functional.root() || rebound.arena() != functional.arena() {
            return Err(graph_mismatch());
        }
        let plans = compile(&functional, &data, &requests, limits, ctx)?;
        Ok(PreparedMzTransport {
            graph,
            functional,
            search,
            data,
            requests,
            limits,
            plans,
            empirical: false,
        })
    }

    /// Prepare an empirical plug-in evaluation; every law must carry counts.
    ///
    /// # Errors
    /// A law without counts, or any [`Self::mz_transport`] failure.
    pub fn mz_transport_empirical(
        graph: antecedent_graph::Admg,
        functional: BoundMzTransportFunctional,
        search: SearchLimits,
        data: ExactTransportData,
        requests: Vec<Assignment>,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMzTransport, IoError> {
        require_counts(&data)?;
        let mut prepared =
            Self::mz_transport(graph, functional, search, data, requests, limits, ctx)?;
        prepared.empirical = true;
        Ok(prepared)
    }
}

/// The refusal of preparing a functional that was decided on another graph.
fn graph_mismatch() -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message:
            "mz_transport.functional_graph_mismatch: the functional was not decided on this graph"
                .into(),
    }
}

fn compile(
    functional: &BoundMzTransportFunctional,
    data: &ExactTransportData,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<Vec<ExactEvaluationPlan>, IoError> {
    compile_plans(requests, |request| {
        antecedent_estimate::prepare_exact_mz_transport(
            functional,
            data.clone(),
            request.clone(),
            limits,
            ctx,
        )
        .map_err(|e| eval_err(&e))
    })
}

fn require_counts(data: &ExactTransportData) -> Result<(), IoError> {
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("transport_missing_provider"),
            message: "mz_transport.empirical_counts_required: every law of an empirical preparation carries counts".into(),
        });
    }
    Ok(())
}

impl PreparedMzTransport {
    /// The bounded theorem scope of this route.
    #[must_use]
    pub fn theorem_scope(&self) -> TheoremScope {
        TheoremScope::mz_transportability()
    }

    /// Evaluate every retained plan and the point contrasts. Exact laws are
    /// point-only. Counted laws also return the nominal pointwise joint
    /// percentile-bootstrap interval (seeded by the execution's master seed) with its
    /// calibration slot; the interval is withheld only for a data or dependence reason,
    /// which the bookkeeping names.
    ///
    /// # Errors
    /// Cancellation or an evaluation refusal.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<MzTransportResult, IoError> {
        let distributions = evaluate_plans("mz-transport estimate", &self.plans, ctx, |plan| {
            plan.evaluate(ctx).map_err(|e| eval_err(&e))
        })?;
        let contrasts = mz_point_contrasts(&distributions)?;
        let uncertainty = if self.data.laws().iter().any(|law| law.empirical_counts().is_none()) {
            MzUncertaintyWire::point_only()
        } else {
            let seed = ctx.rng.master_seed();
            let mut replay = ctx.clone();
            replay.rng = antecedent_core::RngFactory::from_seed(seed);
            let run = antecedent_estimate::mz_transport_bootstrap_interval(
                &self.functional,
                &self.data,
                &self.requests,
                self.limits,
                MZ_INTERVAL_REPLICATES,
                MZ_INTERVAL_LEVEL,
                &replay,
            )
            .map_err(estimate_err)?;
            MzUncertaintyWire::from_bootstrap(&run, MZ_INTERVAL_REPLICATES, MZ_INTERVAL_LEVEL, seed)
        };
        let calibration = interval_calibration(&self.functional, &self.data, &uncertainty);
        Ok(MzTransportResult { distributions, contrasts, uncertainty, calibration })
    }

    /// Replace the laws for the snapshots the frozen catalog binds and recompile.
    /// The proof, bindings and requests are unchanged; laws under any other
    /// snapshot or regime are refused, which forces a new preparation.
    ///
    /// # Errors
    /// A law that does not match the frozen catalog, missing counts on an
    /// empirical preparation, or cancellation.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "mz-transport refresh").map_err(estimate_err)?;
        if self.empirical {
            require_counts(&data)?;
        }
        let plans = compile(&self.functional, &data, &self.requests, self.limits, ctx)?;
        Ok(Self { data, plans, ..self.clone() })
    }

    /// Rebind finite supported requests and laws to the same checked proof and catalog.
    /// # Errors
    /// Missing factors, incompatible snapshots/worlds, support or numerical limits.
    pub(crate) fn refresh_requests(
        &self,
        data: ExactTransportData,
        requests: Vec<Assignment>,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "mz-transport request refresh")
            .map_err(estimate_err)?;
        if self.empirical {
            require_counts(&data)?;
        }
        let plans = compile(&self.functional, &data, &requests, limits, ctx)?;
        Ok(Self { data, requests, limits, plans, ..self.clone() })
    }

    /// The frozen, catalog-bound functional.
    #[must_use]
    pub const fn functional(&self) -> &BoundMzTransportFunctional {
        &self.functional
    }

    /// Retained laws.
    #[must_use]
    pub const fn data(&self) -> &ExactTransportData {
        &self.data
    }

    /// Retained requests; request 0 is the contrast baseline.
    #[must_use]
    pub fn requests(&self) -> &[Assignment] {
        &self.requests
    }

    /// The compiled evaluation plan of every request, in request order.
    #[must_use]
    pub fn plans(&self) -> &[ExactEvaluationPlan] {
        &self.plans
    }
}
