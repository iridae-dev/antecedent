//! Prepared execution of a multi-source limited-experiment (`TR^mz`) formula.
//!
//! Preparation freezes one checked, catalog-bound derivation and compiles its
//! evaluation plan. Estimation evaluates that plan and never searches again.
//! Refresh replaces laws only for the snapshots the frozen catalog binds, so the
//! proof is unchanged; evidence under another snapshot, regime or catalog needs
//! a new decision and a new preparation.
use super::StudyBuilder;
use super::transport_common::{err, estimate_err};
use antecedent_core::{ExecutionContext, SearchLimits, TheoremScope};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
};
use antecedent_identify::BoundMzTransportFunctional;
use antecedent_io::IoError;
use antecedent_io::mz_transport_artifact::{
    MZ_NOMINAL_INTERVAL, MZ_WITHHELD, MzTransportArtifactWire, MzTransportConsumeLimits,
    MzUncertaintyWire,
};

/// A prepared mz formula: frozen proof, bound catalog, laws and compiled plan.
#[derive(Clone, Debug)]
pub struct PreparedMzTransport {
    graph: antecedent_graph::Admg,
    functional: BoundMzTransportFunctional,
    search: SearchLimits,
    data: ExactTransportData,
    request: Assignment,
    limits: ExactEvaluationLimits,
    plan: ExactEvaluationPlan,
    /// Every retained law carries empirical counts; a refresh keeps that contract.
    empirical: bool,
}

/// One executed request: the point and its interval bookkeeping.
#[derive(Clone, Debug)]
pub struct MzTransportResult {
    distribution: ExactDistribution,
    uncertainty: MzUncertaintyWire,
}

impl MzTransportResult {
    /// The point distribution.
    #[must_use]
    pub const fn distribution(&self) -> &ExactDistribution {
        &self.distribution
    }

    /// Interval status, reason, replicate accounting and pointwise mean intervals.
    #[must_use]
    pub const fn uncertainty(&self) -> &MzUncertaintyWire {
        &self.uncertainty
    }

    /// Export this result with its checked proof, bindings, laws and bookkeeping.
    ///
    /// # Errors
    /// The premises do not encode or the bookkeeping does not check.
    pub fn export(&self, prepared: &PreparedMzTransport) -> Result<Vec<u8>, IoError> {
        MzTransportArtifactWire::checked(
            &prepared.graph,
            &prepared.functional,
            prepared.search,
            &prepared.data,
            &prepared.request,
            prepared.limits,
            &self.distribution,
            self.uncertainty.clone(),
        )?
        .export()
    }
}

/// Independently consume an mz artifact: re-decide, re-bind, recompute the point
/// and recheck the interval bookkeeping, without fetching data or providers.
///
/// # Errors
/// Any reconstruction or replay failure.
pub fn consume_mz_transport_artifact(
    bytes: &[u8],
    limits: MzTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<MzTransportResult, IoError> {
    let consumed = MzTransportArtifactWire::consume_with_limits(bytes, limits, ctx)?;
    Ok(MzTransportResult {
        distribution: consumed.distribution,
        uncertainty: consumed.wire.uncertainty,
    })
}

impl StudyBuilder {
    /// Prepare exact-law evaluation of a checked mz functional. `search` records
    /// the limits it was decided under, so a consumer can refuse larger ones.
    ///
    /// # Errors
    /// The functional was not decided on `graph`, or the laws, request or
    /// resources do not compile.
    pub fn mz_transport(
        graph: antecedent_graph::Admg,
        functional: BoundMzTransportFunctional,
        search: SearchLimits,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMzTransport, IoError> {
        let rebound = antecedent_identify::bind_mz_transport_catalog(
            &graph,
            functional.derivation(),
            functional.catalog(),
        )?;
        if rebound.root() != functional.root() || rebound.arena() != functional.arena() {
            return Err(err("mz_transport.functional_graph_mismatch"));
        }
        let plan = compile(&functional, &data, &request, limits, ctx)?;
        Ok(PreparedMzTransport {
            graph,
            functional,
            search,
            data,
            request,
            limits,
            plan,
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
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMzTransport, IoError> {
        require_counts(&data)?;
        let mut prepared =
            Self::mz_transport(graph, functional, search, data, request, limits, ctx)?;
        prepared.empirical = true;
        Ok(prepared)
    }
}

fn compile(
    functional: &BoundMzTransportFunctional,
    data: &ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<ExactEvaluationPlan, IoError> {
    antecedent_estimate::prepare_exact_mz_transport(
        functional,
        data.clone(),
        request.clone(),
        limits,
        ctx,
    )
    .map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))
}

fn require_counts(data: &ExactTransportData) -> Result<(), IoError> {
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Err(err("mz_transport.empirical_counts_required"));
    }
    Ok(())
}

impl PreparedMzTransport {
    /// The bounded theorem scope of this route.
    #[must_use]
    pub fn theorem_scope(&self) -> TheoremScope {
        TheoremScope::mz_transportability()
    }

    /// Evaluate the retained plan. Exact laws are point-only; empirical tables
    /// attach a nominal joint bootstrap, or the stated reason it is withheld.
    ///
    /// # Errors
    /// Cancellation or an evaluation refusal.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<MzTransportResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "mz-transport estimate")
            .map_err(estimate_err)?;
        let distribution = self
            .plan
            .evaluate(ctx)
            .map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))?;
        if !self.empirical {
            return Ok(MzTransportResult {
                distribution,
                uncertainty: MzUncertaintyWire::point_only(),
            });
        }
        let options = antecedent_estimate::EmpiricalTableOptions::default();
        let interval = antecedent_estimate::mz_transport_bootstrap_interval(
            &self.functional,
            &self.data,
            std::slice::from_ref(&self.request),
            self.limits,
            options.bootstrap_replicates,
            options.coverage_level,
            ctx,
        )
        .map_err(estimate_err)?;
        let uncertainty = match interval {
            Ok(interval) => MzUncertaintyWire {
                status: MZ_NOMINAL_INTERVAL.into(),
                reason: interval.reason.to_string(),
                method: Some(interval.method.to_string()),
                coverage_target: Some(interval.coverage_target),
                replicates_requested: interval.replicates_requested,
                replicates_ok: interval.replicates_ok,
                replicates_failed: interval.replicates_failed,
                mean_intervals: interval.requests[0]
                    .mean_intervals
                    .iter()
                    .map(|(v, lo, hi)| (v.raw(), *lo, *hi))
                    .collect(),
            },
            Err(reason) => MzUncertaintyWire {
                status: MZ_WITHHELD.into(),
                reason: reason.into(),
                ..MzUncertaintyWire::point_only()
            },
        };
        Ok(MzTransportResult { distribution, uncertainty })
    }

    /// Replace the laws for the snapshots the frozen catalog binds and recompile.
    /// The proof, bindings and request are unchanged; laws under any other
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
        let plan = compile(&self.functional, &data, &self.request, self.limits, ctx)?;
        Ok(Self { data, plan, ..self.clone() })
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
}
