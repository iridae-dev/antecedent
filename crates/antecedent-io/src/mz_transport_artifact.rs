//! Independent point-result artifacts for multi-source limited-experiment transport.
//!
//! Format version 1. A consumer trusts nothing in the artifact. It re-runs the
//! bounded `TR^mz` decision on the stored graph, query and catalog under its own
//! search limits and accepts only the identical derivation, including the search
//! receipt of the stages that preceded it; re-binds every leaf to the regime of
//! its own population and compares the source-specific bindings; re-validates the
//! laws; recomputes every request's point and every contrast bit for bit; and
//! re-derives the interval bookkeeping. The verified identity is two digests: the
//! premises digest (graph, query, proof, expression, search and evaluation
//! limits, requests and the variable-name mapping) and the data-identity digest
//! (catalog snapshot and dataset ids and every law's snapshot, which a refresh
//! may replace). A stored interval carries its seed and replicate spec and is
//! recomputed bit for bit. The public multi-source interval route is closed
//! (`cell_not_licensed`) until its coverage records exist, so public producers
//! store no interval and public consumers refuse one
//! ([`refuse_unlicensed_interval`]). Limits an artifact records are provenance;
//! a stored limit larger than the consumer's refuses.

use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, IdentityDomain, InterventionAssignment, SearchLimits,
    VariableId,
};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMzTransportFunctional, IdentificationError, MzTransportDerivation,
    MzTransportDerivationRecord, MzTransportQuery, ZTransportSourceSpec, bind_mz_transport_catalog,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const MZ_TRANSPORT_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const MZ_TRANSPORT_ARTIFACT_FEATURE: &str = "checked_mz_transport_point_v1";
/// Uncertainty status: exact laws, no sampling uncertainty.
pub const MZ_POINT_ONLY: &str = "point_only";
/// Uncertainty status: no interval, for the stated reason.
pub const MZ_WITHHELD: &str = "withheld";
/// Uncertainty status: a nominal pointwise percentile-bootstrap interval of the
/// internal estimator. Public routes never publish one while the route is closed.
pub const MZ_NOMINAL_INTERVAL: &str = "nominal_interval";
/// Why the public multi-source interval route publishes no interval: the joint
/// bootstrap is registered closed until its coverage records are measured.
pub const MZ_INTERVAL_NOT_LICENSED: &str = "cell_not_licensed";
/// Detail of the closed interval route: a public consumer refuses an artifact
/// that carries an interval with `cell_not_licensed` and this detail.
pub const MZ_INTERVAL_NOT_LICENSED_DETAIL: &str = "mz_transport.interval_withheld";

/// Why an mz-transport artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MzTransportArtifactError {
    /// The feature marker or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The re-run decision did not reproduce the stored derivation.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The verified derivation does not bind to the stored catalog.
    #[error("catalog binding failed: {0}")]
    CatalogBinding(IdentificationError),
    /// The stored source-specific bindings differ from the re-derived ones.
    #[error("source bindings do not match the checked binding")]
    BindingMismatch,
    /// A stored law is not a valid law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// A recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// A recomputed contrast differs from the stored contrast.
    #[error("contrast does not replay")]
    ContrastMismatch,
    /// The stored interval bookkeeping contradicts the licensing decision or
    /// does not recompute bit for bit from its seed and replicate spec.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct MzTransportConsumeLimits {
    /// Limits of the re-run identification search.
    pub search: SearchLimits,
    /// Exact-evaluation operation and depth limits of the point replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
    /// Most cells one stored law may carry.
    pub max_law_cells: usize,
    /// Most requests (and so contrasts per outcome) an artifact may carry.
    pub max_requests: usize,
    /// Most bootstrap replicates a stored interval may ask the consumer to recompute.
    pub max_bootstrap_replicates: u32,
}

impl Default for MzTransportConsumeLimits {
    fn default() -> Self {
        Self {
            search: antecedent_identify::MZ_TRANSPORT_DEFAULT_LIMITS,
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
            max_law_cells: 1_000_000,
            max_requests: 64,
            max_bootstrap_replicates: 10_000,
        }
    }
}

/// One source of a stored query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzSourceWire {
    /// Source population.
    pub population: String,
    /// Declared controllable variables.
    pub controllable: Vec<u32>,
    /// Concrete experiment levels.
    pub experiment_assignment: Vec<(u32, ValueWire)>,
    /// Selection targets on this source's diagram.
    pub selection_targets: Vec<u32>,
}

/// Portable canonical mz query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzTransportQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Target population.
    pub target: String,
    /// Sources in canonical order.
    pub sources: Vec<MzSourceWire>,
}

impl MzTransportQueryWire {
    /// Encode a query in canonical form.
    #[must_use]
    pub fn from_query(query: &MzTransportQuery) -> Self {
        let query = query.canonical();
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            outcomes: raw(&query.outcomes),
            treatments: raw(&query.treatments),
            target: query.target.to_string(),
            sources: query
                .sources
                .iter()
                .map(|s| MzSourceWire {
                    population: s.population.to_string(),
                    controllable: raw(&s.controllable),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                        .collect(),
                    selection_targets: raw(&s.selection_targets),
                })
                .collect(),
        }
    }

    /// Decode. Validation happens when the query is decided.
    #[must_use]
    pub fn to_query(&self) -> MzTransportQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        MzTransportQuery {
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
            target: Arc::from(self.target.as_str()),
            sources: self
                .sources
                .iter()
                .map(|s| ZTransportSourceSpec {
                    population: Arc::from(s.population.as_str()),
                    controllable: ids(&s.controllable).into(),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|(v, x)| InterventionAssignment {
                            variable: VariableId::from_raw(*v),
                            value: x.to_value(),
                        })
                        .collect::<Vec<_>>()
                        .into(),
                    selection_targets: ids(&s.selection_targets).into(),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }
}

/// Point distribution in canonical atom order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzTransportPointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl MzTransportPointWire {
    /// Encode a distribution.
    #[must_use]
    pub fn from_distribution(distribution: &ExactDistribution) -> Self {
        Self {
            outcomes: distribution.outcomes.iter().map(|v| v.raw()).collect(),
            atoms: distribution
                .atoms
                .iter()
                .map(|atom| atom.iter().map(ValueWire::from_value).collect())
                .collect(),
            probabilities: distribution.probabilities.to_vec(),
        }
    }

    /// Bit-exact replay; non-finite probabilities fail closed.
    fn replays(&self, recomputed: &Self) -> bool {
        self.outcomes == recomputed.outcomes
            && self.atoms == recomputed.atoms
            && self.probabilities.len() == recomputed.probabilities.len()
            && self.probabilities.iter().zip(&recomputed.probabilities).all(|(stored, fresh)| {
                stored.is_finite() && fresh.is_finite() && stored.to_bits() == fresh.to_bits()
            })
    }
}

/// Point contrast of one outcome mean: request `request` minus request 0.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzContrastWire {
    /// Index of the contrasted request (at least 1).
    pub request: u32,
    /// Outcome coordinate.
    pub outcome: u32,
    /// `E[outcome | request] - E[outcome | request 0]`.
    pub estimate: f64,
}

/// Point contrasts of every outcome mean, each request against request 0, in
/// request then outcome order.
///
/// # Errors
/// An outcome without a finite numeric mean.
pub fn mz_point_contrasts(
    distributions: &[ExactDistribution],
) -> Result<Vec<MzContrastWire>, IoError> {
    let Some(base) = distributions.first() else {
        return Ok(Vec::new());
    };
    let mean = |distribution: &ExactDistribution, outcome: VariableId| {
        distribution.mean(outcome).map_err(|error| antecedent_estimate::refuse_eval(&error))
    };
    let mut out = Vec::new();
    for (k, distribution) in distributions.iter().enumerate().skip(1) {
        for outcome in base.outcomes.iter() {
            out.push(MzContrastWire {
                request: u32::try_from(k).unwrap_or(u32::MAX),
                outcome: outcome.raw(),
                estimate: mean(distribution, *outcome)? - mean(base, *outcome)?,
            });
        }
    }
    Ok(out)
}

/// Interval bookkeeping of one execution over all its requests.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzUncertaintyWire {
    /// [`MZ_POINT_ONLY`], [`MZ_WITHHELD`] or [`MZ_NOMINAL_INTERVAL`].
    pub status: String,
    /// `no_interval_reported`, [`MZ_INTERVAL_NOT_LICENSED`], the internal
    /// estimator's withheld reason, or `estimator_grid_not_measured` for its
    /// nominal interval.
    pub reason: String,
    /// On a closed-route withholding, the declared-sampling reason the internal
    /// estimator would also withhold for (for example
    /// `sampling_dependence_unknown` or `transport.unsupported_dependence`).
    pub dependence_reason: Option<String>,
    /// Interval method of an internal estimator run.
    pub method: Option<String>,
    /// Nominal level of an internal estimator run.
    pub coverage_target: Option<f64>,
    /// Master seed of an internal estimator run; a consumer recomputes from it.
    pub seed: Option<u64>,
    /// Requested bootstrap replicates; zero when no bootstrap ran.
    pub replicates_requested: u32,
    /// Replicates where every request evaluated.
    pub replicates_ok: u32,
    /// Replicates that failed.
    pub replicates_failed: u32,
    /// Per request, `(outcome, lower, upper)` pointwise mean intervals.
    pub mean_intervals: Vec<Vec<(u32, f64, f64)>>,
    /// `(request, outcome, lower, upper)` within-replicate contrast intervals.
    pub contrast_intervals: Vec<(u32, u32, f64, f64)>,
}

impl MzUncertaintyWire {
    /// Exact laws: no sampling uncertainty.
    #[must_use]
    pub fn point_only() -> Self {
        Self {
            status: MZ_POINT_ONLY.into(),
            reason: "no_interval_reported".into(),
            dependence_reason: None,
            method: None,
            coverage_target: None,
            seed: None,
            replicates_requested: 0,
            replicates_ok: 0,
            replicates_failed: 0,
            mean_intervals: Vec::new(),
            contrast_intervals: Vec::new(),
        }
    }

    /// Counted laws on the closed public route: no interval, reason
    /// [`MZ_INTERVAL_NOT_LICENSED`], with the declared-sampling reason the
    /// internal estimator would also withhold for, if any.
    #[must_use]
    pub fn not_licensed(dependence_reason: Option<&str>) -> Self {
        Self {
            status: MZ_WITHHELD.into(),
            reason: MZ_INTERVAL_NOT_LICENSED.into(),
            dependence_reason: dependence_reason.map(Into::into),
            ..Self::point_only()
        }
    }

    /// Bookkeeping of one internal joint-bootstrap run under `seed`.
    #[must_use]
    pub fn from_bootstrap(
        run: &Result<antecedent_estimate::MzTransportIntervals, &'static str>,
        replicates: u32,
        coverage_level: f64,
        seed: u64,
    ) -> Self {
        let base = Self {
            method: Some(antecedent_estimate::PERCENTILE_BOOTSTRAP.into()),
            coverage_target: Some(coverage_level),
            seed: Some(seed),
            replicates_requested: replicates,
            ..Self::point_only()
        };
        match run {
            Ok(intervals) => Self {
                status: MZ_NOMINAL_INTERVAL.into(),
                reason: intervals.reason.to_string(),
                replicates_ok: intervals.replicates_ok,
                replicates_failed: intervals.replicates_failed,
                mean_intervals: intervals
                    .requests
                    .iter()
                    .map(|request| {
                        request
                            .mean_intervals
                            .iter()
                            .map(|(v, lo, hi)| (v.raw(), *lo, *hi))
                            .collect()
                    })
                    .collect(),
                contrast_intervals: intervals
                    .contrasts
                    .iter()
                    .map(|(k, v, lo, hi)| {
                        (u32::try_from(*k).unwrap_or(u32::MAX), v.raw(), *lo, *hi)
                    })
                    .collect(),
                ..base
            },
            Err(reason) => Self { status: MZ_WITHHELD.into(), reason: (*reason).into(), ..base },
        }
    }

    /// Whether any interval is published.
    #[must_use]
    pub fn available(&self) -> bool {
        self.status == MZ_NOMINAL_INTERVAL
    }
}

/// Versioned mz point execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MzTransportArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Shared causal graph.
    pub graph: AdmgWire,
    /// Canonical query with each source's selection targets and controllables.
    pub query: MzTransportQueryWire,
    /// Search operation limit the producer decided under; bound by the premises digest.
    pub search_operations: usize,
    /// Search depth limit the producer decided under; bound by the premises digest.
    pub search_depth: usize,
    /// Checked proof premises, including the search receipt.
    pub proof: MzTransportDerivationRecord,
    /// Expression arena of the checked derivation.
    pub expression: ExprArenaWire,
    /// `(regime, population)` of every cited regime: the source-specific bindings.
    pub bindings: Vec<(u32, String)>,
    /// Evidence catalog, including declared sampling and study identities.
    pub catalog: EvidenceCatalogWire,
    /// Target and source laws in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget the producer evaluated under.
    pub max_support_rows: usize,
    /// Target requests; request 0 is the contrast baseline.
    pub requests: Vec<Vec<(u32, ValueWire)>>,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// Variable name of every graph coordinate, or empty when the producer
    /// named none; bound by the premises digest.
    pub variable_names: Vec<String>,
    /// Point result of every request, in request order.
    pub results: Vec<MzTransportPointWire>,
    /// Point contrasts of every outcome mean against request 0.
    pub contrasts: Vec<MzContrastWire>,
    /// Interval bookkeeping.
    pub uncertainty: MzUncertaintyWire,
    /// Digest of the canonical premises: graph, query, proof, expression,
    /// search and evaluation limits, requests and variable names.
    pub premises_digest: String,
    /// Digest of the data identity: catalog snapshot and dataset ids and every
    /// law's snapshot (refresh-replaceable).
    pub data_digest: String,
}

/// Everything a replay reconstructed and recomputed.
pub struct ConsumedMzTransport {
    /// The re-decided, re-bound functional.
    pub functional: BoundMzTransportFunctional,
    /// The re-validated laws.
    pub data: ExactTransportData,
    /// The target requests.
    pub requests: Vec<Assignment>,
    /// The recomputed point of every request.
    pub distributions: Vec<ExactDistribution>,
    /// The recomputed point contrasts.
    pub contrasts: Vec<MzContrastWire>,
    /// The decoded artifact.
    pub wire: MzTransportArtifactWire,
}

/// Everything the premises digest binds, in canonical form.
#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    query: &'a MzTransportQueryWire,
    proof: &'a MzTransportDerivationRecord,
    expression: &'a ExprArenaWire,
    search: (usize, usize),
    evaluation: (usize, usize, usize),
    requests: &'a [Vec<(u32, ValueWire)>],
    variable_names: &'a [String],
}

impl MzTransportArtifactWire {
    fn premises_view(&self) -> PremisesView<'_> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        PremisesView {
            tag: "mz_transport_point_v1",
            graph,
            query: &self.query,
            proof: &self.proof,
            expression: &self.expression,
            search: (self.search_operations, self.search_depth),
            evaluation: (self.operation_limit, self.depth_limit, self.max_support_rows),
            requests: &self.requests,
            variable_names: &self.variable_names,
        }
    }
}

/// `(regime, snapshot, dataset)` of every catalog binding and
/// `(population, regime, snapshot)` of every law, sorted.
fn data_digest(catalog: &EvidenceCatalog, laws: &[ExactLawWire]) -> Result<String, IoError> {
    let mut bindings = catalog
        .bindings
        .iter()
        .map(|b| {
            (
                b.regime.raw(),
                b.snapshot_identity.to_string(),
                b.dataset_identity.as_deref().map(str::to_owned),
            )
        })
        .collect::<Vec<_>>();
    bindings.sort();
    let mut snapshots = laws
        .iter()
        .map(|law| (law.population.clone(), law.regime, law.snapshot.clone()))
        .collect::<Vec<_>>();
    snapshots.sort();
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("mz_transport_data_v1", bindings, snapshots),
    )?
    .to_hex())
}

fn canonical_laws(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            crate::to_cbor(&(&wire.population, wire.regime, &wire.interventions))
                .map(|key| (key, wire))
        })
        .collect::<Result<Vec<_>, _>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

fn bindings_of(functional: &BoundMzTransportFunctional) -> Vec<(u32, String)> {
    functional.cited_populations().into_iter().map(|(id, p)| (id.raw(), p.to_string())).collect()
}

fn request_wire(request: &Assignment) -> Vec<(u32, ValueWire)> {
    request.entries().iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect()
}

fn request_of(wire: &[(u32, ValueWire)]) -> Assignment {
    Assignment::from_pairs(wire.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())))
}

/// Bit-exact equality of two encodable values (floats compare by bits).
fn same_bits<T: Serialize>(a: &T, b: &T) -> bool {
    matches!((crate::to_cbor(a), crate::to_cbor(b)), (Ok(a), Ok(b)) if a == b)
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// The checked premises and results an artifact is built from.
pub struct MzTransportArtifactInput<'a> {
    /// Shared causal graph the functional was decided on.
    pub graph: &'a antecedent_graph::Admg,
    /// The checked, catalog-bound functional.
    pub functional: &'a BoundMzTransportFunctional,
    /// Search limits the functional was decided under.
    pub search: SearchLimits,
    /// Retained laws.
    pub data: &'a ExactTransportData,
    /// Requests, in order; request 0 is the contrast baseline.
    pub requests: &'a [Assignment],
    /// Evaluation limits.
    pub limits: ExactEvaluationLimits,
    /// Variable name of every graph coordinate, or empty.
    pub variable_names: &'a [String],
    /// The point of every request.
    pub results: &'a [ExactDistribution],
    /// Interval bookkeeping.
    pub uncertainty: MzUncertaintyWire,
}

impl MzTransportArtifactWire {
    /// Build an artifact from checked premises, points and bookkeeping.
    ///
    /// # Errors
    /// The premises do not encode, or the results or bookkeeping are inconsistent.
    pub fn checked(
        input: MzTransportArtifactInput<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let graph = admg_to_wire(input.graph)?;
        let derivation = input.functional.derivation();
        let laws = canonical_laws(input.data)?;
        let mut wire = Self {
            version: MZ_TRANSPORT_ARTIFACT_VERSION,
            required_features: vec![MZ_TRANSPORT_ARTIFACT_FEATURE.into()],
            graph,
            query: MzTransportQueryWire::from_query(derivation.query()),
            search_operations: input.search.operations,
            search_depth: input.search.depth,
            proof: derivation.to_record(),
            expression: expr_arena_to_wire(derivation.arena())?,
            bindings: bindings_of(input.functional),
            catalog: EvidenceCatalogWire::from_catalog(input.functional.catalog()),
            data_digest: data_digest(input.functional.catalog(), &laws)?,
            laws,
            max_support_rows: input.data.max_support_rows(),
            requests: input.requests.iter().map(request_wire).collect(),
            operation_limit: input.limits.operations,
            depth_limit: input.limits.depth,
            variable_names: input.variable_names.to_vec(),
            results: input.results.iter().map(MzTransportPointWire::from_distribution).collect(),
            contrasts: mz_point_contrasts(input.results)?,
            uncertainty: input.uncertainty,
            premises_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.validate_shape()?;
        check_uncertainty(
            input.functional,
            input.data,
            input.requests,
            input.limits,
            &wire.uncertainty,
            &wire.results,
            ctx,
        )?;
        Ok(wire)
    }

    /// The digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-decides and re-binds every premise.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &self.premises_view(),
        )?
        .to_hex())
    }

    /// The data-identity digest the stored snapshot ids should carry.
    ///
    /// # Errors
    /// The catalog or laws do not decode or encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_digest(&self.catalog.to_catalog()?, &self.laws)
    }

    fn validate_shape(&self) -> Result<(), MzTransportArtifactError> {
        let unsupported = MzTransportArtifactError::UnsupportedSemantics;
        if self.required_features != [MZ_TRANSPORT_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        if self.requests.is_empty() || self.results.len() != self.requests.len() {
            return Err(unsupported("one point result per request"));
        }
        if self.results.iter().any(|result| {
            result.atoms.len() != result.probabilities.len()
                || result.outcomes != self.query.outcomes
        }) {
            return Err(unsupported("point result shape"));
        }
        let nodes = self.graph.node_count;
        if !self.variable_names.is_empty()
            && (u32::try_from(self.variable_names.len()).ok() != Some(nodes)
                || self.variable_names.iter().any(|name| name.trim().is_empty())
                || self.variable_names.iter().collect::<std::collections::BTreeSet<_>>().len()
                    != self.variable_names.len())
        {
            return Err(unsupported("variable names"));
        }
        Ok(())
    }

    /// Check a caller's variable names against the verified name mapping.
    ///
    /// # Errors
    /// [`MzTransportArtifactError::NamesMismatch`] unless `names` equals the
    /// stored mapping (an artifact without a mapping accepts none but the empty one).
    pub fn check_variable_names(&self, names: &[String]) -> Result<(), MzTransportArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(MzTransportArtifactError::NamesMismatch)
        }
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != MZ_TRANSPORT_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &MzTransportConsumeLimits,
    ) -> Result<(), MzTransportArtifactError> {
        let exceeded = MzTransportArtifactError::LimitsExceeded;
        if self.search_operations > limits.search.operations {
            return Err(exceeded("search operation limit"));
        }
        if self.search_depth > limits.search.depth {
            return Err(exceeded("search depth limit"));
        }
        if self.operation_limit > limits.evaluation.operations {
            return Err(exceeded("operation limit"));
        }
        if self.depth_limit > limits.evaluation.depth {
            return Err(exceeded("depth limit"));
        }
        if self.max_support_rows > limits.max_support_rows {
            return Err(exceeded("support rows"));
        }
        if self.laws.len() > limits.max_laws {
            return Err(exceeded("law count"));
        }
        if self.requests.len() > limits.max_requests {
            return Err(exceeded("request count"));
        }
        if self.uncertainty.replicates_requested > limits.max_bootstrap_replicates {
            return Err(exceeded("bootstrap replicates"));
        }
        if self.laws.iter().any(|law| {
            law.probabilities.len() > limits.max_law_cells
                || law.empirical_counts.as_ref().is_some_and(|c| c.len() > limits.max_law_cells)
        }) {
            return Err(exceeded("law cells"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then recompute
    /// every point and contrast and any stored interval. No external provider is
    /// accessed.
    ///
    /// # Errors
    /// Any reconstruction failure, an evaluation refusal, a point, contrast or
    /// interval that does not replay, or bookkeeping that contradicts the
    /// licensing decision.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: MzTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedMzTransport, IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(MzTransportArtifactError::PremisesMismatch.into());
        }
        let catalog = wire.catalog.to_catalog()?;
        if data_digest(&catalog, &wire.laws)? != wire.data_digest {
            return Err(MzTransportArtifactError::DataIdentityMismatch.into());
        }
        let graph = admg_from_wire(&wire.graph)?;
        let query = wire.query.to_query();
        let arena = expr_arena_from_wire(&wire.expression)?;
        let derivation = MzTransportDerivation::from_record_checked(
            &graph,
            &query,
            &catalog,
            &wire.proof,
            &arena,
            limits.search,
            ctx,
        )
        .map_err(MzTransportArtifactError::ProofMismatch)?;
        let functional = bind_mz_transport_catalog(&graph, &derivation, &catalog)
            .map_err(MzTransportArtifactError::CatalogBinding)?;
        if bindings_of(&functional) != wire.bindings {
            return Err(MzTransportArtifactError::BindingMismatch.into());
        }
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| MzTransportArtifactError::LawInvalid(error.to_string()))?;
        let data = ExactTransportData::try_new(laws, limits.max_support_rows)
            .map_err(|error| MzTransportArtifactError::LawInvalid(error.to_string()))?;
        let requests = wire.requests.iter().map(|r| request_of(r)).collect::<Vec<_>>();
        let mut distributions = Vec::with_capacity(requests.len());
        for (request, stored) in requests.iter().zip(&wire.results) {
            let distribution = antecedent_estimate::evaluate_exact_mz_transport(
                &functional,
                data.clone(),
                request.clone(),
                limits.evaluation,
                ctx,
            )
            .map_err(|error| antecedent_estimate::refuse_eval(&error))?;
            if !stored.replays(&MzTransportPointWire::from_distribution(&distribution)) {
                return Err(MzTransportArtifactError::PointMismatch.into());
            }
            distributions.push(distribution);
        }
        let contrasts = mz_point_contrasts(&distributions)?;
        if !same_bits(&contrasts, &wire.contrasts) {
            return Err(MzTransportArtifactError::ContrastMismatch.into());
        }
        let producer_limits =
            ExactEvaluationLimits { operations: wire.operation_limit, depth: wire.depth_limit };
        check_uncertainty(
            &functional,
            &data,
            &requests,
            producer_limits,
            &wire.uncertainty,
            &wire.results,
            ctx,
        )?;
        Ok(ConsumedMzTransport { functional, data, requests, distributions, contrasts, wire })
    }

    /// [`Self::consume_with_limits`] under the default consumer limits.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<ConsumedMzTransport, IoError> {
        Self::consume_with_limits(bytes, MzTransportConsumeLimits::default(), ctx)
    }
}

/// The public-route policy on consumed bookkeeping: the multi-source interval
/// route is closed until its coverage records exist, so a public consumer
/// refuses any artifact carrying an internal estimator run, even one that
/// recomputes bit for bit.
///
/// # Errors
/// [`IoError::Refused`] with `cell_not_licensed` and detail
/// [`MZ_INTERVAL_NOT_LICENSED_DETAIL`].
pub fn refuse_unlicensed_interval(uncertainty: &MzUncertaintyWire) -> Result<(), IoError> {
    if uncertainty.seed.is_some() || uncertainty.available() {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            message: format!(
                "{MZ_INTERVAL_NOT_LICENSED_DETAIL}: the multi-source interval route is closed until its coverage records are measured"
            ),
        });
    }
    Ok(())
}

/// Re-derive the interval bookkeeping and check the stored one against it.
///
/// Exact laws are point-only. Counted laws without an estimator run carry the
/// closed-route bookkeeping ([`MzUncertaintyWire::not_licensed`]) with the
/// declared-sampling reason recomputed from the catalog and laws. An internal
/// estimator run carries its seed and replicate spec and must recompute bit
/// for bit.
fn check_uncertainty(
    functional: &BoundMzTransportFunctional,
    data: &ExactTransportData,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    stored: &MzUncertaintyWire,
    results: &[MzTransportPointWire],
    ctx: &ExecutionContext,
) -> Result<(), IoError> {
    let bad = |what| Err(MzTransportArtifactError::UncertaintyMismatch(what).into());
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return if *stored == MzUncertaintyWire::point_only() {
            Ok(())
        } else {
            bad("exact laws carry no interval")
        };
    }
    let Some(seed) = stored.seed else {
        let reason = antecedent_estimate::mz_interval_withheld_reason(functional, data)?;
        return if same_bits(stored, &MzUncertaintyWire::not_licensed(reason)) {
            Ok(())
        } else {
            bad("closed interval route bookkeeping")
        };
    };
    let (Some(coverage), true) = (
        stored.coverage_target,
        stored.method.as_deref() == Some(antecedent_estimate::PERCENTILE_BOOTSTRAP),
    ) else {
        return bad("interval run spec");
    };
    if !(coverage.is_finite() && 0.0 < coverage && coverage < 1.0)
        || stored.replicates_requested < 2
        || results.len() != requests.len()
    {
        return bad("interval run spec");
    }
    let mut replay = ctx.clone();
    replay.rng = antecedent_core::RngFactory::from_seed(seed);
    let run = antecedent_estimate::mz_transport_bootstrap_interval(
        functional,
        data,
        requests,
        limits,
        stored.replicates_requested,
        coverage,
        &replay,
    )?;
    let expected =
        MzUncertaintyWire::from_bootstrap(&run, stored.replicates_requested, coverage, seed);
    if same_bits(stored, &expected) { Ok(()) } else { bad("interval does not recompute") }
}
