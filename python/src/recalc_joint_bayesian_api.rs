//! Checked joint transport recalculation: one native aligned candidate posterior over the
//! sources, with measured work and no calibrated interval.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::recalc_api::{RunPayload, bounded_context, invalid, parse_json};
use crate::recalc_family_api::{caps, payload, resume, run_error};
use antecedent::analysis::recalc_bayesian::PosteriorSummarySpec;
use antecedent::analysis::recalc_joint_bayesian::{
    JointBayesianRequest, JointBayesianSession, execute_joint_bayesian_with_receipt,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_io::joint_bayesian_transport_artifact::{
    DataIdentityWire, GaussianPriorWire, JointBayesianConsumeLimits, PriorProvenanceWire,
};
use antecedent_io::recalc_receipt_artifact::{CapabilitiesWire, identities_to_wire};
use pyo3::prelude::*;
use serde::Deserialize;

const REFUSED: &str = "recalc.joint_bayesian_refused";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceWire {
    id: String,
    identity: DataIdentityWire,
    treatment: Vec<bool>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    noise_variance: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetWire {
    identity: DataIdentityWire,
    rows: usize,
    covariates: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    names: Vec<String>,
    edges: Vec<(String, String)>,
    selections: Vec<String>,
    treatment: String,
    outcome: String,
    source_population: String,
    target_population: String,
    features: Vec<String>,
    varying: String,
    sharing: String,
    dependence: String,
    invariant: GaussianPriorWire,
    varying_prior: GaussianPriorWire,
    sources: Vec<SourceWire>,
    target: TargetWire,
    draws: usize,
    seed: u64,
    max_unsupported_mass: f64,
    conflict_z_threshold: f64,
    lower_probability: f64,
    upper_probability: f64,
    threshold: f64,
    benefit_per_unit: f64,
    cost: f64,
}
fn identity(wire: DataIdentityWire) -> DataIdentity {
    DataIdentity { snapshot_digest: wire.snapshot_digest, datum_ids: wire.datum_ids }
}
fn prior(wire: GaussianPriorWire) -> GaussianPrior {
    GaussianPrior {
        mean: wire.mean,
        covariance: wire.covariance,
        provenance: match wire.provenance {
            PriorProvenanceWire::Declared => PriorProvenance::Declared,
            PriorProvenanceWire::Bank { bank_id, consumed } => PriorProvenance::Bank {
                bank_id,
                consumed: consumed.into_iter().map(identity).collect(),
            },
        },
    }
}
fn check_identity(identity: &DataIdentityWire, rows: usize) -> PyResult<()> {
    let ids: std::collections::HashSet<_> = identity.datum_ids.iter().collect();
    if identity.snapshot_digest.trim().is_empty()
        || identity.datum_ids.len() != rows
        || ids.len() != rows
        || identity.datum_ids.iter().any(|id| id.trim().is_empty())
    {
        return Err(invalid(
            "recalc.joint_bayesian_invalid_data",
            "observation identity must name every unit once",
        ));
    }
    Ok(())
}
fn request(specification: &str) -> PyResult<JointBayesianRequest> {
    let wire: RequestWire = parse_json(specification, "joint Bayesian request")?;
    let bad = |text: &str| invalid("recalc.joint_bayesian_invalid_query", text);
    if wire.names.len() > 64 {
        return Err(invalid("recalc.limits_exceeded", "joint transport variable count exceeds 64"));
    }
    let locate = |name: &str| -> PyResult<VariableId> {
        let Some(i) = wire.names.iter().position(|n| n == name) else {
            return Err(bad("unknown role variable"));
        };
        Ok(VariableId::from_raw(crate::fit_u32(i)?))
    };
    for source in &wire.sources {
        check_identity(&source.identity, source.outcome.len())?;
    }
    check_identity(&wire.target.identity, wire.target.rows)?;
    let treatment = locate(&wire.treatment)?;
    let outcome = locate(&wire.outcome)?;
    let mut graph = Admg::with_variables(crate::fit_u32(wire.names.len())?);
    for (from, to) in &wire.edges {
        graph
            .insert_directed(
                DenseNodeId::from_raw(locate(from)?.raw()),
                DenseNodeId::from_raw(locate(to)?.raw()),
            )
            .map_err(|e| bad(&e.to_string()))?;
    }
    let diagram = SelectionDiagram::try_new(
        graph,
        wire.selections.iter().map(|name| locate(name)).collect::<PyResult<Vec<_>>>()?,
    )
    .map_err(|e| bad(&e.to_string()))?;
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment, GridSpec::Values(vec![0.0, 1.0].into())),
        }),
        wire.source_population,
        wire.target_population,
        vec![treatment],
    );
    let features = wire
        .features
        .iter()
        .map(|name| locate(name).map(VariableId::raw))
        .collect::<PyResult<Vec<u32>>>()?;
    let varying = match wire.varying.as_str() {
        "intercept" => VaryingBlock::Intercept,
        "intercept_and_covariates" => VaryingBlock::InterceptAndCovariates,
        _ => return Err(bad("unknown varying block")),
    };
    let sharing = match wire.sharing.as_str() {
        "independent_varying_blocks" => SourceSharing::IndependentVaryingBlocks,
        "shared_varying_block" => SourceSharing::SharedVaryingBlock,
        _ => return Err(bad("unknown source sharing")),
    };
    let dependence = match wire.dependence.as_str() {
        "independent_samples" => SourceDependence::IndependentSamples,
        "overlapping_units" => SourceDependence::OverlappingUnits,
        "unknown" => SourceDependence::Unknown,
        _ => return Err(bad("unknown source dependence")),
    };
    // The same consumer bounds the sealed artifact is read under, checked before any work.
    let limits = JointBayesianConsumeLimits::default();
    let source_rows = wire.sources.iter().fold(0usize, |n, s| n.saturating_add(s.outcome.len()));
    let basis = features.len();
    let varying_dim = if varying == VaryingBlock::Intercept { 1 } else { 1 + basis };
    let invariant_dim = if varying == VaryingBlock::Intercept { 1 + 2 * basis } else { 1 + basis };
    let blocks = if sharing == SourceSharing::SharedVaryingBlock { 1 } else { wire.sources.len() };
    let width = invariant_dim
        .saturating_add(varying_dim.saturating_mul(blocks))
        .saturating_add(wire.sources.len())
        .saturating_add(1);
    if wire.sources.len() > limits.max_sources
        || source_rows > limits.max_source_rows
        || wire.target.rows > limits.max_target_rows
        || features.len() > limits.max_features
        || wire.draws > limits.max_draws
        || wire.draws.saturating_mul(width) > limits.max_draw_cells
    {
        return Err(invalid("recalc.limits_exceeded", "joint transport exceeds consumer bounds"));
    }
    let sources = wire
        .sources
        .into_iter()
        .map(|s| SourceData {
            id: s.id,
            identity: identity(s.identity),
            treatment: s.treatment,
            outcome: s.outcome,
            covariates: s.covariates,
            noise_variance: s.noise_variance,
        })
        .collect();
    Ok(JointBayesianRequest {
        diagram,
        query,
        model: JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features,
            varying,
            sharing,
            dependence,
            priors: JointPriors {
                invariant: prior(wire.invariant),
                varying: prior(wire.varying_prior),
            },
            max_unsupported_mass: wire.max_unsupported_mass,
            conflict_z_threshold: wire.conflict_z_threshold,
        },
        sources,
        target: TargetData {
            identity: identity(wire.target.identity),
            rows: wire.target.rows,
            covariates: wire.target.covariates,
        },
        options: JointTransportOptions { draws: wire.draws, seed: wire.seed },
        summary: PosteriorSummarySpec {
            lower_probability: wire.lower_probability,
            upper_probability: wire.upper_probability,
            threshold: wire.threshold,
        },
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}
#[pyclass(name = "JointBayesianSessionHandle")]
struct PyJointBayesianSession {
    inner: JointBayesianSession,
}
#[pymethods]
impl PyJointBayesianSession {
    #[new]
    fn new() -> Self {
        Self { inner: JointBayesianSession::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let (ids, ctx) = resume(previous_json, resume_json)?;
        Ok(Self { inner: JointBayesianSession::resume(ids, ctx) })
    }
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(crate::py_msg)
    }
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&caps(self.inner.boundary())))
            .map_err(crate::py_msg)
    }
    #[pyo3(signature=(specification,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<RunPayload> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = request(specification)?;
        let previous = self.inner.identities().clone();
        let capabilities = caps(self.inner.boundary());
        let mut session = std::mem::take(&mut self.inner);
        let (session, ran) = crate::detach_catch(py, move || {
            let ran = execute_joint_bayesian_with_receipt(&mut session, &request, &ctx);
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ok(out) => {
                let fit = self.inner.fit().map(|fit| {
                    serde_json::json!({
                        "target_effect_mean": fit.target_effect_mean,
                        "target_effect_variance": fit.target_effect_variance,
                        "effect_names": fit.effect_names,
                        "effect_means": fit.effect_means,
                    })
                });
                payload(
                    &previous,
                    self.inner.identities(),
                    &capabilities,
                    &out.plan,
                    &out.receipt,
                    serde_json::json!({
                        "law": {
                            "mean": out.law.mean,
                            "standard_deviation": out.law.standard_deviation,
                            "lower_quantile": out.law.lower_quantile,
                            "upper_quantile": out.law.upper_quantile,
                            "probability_below": out.law.probability_below,
                            "draws": out.law.draws,
                        },
                        "decision": {"net_benefit": out.decision.net_benefit, "treat": out.decision.treat},
                        "fit": fit,
                    }),
                )
            }
            Err(error) => Ok((None, None, Some(run_error(&error, REFUSED)))),
        }
    }
    #[pyo3(signature=(*,seed=1,threads=None))]
    fn export_result(&self, seed: u64, threads: Option<u32>) -> (Option<Vec<u8>>, Option<String>) {
        // The sealed result was produced at execution; export samples and fits nothing.
        let _ = (seed, threads);
        match self.inner.export_result() {
            Ok(bytes) => (Some(bytes), None),
            Err(error) => (None, Some(run_error(&error, REFUSED))),
        }
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyJointBayesianSession>()
}
