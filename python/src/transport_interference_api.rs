//! Python bindings for structural transport and randomized interference.
//!
//! Transport identification is a stage (`identify_transport`). Estimation of
//! both licensed cells runs on the prepared study (`PreparedAnalysis.
//! prepare_transport` / `prepare_interference`); this module owns the query
//! builders those entry points share and the result sections they report.

use std::sync::Arc;

use antecedent::estimate::{
    NetworkData, NetworkEdge, estimate_interference as facade_estimate_interference,
    trial_to_target_effect,
};
use antecedent::identify::{TransportFormula, TransportIdentification, TransportIdentifier};
use antecedent_core::{
    AssignmentDesign, CausalSchema, DerivativeScale, DerivativeWeighting, ExposureLevel,
    ExposureMapping, InterferenceFunctional, InterferenceQuery, ResponseQuery, TransportQuery,
    VariableId,
};
use antecedent_graph::SelectionDiagram;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use antecedent_expr::CausalExprArena;
use antecedent_identify::lower_transport_formula;
use antecedent_io::{expr_arena_from_wire, expr_arena_to_wire};

use crate::graphs::Admg;
use crate::response_api::build_functional;
use crate::{CausalIdentifyError, columns_to_batch, detach_catch, py_err};

#[pyclass(skip_from_py_object)]
struct TransportIdentificationResult {
    #[pyo3(get)]
    transportable: bool,
    #[pyo3(get)]
    outcome: String,
    #[pyo3(get)]
    formula_kind: Option<String>,
    #[pyo3(get)]
    rule: Option<String>,
    #[pyo3(get)]
    reason: Option<String>,
    #[pyo3(get)]
    message: Option<String>,
    #[pyo3(get)]
    selection_targets: Vec<String>,
    #[pyo3(get)]
    marginalize: Vec<String>,
    #[pyo3(get)]
    factor_populations: Vec<String>,
    #[pyo3(get)]
    factor_variables: Vec<Vec<String>>,
    #[pyo3(get)]
    factor_conditioned_on: Vec<Vec<String>>,
    #[pyo3(get)]
    factor_interventions: Vec<Vec<String>>,
    #[pyo3(get)]
    factor_regimes: Vec<Option<u32>>,
    #[pyo3(get)]
    expr_pretty: Option<String>,
    #[pyo3(get)]
    expr_latex: Option<String>,
    #[pyo3(get)]
    leaf_populations: Vec<String>,
    #[pyo3(get)]
    leaf_regimes: Vec<Option<u32>>,
    #[pyo3(get)]
    expr_root: Option<u32>,
    #[pyo3(get)]
    expr_wire_json: Option<String>,
    // Deliberately NOT exposed via `#[pyo3(get)]`. `estimate_trial_transport` gates on this
    // field, so it must hold the certificate this class was actually constructed with rather
    // than a value a caller could set from Python (this pyclass has no `#[new]` and no
    // setters, so a Python caller cannot forge or mutate one either).
    identification: TransportIdentification,
}

#[pyclass(skip_from_py_object)]
struct TrialTransportResult {
    /// Rule id of the transport certificate that authorized this estimate, so the result
    /// carries its own identification provenance instead of relying on the caller to
    /// remember which `TransportIdentificationResult` it passed in.
    #[pyo3(get)]
    rule: String,
    #[pyo3(get)]
    ipw: f64,
    #[pyo3(get)]
    aipw: Option<f64>,
    #[pyo3(get)]
    selection_probability_min: f64,
    #[pyo3(get)]
    selection_probability_max: f64,
    #[pyo3(get)]
    selection_effective_sample_size: f64,
    #[pyo3(get)]
    selection_extreme_weight_count: usize,
    #[pyo3(get)]
    treatment_probability_min: f64,
    #[pyo3(get)]
    treatment_probability_max: f64,
    #[pyo3(get)]
    treatment_effective_sample_size: f64,
    #[pyo3(get)]
    treatment_extreme_weight_count: usize,
}

#[pyclass(skip_from_py_object)]
struct InterferenceAnalysisResult {
    #[pyo3(get)]
    horvitz_thompson: f64,
    #[pyo3(get)]
    hajek: f64,
    #[pyo3(get)]
    conservative_variance: f64,
    #[pyo3(get)]
    from_probability_method: String,
    #[pyo3(get)]
    to_probability_method: String,
    #[pyo3(get)]
    minimum_exposure_probability: f64,
}

/// Trial-to-target transport estimate of one execution, with its two overlap diagnostics.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct TransportSection {
    #[pyo3(get)]
    ipw: f64,
    #[pyo3(get)]
    selection_probability_min: f64,
    #[pyo3(get)]
    selection_probability_max: f64,
    #[pyo3(get)]
    selection_effective_sample_size: f64,
    #[pyo3(get)]
    selection_extreme_weight_count: usize,
    #[pyo3(get)]
    treatment_probability_min: f64,
    #[pyo3(get)]
    treatment_probability_max: f64,
    #[pyo3(get)]
    treatment_effective_sample_size: f64,
    #[pyo3(get)]
    treatment_extreme_weight_count: usize,
}

impl TransportSection {
    pub(crate) fn from_estimate(estimate: &antecedent::estimate::TransportEffectEstimate) -> Self {
        let overlap = &estimate.overlap;
        Self {
            ipw: estimate.ipw,
            selection_probability_min: overlap.selection.probability_min,
            selection_probability_max: overlap.selection.probability_max,
            selection_effective_sample_size: overlap.selection.effective_sample_size,
            selection_extreme_weight_count: overlap.selection.extreme_weight_count,
            treatment_probability_min: overlap.treatment.probability_min,
            treatment_probability_max: overlap.treatment.probability_max,
            treatment_effective_sample_size: overlap.treatment.effective_sample_size,
            treatment_extreme_weight_count: overlap.treatment.extreme_weight_count,
        }
    }
}

/// Randomized exposure contrast of one execution.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct InterferenceSection {
    #[pyo3(get)]
    horvitz_thompson: f64,
    #[pyo3(get)]
    hajek: f64,
    #[pyo3(get)]
    conservative_variance: f64,
    #[pyo3(get)]
    from_probability_method: String,
    #[pyo3(get)]
    to_probability_method: String,
    #[pyo3(get)]
    minimum_exposure_probability: f64,
}

impl InterferenceSection {
    pub(crate) fn from_estimate(estimate: &antecedent::estimate::InterferenceEstimate) -> Self {
        Self {
            horvitz_thompson: estimate.contrast.horvitz_thompson,
            hajek: estimate.contrast.hajek,
            conservative_variance: estimate.contrast.conservative_variance,
            from_probability_method: probability_method(estimate.from_probability_method),
            to_probability_method: probability_method(estimate.to_probability_method),
            minimum_exposure_probability: estimate.minimum_exposure_probability,
        }
    }
}

/// Response functional arguments of a transport query, as `transport._response_args` sends them.
pub(crate) struct ResponseArgs {
    pub(crate) kind: String,
    pub(crate) treatments: Vec<String>,
    pub(crate) outcomes: Vec<String>,
    pub(crate) grid: Option<Vec<f64>>,
    pub(crate) at: Option<Vec<f64>>,
    pub(crate) direction: Option<Vec<f64>>,
    pub(crate) order: u8,
    pub(crate) scale: String,
    pub(crate) weighting: String,
}

/// Single-source transport query over names resolved by `resolve`.
pub(crate) fn transport_query(
    response: ResponseArgs,
    source_population: String,
    target_population: String,
    source_experiments: &[String],
    resolve: impl Fn(&[String]) -> PyResult<Arc<[VariableId]>>,
) -> PyResult<TransportQuery> {
    let functional = build_functional(
        &response.kind,
        &resolve(&response.treatments)?,
        &resolve(&response.outcomes)?,
        response.grid,
        response.at,
        response.direction,
        None,
        None,
        response.order,
        parse_scale(&response.scale)?,
        parse_weighting(&response.weighting)?,
    )?;
    Ok(TransportQuery::new(
        ResponseQuery::new(functional),
        source_population,
        target_population,
        resolve(source_experiments)?,
    ))
}

/// Resolve variable names against a data schema.
pub(crate) fn schema_ids(schema: &CausalSchema, names: &[String]) -> PyResult<Arc<[VariableId]>> {
    names
        .iter()
        .map(|name| crate::graph_build::schema_var_id(schema, name))
        .collect::<PyResult<Vec<_>>>()
        .map(Into::into)
}

/// Assignment design, exposure mapping and contrast of an interference query.
pub(crate) struct InterferenceArgs {
    pub(crate) assignment_kind: String,
    pub(crate) assignment_probabilities: Vec<f64>,
    pub(crate) treated: usize,
    pub(crate) clusters: Vec<u32>,
    pub(crate) treated_clusters: usize,
    pub(crate) exposure: String,
    pub(crate) from_level: (f64, f64),
    pub(crate) to_level: (f64, f64),
    pub(crate) probability_draws: u32,
}

/// Interference query on `outcome`.
pub(crate) fn interference_query(
    args: InterferenceArgs,
    outcome: VariableId,
) -> PyResult<InterferenceQuery> {
    let assignment = match args.assignment_kind.as_str() {
        "bernoulli" => {
            AssignmentDesign::Bernoulli { probabilities: args.assignment_probabilities.into() }
        }
        "complete" => AssignmentDesign::CompleteRandomization { treated: args.treated },
        "cluster" => AssignmentDesign::ClusterRandomization {
            clusters: args.clusters.into(),
            treated_clusters: args.treated_clusters,
        },
        _ => return Err(PyValueError::new_err("unknown assignment design")),
    };
    let exposure = match args.exposure.as_str() {
        "own_treatment" => ExposureMapping::OwnTreatment,
        "neighbor_count" => ExposureMapping::NeighborCount,
        "neighbor_fraction" => ExposureMapping::NeighborFraction,
        "weighted_neighbor_exposure" => ExposureMapping::WeightedNeighborExposure,
        _ => return Err(PyValueError::new_err("unknown exposure mapping")),
    };
    Ok(InterferenceQuery {
        assignment,
        exposure,
        functional: InterferenceFunctional::ExposureContrast {
            outcome,
            from: ExposureLevel { own: args.from_level.0, neighbors: args.from_level.1 },
            to: ExposureLevel { own: args.to_level.0, neighbors: args.to_level.1 },
        },
        probability_draws: args.probability_draws,
    })
}

#[pyfunction]
#[pyo3(signature = (
    graph, selections, source_population, target_population, source_experiments,
    kind, treatments, outcomes, *, grid=None, at=None, direction=None, order=1,
    scale="identity", weighting="observed", catalog=None
))]
#[allow(clippy::too_many_arguments)]
fn identify_transport(
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source_population: String,
    target_population: String,
    source_experiments: Vec<String>,
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    grid: Option<Vec<f64>>,
    at: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    order: u8,
    scale: &str,
    weighting: &str,
    catalog: Option<&Bound<'_, PyAny>>,
) -> PyResult<TransportIdentificationResult> {
    let resolve = |names: &[String]| resolve_names(&graph, names);
    let mut query = transport_query(
        ResponseArgs {
            kind,
            treatments,
            outcomes,
            grid,
            at,
            direction,
            order,
            scale: scale.to_owned(),
            weighting: weighting.to_owned(),
        },
        source_population,
        target_population,
        &source_experiments,
        resolve,
    )?;
    if let Some(catalog) = catalog {
        query = query
            .with_catalog(parse_catalog(catalog, &graph)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    let selection_ids = resolve_names(&graph, &selections)?;
    let diagram = SelectionDiagram::try_new(graph.aligned_to_names(&graph.names)?, selection_ids)
        .map_err(py_err)?;
    let result = TransportIdentifier::new()
        .identify(&diagram, &query)
        .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
    transport_result(result, &graph.names)
}

pub(crate) fn parse_catalog(
    catalog: &Bound<'_, PyAny>,
    graph: &Admg,
) -> PyResult<antecedent_core::EvidenceCatalog> {
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
        EvidenceRegime, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, TargetSampling,
        VariableCoordinate, VariableDomain,
    };
    let names = |object: &Bound<'_, PyAny>, field: &str| -> PyResult<Arc<[VariableId]>> {
        resolve_names(graph, &object.getattr(field)?.extract::<Vec<String>>()?)
    };
    let mut environments = Vec::new();
    for env in catalog.getattr("environments")?.try_iter()? {
        let env = env?;
        let mut coordinates = Vec::new();
        for coordinate in env.getattr("variables")?.try_iter()? {
            let coordinate = coordinate?;
            let name: String = coordinate.getattr("name")?.extract()?;
            let variable = resolve_names(graph, &[name])?[0];
            let domain = match coordinate.getattr("domain")?.extract::<String>()?.as_str() {
                "unspecified" => VariableDomain::Unspecified,
                "continuous" => VariableDomain::Continuous,
                "binary" => VariableDomain::Binary,
                "count" => VariableDomain::Count,
                "categorical" => VariableDomain::Categorical {
                    cardinality: coordinate.getattr("cardinality")?.extract()?,
                },
                _ => return Err(PyValueError::new_err("unknown variable domain")),
            };
            let unit: Option<String> = coordinate.getattr("unit")?.extract()?;
            coordinates.push(VariableCoordinate { variable, domain, unit: unit.map(Arc::from) });
        }
        environments.push(
            Environment::try_new(
                env.getattr("identity")?.extract::<String>()?,
                coordinates,
                names(&env, "selection_targets")?,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    let mut regimes = Vec::new();
    let mut ids = std::collections::HashMap::new();
    let mut entries = catalog
        .getattr("regimes")?
        .try_iter()?
        .map(|regime| {
            let regime = regime?;
            let label = regime.getattr("id")?.extract::<String>()?;
            Ok((label, regime))
        })
        .collect::<PyResult<Vec<_>>>()?;
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (index, (name, regime)) in entries.into_iter().enumerate() {
        let id = RegimeId::from_raw(
            u32::try_from(index).map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        if ids.insert(name.clone(), id).is_some() {
            return Err(PyValueError::new_err("duplicate regime id"));
        }
        let kind = match regime.getattr("kind")?.extract::<String>()?.as_str() {
            "observational" => RegimeKind::Observational,
            "experimental" => RegimeKind::Experimental,
            _ => return Err(PyValueError::new_err("unknown regime kind")),
        };
        let evidence_kind = match regime.getattr("evidence_kind")?.extract::<String>()?.as_str() {
            "available" => EvidenceKind::Available,
            "manipulable" => EvidenceKind::Manipulable,
            "proposed" => EvidenceKind::Proposed,
            _ => return Err(PyValueError::new_err("unknown evidence kind")),
        };
        let measured = names(&regime, "measured")?;
        let distribution = match regime.getattr("distribution")?.extract::<String>()?.as_str() {
            "joint" => DistributionAvailability::Joint,
            "separate_marginals" => {
                DistributionAvailability::SeparateMarginals { variables: Arc::clone(&measured) }
            }
            _ => return Err(PyValueError::new_err("unknown distribution availability")),
        };
        let values: std::collections::BTreeMap<String, f64> =
            regime.getattr("intervention_values")?.extract()?;
        let assignments = values
            .into_iter()
            .map(|(name, value)| {
                Ok(antecedent_core::InterventionAssignment {
                    variable: resolve_names(graph, &[name])?[0],
                    value: antecedent_core::Value::f64(value),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let conditioned_on = names(&regime, "conditioned_on")?;
        regimes.push(
            EvidenceRegime::try_new(
                id,
                kind,
                evidence_kind,
                names(&regime, "interventions")?,
                assignments,
                measured,
                regime.getattr("population")?.extract::<String>()?,
                distribution,
            )
            .and_then(|mut r| {
                r.label = Some(Arc::from(name.as_str()));
                r.project(antecedent_core::EvidenceProjection::Condition { on: conditioned_on })
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    let mut bindings = Vec::new();
    for binding in catalog.getattr("bindings")?.try_iter()? {
        let binding = binding?;
        let name: String = binding.getattr("regime")?.extract()?;
        let regime =
            *ids.get(&name).ok_or_else(|| PyValueError::new_err("unknown bound regime"))?;
        let sampling = match binding.getattr("sampling")?.extract::<String>()?.as_str() {
            "independent" => SamplingDesign::Independent,
            "clustered" => SamplingDesign::Clustered,
            "unknown" => SamplingDesign::Unknown,
            _ => return Err(PyValueError::new_err("unknown sampling design")),
        };
        let dependence = match binding.getattr("dependence")?.extract::<String>()?.as_str() {
            "independent_studies" => DependenceGroup::IndependentStudies,
            "linked_units" => DependenceGroup::LinkedUnits,
            "unknown_dependence" => DependenceGroup::UnknownDependence,
            _ => return Err(PyValueError::new_err("unknown dependence group")),
        };
        bindings.push(RegimeBinding {
            dataset_identity: binding
                .getattr("dataset_identity")?
                .extract::<Option<String>>()?
                .map(Arc::from),
            regime,
            snapshot_identity: Arc::from(
                binding.getattr("snapshot_identity")?.extract::<String>()?,
            ),
            schema_names: binding
                .getattr("schema_names")?
                .extract::<Vec<String>>()?
                .into_iter()
                .map(Arc::from)
                .collect::<Vec<_>>()
                .into(),
            sampling,
            dependence,
            weights: binding
                .getattr("weights_snapshot")?
                .extract::<Option<String>>()?
                .map(|s| antecedent_core::LicensedWeights { snapshot_identity: Arc::from(s) }),
        });
    }
    let target_sampling =
        match catalog.getattr("target_sampling")?.extract::<Option<String>>()?.as_deref() {
            None => None,
            Some("supplied_population_law") => Some(TargetSampling::SuppliedPopulationLaw),
            Some("representative_sample") => Some(TargetSampling::RepresentativeSample),
            Some("licensed_weighted_design") => Some(TargetSampling::LicensedWeightedDesign),
            Some("convenience_sample") => Some(TargetSampling::ConvenienceSample),
            _ => return Err(PyValueError::new_err("unknown target sampling")),
        };
    EvidenceCatalog::try_new(environments, regimes, bindings, target_sampling)
        .and_then(|catalog| catalog.canonicalized())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn resolve_names(graph: &Admg, names: &[String]) -> PyResult<Arc<[VariableId]>> {
    names
        .iter()
        .map(|name| graph.name_index(name).map(|id| VariableId::from_raw(id.raw())))
        .collect::<PyResult<Vec<_>>>()
        .map(Into::into)
}

fn parse_scale(value: &str) -> PyResult<DerivativeScale> {
    match value {
        "identity" => Ok(DerivativeScale::Identity),
        "log_treatment" => Ok(DerivativeScale::LogTreatment),
        "log_outcome" => Ok(DerivativeScale::LogOutcome),
        "log_log" => Ok(DerivativeScale::LogLog),
        _ => Err(PyValueError::new_err("unknown derivative scale")),
    }
}

fn parse_weighting(value: &str) -> PyResult<DerivativeWeighting> {
    match value {
        "observed" => Ok(DerivativeWeighting::Observed),
        _ => Err(PyValueError::new_err(
            "transport identification currently supports weighting='observed'",
        )),
    }
}

fn variable_names(ids: &[VariableId], names: &[String]) -> PyResult<Vec<String>> {
    ids.iter()
        .map(|id| {
            names
                .get(id.as_usize())
                .cloned()
                .ok_or_else(|| PyValueError::new_err("transport formula contains unknown variable"))
        })
        .collect()
}

fn transport_result(
    result: TransportIdentification,
    names: &[String],
) -> PyResult<TransportIdentificationResult> {
    let mut out = TransportIdentificationResult {
        transportable: false,
        outcome: result.outcome().kind.as_str().to_owned(),
        formula_kind: None,
        rule: None,
        reason: None,
        message: None,
        selection_targets: Vec::new(),
        marginalize: Vec::new(),
        factor_populations: Vec::new(),
        factor_variables: Vec::new(),
        factor_conditioned_on: Vec::new(),
        factor_interventions: Vec::new(),
        factor_regimes: Vec::new(),
        expr_pretty: None,
        expr_latex: None,
        leaf_populations: Vec::new(),
        leaf_regimes: Vec::new(),
        expr_root: None,
        expr_wire_json: None,
        identification: result.clone(),
    };
    match result {
        TransportIdentification::NotCertified(certificate) => {
            out.reason = Some(certificate.reason.to_string());
            out.message = Some(certificate.message.to_string());
            out.selection_targets = variable_names(&certificate.witness, names)?;
        }
        TransportIdentification::MissingEvidence(certificate) => {
            out.reason = Some(certificate.reason.to_string());
            out.message = Some(certificate.message.to_string());
            out.selection_targets = variable_names(&certificate.missing, names)?;
        }
        TransportIdentification::Transportable { formula, certificate } => {
            out.transportable = true;
            out.rule = Some(certificate.rule.to_string());
            out.selection_targets = variable_names(&certificate.selection_targets, names)?;
            let factors = match &formula {
                TransportFormula::Direct(factor) => {
                    out.formula_kind = Some("direct".into());
                    vec![factor.clone()]
                }
                TransportFormula::Standardize { over, source_response, target_law } => {
                    out.formula_kind = Some("standardize".into());
                    out.marginalize = variable_names(over, names)?;
                    vec![source_response.clone(), target_law.clone()]
                }
                TransportFormula::RecursiveFactorization { sum_out, factors } => {
                    out.formula_kind = Some("recursive_factorization".into());
                    out.marginalize = variable_names(sum_out, names)?;
                    factors.to_vec()
                }
            };
            for factor in &factors {
                out.factor_regimes.push(factor.regime.map(antecedent_core::RegimeId::raw));
                out.factor_populations.push(factor.population.to_string());
                out.factor_variables.push(variable_names(&factor.variables, names)?);
                out.factor_conditioned_on.push(variable_names(&factor.conditioned_on, names)?);
                out.factor_interventions.push(variable_names(&factor.interventions, names)?);
            }
            let mut arena = CausalExprArena::new();
            let root = lower_transport_formula(&mut arena, &formula);
            antecedent_identify::bind_transport_derivation(
                &mut arena,
                root,
                &certificate,
                "catalog-bound transport formula",
            );
            out.expr_pretty = Some(arena.pretty(root));
            out.expr_latex = Some(arena.latex(root));
            for binding in arena.leaf_bindings(root) {
                out.leaf_populations.push(binding.population.to_string());
                out.leaf_regimes.push(binding.regime.map(antecedent_core::RegimeId::raw));
            }
            out.expr_root = Some(root.raw());
            let wire = expr_arena_to_wire(&arena).map_err(py_err)?;
            out.expr_wire_json = Some(
                serde_json::to_string(&wire).map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
    }
    Ok(out)
}

type ReloadedExpression = (String, String, Vec<String>, Vec<Option<u32>>, Vec<u32>);

#[pyfunction]
fn roundtrip_expr_arena(wire_json: &str, root: u32) -> PyResult<ReloadedExpression> {
    let wire: antecedent_io::ExprArenaWire =
        serde_json::from_str(wire_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let arena = expr_arena_from_wire(&wire).map_err(py_err)?;
    if root as usize >= arena.len() {
        return Err(PyValueError::new_err("expression root is out of range"));
    }
    let root = antecedent_expr::ExprId::from_raw(root);
    let pretty = arena.pretty(root);
    let latex = arena.latex(root);
    let mut populations = Vec::new();
    let mut regimes = Vec::new();
    for binding in arena.leaf_bindings(root) {
        populations.push(binding.population.to_string());
        regimes.push(binding.regime.map(antecedent_core::RegimeId::raw));
    }
    let free = arena.free_variables(root).into_iter().map(VariableId::raw).collect();
    Ok((pretty, latex, populations, regimes, free))
}

#[pyfunction]
#[pyo3(signature = (
    identification, outcome, treatment, trial, selection_probability, treatment_probability, *,
    mu0=None, mu1=None
))]
#[allow(clippy::too_many_arguments)]
fn estimate_trial_transport(
    py: Python<'_>,
    identification: PyRef<'_, TransportIdentificationResult>,
    outcome: PyReadonlyArray1<'_, f64>,
    treatment: Vec<bool>,
    trial: Vec<bool>,
    selection_probability: PyReadonlyArray1<'_, f64>,
    treatment_probability: PyReadonlyArray1<'_, f64>,
    mu0: Option<PyReadonlyArray1<'_, f64>>,
    mu1: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<TrialTransportResult> {
    if mu0.is_some() != mu1.is_some() {
        return Err(PyValueError::new_err("mu0 and mu1 must be supplied together"));
    }
    // Gate on the certificate the `TransportIdentificationResult` was actually built from, not
    // on its `transportable` getter: estimating a transported contrast without a positive
    // identification certificate would silently report a number for a quantity that was never
    // shown to be identified.
    let rule = match &identification.identification {
        TransportIdentification::NotCertified(certificate) => {
            return Err(py_err(antecedent::estimate::EstimationError::not_certified(
                "trial-to-target effect",
                &certificate.reason,
                &certificate.message,
            )));
        }
        TransportIdentification::MissingEvidence(certificate) => {
            return Err(py_err(antecedent::estimate::EstimationError::not_certified(
                "trial-to-target effect",
                &certificate.reason,
                &certificate.message,
            )));
        }
        TransportIdentification::Transportable { certificate, .. } => certificate.rule.to_string(),
    };
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let selection = selection_probability.as_array().iter().copied().collect::<Vec<_>>();
    let propensity = treatment_probability.as_array().iter().copied().collect::<Vec<_>>();
    let mu0 = mu0.map(|values| values.as_array().iter().copied().collect::<Vec<_>>());
    let mu1 = mu1.map(|values| values.as_array().iter().copied().collect::<Vec<_>>());
    let identification = identification.identification.clone();
    // Buffers are owned copies now; run the estimator off the GIL like the
    // interference path below.
    let estimate = detach_catch(py, move || {
        let regressions = mu0.as_deref().zip(mu1.as_deref());
        trial_to_target_effect(
            &identification,
            &y,
            &treatment,
            &trial,
            &selection,
            &propensity,
            regressions,
        )
        .map_err(py_err)
    })?;
    Ok(TrialTransportResult {
        rule,
        ipw: estimate.ipw,
        aipw: estimate.aipw,
        selection_probability_min: estimate.overlap.selection.probability_min,
        selection_probability_max: estimate.overlap.selection.probability_max,
        selection_effective_sample_size: estimate.overlap.selection.effective_sample_size,
        selection_extreme_weight_count: estimate.overlap.selection.extreme_weight_count,
        treatment_probability_min: estimate.overlap.treatment.probability_min,
        treatment_probability_max: estimate.overlap.treatment.probability_max,
        treatment_effective_sample_size: estimate.overlap.treatment.effective_sample_size,
        treatment_extreme_weight_count: estimate.overlap.treatment.extreme_weight_count,
    })
}

#[pyfunction]
#[pyo3(signature = (
    outcome, assignment, edges, assignment_kind, assignment_probabilities, treated,
    clusters, treated_clusters, exposure, from_level, to_level, *, probability_draws=10_000,
    seed=1
))]
#[allow(clippy::too_many_arguments)]
fn estimate_network_interference(
    py: Python<'_>,
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    edges: Vec<(u32, u32, f64)>,
    assignment_kind: String,
    assignment_probabilities: Vec<f64>,
    treated: usize,
    clusters: Vec<u32>,
    treated_clusters: usize,
    exposure: String,
    from_level: (f64, f64),
    to_level: (f64, f64),
    probability_draws: u32,
    seed: u64,
) -> PyResult<InterferenceAnalysisResult> {
    let names = vec!["outcome".to_owned()];
    let batch = columns_to_batch(&names, &[outcome])?;
    let query = interference_query(
        InterferenceArgs {
            assignment_kind,
            assignment_probabilities,
            treated,
            clusters,
            treated_clusters,
            exposure,
            from_level,
            to_level,
            probability_draws,
        },
        VariableId::from_raw(0),
    )?;
    detach_catch(py, move || {
        let loaded = antecedent_data::tabular_from_record_batch(&batch).map_err(py_err)?;
        let network_edges = edges
            .into_iter()
            .map(|(from, to, weight)| NetworkEdge { from, to, weight })
            .collect::<Vec<_>>();
        let network = NetworkData::try_new(loaded.data, network_edges).map_err(py_err)?;
        let result =
            facade_estimate_interference(&query, &network, &assignment, seed).map_err(py_err)?;
        Ok(InterferenceAnalysisResult {
            horvitz_thompson: result.contrast.horvitz_thompson,
            hajek: result.contrast.hajek,
            conservative_variance: result.contrast.conservative_variance,
            from_probability_method: probability_method(result.from_probability_method),
            to_probability_method: probability_method(result.to_probability_method),
            minimum_exposure_probability: result.minimum_exposure_probability,
        })
    })
}

fn probability_method(method: antecedent_stats::ExposureProbabilityMethod) -> String {
    match method {
        antecedent_stats::ExposureProbabilityMethod::Exact => "exact".into(),
        antecedent_stats::ExposureProbabilityMethod::MonteCarlo { draws, seed } => {
            format!("monte_carlo(draws={draws},seed={seed})")
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TransportIdentificationResult>()?;
    m.add_class::<TrialTransportResult>()?;
    m.add_class::<InterferenceAnalysisResult>()?;
    m.add_class::<TransportSection>()?;
    m.add_class::<InterferenceSection>()?;
    m.add_function(wrap_pyfunction!(identify_transport, m)?)?;
    m.add_function(wrap_pyfunction!(roundtrip_expr_arena, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_trial_transport, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_network_interference, m)?)?;
    Ok(())
}
