//! Evidence relations, operations and atom combination.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Evidence relations, declared operations and atom combination.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationKindWire {
    kind: String,
    #[serde(default)]
    ids: Vec<String>,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteWire {
    kind: String,
    id: String,
    source_input: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationWire {
    left: String,
    right: String,
    relation: RelationKindWire,
    #[serde(default)]
    route: Option<RouteWire>,
}

fn relation_from(wire: RelationKindWire) -> Result<EvidenceRelation, BoundaryError> {
    let RelationKindWire { kind, ids, id } = wire;
    let single = |id: Option<String>| id.ok_or(BoundaryError::InvalidInput("a relation identity"));
    Ok(match kind.as_str() {
        "independent_sources" => EvidenceRelation::IndependentSources,
        "shared_data" => EvidenceRelation::SharedData { ids },
        "shared_prior" => EvidenceRelation::SharedPrior { id: single(id)? },
        "shared_fitted_model" => EvidenceRelation::SharedFittedModel { id: single(id)? },
        "unknown_dependence" => EvidenceRelation::UnknownDependence,
        _ => return Err(BoundaryError::InvalidInput("an evidence relation kind")),
    })
}

fn route_from(wire: RouteWire) -> Result<DependenceRoute, BoundaryError> {
    let kind = match wire.kind.as_str() {
        "covariance" => RouteKind::Covariance,
        "joint_law" => RouteKind::JointLaw,
        _ => return Err(BoundaryError::InvalidInput("a dependence route kind")),
    };
    Ok(DependenceRoute { kind, id: wire.id, source_input: wire.source_input })
}

fn relations_from(json: &str) -> Result<Vec<PairRelation>, BoundaryError> {
    let wires: Vec<RelationWire> = serde_json::from_str(json)
        .map_err(|_| BoundaryError::InvalidInput("a relation declaration"))?;
    wires
        .into_iter()
        .map(|wire| {
            Ok(PairRelation {
                left: wire.left,
                right: wire.right,
                relation: relation_from(wire.relation)?,
                route: wire.route.map(route_from).transpose()?,
            })
        })
        .collect()
}

fn operation_from(name: Option<&str>) -> Result<Option<CompositionOperation>, BoundaryError> {
    Ok(match name {
        None => None,
        Some("statistical_pooling") => Some(CompositionOperation::StatisticalPooling),
        Some("bayesian_borrowing") => Some(CompositionOperation::BayesianBorrowing),
        Some("causal_transport") => Some(CompositionOperation::CausalTransport),
        Some("evidence_reuse") => Some(CompositionOperation::EvidenceReuse),
        Some(_) => return Err(BoundaryError::InvalidInput("a composition operation")),
    })
}

const fn operation_name(operation: CompositionOperation) -> &'static str {
    match operation {
        CompositionOperation::StatisticalPooling => "statistical_pooling",
        CompositionOperation::BayesianBorrowing => "bayesian_borrowing",
        CompositionOperation::CausalTransport => "causal_transport",
        CompositionOperation::EvidenceReuse => "evidence_reuse",
    }
}

fn composition_receipt_json(
    receipt: &antecedent::design::composition_boundary::CompositionReceipt,
) -> String {
    json!({
        "operation": operation_name(receipt.operation),
        "independence_assumed": receipt.independence_assumed,
        "routes": receipt.routes,
        "shared_evidence": receipt.shared_evidence,
    })
    .to_string()
}

/// Check that inputs may be combined under one declared operation.
#[pyfunction]
#[pyo3(signature = (inputs, relations_json, operation=None))]
#[allow(clippy::needless_pass_by_value)]
fn composition_check(
    inputs: Vec<PyRef<'_, PyDecisionInput>>,
    relations_json: &str,
    operation: Option<&str>,
) -> PyResult<Pair> {
    check_size(relations_json)?;
    let owned: Vec<DecisionInput> = inputs.iter().map(|input| input.inner.clone()).collect();
    let checked = relations_from(relations_json).and_then(|relations| {
        let operation = operation_from(operation)?;
        check_composition(&owned, &relations, operation)
    });
    Ok(match checked {
        Ok(receipt) => (Some(composition_receipt_json(&receipt)), None),
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

/// Check that the draws of inputs may be paired row by row.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn composition_check_paired_draws(
    inputs: Vec<PyRef<'_, PyDecisionInput>>,
    relations_json: &str,
) -> PyResult<Pair> {
    check_size(relations_json)?;
    let owned: Vec<DecisionInput> = inputs.iter().map(|input| input.inner.clone()).collect();
    let checked =
        relations_from(relations_json).and_then(|relations| check_paired_draws(&owned, &relations));
    Ok(match checked {
        Ok(receipt) => (Some(composition_receipt_json(&receipt)), None),
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomWire {
    id: String,
    #[serde(default)]
    probability: Option<f64>,
}

/// Check that conflicting structural atoms may be combined as requested.
#[pyfunction]
fn composition_check_atoms(atoms_json: &str, combination: &str) -> PyResult<Pair> {
    check_size(atoms_json)?;
    let wires: Vec<AtomWire> = serde_json::from_str(atoms_json)
        .map_err(|e| crate::value_err(format!("invalid atoms: {e}")))?;
    let combination = match combination {
        "report_each" => AtomCombination::ReportEach,
        "worst_case" => AtomCombination::WorstCase,
        "weighted_by_declared_probabilities" => AtomCombination::WeightedByDeclaredProbabilities,
        other => {
            return Err(crate::value_err(format!("unknown atom combination {other:?}")));
        }
    };
    let atoms: Vec<StructuralAtom> = wires
        .into_iter()
        .map(|wire| StructuralAtom {
            id: wire.id,
            probability: wire.probability,
            evidence: AtomEvidence::Unidentified,
        })
        .collect();
    Ok(match check_atom_combination(&atoms, combination) {
        Ok(AtomPlan::ReportEach) => (Some(json!({"kind": "report_each"}).to_string()), None),
        Ok(AtomPlan::WorstCase) => (Some(json!({"kind": "worst_case"}).to_string()), None),
        Ok(AtomPlan::Weighted(weights)) => {
            (Some(json!({"kind": "weighted", "weights": weights}).to_string()), None)
        }
        Err(error) => (None, Some(boundary_refusal(&error))),
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(composition_check, m)?)?;
    m.add_function(wrap_pyfunction!(composition_check_paired_draws, m)?)?;
    m.add_function(wrap_pyfunction!(composition_check_atoms, m)?)?;
    Ok(())
}
