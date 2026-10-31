//! Helpers shared by the transport bindings: one error mapper, one variable
//! resolver, one law parser, and one execution-context builder.
use crate::graphs::Admg;
use antecedent_core::{ExecutionContext, Value, VariableId, reason_code};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactTransportData, InterventionAssignment,
    LawTolerance,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// The Python exception for one native transport failure.
///
/// Every binding routes its failures through here. A typed crate error is
/// matched on its kind ([`TransportPyErr`]); only a failure that is still a
/// plain message falls back to [`classify`].
pub(crate) fn error<E: TransportPyErr>(e: E) -> PyErr {
    e.into_transport_py_err()
}

/// The transport bindings' name for [`crate::bridge::serialization`], the one mapping of a
/// failure that can only mean the bytes do not describe what they claim.
pub(crate) use crate::bridge::serialization as serialization_error;

/// One native transport failure kind, mapped to its `Causal*` class.
pub(crate) trait TransportPyErr {
    fn into_transport_py_err(self) -> PyErr;
}

/// An exhausted identification budget is a resource refusal, never a finding.
fn budget_error(budget: antecedent_identify::IdentificationBudget) -> PyErr {
    crate::CausalResourceError::new_err(budget.code().to_string())
}

/// A reason-coded refusal. Two codes still need their message read:
/// `transport_budget_cancel` covers both a fired cancellation token and an
/// exhausted budget (the crates spell the former `<stage> cancelled`), and the
/// exact evaluator reports its operation, memory, support and recursion
/// budgets as provider-kind failures spelled `<budget> budget exceeded`, which
/// `refuse_eval` codes as `transport_missing_provider`.
fn coded_refusal(code: &'static str, message: String) -> PyErr {
    let lower = message.to_ascii_lowercase();
    if code == reason_code!("transport_budget_cancel") {
        if lower.contains("cancel") {
            return crate::CausalCancelledError::new_err(message);
        }
        return crate::CausalResourceError::new_err(message);
    }
    if code == reason_code!("transport_missing_provider") && lower.contains("budget exceeded") {
        return crate::CausalResourceError::new_err(message);
    }
    crate::refusal(code, message)
}

impl TransportPyErr for antecedent_identify::IdentificationError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_identify::IdentificationError as E;
        match self {
            E::Cancelled => crate::CausalCancelledError::new_err(self.to_string()),
            E::Budget { budget } => budget_error(budget),
            E::MissingEvidence { .. } => {
                crate::refusal(reason_code!("transport_missing_evidence"), self.to_string())
            }
            // A record that no longer checks against its inputs.
            E::InvalidDerivation { .. } => serialization_error(self),
            E::InvalidInput { .. }
            | E::InvalidCatalog { .. }
            | E::UnknownVariable { .. }
            | E::InvalidQuery { .. }
            | E::Graph(_) => crate::value_err(self.to_string()),
            E::UnsupportedInput { .. }
            | E::UnsupportedQuery { .. }
            | E::SustainedPolicyUnsupported
            | E::NotCertified { .. }
            | E::ResultLimitExceeded { .. }
            | E::InvariantViolated { .. } => {
                crate::refusal(reason_code!("transport_not_certified"), self.to_string())
            }
            // `IdentificationError` is `#[non_exhaustive]`; a new variant lands on
            // the identification class until its Python-facing category is decided.
            other => crate::CausalIdentifyError::new_err(other.to_string()),
        }
    }
}

impl TransportPyErr for antecedent_estimate::EstimationError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_estimate::EstimationError as E;
        match self {
            E::Refused { code, message } => coded_refusal(code, message.into_string()),
            other => crate::py_err(other),
        }
    }
}

impl TransportPyErr for antecedent_io::z_transport_artifact::ZTransportArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::z_transport_artifact::ZTransportArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            E::ProofMismatch(inner) | E::CatalogBinding(inner) if inner.is_budget_or_cancel() => {
                inner.into_transport_py_err()
            }
            E::ProofMismatch(_)
            | E::CatalogBinding(_)
            | E::UnsupportedSemantics(_)
            | E::LawInvalid(_)
            | E::ProgramMismatch(_)
            | E::PointMismatch
            | E::PremisesMismatch => serialization_error(self),
            // `#[non_exhaustive]`: a new consumer check is a consistency failure
            // until its class is decided.
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_io::mixed_source_artifact::MixedSourceArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::mixed_source_artifact::MixedSourceArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            E::ProofMismatch(inner) | E::CatalogBinding(inner) if inner.is_budget_or_cancel() => {
                inner.into_transport_py_err()
            }
            // Every other consumer check is a consistency failure of the artifact.
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_io::mz_transport_artifact::MzTransportArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::mz_transport_artifact::MzTransportArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            E::ProofMismatch(inner) | E::CatalogBinding(inner) if inner.is_budget_or_cancel() => {
                inner.into_transport_py_err()
            }
            // Every other consumer check is a consistency failure of the artifact;
            // a proof or binding that does not check also carries the X1 record's
            // frozen `(reason code, mz_transport.* detail)` pair.
            other => match other.refusal() {
                Some((code, detail)) => {
                    crate::with_reason_code(serialization_error(format!("{detail}: {other}")), code)
                }
                None => serialization_error(other),
            },
        }
    }
}

impl TransportPyErr for antecedent_io::learned_continuous_artifact::LearnedContinuousArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::learned_continuous_artifact::LearnedContinuousArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            // Every other consumer check is a consistency failure of the artifact.
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_io::smoothed_dose_artifact::SmoothedDoseArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::smoothed_dose_artifact::SmoothedDoseArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            // A replay the consumer cancelled is a cancellation, still reason-coded.
            E::ReplayRefused { code, message }
                if code == reason_code!("transport_budget_cancel") =>
            {
                crate::with_reason_code(crate::CausalCancelledError::new_err(message), code)
            }
            // A replay the request no longer passes carries the record's reason code.
            E::ReplayRefused { code, message } => {
                crate::with_reason_code(serialization_error(message), code)
            }
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_io::transport_scenario_artifact::TransportScenarioArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::transport_scenario_artifact::TransportScenarioArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_io::temporal_transport_artifact::TemporalTransportArtifactError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::temporal_transport_artifact::TemporalTransportArtifactError as E;
        match self {
            E::LimitsExceeded(_) => crate::CausalResourceError::new_err(self.to_string()),
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_io::IoError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::IoError as E;
        match self {
            E::Refused { code, message } => coded_refusal(code, message),
            E::ZTransport(inner) => inner.into_transport_py_err(),
            E::MzTransport(inner) => inner.into_transport_py_err(),
            E::MixedSource(inner) => inner.into_transport_py_err(),
            E::TransportScenario(inner) => inner.into_transport_py_err(),
            E::TemporalTransport(inner) => inner.into_transport_py_err(),
            E::LearnedContinuous(inner) => inner.into_transport_py_err(),
            E::SmoothedDose(inner) => inner.into_transport_py_err(),
            E::UnsupportedVersion { .. } => serialization_error(self),
            // The facade still carries some refusals as converted messages.
            E::Convert(message) => classify(&message),
            other => serialization_error(other),
        }
    }
}

impl TransportPyErr for antecedent_design::ZTransportPlanningError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_design::ZTransportPlanningError as E;
        match self {
            E::Invalid(_) | E::InvalidSpec(_) | E::Catalog(_) => crate::value_err(self.to_string()),
            E::UnsupportedVersion { .. } | E::DigestMismatch => serialization_error(self),
            E::Io(inner) => inner.into_transport_py_err(),
            E::Identification(inner) => inner.into_transport_py_err(),
        }
    }
}

impl TransportPyErr for antecedent_validate::ZTransportSensitivityError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_validate::ZTransportSensitivityError as E;
        match self {
            E::Cancelled => {
                crate::CausalCancelledError::new_err("z-transport sensitivity cancelled")
            }
            E::InvalidProof | E::ProviderMismatch => serialization_error(self),
            E::IncompatibleFormula | E::UnsupportedDomain | E::IncompleteKernel => {
                crate::refusal(reason_code!("transport_unsupported_evaluator"), self.to_string())
            }
            E::InvalidSensitivity(_) => crate::value_err(self.to_string()),
        }
    }
}

impl TransportPyErr for antecedent_validate::FixedGraphSensitivityError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_validate::FixedGraphSensitivityError as E;
        match self {
            E::InvalidTransportProof | E::SnapshotMismatch => serialization_error(self),
            E::UnsupportedGraph | E::UnboundOutcomeFactor => {
                crate::refusal(reason_code!("transport_unsupported_evaluator"), self.to_string())
            }
            E::InvalidQuery | E::InvalidParentLaw | E::Kernel(_) => {
                crate::value_err(self.to_string())
            }
        }
    }
}

impl TransportPyErr for antecedent::CausalError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent::CausalError as E;
        match self {
            E::Identify(inner) => inner.into_transport_py_err(),
            E::Estimate(inner) => inner.into_transport_py_err(),
            E::Serialization(inner) => inner.into_transport_py_err(),
            other => crate::py_err(other),
        }
    }
}

impl TransportPyErr for antecedent_expr::ExactLawError {
    fn into_transport_py_err(self) -> PyErr {
        crate::value_err(self.to_string())
    }
}

impl TransportPyErr for antecedent_core::QueryError {
    fn into_transport_py_err(self) -> PyErr {
        crate::value_err(self.to_string())
    }
}

impl TransportPyErr for antecedent_graph::GraphError {
    fn into_transport_py_err(self) -> PyErr {
        crate::value_err(self.to_string())
    }
}

impl TransportPyErr for std::num::TryFromIntError {
    fn into_transport_py_err(self) -> PyErr {
        crate::value_err(self.to_string())
    }
}

impl TransportPyErr for serde_json::Error {
    fn into_transport_py_err(self) -> PyErr {
        serialization_error(self)
    }
}

/// A message the bindings composed themselves.
impl TransportPyErr for String {
    fn into_transport_py_err(self) -> PyErr {
        classify(&self)
    }
}

impl TransportPyErr for &str {
    fn into_transport_py_err(self) -> PyErr {
        classify(self)
    }
}

/// How an exact structured transport detail is raised.
enum Kind {
    Serialization,
    Refusal(&'static str),
    Cancelled,
    Resource,
}

/// Classify recognized transport details without interpreting user-supplied text.
/// Untyped messages keep the generic value error; typed errors use their own mappings.
pub(crate) fn classify(message: &str) -> PyErr {
    // Legacy sensitivity consumers emit these complete, fixed messages with no
    // user text. Preserve their serialization category without substring matching.
    if matches!(
        message,
        "z-transport sensitivity baseline digest mismatch"
            | "z-transport sensitivity baseline premises digest mismatch"
            | "z-transport sensitivity result or provider binding mismatch"
    ) {
        return serialization_error(message);
    }
    // The exact evaluator's budget and cancellation errors are complete fixed messages.
    if message == "exact evaluation cancelled" {
        return crate::CausalCancelledError::new_err(message.to_owned());
    }
    if matches!(
        message,
        "exact domain budget exceeded"
            | "exact cache memory/operation budget exceeded"
            | "exact intervention coverage budget exceeded"
            | "exact memory budget exceeded"
            | "exact operation budget exceeded"
            | "exact preflight budget exceeded"
            | "exact recursion budget exceeded"
    ) {
        return crate::CausalResourceError::new_err(message.to_owned());
    }
    if let Some(kind) = dotted_token(message).and_then(refusal_kind) {
        return match kind {
            Kind::Serialization => serialization_error(message),
            Kind::Refusal(code) => crate::refusal(code, message),
            Kind::Cancelled => crate::CausalCancelledError::new_err(message.to_owned()),
            Kind::Resource => crate::CausalResourceError::new_err(message.to_owned()),
        };
    }
    crate::value_err(message)
}

/// A structured transport detail is recognized only at the start of a message.
fn dotted_token(message: &str) -> Option<&str> {
    message
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '"' | '\''))
        .map(|word| word.trim_end_matches(':'))
        .find(|word| !word.is_empty())
        .filter(|word| word.starts_with("transport.") || word.starts_with("z_transport."))
}

/// Only known stable detail spellings carry a classification.
fn refusal_kind(token: &str) -> Option<Kind> {
    match token {
        "transport.cancelled" => return Some(Kind::Cancelled),
        "transport.identification_budget"
        | "transport.identification_memory_budget"
        | "transport.binding_budget"
        | "transport.memory_budget"
        | "z_transport.exhausted_computation" => return Some(Kind::Resource),
        _ => {}
    }
    let detail = token.split_once('.').map_or(token, |(_, rest)| rest);
    Some(match detail {
        "obstruction_assignment_mismatch"
        | "obstruction_input_mismatch"
        | "obstruction_record_mismatch"
        | "obstruction_replay_mismatch"
        | "obstruction_not_reproduced"
        | "proof_formula_mismatch"
        | "proof_graph_mismatch"
        | "proof_input_mismatch"
        | "proof_rule_mismatch"
        | "invalid_derivation"
        | "invalid_derivation_record"
        | "invalid_s_hedge"
        | "invalid_s_hedge_evidence" => Kind::Serialization,
        "missing_evidence"
        | "obstruction_missing_evidence"
        | "source_experiment_missing"
        | "target_joint_observational_law_required"
        | "experiment_assignment_required" => {
            Kind::Refusal(reason_code!("transport_missing_evidence"))
        }
        "empirical_counts_required" | "samples_not_embedded" => {
            Kind::Refusal(reason_code!("transport_missing_provider"))
        }
        "no_execution_claim" => Kind::Refusal(reason_code!("not_executed")),
        "unsupported_dependence" => Kind::Refusal(reason_code!("transport_unsupported_evaluator")),
        "bootstrap_failure_fraction" => Kind::Refusal(reason_code!("transport_numerical_failure")),
        "stale_request" | "stale_result" | "reprepare_required" => {
            Kind::Refusal(reason_code!("invalid_argument"))
        }
        "unknown_variable" => Kind::Refusal(reason_code!("unknown_variable")),
        "not_certified" => Kind::Refusal(reason_code!("transport_not_certified")),
        _ => return None,
    })
}

/// The coordinate of a named variable.
pub(crate) fn resolve(names: &[String], name: &str) -> PyResult<VariableId> {
    let index = names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| crate::value_err(format!("unknown variable {name}")))?;
    Ok(VariableId::from_raw(u32::try_from(index).map_err(error)?))
}

/// A concrete intervention world from `{name: value}`.
pub(crate) fn assignment_from_pairs(
    names: &[String],
    pairs: BTreeMap<String, f64>,
) -> PyResult<Assignment> {
    Ok(Assignment::from_pairs(
        pairs
            .into_iter()
            .map(|(name, value)| Ok((resolve(names, &name)?, Value::f64(value))))
            .collect::<PyResult<Vec<_>>>()?,
    ))
}

/// Whether a parsed law must agree with its catalog regime's intervention set
/// and measured margin. Statistical payloads mix supplied laws with samples
/// whose regimes are declared from the samples, so they stay lenient.
#[derive(Clone, Copy)]
pub(crate) enum RegimeCheck {
    Strict,
    Lenient,
}

/// Parse one Python `ExactDiscreteLaw` against the catalog it cites.
pub(crate) fn parse_law_table(
    table: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    check: RegimeCheck,
) -> PyResult<ExactDiscreteLaw> {
    let population: String = table.getattr("population")?.extract()?;
    let label: String = table.getattr("regime")?.extract()?;
    let regime = catalog
        .regimes
        .iter()
        .find(|r| r.label.as_deref() == Some(label.as_str()) && r.population.as_ref() == population)
        .ok_or_else(|| crate::value_err("exact law names an unknown population/regime"))?;
    let axes: Vec<(String, Vec<f64>)> = table.getattr("axes")?.extract()?;
    let axes = axes
        .into_iter()
        .map(|(name, values)| {
            Ok(DiscreteAxis {
                variable: resolve(&graph.names, &name)?,
                values: values.into_iter().map(Value::f64).collect(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let interventions: Vec<(String, f64)> = table.getattr("interventions")?.extract()?;
    let interventions = interventions
        .into_iter()
        .map(|(name, value)| {
            Ok(InterventionAssignment {
                variable: resolve(&graph.names, &name)?,
                value: Value::f64(value),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    if matches!(check, RegimeCheck::Strict)
        && (interventions.len() != regime.interventions.len()
            || !interventions.iter().all(|a| regime.interventions.contains(&a.variable))
            || axes.iter().any(|a| !regime.measured.contains(&a.variable)))
    {
        return Err(crate::value_err("exact law disagrees with its evidence regime"));
    }
    let probabilities: Vec<f64> = table.getattr("probabilities")?.extract()?;
    let snapshot: String = table.getattr("snapshot_identity")?.extract()?;
    let absolute: f64 = table.getattr("absolute_tolerance")?.extract()?;
    let relative: f64 = table.getattr("relative_tolerance")?.extract()?;
    let mut law = ExactDiscreteLaw::try_new(
        population,
        regime.id,
        interventions,
        axes,
        probabilities,
        snapshot,
        LawTolerance { absolute, relative },
    )
    .map_err(error)?;
    let counts = match table.getattr("empirical_counts") {
        Ok(counts) => Some(counts),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(table.py()) => {
            None
        }
        Err(error) => return Err(error),
    };
    if let Some(counts) = counts {
        if !counts.is_none() {
            let counts: Vec<i64> = counts.extract()?;
            let counts = counts
                .into_iter()
                .map(|count| {
                    u64::try_from(count)
                        .map_err(|_| crate::value_err("empirical counts must be non-negative"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            law = law.with_empirical_counts(counts).map_err(error)?;
        }
    }
    Ok(law)
}

/// Parse an iterable of Python laws into validated exact transport data.
pub(crate) fn parse_laws(
    laws: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
    check: RegimeCheck,
) -> PyResult<ExactTransportData> {
    let mut tables = Vec::new();
    for table in laws.try_iter()? {
        tables.push(parse_law_table(&table?, catalog, graph, check)?);
    }
    ExactTransportData::try_new(tables, max_support_rows).map_err(error)
}

/// A production execution context with the caller's memory limit and token.
pub(crate) fn execution_context(
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = ExecutionContext::production_default(seed);
    ctx.set_memory_limit(memory_bytes);
    crate::apply_cancel(&mut ctx, cancel);
    crate::execution_api::apply(&mut ctx);
    ctx
}

/// Spell every `VariableId(k)` a crate message carries as the variable's name.
///
/// Identification obligations are rendered by the crates in coordinates; the
/// Python caller only knows names, so the message is rewritten once here.
pub(crate) fn resolve_variable_ids(text: &str, names: &[String]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("VariableId(") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "VariableId(".len()..];
        let Some(end) = after.find(')') else {
            out.push_str("VariableId(");
            rest = after;
            continue;
        };
        let name = after[..end]
            .parse::<usize>()
            .ok()
            .and_then(|index| names.get(index))
            .map_or_else(|| format!("VariableId({})", &after[..end]), String::clone);
        out.push_str(&name);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// The refusal for a bounded catalog search that did not bind: the strategies
/// searched and their obligations are rendered in prose with variable names,
/// never as a debug dump.
pub(crate) fn catalog_search_refusal(
    result: &antecedent_identify::CatalogTransportResult,
    names: &[String],
) -> PyErr {
    use antecedent_identify::CatalogTransportResult;
    let (code, searched, obligations) = match result {
        CatalogTransportResult::Identified(_) => {
            return crate::value_err("bounded catalog search identified the formula");
        }
        CatalogTransportResult::MissingEvidence { searched, obligations } => {
            ("transport.missing_evidence", searched, obligations)
        }
        CatalogTransportResult::NotCertified { searched, obligations } => {
            ("transport.not_certified", searched, obligations)
        }
    };
    let searched = searched.iter().map(AsRef::as_ref).collect::<Vec<_>>().join(", ");
    let obligations = obligations
        .iter()
        .map(|obligation| resolve_variable_ids(obligation, names))
        .collect::<Vec<_>>()
        .join("; ");
    error(format!(
        "{code}: bounded catalog search ({searched}) did not bind the certified formula: {obligations}"
    ))
}

/// `[names, artifact]` CBOR framing behind a magic prefix, shared by the
/// transport artifacts that travel with their variable names.
pub(crate) fn frame_named_artifact(
    prefix: &[u8],
    names: &[String],
    artifact: Vec<u8>,
) -> PyResult<Vec<u8>> {
    let mut framed = prefix.to_vec();
    framed.extend(antecedent_io::to_cbor(&(names.to_vec(), artifact)).map_err(error)?);
    Ok(framed)
}

/// Undo [`frame_named_artifact`], refusing bytes that do not carry the prefix.
pub(crate) fn unframe_named_artifact(
    prefix: &[u8],
    bytes: &[u8],
    what: &str,
) -> PyResult<(Vec<String>, Vec<u8>)> {
    let payload = bytes
        .strip_prefix(prefix)
        .ok_or_else(|| serialization_error(format!("invalid {what} artifact format")))?;
    antecedent_io::from_cbor(payload).map_err(serialization_error)
}

/// A native support slot for an inspection payload: its matrix status and
/// coordinate and its empirical support, or the reason it is unavailable.
pub(crate) fn support_slot_json(
    slot: &antecedent_contract::SlotAvailability<antecedent_contract::SupportSlot>,
) -> serde_json::Value {
    match slot.as_ref() {
        Some(support) => serde_json::json!({
            "matrix_status": support.matrix_status.as_ref(),
            "matrix_coordinate": support.matrix_coordinate.as_deref(),
            "empirical": support.empirical.label(ToString::to_string),
        }),
        None => serde_json::json!({ "unavailable": slot.label(|_| String::new()) }),
    }
}

/// Obligation ids of a native assumption slot, empty when it is unavailable.
pub(crate) fn obligation_ids(
    slot: &antecedent_contract::SlotAvailability<antecedent_contract::AssumptionSlot>,
) -> Vec<String> {
    slot.as_ref().map_or_else(Vec::new, |assumptions| {
        assumptions.obligations.iter().map(|obligation| obligation.id.to_string()).collect()
    })
}

#[cfg(test)]
mod structured_classification_tests {
    use super::classify;
    use pyo3::prelude::*;

    #[test]
    fn user_names_cannot_turn_structured_refusals_into_resource_or_cancel_errors() {
        Python::initialize();
        Python::attach(|py| {
            for name in ["ad_budget", "memory_use", "digest_score", "cancel_flag"] {
                let error = classify(&format!("transport.missing_evidence: variable {name}"));
                assert!(error.is_instance_of::<crate::CausalUnsupportedError>(py));
            }
            for text in [
                "unknown memory_budget",
                "unknown cancel_flag",
                "unknown digest_score",
                "unknown variable transport.cancelled",
                "unknown variable transport.identification_budget",
            ] {
                let error = classify(text);
                assert!(!error.is_instance_of::<crate::CausalResourceError>(py));
                assert!(!error.is_instance_of::<crate::CausalCancelledError>(py));
                assert!(!error.is_instance_of::<crate::CausalSerializationError>(py));
            }
            for message in [
                "z-transport sensitivity baseline digest mismatch",
                "z-transport sensitivity baseline premises digest mismatch",
                "z-transport sensitivity result or provider binding mismatch",
            ] {
                assert!(classify(message).is_instance_of::<crate::CausalSerializationError>(py));
                let named = format!("unknown variable {message}");
                assert!(!classify(&named).is_instance_of::<crate::CausalSerializationError>(py));
            }
            assert!(
                classify("transport.cancelled").is_instance_of::<crate::CausalCancelledError>(py)
            );
            assert!(
                classify("transport.identification_budget")
                    .is_instance_of::<crate::CausalResourceError>(py)
            );
            assert!(
                classify("z_transport.proof_graph_mismatch")
                    .is_instance_of::<crate::CausalSerializationError>(py)
            );
        });
    }
}
