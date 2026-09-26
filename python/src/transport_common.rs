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

/// A failure that can only mean the bytes do not describe what they claim.
pub(crate) fn serialization_error(e: impl std::fmt::Display) -> PyErr {
    crate::CausalSerializationError::new_err(e.to_string())
}

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
            | E::InvariantViolated { .. }
            | E::Message(_) => {
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
            E::Refused { code, message } => coded_refusal(code, message),
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

impl TransportPyErr for antecedent_io::IoError {
    fn into_transport_py_err(self) -> PyErr {
        use antecedent_io::IoError as E;
        match self {
            E::Refused { code, message } => coded_refusal(code, message),
            E::ZTransport(inner) => inner.into_transport_py_err(),
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

/// How a dotted `transport.` / `z_transport.` detail is raised.
enum Kind {
    /// An artifact, proof or record failed its own consistency check.
    Serialization,
    /// A refusal carrying a registered runtime reason code.
    Refusal(&'static str),
}

/// Classify one plain message: the fallback for failures the crates still
/// report as strings (`IoError::Convert` from the facade and the bindings'
/// own formatted refusals). Cancellation and budgets win over any dotted
/// detail because the budget message may itself be spelled
/// `transport.<x>_budget`.
pub(crate) fn classify(message: &str) -> PyErr {
    let lower = message.to_ascii_lowercase();
    if lower.contains("cancel") {
        return crate::CausalCancelledError::new_err(message.to_string());
    }
    if lower.contains("memory")
        || lower.contains("budget")
        || lower.contains("exhausted_computation")
    {
        return crate::CausalResourceError::new_err(message.to_string());
    }
    // A record that no longer agrees with itself (digest, identity, replayed
    // result) is a serialization failure whatever spelling the crate chose.
    if lower.contains("mismatch") || lower.contains("digest") || lower.contains("identity changed")
    {
        return serialization_error(message);
    }
    if let Some(token) = dotted_token(message) {
        return match refusal_kind(token) {
            Kind::Serialization => serialization_error(message),
            Kind::Refusal(code) => crate::refusal(code, message),
        };
    }
    crate::value_err(message)
}

/// The first `transport.<detail>` or `z_transport.<detail>` token in a message.
fn dotted_token(message: &str) -> Option<&str> {
    message
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '"' | '\''))
        .map(|word| word.trim_end_matches(':'))
        .find(|word| word.starts_with("transport.") || word.starts_with("z_transport."))
}

/// The registered reason code (or serialization class) for one dotted detail.
fn refusal_kind(token: &str) -> Kind {
    let detail = token.split_once('.').map_or(token, |(_, rest)| rest);
    let consistency = detail.ends_with("_mismatch")
        || detail.starts_with("proof_")
        || detail.starts_with("functional_program")
        || detail.starts_with("invalid_derivation")
        || detail.starts_with("invalid_s_hedge")
        || matches!(detail, "obstruction_not_reproduced" | "obstruction_replay_mismatch");
    if consistency {
        return Kind::Serialization;
    }
    Kind::Refusal(match detail {
        "missing_evidence"
        | "obstruction_missing_evidence"
        | "source_experiment_missing"
        | "target_joint_observational_law_required"
        | "experiment_assignment_required" => reason_code!("transport_missing_evidence"),
        "empirical_counts_required" | "samples_not_embedded" => {
            reason_code!("transport_missing_provider")
        }
        "no_execution_claim" => reason_code!("not_executed"),
        "unsupported_dependence" => reason_code!("transport_unsupported_evaluator"),
        "bootstrap_failure_fraction" => reason_code!("transport_numerical_failure"),
        "stale_request" | "stale_result" | "reprepare_required" => reason_code!("invalid_argument"),
        "unknown_variable" => reason_code!("unknown_variable"),
        _ => reason_code!("transport_not_certified"),
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
    if let Ok(counts) = table.getattr("empirical_counts") {
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
    ctx.memory.hard_limit_bytes = memory_bytes;
    crate::apply_cancel(&mut ctx, cancel);
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
