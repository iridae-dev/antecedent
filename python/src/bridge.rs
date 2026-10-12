//! The pieces every declared-JSON artifact binding shares: the byte-bound
//! check on a declaration, the mapping of a serialization failure and the
//! structured-refusal JSON the Python layer raises as its own exception.
//!
//! Each binding keeps its own public `MAX_*_BYTES` limit and passes it in, so
//! the limits stay per-artifact while the error spelling lives here once.

use pyo3::prelude::*;

const MIB: usize = 1024 * 1024;

/// A declaration of options and scalar settings: 1 MiB.
pub(crate) const MAX_CONFIG_DECLARATION_BYTES: usize = MIB;

/// A declaration of a structure or a grid (an edge list, a scenario set): 8 MiB.
pub(crate) const MAX_SPEC_DECLARATION_BYTES: usize = 8 * MIB;

/// The byte limit of a declaration or sealed artifact that has no tighter bound
/// of its own: 16 MiB.
pub(crate) const MAX_DECLARED_ARTIFACT_BYTES: usize = 16 * MIB;

/// A declaration that carries data columns or per-row values: 64 MiB.
pub(crate) const MAX_DATA_DECLARATION_BYTES: usize = 64 * MIB;

/// A sealed artifact that embeds the data and law it was computed from: 128 MiB.
pub(crate) const MAX_SEALED_ARTIFACT_BYTES: usize = 128 * MIB;

/// Refuse `len` bytes beyond `limit`, as an invalid argument.
///
/// The message is `"{what} is too large"`; `what` names the declaration, the
/// artifact or the identity as its binding always spelled it.
pub(crate) fn check_declared_len(len: usize, limit: usize, what: &str) -> PyResult<()> {
    if len > limit {
        return Err(crate::with_reason_code(
            crate::value_err(format!("{what} is too large")),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    Ok(())
}

/// Refuse a declared JSON text longer than `limit` bytes
/// (`"{what} declaration is too large"`).
pub(crate) fn check_declared_size(text: &str, limit: usize, what: &str) -> PyResult<()> {
    check_declared_len(text.len(), limit, &format!("{what} declaration"))
}

/// A failure that can only mean the bytes do not describe what they claim.
pub(crate) fn serialization(e: impl std::fmt::Display) -> PyErr {
    crate::serialization_err(e.to_string())
}

/// The refusal JSON of one coded refusal, as the Python layer reads it.
///
/// Every refusal carries the same eight keys (`code`, `stage`, `detail`,
/// `offending`, `expected`, `supplied`, `remedy`, `message`); the ones a
/// refusal does not know are `null`. A binding with more to say (a plan, a
/// witness) adds named fields with [`RefusalJson::field`].
pub(crate) struct RefusalJson(serde_json::Map<String, serde_json::Value>);

impl RefusalJson {
    /// A refusal with its code, stage, stable detail id and human message.
    pub(crate) fn new(code: &str, stage: &str, detail: &str, message: &str) -> Self {
        use serde_json::Value::{Null, String as Text};
        let mut map = serde_json::Map::new();
        map.insert("code".into(), Text(code.to_owned()));
        map.insert("stage".into(), Text(stage.to_owned()));
        map.insert("detail".into(), Text(detail.to_owned()));
        map.insert("offending".into(), Null);
        map.insert("expected".into(), Null);
        map.insert("supplied".into(), Null);
        map.insert("remedy".into(), Null);
        map.insert("message".into(), Text(message.to_owned()));
        Self(map)
    }

    /// A refusal with no human message beyond its detail id (`message` is `null`).
    pub(crate) fn bare(code: &str, stage: &str, detail: &str) -> Self {
        Self::new(code, stage, detail, "").field("message", serde_json::Value::Null)
    }

    /// A refusal whose message is `"{detail}: {text}"` when it starts with a
    /// dotted identifier, and the whole message under an empty detail otherwise.
    pub(crate) fn from_coded_message(code: &str, stage: &str, message: &str) -> Self {
        let (detail, text) = split_detail(message);
        Self::new(code, stage, detail, text)
    }

    /// The thing the refusal is about (a field, a stage label).
    pub(crate) fn offending(self, offending: Option<&str>) -> Self {
        self.field("offending", offending)
    }

    /// What the caller can do about it.
    pub(crate) fn remedy(self, remedy: Option<&str>) -> Self {
        self.field("remedy", remedy)
    }

    /// Set or add one named field.
    ///
    /// The value is strings, options, or a plan of those. Builders stay on
    /// `String`, so a fallible `Serialize` is not threaded out as `PyResult`.
    pub(crate) fn field(mut self, key: &str, value: impl RefusalField) -> Self {
        self.0.insert(key.to_owned(), value.into_refusal_value());
        self
    }

    /// The JSON value, for a refusal nested in a larger report.
    pub(crate) fn value(self) -> serde_json::Value {
        serde_json::Value::Object(self.0)
    }

    /// The JSON text.
    pub(crate) fn json(self) -> String {
        self.value().to_string()
    }
}

/// Split `"{detail}: {text}"` where `detail` is a dotted identifier such as
/// `recalc.refused`; any other message has an empty detail.
pub(crate) fn split_detail(message: &str) -> (&str, &str) {
    match message.split_once(": ") {
        Some((head, rest))
            if head.contains('.')
                && head.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') =>
        {
            (head, rest)
        }
        _ => ("", message),
    }
}

/// The refusal JSON of an artifact error's `refusal()` triple
/// `(code, detail, message)`, with the identity field it names as `offending`.
pub(crate) fn artifact_refusal_json(
    stage: &str,
    refusal: (&str, impl AsRef<str>, impl AsRef<str>),
    offending: Option<&str>,
) -> String {
    let (code, detail, message) = refusal;
    RefusalJson::new(code, stage, detail.as_ref(), message.as_ref()).offending(offending).json()
}

/// A refusal field whose JSON is strings, nulls, bools, or a plan of those.
///
/// `serde_json::to_value` returns `Result` because some `Serialize` impls fail
/// (non-finite numbers, maps with non-string keys). These fields do not, so
/// the conversion stays inside the builder and the JSON text is unchanged.
pub(crate) trait RefusalField {
    fn into_refusal_value(self) -> serde_json::Value;
}

impl RefusalField for serde_json::Value {
    fn into_refusal_value(self) -> serde_json::Value {
        self
    }
}

impl RefusalField for &str {
    fn into_refusal_value(self) -> serde_json::Value {
        serde_json::Value::String(self.to_owned())
    }
}

impl RefusalField for &String {
    fn into_refusal_value(self) -> serde_json::Value {
        serde_json::Value::String(self.clone())
    }
}

impl RefusalField for Option<&str> {
    fn into_refusal_value(self) -> serde_json::Value {
        self.map_or(serde_json::Value::Null, |text| serde_json::Value::String(text.to_owned()))
    }
}

impl RefusalField for Option<antecedent_io::recalc_receipt_artifact::PlanWire> {
    fn into_refusal_value(self) -> serde_json::Value {
        match self {
            None => serde_json::Value::Null,
            Some(plan) => plan_wire_value(&plan),
        }
    }
}

fn plan_wire_value(plan: &antecedent_io::recalc_receipt_artifact::PlanWire) -> serde_json::Value {
    let entries = plan
        .entries
        .iter()
        .map(|entry| {
            serde_json::json!({
                "stage": entry.stage,
                "tag": entry.tag,
                "detail": entry.detail,
                "status": entry.status,
                "identity": entry.identity,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "entries": entries,
        "identity": plan.identity,
        "executable": plan.executable,
    })
}

/// The wire JSON of one structured external refusal.
///
/// Every `ExternalRefusal` a binding returns to Python is spelled here once,
/// through the same `RefusalWire` the external-binding bridge uses, so the
/// missing operation of a capability refusal (`capability`) reaches Python.
/// Key order matches `serde_json::to_string` on that wire.
pub(crate) fn external_refusal_json(value: &antecedent_contract::ExternalRefusal) -> String {
    wire_json(&antecedent_io::external_binding_wire::RefusalWire::from(value.clone()))
}

fn json_string(text: &str) -> String {
    serde_json::Value::String(text.to_owned()).to_string()
}

fn json_opt(text: &Option<String>) -> String {
    match text {
        Some(text) => json_string(text),
        None => "null".to_owned(),
    }
}

fn wire_json(wire: &antecedent_io::external_binding_wire::RefusalWire) -> String {
    format!(
        "{{\"code\":{},\"stage\":{},\"detail\":{},\"offending\":{},\"expected\":{},\"supplied\":{},\"capability\":{},\"remedy\":{}}}",
        json_string(&wire.code),
        json_string(&wire.stage),
        json_string(&wire.detail),
        json_opt(&wire.offending),
        json_opt(&wire.expected),
        json_opt(&wire.supplied),
        json_opt(&wire.capability),
        json_opt(&wire.remedy),
    )
}

#[cfg(test)]
mod tests {
    use super::external_refusal_json;
    use antecedent_contract::{ExternalCapability, ExternalRefusal};

    fn refusal(capability: Option<ExternalCapability>) -> ExternalRefusal {
        ExternalRefusal {
            code: antecedent_core::reason_code!("external_capability_missing"),
            stage: "negotiate",
            detail: "provider_capability_negotiation.capability_missing".to_owned(),
            offending: None,
            expected: None,
            supplied: None,
            capability,
            remedy: None,
        }
    }

    #[test]
    fn the_missing_operation_reaches_the_wire() {
        let json = external_refusal_json(&refusal(Some(ExternalCapability::Cdf)));
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["capability"], "cdf");
        assert_eq!(value["code"], "external_capability_missing");
    }

    #[test]
    fn a_refusal_without_one_carries_a_null_capability() {
        let json = external_refusal_json(&refusal(None));
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert!(value.get("capability").is_some_and(serde_json::Value::is_null));
    }

    #[test]
    fn external_refusal_json_matches_the_wire_serializer() {
        use antecedent_io::external_binding_wire::RefusalWire;
        for capability in [Some(ExternalCapability::Cdf), None] {
            let value = refusal(capability);
            let json = external_refusal_json(&value);
            let via_serde = serde_json::to_string(&RefusalWire::from(value)).expect("wire");
            assert_eq!(json, via_serde);
        }
        let quoted = ExternalRefusal {
            detail: "family.slot \"quoted\" \\ tail".to_owned(),
            offending: Some("line\nbreak".to_owned()),
            ..refusal(None)
        };
        let json = external_refusal_json(&quoted);
        let via_serde = serde_json::to_string(&RefusalWire::from(quoted)).expect("wire");
        assert_eq!(json, via_serde);
    }

    #[test]
    fn a_plan_field_matches_serde_of_the_wire() {
        use super::RefusalJson;
        use antecedent_io::recalc_receipt_artifact::{PlanEntryWire, PlanWire};
        let plan = PlanWire {
            entries: vec![PlanEntryWire {
                stage: "estimate".to_owned(),
                tag: "recomputed".to_owned(),
                detail: "own:utility".to_owned(),
                status: "recomputed(own:utility)".to_owned(),
                identity: "ab".to_owned(),
            }],
            identity: "cd".to_owned(),
            executable: true,
        };
        let json = RefusalJson::new("c", "s", "d", "m").field("plan", Some(plan.clone())).json();
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        let via_serde = serde_json::to_value(plan).expect("plan");
        assert_eq!(value["plan"], via_serde);
        assert_eq!(value["message"], "m");
        assert!(value["offending"].is_null());
    }
}
