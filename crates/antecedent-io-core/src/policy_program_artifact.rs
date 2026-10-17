//! Versioned portable encoding of intervention programs.
//!
//! A program (a list of interventions, any of which may be a [`Intervention::Transform`]
//! or [`Intervention::Assign`] over a [`ValueExpr`](antecedent_core::ValueExpr), including
//! lagged history reads) is encoded as its canonical wire form
//! ([`InterventionWire`]) under an encoding version and the digest of that wire form. A
//! consumer reads the version before interpreting the body, refuses a version it does not
//! define, recomputes the digest rather than trusting the stored one, and re-validates
//! every program it decodes (declared reads must equal the referenced variables), so an
//! edited expression or edited reads never replays.
//!
//! A program that cannot leave the process (it holds a callback bound to the running
//! interpreter) has no encoding: it is refused with `policy_callback_not_portable`
//! before any byte is written.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::Intervention;
use serde::{Deserialize, Serialize};

use crate::error::IoError;
use crate::identity::payload_digest_of;
use crate::query_wire::{InterventionWire, ValueExprWire};

/// Format marker of the encoding.
pub const POLICY_PROGRAM_MAGIC: &[u8] = b"ANTPOLICYPROG";
/// Encoding version this library writes.
pub const POLICY_PROGRAM_VERSION: u16 = 1;
/// Encoding versions this library reads.
pub const POLICY_PROGRAM_SUPPORTED_VERSIONS: &[u16] = &[1];
/// Largest encoding, in bytes, read or written.
pub const POLICY_PROGRAM_MAX_BYTES: usize = 16_000_000;
/// Most interventions one encoding carries.
pub const POLICY_PROGRAM_MAX_INTERVENTIONS: usize = 4096;

/// Payload family label under which the program digest is taken.
const IDENTITY_KIND: &str = "policy_program.v1";

#[derive(Serialize, Deserialize)]
struct PolicyProgramEnvelope {
    version: u16,
    identity: [u8; 32],
    interventions: Vec<InterventionWire>,
}

fn wires(interventions: &[Intervention]) -> Result<Vec<InterventionWire>, IoError> {
    if interventions.len() > POLICY_PROGRAM_MAX_INTERVENTIONS {
        return Err(IoError::TooLarge);
    }
    interventions.iter().map(InterventionWire::from_domain).collect()
}

/// Identity of a program: the digest of its canonical wire form.
///
/// Equal programs have equal identities, and any change to an expression, a lag, a
/// declared read, a target variable or a program kind changes it.
///
/// # Errors
///
/// A program that cannot leave the process, or an encoding failure.
pub fn policy_program_identity(interventions: &[Intervention]) -> Result<[u8; 32], IoError> {
    payload_digest_of(IDENTITY_KIND, &wires(interventions)?)
}

/// Encode a program with its encoding version and identity.
///
/// # Errors
///
/// `policy_callback_not_portable` for a runtime-bound program, `TooLarge` over the bounds,
/// or an encoding failure.
pub fn encode_policy_program(interventions: &[Intervention]) -> Result<Vec<u8>, IoError> {
    let interventions = wires(interventions)?;
    let envelope = PolicyProgramEnvelope {
        version: POLICY_PROGRAM_VERSION,
        identity: payload_digest_of(IDENTITY_KIND, &interventions)?,
        interventions,
    };
    let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
    ciborium::into_writer(&envelope, &mut bytes).map_err(|e| IoError::Cbor(e.to_string()))?;
    if bytes.len() > POLICY_PROGRAM_MAX_BYTES {
        return Err(IoError::TooLarge);
    }
    Ok(bytes)
}

/// Decode and verify a program: version first, then shape, identity and validity.
///
/// # Errors
///
/// `BadMagic`; `UnsupportedVersion` for a version outside
/// [`POLICY_PROGRAM_SUPPORTED_VERSIONS`] (read before the body is interpreted); a manifest
/// mismatch when the stored identity does not match the expressions; or a validation
/// failure of a decoded program (for instance declared reads that differ from the
/// expression's).
pub fn decode_policy_program(bytes: &[u8]) -> Result<Vec<Intervention>, IoError> {
    if bytes.len() > POLICY_PROGRAM_MAX_BYTES {
        return Err(IoError::TooLarge);
    }
    let body = bytes.strip_prefix(POLICY_PROGRAM_MAGIC).ok_or(IoError::BadMagic)?;
    let value: ciborium::Value =
        ciborium::from_reader(body).map_err(|e| IoError::Cbor(e.to_string()))?;
    let version = value
        .as_map()
        .and_then(|entries| {
            entries.iter().find_map(|(key, found)| {
                (key.as_text() == Some("version")).then(|| found.as_integer())
            })
        })
        .flatten()
        .and_then(|raw| u32::try_from(i128::from(raw)).ok())
        .ok_or(IoError::ManifestMismatch { message: "policy program has no encoding version" })?;
    if !POLICY_PROGRAM_SUPPORTED_VERSIONS.iter().any(|v| u32::from(*v) == version) {
        return Err(IoError::UnsupportedVersion { version });
    }
    let envelope: PolicyProgramEnvelope =
        value.deserialized().map_err(|e| IoError::Cbor(e.to_string()))?;
    if envelope.interventions.len() > POLICY_PROGRAM_MAX_INTERVENTIONS {
        return Err(IoError::TooLarge);
    }
    if payload_digest_of(IDENTITY_KIND, &envelope.interventions)? != envelope.identity {
        return Err(IoError::ManifestMismatch {
            message: "policy program identity does not match its expressions",
        });
    }
    envelope
        .interventions
        .iter()
        .map(|wire| {
            let intervention = wire.to_domain();
            intervention.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            Ok(intervention)
        })
        .collect()
}

/// Program text with variables spelled by name: a JSON array of
/// `{"mode": "transform" | "assign", "variable": <name>, "expr": <expression>}` whose
/// expression is the [`ValueExprWire`] JSON with every `var` / `lag` node naming its
/// variable. This is how a host language with named columns hands a program to the
/// library; the declared reads are inferred from the expression.
///
/// # Errors
///
/// Malformed JSON or shape, an unknown variable name, an unknown mode, or a program that
/// does not validate.
pub fn interventions_from_named_json(
    text: &str,
    resolve: &dyn Fn(&str) -> Option<u32>,
) -> Result<Vec<Intervention>, IoError> {
    let mut items: Vec<serde_json::Value> =
        serde_json::from_str(text).map_err(|e| IoError::Convert(format!("policy program: {e}")))?;
    if items.len() > POLICY_PROGRAM_MAX_INTERVENTIONS {
        return Err(IoError::TooLarge);
    }
    let named = |value: Option<&serde_json::Value>, what: &str| -> Result<String, IoError> {
        value
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| IoError::Convert(format!("policy program: `{what}` must be text")))
    };
    items
        .iter_mut()
        .map(|item| {
            let mode = named(item.get("mode"), "mode")?;
            let name = named(item.get("variable"), "variable")?;
            let variable =
                resolve(&name).map(antecedent_core::VariableId::from_raw).ok_or_else(|| {
                    IoError::Convert(format!("policy program: unknown variable {name:?}"))
                })?;
            let mut expr = item
                .get_mut("expr")
                .map(serde_json::Value::take)
                .ok_or_else(|| IoError::Convert("policy program: missing `expr`".into()))?;
            rename_variables(&mut expr, &|v| match v {
                serde_json::Value::String(name) => {
                    resolve(name).map(serde_json::Value::from).ok_or_else(|| {
                        IoError::Convert(format!("policy program: unknown variable {name:?}"))
                    })
                }
                other => Ok(other.clone()),
            })?;
            let expr = serde_json::from_value::<ValueExprWire>(expr)
                .map_err(|e| IoError::Convert(format!("policy program expression: {e}")))?
                .to_domain();
            let intervention = match mode.as_str() {
                "transform" => Intervention::transform(variable, expr),
                "assign" => Intervention::assign(variable, expr),
                other => {
                    return Err(IoError::Convert(format!(
                        "policy program: mode {other:?} is not transform or assign"
                    )));
                }
            };
            intervention.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            Ok(intervention)
        })
        .collect()
}

/// The named-variable program text of [`interventions_from_named_json`] for a program,
/// with `names[id]` spelling variable `id`.
///
/// # Errors
///
/// A program that cannot leave the process, an intervention that is not a transform or
/// assignment, or a variable id outside `names`.
pub fn named_json_from_interventions(
    interventions: &[Intervention],
    names: &[String],
) -> Result<String, IoError> {
    let spell = |id: u32| {
        usize::try_from(id).ok().and_then(|i| names.get(i)).ok_or_else(|| {
            IoError::Convert(format!("policy program: variable id {id} has no name"))
        })
    };
    let mut out = Vec::with_capacity(interventions.len());
    for wire in wires(interventions)? {
        let (mode, variable, expr) = match wire {
            InterventionWire::Transform { variable, expr, .. } => ("transform", variable, expr),
            InterventionWire::Assign { variable, expr, .. } => ("assign", variable, expr),
            _ => {
                return Err(IoError::Convert(
                    "policy program: only transforms and assignments have a named form".into(),
                ));
            }
        };
        let mut expr = serde_json::to_value(&expr).map_err(|e| IoError::Convert(e.to_string()))?;
        rename_variables(&mut expr, &|v| match v.as_u64().and_then(|id| u32::try_from(id).ok()) {
            Some(id) => spell(id).map(|name| serde_json::Value::from(name.as_str())),
            None => Err(IoError::Convert("policy program: variable id is not a u32".into())),
        })?;
        out.push(serde_json::json!({
            "mode": mode,
            "variable": spell(variable)?,
            "expr": expr,
        }));
    }
    serde_json::to_string(&out).map_err(|e| IoError::Convert(e.to_string()))
}

/// Rewrite the `variable` field of every `var` and `lag` node of an expression tree.
fn rename_variables(
    node: &mut serde_json::Value,
    rewrite: &dyn Fn(&serde_json::Value) -> Result<serde_json::Value, IoError>,
) -> Result<(), IoError> {
    match node {
        serde_json::Value::Object(map) => {
            for (tag, child) in map.iter_mut() {
                if matches!(tag.as_str(), "var" | "lag") {
                    if let Some(field) = child.get_mut("variable") {
                        *field = rewrite(field)?;
                    }
                } else {
                    rename_variables(child, rewrite)?;
                }
            }
            Ok(())
        }
        serde_json::Value::Array(children) => {
            children.iter_mut().try_for_each(|child| rename_variables(child, rewrite))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec, Value,
        ValueExpr, VariableId,
    };

    use super::*;

    fn price() -> VariableId {
        VariableId::from_raw(0)
    }

    fn floor(slope: f64) -> Intervention {
        Intervention::transform(
            price(),
            ValueExpr::constant(10.0).max(ValueExpr::constant(slope).mul(ValueExpr::generated())),
        )
    }

    fn lagged() -> Intervention {
        Intervention::assign(
            price(),
            ValueExpr::constant(10.0).max(
                ValueExpr::constant(0.85).mul(ValueExpr::lag(price(), 1)).add(
                    ValueExpr::var(VariableId::from_raw(1))
                        .gt(ValueExpr::constant(0.0))
                        .if_then_else(ValueExpr::lag(price(), 2), ValueExpr::constant(0.0)),
                ),
            ),
        )
    }

    #[derive(Debug)]
    struct InProcess(CallbackSpec);

    impl BatchCallback for InProcess {
        fn spec(&self) -> &CallbackSpec {
            &self.0
        }

        fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
            out.copy_from_slice(inputs[0]);
            Ok(())
        }
    }

    #[test]
    fn a_program_round_trips_with_lags_and_keeps_its_identity() {
        let program = vec![floor(0.85), lagged(), Intervention::set(price(), Value::f64(1.0))];
        let bytes = encode_policy_program(&program).unwrap();
        assert!(bytes.starts_with(POLICY_PROGRAM_MAGIC));
        let decoded = decode_policy_program(&bytes).unwrap();
        assert_eq!(decoded, program);
        assert_eq!(
            policy_program_identity(&decoded).unwrap(),
            policy_program_identity(&program).unwrap()
        );
        assert_eq!(encode_policy_program(&decoded).unwrap(), bytes, "encoding is canonical");
    }

    #[test]
    fn a_program_written_with_names_is_the_program_written_with_ids() {
        let names = vec!["price".to_string(), "demand".to_string()];
        let resolve =
            |name: &str| names.iter().position(|n| n == name).and_then(|i| u32::try_from(i).ok());
        let text = r#"[{"mode":"assign","variable":"price","expr":{"arith":{"op":"max",
            "lhs":{"const":{"value":10.0}},
            "rhs":{"arith":{"op":"mul","lhs":{"const":{"value":0.85}},
                "rhs":{"lag":{"variable":"price","lag":1}}}}}}},
            {"mode":"transform","variable":"demand","expr":{"arith":{"op":"add",
            "lhs":"generated","rhs":{"var":{"variable":"price"}}}}}]"#;
        let program = interventions_from_named_json(text, &resolve).unwrap();
        let expected = vec![
            Intervention::assign(
                price(),
                ValueExpr::constant(10.0)
                    .max(ValueExpr::constant(0.85).mul(ValueExpr::lag(price(), 1))),
            ),
            Intervention::transform(
                VariableId::from_raw(1),
                ValueExpr::generated().add(ValueExpr::var(price())),
            ),
        ];
        assert_eq!(program, expected);
        let spelled = named_json_from_interventions(&program, &names).unwrap();
        assert_eq!(interventions_from_named_json(&spelled, &resolve).unwrap(), expected);
        let unknown = text.replace("\"demand\"", "\"supply\"");
        assert!(matches!(
            interventions_from_named_json(&unknown, &resolve),
            Err(IoError::Convert(message)) if message.contains("supply")
        ));
    }

    /// The digest a host language reports for the same program must be this one: the
    /// Python binding asserts the same literal.
    #[test]
    fn the_price_floor_program_has_a_pinned_identity() {
        let digest = policy_program_identity(&[floor(0.85)]).unwrap();
        let hex = blake3::Hash::from_bytes(digest).to_hex().to_string();
        assert_eq!(hex, "4828012ed5bdb86f66847911617623031457a3373151fb18a2b4c54d3a051d04");
    }

    #[test]
    fn identity_moves_with_every_part_of_the_program() {
        let base = policy_program_identity(&[floor(0.85)]).unwrap();
        assert_eq!(base, policy_program_identity(&[floor(0.85)]).unwrap());
        assert_ne!(base, policy_program_identity(&[floor(0.9)]).unwrap());
        let assigned = Intervention::assign(price(), ValueExpr::constant(3.0));
        let transformed = Intervention::transform(price(), ValueExpr::constant(3.0));
        assert_ne!(
            policy_program_identity(&[assigned]).unwrap(),
            policy_program_identity(&[transformed]).unwrap()
        );
        let one_back = Intervention::assign(price(), ValueExpr::lag(price(), 1));
        let two_back = Intervention::assign(price(), ValueExpr::lag(price(), 2));
        assert_ne!(
            policy_program_identity(&[one_back]).unwrap(),
            policy_program_identity(&[two_back]).unwrap()
        );
    }

    #[test]
    fn a_future_encoding_version_is_refused_before_its_body_is_read() {
        let value = ciborium::Value::Map(vec![
            (ciborium::Value::Text("version".into()), ciborium::Value::Integer(2.into())),
            (
                ciborium::Value::Text("shape_this_reader_does_not_know".into()),
                ciborium::Value::Null,
            ),
        ]);
        let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
        ciborium::into_writer(&value, &mut bytes).unwrap();
        assert_eq!(decode_policy_program(&bytes), Err(IoError::UnsupportedVersion { version: 2 }));
        assert_eq!(decode_policy_program(b"not a program"), Err(IoError::BadMagic));
    }

    #[test]
    fn an_edited_expression_is_an_identity_mismatch() {
        let bytes = encode_policy_program(&[floor(0.85)]).unwrap();
        let mut envelope: PolicyProgramEnvelope =
            ciborium::from_reader(&bytes[POLICY_PROGRAM_MAGIC.len()..]).unwrap();
        let edited = InterventionWire::from_domain(&floor(0.9)).unwrap();
        envelope.interventions = vec![edited];
        let mut tampered = POLICY_PROGRAM_MAGIC.to_vec();
        ciborium::into_writer(&envelope, &mut tampered).unwrap();
        assert!(matches!(
            decode_policy_program(&tampered),
            Err(IoError::ManifestMismatch { message }) if message.contains("identity")
        ));
    }

    #[test]
    fn edited_declared_reads_are_refused_even_with_a_matching_identity() {
        let mut wire = InterventionWire::from_domain(&lagged()).unwrap();
        if let InterventionWire::Assign { reads, .. } = &mut wire {
            reads.history.clear();
        }
        let interventions = vec![wire];
        let envelope = PolicyProgramEnvelope {
            version: POLICY_PROGRAM_VERSION,
            identity: payload_digest_of(IDENTITY_KIND, &interventions).unwrap(),
            interventions,
        };
        let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
        ciborium::into_writer(&envelope, &mut bytes).unwrap();
        assert!(matches!(decode_policy_program(&bytes), Err(IoError::Convert(_))));
    }

    #[test]
    fn a_runtime_bound_callback_has_no_encoding() {
        let spec = CallbackSpec {
            name: "interactive_floor".into(),
            inputs: Arc::from(["price".to_string()]),
            output: CallbackOutput::Real,
            dependencies: Arc::from([]),
            implementation: CallbackImplementation::Runtime,
        };
        let callback: Arc<dyn BatchCallback> = Arc::new(InProcess(spec));
        let expr = ValueExpr::callback(callback, [ValueExpr::generated()]);
        let program = [Intervention::transform(price(), expr)];
        for refused in
            [encode_policy_program(&program), policy_program_identity(&program).map(|_| Vec::new())]
        {
            match refused {
                Err(IoError::Refused { code, message }) => {
                    assert_eq!(code, "policy_callback_not_portable");
                    assert!(message.contains("interactive_floor"), "{message}");
                }
                other => panic!("expected a coded refusal, got {other:?}"),
            }
        }
    }
}
