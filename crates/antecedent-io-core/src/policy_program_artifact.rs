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
//! A program whose callbacks are all recoverable (a content digest of self-contained
//! source, an entry point and pinned dependency versions) is written under a second
//! encoding version, in which each callback is its recovery record and never its code.
//! Programs without callbacks keep version 1 and their released bytes and identities. A
//! consumer supplies an implementation for every record and
//! [`decode_policy_program_with`] re-verifies each against its record (source digest,
//! entry point, inputs, output and dependency pins) before the program is returned for
//! replay; [`decode_policy_program`] has no implementations to offer and refuses a
//! program that needs them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{BatchCallback, Intervention, Portability, RecoveryRecord, reason_code};
use serde::{Deserialize, Serialize};

use crate::error::IoError;
use crate::identity::payload_digest_of;
use crate::query_wire::{InterventionWire, ValueExprWire, unresolved_callback};

/// Format marker of the encoding.
pub const POLICY_PROGRAM_MAGIC: &[u8] = b"ANTPOLICYPROG";
/// Encoding version of a program without callbacks.
pub const POLICY_PROGRAM_VERSION: u16 = 1;
/// Encoding version of a program whose callbacks are recoverable.
pub const POLICY_PROGRAM_RECOVERABLE_VERSION: u16 = 2;
/// Encoding versions this library reads.
pub const POLICY_PROGRAM_SUPPORTED_VERSIONS: &[u16] =
    &[POLICY_PROGRAM_VERSION, POLICY_PROGRAM_RECOVERABLE_VERSION];
/// Largest encoding, in bytes, read or written.
pub const POLICY_PROGRAM_MAX_BYTES: usize = 16_000_000;
/// Most interventions one encoding carries.
pub const POLICY_PROGRAM_MAX_INTERVENTIONS: usize = 4096;

/// Payload family label under which the digest of a program without callbacks is taken.
const IDENTITY_KIND: &str = "policy_program.v1";
/// Payload family label under which the digest of a recoverable-callback program is taken.
const RECOVERABLE_IDENTITY_KIND: &str = "policy_program.recoverable.v1";

#[derive(Serialize, Deserialize)]
struct PolicyProgramEnvelope {
    version: u16,
    identity: [u8; 32],
    interventions: Vec<InterventionWire>,
}

/// The wire forms of a program and whether any callback in it is recoverable. A
/// runtime-bound callback is refused here, before any wire form exists.
fn wires(interventions: &[Intervention]) -> Result<(Vec<InterventionWire>, bool), IoError> {
    if interventions.len() > POLICY_PROGRAM_MAX_INTERVENTIONS {
        return Err(IoError::TooLarge);
    }
    let recoverable = interventions
        .iter()
        .try_fold(Portability::Portable, |acc, iv| {
            let portability = iv.portability();
            match portability.export_refusal() {
                Some(refusal) => {
                    Err(IoError::Refused { code: refusal.code, message: refusal.message })
                }
                None => Ok(acc.and(portability)),
            }
        })
        .map(|portability| matches!(portability, Portability::Recoverable(_)))?;
    let wires = interventions
        .iter()
        .map(|iv| {
            if recoverable {
                InterventionWire::from_domain_recoverable(iv)
            } else {
                InterventionWire::from_domain(iv)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((wires, recoverable))
}

fn identity_kind(recoverable: bool) -> &'static str {
    if recoverable { RECOVERABLE_IDENTITY_KIND } else { IDENTITY_KIND }
}

/// Identity of a program: the digest of its canonical wire form.
///
/// Equal programs have equal identities, and any change to an expression, a lag, a
/// declared read, a target variable or a program kind changes it. A program with
/// recoverable callbacks is identified by their recovery records, so a different source,
/// entry point or dependency pin is a different program; one without callbacks keeps the
/// identity it always had.
///
/// # Errors
///
/// A program that cannot leave the process, or an encoding failure.
pub fn policy_program_identity(interventions: &[Intervention]) -> Result<[u8; 32], IoError> {
    let (wires, recoverable) = wires(interventions)?;
    payload_digest_of(identity_kind(recoverable), &wires)
}

/// Canonical identity of a portable expression in the existing programmable syntax.
///
/// This identifies the expression tree and exact variable IDs; feature scale and column
/// bindings belong to the enclosing compiled program. No fake intervention target is needed.
///
/// # Errors
/// Malformed expressions, runtime-bound callbacks or wire encoding failures are refused.
pub fn value_expression_identity(expr: &antecedent_core::ValueExpr) -> Result<[u8; 32], IoError> {
    expr.validate_program(true, &expr.reads())
        .map_err(|error| IoError::Convert(error.to_string()))?;
    let wire = ValueExprWire::from_domain(expr)?;
    payload_digest_of("value_expression.v1", &wire)
}

/// Encode a program with its encoding version and identity.
///
/// # Errors
///
/// `policy_callback_not_portable` for a runtime-bound program, `TooLarge` over the bounds,
/// or an encoding failure.
pub fn encode_policy_program(interventions: &[Intervention]) -> Result<Vec<u8>, IoError> {
    let (interventions, recoverable) = wires(interventions)?;
    let envelope = PolicyProgramEnvelope {
        version: if recoverable {
            POLICY_PROGRAM_RECOVERABLE_VERSION
        } else {
            POLICY_PROGRAM_VERSION
        },
        identity: payload_digest_of(identity_kind(recoverable), &interventions)?,
        interventions,
    };
    let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
    ciborium::into_writer(&envelope, &mut bytes).map_err(IoError::cbor_encode)?;
    if bytes.len() > POLICY_PROGRAM_MAX_BYTES {
        return Err(IoError::TooLarge);
    }
    Ok(bytes)
}

/// Read the envelope and verify version, identity and shape. The interventions are not yet
/// decoded.
fn open_envelope(bytes: &[u8]) -> Result<PolicyProgramEnvelope, IoError> {
    if bytes.len() > POLICY_PROGRAM_MAX_BYTES {
        return Err(IoError::TooLarge);
    }
    let body = bytes.strip_prefix(POLICY_PROGRAM_MAGIC).ok_or(IoError::BadMagic)?;
    // Read the version alone first so an unsupported encoding is refused before the
    // body is interpreted, without building a value tree of every expression.
    #[derive(Deserialize)]
    struct Peek {
        version: Option<u64>,
    }
    let no_version = IoError::ManifestMismatch { message: "policy program has no encoding version" };
    let version = match ciborium::from_reader::<Peek, _>(body) {
        Ok(peek) => peek.version,
        Err(ciborium::de::Error::Semantic(..)) => return Err(no_version),
        Err(error) => return Err(IoError::cbor_decode(error)),
    }
    .and_then(|raw| u32::try_from(raw).ok())
    .ok_or(no_version)?;
    if !POLICY_PROGRAM_SUPPORTED_VERSIONS.iter().any(|v| u32::from(*v) == version) {
        return Err(IoError::UnsupportedVersion { version });
    }
    let envelope: PolicyProgramEnvelope =
        ciborium::from_reader(body).map_err(IoError::cbor_decode)?;
    if envelope.interventions.len() > POLICY_PROGRAM_MAX_INTERVENTIONS {
        return Err(IoError::TooLarge);
    }
    let recoverable = envelope.version == POLICY_PROGRAM_RECOVERABLE_VERSION;
    if payload_digest_of(identity_kind(recoverable), &envelope.interventions)? != envelope.identity
    {
        return Err(IoError::ManifestMismatch {
            message: "policy program identity does not match its expressions",
        });
    }
    Ok(envelope)
}

/// Decode and validate the interventions of an opened envelope, binding each callback to the
/// implementation `resolve` returns for its record. The encoding version must agree with the
/// content: callbacks only under the recoverable version, and at least one there.
fn decode_interventions(
    envelope: &PolicyProgramEnvelope,
    resolve: &mut crate::query_wire::ResolveCallback<'_, IoError>,
) -> Result<Vec<Intervention>, IoError> {
    let program = envelope
        .interventions
        .iter()
        .map(|wire| {
            let intervention = wire.to_domain_with(&mut *resolve)?;
            intervention.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            Ok(intervention)
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    let has_callbacks =
        program.iter().any(|iv| matches!(iv.portability(), Portability::Recoverable(_)));
    if has_callbacks != (envelope.version == POLICY_PROGRAM_RECOVERABLE_VERSION) {
        return Err(IoError::ManifestMismatch {
            message: "policy program encoding version does not match whether it carries callbacks",
        });
    }
    Ok(program)
}

/// Decode and verify a program: version first, then shape, identity and validity.
///
/// # Errors
///
/// `BadMagic`; `UnsupportedVersion` for a version outside
/// [`POLICY_PROGRAM_SUPPORTED_VERSIONS`] (read before the body is interpreted); a manifest
/// mismatch when the stored identity does not match the expressions; a validation
/// failure of a decoded program (for instance declared reads that differ from the
/// expression's); or `policy_callback_unrecoverable` for a program that carries callback
/// records, which need [`decode_policy_program_with`].
pub fn decode_policy_program(bytes: &[u8]) -> Result<Vec<Intervention>, IoError> {
    decode_policy_program_with(bytes, &|_| None)
}

/// Decode and verify a program, binding each callback record to the implementation
/// `supply` returns for it and re-verifying that implementation against the record.
///
/// Nothing is returned for replay unless every record verifies: a missing implementation or a
/// source digest, entry point, input label, output declaration or dependency pin that differs
/// from the record is `policy_callback_unrecoverable`.
///
/// # Errors
///
/// As [`decode_policy_program`], and `policy_callback_unrecoverable` naming the callback and
/// the first differing field.
pub fn decode_policy_program_with(
    bytes: &[u8],
    supply: &dyn Fn(&RecoveryRecord) -> Option<Arc<dyn BatchCallback>>,
) -> Result<Vec<Intervention>, IoError> {
    let envelope = open_envelope(bytes)?;
    decode_interventions(&envelope, &mut |record| {
        let supplied = supply(record).ok_or_else(|| IoError::Refused {
            code: reason_code!("policy_callback_unrecoverable"),
            message: format!(
                "policy_callback.implementation_missing: callback {} needs its implementation \
                 to replay; supply the source its record identifies, with the pinned dependency \
                 versions installed",
                record.name
            ),
        })?;
        record.verify(supplied.spec()).map_err(|mismatch| {
            let refusal = mismatch.refusal();
            IoError::Refused { code: refusal.code, message: refusal.message }
        })?;
        Ok(supplied)
    })
}

/// The recovery records a program carries, in the order its callbacks run, so a consumer
/// knows which implementations to supply to [`decode_policy_program_with`]. The artifact's
/// version, identity and validity are checked; nothing is bound.
///
/// # Errors
///
/// As [`decode_policy_program`], without the implementation requirement.
pub fn policy_program_callback_records(bytes: &[u8]) -> Result<Vec<RecoveryRecord>, IoError> {
    let envelope = open_envelope(bytes)?;
    let program = decode_interventions(&envelope, &mut |record| Ok(unresolved_callback(record)))?;
    Ok(program
        .iter()
        .flat_map(|iv| match iv.portability() {
            Portability::Recoverable(records) => records,
            _ => Vec::new(),
        })
        .collect())
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
            rename_variables(&mut expr, &|v| {
                if let serde_json::Value::String(name) = v {
                    *v = resolve(name).map(serde_json::Value::from).ok_or_else(|| {
                        IoError::Convert(format!("policy program: unknown variable {name:?}"))
                    })?;
                }
                Ok(())
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
    for wire in wires(interventions)?.0 {
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
        rename_variables(&mut expr, &|v| {
            let id = v.as_u64().and_then(|id| u32::try_from(id).ok()).ok_or_else(|| {
                IoError::Convert("policy program: variable id is not a u32".into())
            })?;
            *v = serde_json::Value::from(spell(id)?.as_str());
            Ok(())
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
    rewrite: &dyn Fn(&mut serde_json::Value) -> Result<(), IoError>,
) -> Result<(), IoError> {
    match node {
        serde_json::Value::Object(map) => {
            for (tag, child) in map.iter_mut() {
                if matches!(tag.as_str(), "var" | "lag") {
                    if let Some(field) = child.get_mut("variable") {
                        rewrite(field)?;
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
        let hex = antecedent_core::to_hex(&digest);
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
            (ciborium::Value::Text("version".into()), ciborium::Value::Integer(3.into())),
            (
                ciborium::Value::Text("shape_this_reader_does_not_know".into()),
                ciborium::Value::Null,
            ),
        ]);
        let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
        ciborium::into_writer(&value, &mut bytes).unwrap();
        assert_eq!(decode_policy_program(&bytes), Err(IoError::UnsupportedVersion { version: 3 }));
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

    /// A recoverable callback that doubles its only input.
    #[derive(Debug)]
    struct Doubler(CallbackSpec);

    impl BatchCallback for Doubler {
        fn spec(&self) -> &CallbackSpec {
            &self.0
        }

        fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
            for (o, i) in out.iter_mut().zip(inputs[0]) {
                *o = 2.0 * i;
            }
            Ok(())
        }
    }

    fn recoverable_spec(digest: &str) -> CallbackSpec {
        CallbackSpec {
            name: "doubler".into(),
            inputs: Arc::from(["price".to_string()]),
            output: CallbackOutput::Range { lower: -1.0e6, upper: 1.0e6 },
            dependencies: Arc::from([antecedent_core::PinnedDependency {
                name: "numpy".into(),
                version: "2.0.0".into(),
            }]),
            implementation: CallbackImplementation::Recoverable {
                source_sha256: digest.into(),
                entry_point: "doubler".into(),
            },
        }
    }

    fn doubling(digest: &str) -> (Arc<dyn BatchCallback>, Intervention) {
        let callback: Arc<dyn BatchCallback> = Arc::new(Doubler(recoverable_spec(digest)));
        let program = Intervention::transform(
            VariableId::from_raw(1),
            ValueExpr::callback(Arc::clone(&callback), [ValueExpr::var(price())]),
        );
        (callback, program)
    }

    #[test]
    fn a_recoverable_callback_is_encoded_as_its_record_under_its_own_version() {
        let digest = "a".repeat(64);
        let (callback, program) = doubling(&digest);
        let saved = vec![floor(0.85), program];
        let bytes = encode_policy_program(&saved).unwrap();
        let envelope: PolicyProgramEnvelope =
            ciborium::from_reader(&bytes[POLICY_PROGRAM_MAGIC.len()..]).unwrap();
        assert_eq!(envelope.version, POLICY_PROGRAM_RECOVERABLE_VERSION);
        // The bytes carry the record and no code: only the digest of the source is there.
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(&digest) && text.contains("doubler") && text.contains("numpy"));

        let records = policy_program_callback_records(&bytes).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source_sha256, digest);

        let supplied = Arc::clone(&callback);
        let loaded = decode_policy_program_with(&bytes, &|record| {
            (record.name == "doubler").then(|| Arc::clone(&supplied))
        })
        .unwrap();
        // The loaded program calls the supplied implementation and has the saved identity.
        assert_eq!(loaded, saved);
        assert_eq!(
            policy_program_identity(&loaded).unwrap(),
            policy_program_identity(&saved).unwrap()
        );
        assert_eq!(encode_policy_program(&loaded).unwrap(), bytes, "encoding is canonical");
        // The identity moves with the source digest.
        let other = vec![floor(0.85), doubling(&"b".repeat(64)).1];
        assert_ne!(
            policy_program_identity(&other).unwrap(),
            policy_program_identity(&saved).unwrap()
        );
    }

    #[test]
    fn a_recoverable_program_is_refused_without_a_verified_implementation() {
        let digest = "a".repeat(64);
        let (_, program) = doubling(&digest);
        let bytes = encode_policy_program(&[program]).unwrap();
        let refused = |result: Result<Vec<Intervention>, IoError>| match result {
            Err(IoError::Refused { code, message }) => {
                assert_eq!(code, "policy_callback_unrecoverable");
                message
            }
            other => panic!("expected a coded refusal, got {other:?}"),
        };
        let message = refused(decode_policy_program(&bytes));
        assert!(message.contains("doubler"), "{message}");
        let message = refused(decode_policy_program_with(&bytes, &|_| None));
        assert!(message.contains("implementation_missing"), "{message}");

        // An implementation built from different source, entry point or pins is refused.
        let different: Vec<(&str, CallbackSpec)> = vec![
            ("source_sha256", recoverable_spec(&"b".repeat(64))),
            ("dependencies", {
                let mut spec = recoverable_spec(&digest);
                spec.dependencies = Arc::from([antecedent_core::PinnedDependency {
                    name: "numpy".into(),
                    version: "2.1.0".into(),
                }]);
                spec
            }),
            ("implementation", {
                let mut spec = recoverable_spec(&digest);
                spec.implementation = CallbackImplementation::Runtime;
                spec
            }),
        ];
        for (field, spec) in different {
            let callback: Arc<dyn BatchCallback> = Arc::new(Doubler(spec));
            let message =
                refused(decode_policy_program_with(&bytes, &|_| Some(Arc::clone(&callback))));
            assert!(message.contains(field), "{field}: {message}");
        }
    }

    #[test]
    fn the_encoding_version_must_agree_with_the_callbacks_it_carries() {
        let (callback, program) = doubling(&"a".repeat(64));
        let (wires, _) = wires(&[program]).unwrap();
        // Records written under the closed-form version are refused even with a valid digest.
        let forged = PolicyProgramEnvelope {
            version: POLICY_PROGRAM_VERSION,
            identity: payload_digest_of(IDENTITY_KIND, &wires).unwrap(),
            interventions: wires.clone(),
        };
        let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
        ciborium::into_writer(&forged, &mut bytes).unwrap();
        assert!(matches!(
            decode_policy_program_with(&bytes, &|_| Some(Arc::clone(&callback))),
            Err(IoError::ManifestMismatch { message }) if message.contains("version")
        ));
        // The recoverable version without a callback is not an encoding this library writes.
        let closed = vec![InterventionWire::from_domain(&floor(0.85)).unwrap()];
        let forged = PolicyProgramEnvelope {
            version: POLICY_PROGRAM_RECOVERABLE_VERSION,
            identity: payload_digest_of(RECOVERABLE_IDENTITY_KIND, &closed).unwrap(),
            interventions: closed,
        };
        let mut bytes = POLICY_PROGRAM_MAGIC.to_vec();
        ciborium::into_writer(&forged, &mut bytes).unwrap();
        assert!(matches!(decode_policy_program(&bytes), Err(IoError::ManifestMismatch { .. })));
    }

    #[test]
    fn a_closed_form_program_keeps_its_released_bytes() {
        let bytes = encode_policy_program(&[floor(0.85)]).unwrap();
        let envelope: PolicyProgramEnvelope =
            ciborium::from_reader(&bytes[POLICY_PROGRAM_MAGIC.len()..]).unwrap();
        assert_eq!(envelope.version, 1);
        assert_eq!(
            antecedent_core::to_hex(&envelope.identity),
            "4828012ed5bdb86f66847911617623031457a3373151fb18a2b4c54d3a051d04"
        );
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

#[cfg(test)]
mod expression_identity_tests {
    use super::*;
    use antecedent_core::{ValueExpr, VariableId};
    #[test]
    fn expression_identity_survives_existing_wire_roundtrip_and_binds_reads() {
        let a = VariableId::from_raw(1);
        let b = VariableId::from_raw(2);
        let expr = ValueExpr::var(a)
            .mul(ValueExpr::var(b))
            .clip(ValueExpr::constant(0.0), ValueExpr::constant(10.0));
        let wire = ValueExprWire::from_domain(&expr).unwrap();
        let bytes = crate::convert::to_cbor(&wire).unwrap();
        let restored: ValueExprWire = crate::convert::from_cbor(&bytes).unwrap();
        assert_eq!(
            value_expression_identity(&expr).unwrap(),
            value_expression_identity(&restored.to_domain()).unwrap()
        );
        assert_ne!(
            value_expression_identity(&expr).unwrap(),
            value_expression_identity(&ValueExpr::var(a).mul(ValueExpr::var(a))).unwrap()
        );
        assert!(value_expression_identity(&ValueExpr::constant(f64::NAN)).is_err());
    }
}
