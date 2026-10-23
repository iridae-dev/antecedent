//! A minimal host with no agent logic: compile, read the status, tell a
//! supported claim from a refusal, and load an artifact.
//!
//! A host is whatever sits between a caller (a service, a notebook tool, an LLM
//! tool layer) and the library. This one only enforces the contract: it asks
//! the capability report before running, runs only what is open, reports the
//! claim with its status, reports a failure as a coded refusal, and accepts an
//! artifact only against the identity it retained. It plans nothing and chooses
//! nothing.
//!
//! Run: `cargo run -p antecedent --example reference_host`
//! Source: `examples/rust/reference_host.rs`

use antecedent::host::{HostStatus, claim_response, error_response, host_schema, operation_record};
use antecedent::io::consume_analysis_result;
use antecedent::prelude::*;
use antecedent::{CausalError, RefuteSuite};
use serde_json::Value;

/// One request the host serves: a study plus the data it runs on.
struct Request {
    study: Study,
    data: TabularData,
}

fn request() -> Result<Request, CausalError> {
    let schema =
        CausalSchemaBuilder::new().continuous("t").treatment().continuous("y").outcome().build()?;
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.1, 0.9, 2.9, 1.0, 3.0, 1.1, 3.2];
    let columns: [(&str, &[f64]); 2] = [("t", &treatment), ("y", &outcome)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns)?;
    let graph = Dag::from_named_edges(&schema, &[("t", "y")])?;
    let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?;
    Ok(Request { study, data })
}

/// Compile, check the capability report, run only when it is open, and answer
/// with the claim record. Returns the record and the artifact bytes the host
/// would store beside its own retained identity.
fn serve(request: &Request, ctx: &ExecutionContext) -> Result<(Value, Vec<u8>), CausalError> {
    // 1. Inspect before running: this identifies nothing and fits nothing.
    let inspected = request.study.inspect()?.capability();
    println!("before compile: {}", operation_record(&inspected)["status"]);

    // 2. Compile. The contract is what was compiled, not an answer.
    let prepared = request.study.prepare(ctx)?;
    let contract = prepared.contract()?;
    let report = contract.capability();
    let status = HostStatus::of_operation(&report);
    println!("after compile: {}", operation_record(&report)["status"]);
    if status == HostStatus::Unsupported {
        // The report already names the blockers; a host relays them.
        return Ok((operation_record(&report), Vec::new()));
    }

    // 3. Run, then read the claim. A partial claim is relayed as partial.
    let result = prepared.estimate(&request.data, ctx)?;
    let claim = result.claim(&contract, ctx)?;
    let response = claim_response(&claim);
    let bytes = prepared.encode_contracted_result(&result, "reference-host", ctx)?;
    Ok((response, bytes))
}

/// Accept artifact bytes only as a verified claim. Anything else is a refusal
/// the host reports; it never uses the numbers inside.
fn accept(bytes: &[u8]) -> Result<Value, CausalError> {
    let consumed = consume_analysis_result(bytes)?;
    if consumed.acceptance.accepts_as_claim() {
        Ok(serde_json::json!({
            "status": "supported",
            "artifact_digest": consumed.artifact_digest.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "estimate": consumed.body.estimate,
        }))
    } else {
        Ok(serde_json::json!({
            "status": "partial",
            "unresolved": consumed.acceptance.unresolved.iter().map(|u| u.as_ref()).collect::<Vec<_>>(),
            "restriction": consumed.acceptance.restriction.as_deref(),
        }))
    }
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    // The host first checks it speaks the contract version it was written for.
    let schema = host_schema().expect("the committed schema is valid JSON");
    assert_eq!(schema["x-contract-version"], antecedent::host::HOST_CONTRACT_VERSION);

    let ctx = ExecutionContext::for_tests(3);
    let request = request()?;

    // Supported claim.
    let (response, bytes) = serve(&request, &ctx)?;
    println!("response: {response}");
    assert_eq!(response["status"], "supported");
    assert_eq!(response["claim"]["kind"], "point");
    let value = response["claim"]["value"].as_f64().expect("a point claim carries its value");
    // t shifts y by about 2 in this table; the host does not interpret it further.
    assert!((value - 2.0).abs() < 0.2, "effect {value}");

    // Load the artifact against what the host retained, not against its own claim.
    let accepted = accept(&bytes)?;
    println!("loaded: {accepted}");
    assert_eq!(accepted["status"], "supported");

    // A refusal is a coded record, never a number. This one is what a route that
    // cannot answer returns; the host relays code, family and remedy unchanged.
    let refusal = antecedent::unsupported_reason!(
        "attested_not_reverifiable",
        "the attestation cannot be re-verified"
    );
    let refused = error_response(&refusal);
    println!("refusal: {refused}");
    assert_eq!(refused["status"], "unsupported");
    assert_eq!(refused["refusal"]["reason_code"], "attested_not_reverifiable");
    assert_eq!(refused["refusal"]["family"], "unsupported");

    // Altered bytes are refused or accepted only in part; they are never a claim.
    let mut altered = bytes;
    let middle = altered.len() / 2;
    altered[middle] ^= 0xff;
    match accept(&altered) {
        Err(error) => {
            let record = error_response(&error);
            println!("altered bytes: {record}");
            assert_eq!(record["status"], "unsupported");
            assert_eq!(record["refusal"]["family"], "serialization");
        }
        Ok(record) => assert_ne!(record["status"], "supported", "{record}"),
    }
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
