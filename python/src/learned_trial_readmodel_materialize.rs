//! Borrowed historical transport wire encoding with one exactly admitted JSON output.
//! No proof/certificate arrays or strings are cloned for serialization.
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_identify::{PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification};
use pyo3::prelude::*;
use serde::{Serialize, Serializer, ser::SerializeSeq};
use serde_json::Value;
use std::{io::{self, Write}, sync::Arc};

pub(super) struct Ids<'a>(&'a [VariableId]);
impl Serialize for Ids<'_> {
    fn serialize<S: Serializer>(&self, serializer:S)->Result<S::Ok,S::Error> {
        let mut sequence=serializer.serialize_seq(Some(self.0.len()))?;
        for id in self.0 {sequence.serialize_element(&id.raw())?;}
        sequence.end()
    }
}
pub(super) struct Texts<'a>(&'a [Arc<str>]);
impl Serialize for Texts<'_> {
    fn serialize<S: Serializer>(&self, serializer:S)->Result<S::Ok,S::Error> {
        let mut sequence=serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0 {sequence.serialize_element(value.as_ref())?;}
        sequence.end()
    }
}
#[derive(Serialize)]
pub(super) struct Factor<'a> {
    population:&'a str,
    #[serde(skip_serializing_if="Option::is_none")]
    regime:Option<u32>,
    variables:Ids<'a>, conditioned_on:Ids<'a>, interventions:Ids<'a>,
}
impl<'a> From<&'a PopulationFactor> for Factor<'a> {
    fn from(value:&'a PopulationFactor)->Self {
        Self {population:&value.population,regime:value.regime.map(antecedent_core::RegimeId::raw),
            variables:Ids(&value.variables),conditioned_on:Ids(&value.conditioned_on),interventions:Ids(&value.interventions)}
    }
}
pub(super) struct Factors<'a>(&'a [PopulationFactor]);
impl Serialize for Factors<'_> {
    fn serialize<S: Serializer>(&self, serializer:S)->Result<S::Ok,S::Error> {
        let mut sequence=serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0 {sequence.serialize_element(&Factor::from(value))?;}
        sequence.end()
    }
}
#[derive(Serialize)]
#[serde(rename_all="snake_case")]
pub(super) enum Formula<'a> {
    Direct(Factor<'a>),
    Standardize {over:Ids<'a>,source_response:Factor<'a>,target_law:Factor<'a>},
    RecursiveFactorization {sum_out:Ids<'a>,factors:Factors<'a>},
}
impl<'a> From<&'a TransportFormula> for Formula<'a> {
    fn from(value:&'a TransportFormula)->Self {
        match value {
            TransportFormula::Direct(f)=>Self::Direct(Factor::from(f)),
            TransportFormula::Standardize {over,source_response,target_law}=>Self::Standardize {
                over:Ids(over),source_response:Factor::from(source_response),target_law:Factor::from(target_law)},
            TransportFormula::RecursiveFactorization {sum_out,factors}=>Self::RecursiveFactorization {
                sum_out:Ids(sum_out),factors:Factors(factors)},
        }
    }
}
#[derive(Serialize)]
pub(super) struct Certificate<'a> {rule:&'a str,selection_targets:Ids<'a>,premises:Texts<'a>}
impl<'a> From<&'a TransportCertificate> for Certificate<'a> {
    fn from(value:&'a TransportCertificate)->Self {
        Self {rule:&value.rule,selection_targets:Ids(&value.selection_targets),premises:Texts(&value.premises)}
    }
}
#[derive(Serialize)]
pub(super) struct Missing<'a> {reason:&'a str,witness:Ids<'a>,message:&'a str}
#[derive(Serialize)]
#[serde(rename_all="snake_case")]
pub(super) enum Proof<'a> {
    Transportable {formula:Formula<'a>,certificate:Certificate<'a>},
    NotCertified(Missing<'a>), MissingEvidence(Missing<'a>),
}
impl<'a> From<&'a TransportIdentification> for Proof<'a> {
    fn from(value:&'a TransportIdentification)->Self {
        match value {
            TransportIdentification::Transportable {formula,certificate}=>Self::Transportable {
                formula:Formula::from(formula),certificate:Certificate::from(certificate)},
            TransportIdentification::NotCertified(c)=>Self::NotCertified(Missing {reason:&c.reason,witness:Ids(&c.witness),message:&c.message}),
            TransportIdentification::MissingEvidence(c)=>Self::MissingEvidence(Missing {reason:&c.reason,witness:Ids(&c.missing),message:&c.message}),
        }
    }
}
#[derive(Serialize)]
pub(super) struct Payload<'a> {
    pub summary:&'a Value,
    pub transport:Proof<'a>,
    pub children:[();0],
}

struct Counter<'a> {length:usize,context:&'a ExecutionContext}
impl Write for Counter<'_> {
    fn write(&mut self, bytes:&[u8])->io::Result<usize> {
        if self.context.cancellation.is_cancelled() {return Err(io::Error::other("cancelled"));}
        self.length=self.length.checked_add(bytes.len()).ok_or_else(||io::Error::other("JSON length overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self)->io::Result<()> {Ok(())}
}

/// Native output bytes and their real reservation have one owner and no Clone.
/// Dropping the owner drops the Vec before releasing the admission lease.
pub(super) struct AdmittedJson {
    bytes:Vec<u8>,
    _lease:AdmissionLease,
}
impl AdmittedJson {
    pub(super) fn as_str(&self)->PyResult<&str> {
        std::str::from_utf8(&self.bytes).map_err(|error|crate::CausalStateError::new_err(error.to_string()))
    }
}
fn json_error(context:&ExecutionContext,error:impl ToString)->PyErr {
    if context.cancellation.is_cancelled() {
        crate::py_err(antecedent::CausalError::Cancelled {stage:"trial_identification_materialize"})
    } else {crate::CausalStateError::new_err(error.to_string())}
}

pub(super) fn admitted(value:&impl Serialize,context:&ExecutionContext)->PyResult<AdmittedJson> {
    let mut counter=Counter {length:0,context};
    serde_json::to_writer(&mut counter,value).map_err(|error| {
        if context.cancellation.is_cancelled() {json_error(context,error)}
        else {crate::py_err(antecedent::CausalError::Resource {message:error.to_string()})}
    })?;
    let length=u64::try_from(counter.length).map_err(|error|crate::py_err(antecedent::CausalError::Resource {message:error.to_string()}))?;
    let lease=context.acquire_admission(0,length).map_err(|error|match error {
        AdmissionError::Cancelled=>crate::py_err(antecedent::CausalError::Cancelled {stage:"trial_identification_materialize"}),
        error=>crate::py_err(antecedent::CausalError::Resource {message:error.to_string()}),
    })?;
    let mut bytes=Vec::new();
    bytes.try_reserve_exact(counter.length).map_err(|error|crate::py_err(antecedent::CausalError::Resource {message:error.to_string()}))?;
    serde_json::to_writer(&mut bytes,value).map_err(|error|json_error(context,error))?;
    if context.cancellation.is_cancelled() {return Err(json_error(context,"cancelled"));}
    if bytes.len()!=counter.length {
        return Err(crate::value_err("immutable transport proof changed during serialization"));
    }
    Ok(AdmittedJson {bytes,_lease:lease})
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    #[test]
    fn borrowed_encoding_matches_every_native_transport_wire_variant() {
        let factor=PopulationFactor {population:Arc::from("target\"unicode-λ"),regime:Some(antecedent_core::RegimeId::from_raw(7)),
            variables:Arc::from([VariableId::from_raw(1)]),conditioned_on:Arc::from([VariableId::from_raw(2)]),interventions:Arc::from([VariableId::from_raw(0)])};
        let certificate=TransportCertificate {rule:Arc::from("standardize"),selection_targets:Arc::from([VariableId::from_raw(2)]),premises:Arc::from([Arc::from("first\nsecond")])};
        let proofs=[
            TransportIdentification::Transportable {formula:TransportFormula::Standardize {over:Arc::from([VariableId::from_raw(2)]),source_response:factor.clone(),target_law:factor.clone()},certificate:certificate.clone()},
            TransportIdentification::Transportable {formula:TransportFormula::RecursiveFactorization {sum_out:Arc::from([VariableId::from_raw(2)]),factors:Arc::from([factor])},certificate},
            TransportIdentification::NotCertified(antecedent_identify::NotCertifiedCertificate {reason:Arc::from("no_rule"),witness:Arc::from([VariableId::from_raw(2)]),message:Arc::from("not a completeness claim")}),
            TransportIdentification::MissingEvidence(antecedent_identify::MissingEvidenceCertificate {reason:Arc::from("missing"),missing:Arc::from([VariableId::from_raw(2)]),message:Arc::from("declare source evidence")}),
        ];
        for proof in proofs {
            assert_eq!(serde_json::to_value(Proof::from(&proof)).unwrap(),serde_json::to_value(antecedent_io::transport_identification_to_wire(&proof)).unwrap());
        }
    }

    #[test]
    fn cancellation_after_output_allocation_releases_actual_buffer_lease() {
        use std::sync::atomic::{AtomicUsize,Ordering};
        struct CancelSecond {
            calls:AtomicUsize,
            token:antecedent_core::CancellationToken,
        }
        impl Serialize for CancelSecond {
            fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error> {
                if self.calls.fetch_add(1,Ordering::SeqCst)==1 {self.token.cancel();}
                serializer.serialize_str("native-json-buffer")
            }
        }
        let pool=AdmissionPool::fixed(1,64);
        let ctx=ExecutionContext::for_tests(1).with_admission(pool.clone());
        let value=CancelSecond {calls:AtomicUsize::new(0),token:ctx.cancellation.clone()};
        assert!(admitted(&value,&ctx).is_err());
        assert_eq!(value.calls.load(Ordering::SeqCst),2);
        assert_eq!(pool.available().bytes(),64);
    }

    #[test]
    fn borrowed_encoder_preserves_native_wire_and_admitted_output_lifetime() {
        let factor=PopulationFactor {population:Arc::from("target"),regime:None,
            variables:Arc::from([VariableId::from_raw(1)]),conditioned_on:Arc::from([]),interventions:Arc::from([VariableId::from_raw(0)])};
        let proof=TransportIdentification::Transportable {formula:TransportFormula::Direct(factor),
            certificate:TransportCertificate {rule:Arc::from("direct"),selection_targets:Arc::from([]),premises:Arc::from([Arc::from("actual native premise")])}};
        assert_eq!(serde_json::to_value(Proof::from(&proof)).unwrap(),serde_json::to_value(antecedent_io::transport_identification_to_wire(&proof)).unwrap());
        let source=Proof::from(&proof);
        let length=serde_json::to_vec(&source).unwrap().len() as u64;
        let pool=AdmissionPool::fixed(1,length);
        let ctx=ExecutionContext::for_tests(1).with_admission(pool.clone());
        let owner=Arc::new(admitted(&source,&ctx).unwrap());
        assert_eq!(pool.available().bytes(),0);
        let retained=Arc::clone(&owner);drop(owner);
        assert_eq!(pool.available().bytes(),0);
        assert_eq!(retained.bytes.len() as u64,length);
        drop(retained);assert_eq!(pool.available().bytes(),length);
        let too_small=AdmissionPool::fixed(1,length-1);
        assert!(admitted(&source,&ExecutionContext::for_tests(1).with_admission(too_small.clone())).is_err());
        assert_eq!(too_small.available().bytes(),length-1);
        let cancelled=ExecutionContext::for_tests(1).with_admission(pool.clone());
        cancelled.cancellation.cancel();
        assert!(admitted(&source,&cancelled).is_err());
        assert_eq!(pool.available().bytes(),length);
    }
}
