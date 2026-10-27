//! Borrowed canonical structural transport proof serialization.
use super::*;
use crate::query_wire::borrowed::RawVariables;
use crate::expr_wire::Sequence;
use serde::ser::{SerializeStruct,SerializeStructVariant};
/// Borrow the existing native structural transport identification and all factors.
pub struct BorrowedTransportIdentificationWire<'a>(pub(crate) &'a TransportIdentification);
impl<'a> BorrowedTransportIdentificationWire<'a> {
    /// Borrow the exact existing structural transport result.
    #[must_use]
    pub const fn new(result: &'a TransportIdentification) -> Self { Self(result) }
}
impl Serialize for BorrowedTransportIdentificationWire<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{match self.0{
    TransportIdentification::Transportable{formula,certificate}=>{let mut wire=s.serialize_struct_variant("TransportIdentificationWire",0,"transportable",2)?;wire.serialize_field("formula",&Formula(formula))?;wire.serialize_field("certificate",&Certificate(certificate))?;wire.end()}
    TransportIdentification::NotCertified(c)=>s.serialize_newtype_variant("TransportIdentificationWire",0,"not_certified",&NotCertified{reason:c.reason.as_ref(),witness:&c.witness,message:c.message.as_ref()}),
    TransportIdentification::MissingEvidence(c)=>s.serialize_newtype_variant("TransportIdentificationWire",0,"missing_evidence",&NotCertified{reason:c.reason.as_ref(),witness:&c.missing,message:c.message.as_ref()}),
}}}
struct Factor<'a>(&'a PopulationFactor);
impl Serialize for Factor<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let f=self.0;let mut wire=s.serialize_struct("PopulationFactorWire",4+usize::from(f.regime.is_some()))?;
    wire.serialize_field("population",f.population.as_ref())?;if let Some(regime)=f.regime{wire.serialize_field("regime",&regime.raw())?;}
    wire.serialize_field("variables",&RawVariables(&f.variables))?;wire.serialize_field("conditioned_on",&RawVariables(&f.conditioned_on))?;wire.serialize_field("interventions",&RawVariables(&f.interventions))?;wire.end()
}}
struct Formula<'a>(&'a TransportFormula);
impl Serialize for Formula<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{match self.0{
    TransportFormula::Direct(f)=>s.serialize_newtype_variant("TransportFormulaWire",0,"direct",&Factor(f)),
    TransportFormula::Standardize{over,source_response,target_law}=>{let mut wire=s.serialize_struct_variant("TransportFormulaWire",0,"standardize",3)?;wire.serialize_field("over",&RawVariables(over))?;wire.serialize_field("source_response",&Factor(source_response))?;wire.serialize_field("target_law",&Factor(target_law))?;wire.end()}
    TransportFormula::RecursiveFactorization{sum_out,factors}=>{let mut wire=s.serialize_struct_variant("TransportFormulaWire",0,"recursive_factorization",2)?;wire.serialize_field("sum_out",&RawVariables(sum_out))?;wire.serialize_field("factors",&Sequence(||factors.iter().map(Factor)))?;wire.end()}
}}}
struct Certificate<'a>(&'a TransportCertificate);
impl Serialize for Certificate<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let c=self.0;let mut wire=s.serialize_struct("InterferenceTransportCertificateWire",3)?;wire.serialize_field("rule",c.rule.as_ref())?;wire.serialize_field("selection_targets",&RawVariables(&c.selection_targets))?;wire.serialize_field("premises",&Sequence(||c.premises.iter().map(|p|p.as_ref())))?;wire.end()}}
struct NotCertified<'a>{reason:&'a str,witness:&'a [antecedent_core::VariableId],message:&'a str}
impl Serialize for NotCertified<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let mut wire=s.serialize_struct("NotCertifiedCertificateWire",3)?;wire.serialize_field("reason",self.reason)?;wire.serialize_field("witness",&RawVariables(self.witness))?;wire.serialize_field("message",self.message)?;wire.end()}}

#[cfg(test)]
mod tests{
    use super::*;
    use antecedent_core::{RegimeId,VariableId};
    fn compare(result:&TransportIdentification){
        let original=transport_identification_to_wire(result);let borrowed=BorrowedTransportIdentificationWire::new(result);
        assert_eq!(serde_json::to_vec(&borrowed).unwrap(),serde_json::to_vec(&original).unwrap());
        let mut old=Vec::new();let mut new=Vec::new();ciborium::into_writer(&original,&mut old).unwrap();ciborium::into_writer(&borrowed,&mut new).unwrap();assert_eq!(new,old);
    }
    #[test]
    fn borrowed_transport_preserves_all_formula_and_refusal_variants(){
        let x=VariableId::from_raw(0);let y=VariableId::from_raw(1);
        let source=PopulationFactor{population:Arc::from("source\\\"λ"),regime:Some(RegimeId::from_raw(3)),variables:Arc::from([y]),conditioned_on:Arc::from([x]),interventions:Arc::from([x])};
        let mut target=source.clone();target.population=Arc::from("target");target.regime=None;
        let certificate=TransportCertificate{rule:Arc::from("rule"),selection_targets:Arc::from([y]),premises:Arc::from([Arc::from("original premise")])};
        for formula in [TransportFormula::Direct(source.clone()),TransportFormula::Standardize{over:Arc::from([x]),source_response:source.clone(),target_law:target.clone()},
            TransportFormula::RecursiveFactorization{sum_out:Arc::from([x]),factors:Arc::from([source,target])}]{
            compare(&TransportIdentification::Transportable{formula,certificate:certificate.clone()});
        }
        compare(&TransportIdentification::NotCertified(NotCertifiedCertificate{reason:Arc::from("not_certified"),witness:Arc::from([x,y]),message:Arc::from("original limitation")}));
        compare(&TransportIdentification::MissingEvidence(MissingEvidenceCertificate{reason:Arc::from("missing_evidence"),missing:Arc::from([y,x]),message:Arc::from("original obligation")}));
    }
}
