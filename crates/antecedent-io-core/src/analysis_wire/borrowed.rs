//! Borrowed canonical identification wire for explicit native read models.
use super::*;
use antecedent_core::CausalQuery;
use crate::expr_wire::{BorrowedExprArenaWire,Sequence};
use crate::query_wire::{BorrowedRetainedQueryWire,borrowed::RawVariables};
use crate::trace::BorrowedAssumptionsWire;
use serde::ser::SerializeStruct;

/// Borrow a retained identification result without cloning its proof arena,
/// derivation, assumptions, diagnostics, estimands or compiled query wire.
pub struct BorrowedIdentificationWire<'a>{
    result:&'a IdentificationResult,
    query:BorrowedRetainedQueryWire<'a>,
    arena:BorrowedExprArenaWire<'a>,
}
impl<'a> BorrowedIdentificationWire<'a>{
    /// Borrow the exact proof and its producer's already compiled query.
    /// # Errors
    /// Existing arena wire coordinate limits are exceeded.
    pub fn new(result:&'a IdentificationResult,source:&'a CausalQuery,encoded:&'a CausalQueryWire)->Result<Self,IoError>{
        Ok(Self{result,query:BorrowedRetainedQueryWire::new(&result.query,source,encoded),arena:BorrowedExprArenaWire::new(&result.arena)?})
    }
}
impl Serialize for BorrowedIdentificationWire<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
    let r=self.result;let mut wire=s.serialize_struct("IdentificationResultWire",9+usize::from(r.hedge.is_some()))?;
    wire.serialize_field("status",identification_status_snake(r.status))?;
    wire.serialize_field("query",&self.query)?;
    wire.serialize_field("estimands",&Sequence(||r.estimands.iter().map(BorrowedEstimand)))?;
    wire.serialize_field("arena",&self.arena)?;
    wire.serialize_field("derivation",&Sequence(||r.derivation.steps.iter().map(Derivation)))?;
    wire.serialize_field("required_assumptions",&BorrowedAssumptionsWire(&r.required_assumptions))?;
    wire.serialize_field("diagnostics",&Sequence(||r.diagnostics.iter().map(Diagnostic)))?;
    wire.serialize_field("candidates_examined",&r.performance.candidates_examined)?;
    wire.serialize_field("sets_returned",&r.performance.sets_returned)?;
    if let Some(hedge)=&r.hedge{wire.serialize_field("hedge",&Hedge(hedge))?;}
    wire.end()
}}

pub(crate) struct BorrowedEstimand<'a>(pub(crate) &'a antecedent_expr::IdentifiedEstimand);
impl Serialize for BorrowedEstimand<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let e=self.0;
    let mut wire=s.serialize_struct("IdentifiedEstimandWire",5+usize::from(e.rd_design.is_some()))?;
    wire.serialize_field("method",e.method.as_ref())?;
    wire.serialize_field("adjustment_set",&RawVariables(&e.adjustment_set))?;
    wire.serialize_field("instruments",&RawVariables(&e.instruments))?;
    wire.serialize_field("mediators",&RawVariables(&e.mediators))?;
    wire.serialize_field("functional",&e.functional.raw())?;
    if let Some(rd)=e.rd_design{wire.serialize_field("rd_design",&RdDesignWire{running_variable:rd.running_variable.raw(),cutoff:rd.cutoff,bandwidth:rd.bandwidth})?;}
    wire.end()
}}
struct Derivation<'a>(&'a antecedent_identify::DerivationStep);
impl Serialize for Derivation<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let mut wire=s.serialize_struct("DerivationStepWire",2)?;wire.serialize_field("rule",self.0.rule.as_ref())?;wire.serialize_field("detail",self.0.detail.as_ref())?;wire.end()}}
struct Diagnostic<'a>(&'a antecedent_core::Diagnostic);
impl Serialize for Diagnostic<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let d=self.0;let mut wire=s.serialize_struct("DiagnosticWire",5+usize::from(!d.fields.is_empty()))?;
    wire.serialize_field("code",d.code.as_ref())?;
    wire.serialize_field("kind",match d.kind{DiagnosticKind::Scientific=>"scientific",DiagnosticKind::Support=>"support",DiagnosticKind::Execution=>"execution"})?;
    wire.serialize_field("severity",match d.severity{DiagnosticSeverity::Info=>"info",DiagnosticSeverity::Warning=>"warning",DiagnosticSeverity::Error=>"error"})?;
    wire.serialize_field("message",d.message.as_ref())?;wire.serialize_field("artifact_id",&d.artifact_id.as_deref())?;
    if !d.fields.is_empty(){wire.serialize_field("fields",&Sequence(||d.fields.iter().map(|(key,value)|(key.as_ref(),value.as_ref()))))?;}
    wire.end()
}}
struct Hedge<'a>(&'a antecedent_identify::HedgeCertificate);
impl Serialize for Hedge<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let h=self.0;let mut wire=s.serialize_struct("HedgeCertificateWire",4+usize::from(h.problem.is_some()))?;
    wire.serialize_field("f",&RawVariables(&h.f))?;wire.serialize_field("f_prime",&RawVariables(&h.f_prime))?;
    wire.serialize_field("f_dense",&Sequence(||h.f_dense.iter().map(|v|v.raw())))?;
    wire.serialize_field("f_prime_dense",&Sequence(||h.f_prime_dense.iter().map(|v|v.raw())))?;
    if let Some(problem)=&h.problem{wire.serialize_field("problem",&Problem(problem))?;}wire.end()
}}
struct Problem<'a>(&'a antecedent_identify::HedgeProblem);
impl Serialize for Problem<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{let p=self.0;let mut wire=s.serialize_struct("HedgeProblemWire",5)?;
    wire.serialize_field("variables",&RawVariables(&p.variables))?;wire.serialize_field("directed",p.directed.as_ref())?;wire.serialize_field("bidirected",p.bidirected.as_ref())?;
    wire.serialize_field("treatments",&RawVariables(&p.treatments))?;wire.serialize_field("outcomes",&RawVariables(&p.outcomes))?;wire.end()
}}
