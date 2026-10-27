//! Borrowed canonical assumption records, preserving original order and scope.
use super::*;
use serde::ser::{SerializeSeq, SerializeStruct};
use std::fmt;

/// Borrowed serialization of the canonical assumption record collection.
/// Strings and variable scopes are formatted directly into the selected
/// serializer; no copied record vector or joined scope string is constructed.
pub struct BorrowedAssumptionsWire<'a>(pub(crate) &'a AssumptionSet);
impl<'a> BorrowedAssumptionsWire<'a> {
    /// Borrow original assumption records without constructing a wire vector.
    #[must_use]
    pub const fn new(records: &'a AssumptionSet) -> Self { Self(records) }
}
impl Serialize for BorrowedAssumptionsWire<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.entries.len()))?;
        for record in &self.0.entries { sequence.serialize_element(&BorrowedRecord(record))?; }
        sequence.end()
    }
}
struct BorrowedRecord<'a>(&'a AssumptionRecord);
impl Serialize for BorrowedRecord<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut record = serializer.serialize_struct("AssumptionRecordWire", 4)?;
        record.serialize_field("assumption", &Tag::from(&self.0.assumption))?;
        record.serialize_field("source", &Source(&self.0.source))?;
        record.serialize_field("scope", &Scope(&self.0.scope))?;
        record.serialize_field("status", self.0.status.as_str())?;
        record.end()
    }
}
struct Source<'a>(&'a AssumptionSource);
impl fmt::Display for Source<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            AssumptionSource::UserDeclared => formatter.write_str("user_declared"),
            AssumptionSource::AlgorithmDefault { algorithm } => write!(formatter, "algorithm_default:{algorithm}"),
            AssumptionSource::Artifact => formatter.write_str("artifact"),
            AssumptionSource::Derived { from } => write!(formatter, "derived:{from}"),
        }
    }
}
impl Serialize for Source<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
struct Scope<'a>(&'a AssumptionScope);
impl fmt::Display for Scope<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            AssumptionScope::Global => formatter.write_str("global"),
            AssumptionScope::Identification => formatter.write_str("identification"),
            AssumptionScope::Estimation => formatter.write_str("estimation"),
            AssumptionScope::Discovery => formatter.write_str("discovery"),
            AssumptionScope::Variables { variables } => {
                formatter.write_str("variables:[")?;
                for (index, variable) in variables.iter().enumerate() {
                    if index > 0 { formatter.write_str(",")?; }
                    write!(formatter, "{}", variable.raw())?;
                }
                formatter.write_str("]")
            }
        }
    }
}
impl Serialize for Scope<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Tag<'a> {
    CausalMarkov, Faithfulness, CausalSufficiency, Consistency, Positivity, NoInterference,
    Stationarity, PiecewiseStationarity, NoSelectionBias, ExclusionRestriction { instrument: u32 },
    Monotonicity,
    ParametricRestriction { id: &'a str, #[serde(skip_serializing_if = "Option::is_none")] description: Option<&'a str> },
    PriorRestriction { id: &'a str, #[serde(skip_serializing_if = "Option::is_none")] description: Option<&'a str> },
    Custom { id: &'a str, #[serde(skip_serializing_if = "Option::is_none")] description: Option<&'a str> },
}
impl<'a> From<&'a Assumption> for Tag<'a> {
    fn from(a: &'a Assumption) -> Self {
        match a {
            Assumption::CausalMarkov => Self::CausalMarkov,
            Assumption::Faithfulness => Self::Faithfulness,
            Assumption::CausalSufficiency => Self::CausalSufficiency,
            Assumption::Consistency => Self::Consistency,
            Assumption::Positivity => Self::Positivity,
            Assumption::NoInterference => Self::NoInterference,
            Assumption::Stationarity => Self::Stationarity,
            Assumption::PiecewiseStationarity => Self::PiecewiseStationarity,
            Assumption::NoSelectionBias => Self::NoSelectionBias,
            Assumption::ExclusionRestriction { instrument } => Self::ExclusionRestriction { instrument: instrument.raw() },
            Assumption::Monotonicity => Self::Monotonicity,
            Assumption::ParametricRestriction(p) => Self::ParametricRestriction { id: p.id.as_ref(), description: Some(p.description.as_ref()) },
            Assumption::PriorRestriction(p) => Self::PriorRestriction { id: p.id.as_ref(), description: Some(p.description.as_ref()) },
            Assumption::Custom { id, description } => Self::Custom { id: id.as_ref(), description: Some(description.as_ref()) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    fn borrowed_assumptions_preserve_all_records_and_original_variable_scope_order() {
        let assumptions = [Assumption::CausalMarkov, Assumption::Faithfulness, Assumption::CausalSufficiency,
            Assumption::Consistency, Assumption::Positivity, Assumption::NoInterference,
            Assumption::Stationarity, Assumption::PiecewiseStationarity, Assumption::NoSelectionBias,
            Assumption::ExclusionRestriction { instrument: VariableId::from_raw(3) }, Assumption::Monotonicity,
            Assumption::ParametricRestriction(antecedent_core::ParametricAssumption { id: Arc::from("linear"), description: Arc::from("linear mean") }),
            Assumption::PriorRestriction(antecedent_core::PriorAssumption { id: Arc::from("normal"), description: Arc::from("normal prior") }),
            Assumption::Custom { id: Arc::from("custom\\\"λ"), description: Arc::from("original\\ntext") }];
        let sources = [AssumptionSource::UserDeclared, AssumptionSource::AlgorithmDefault { algorithm: Arc::from("backdoor") },
            AssumptionSource::Artifact, AssumptionSource::Derived { from: Arc::from("parent\\\"λ") }];
        let scopes = [AssumptionScope::Global, AssumptionScope::Identification, AssumptionScope::Estimation,
            AssumptionScope::Discovery, AssumptionScope::Variables { variables: Arc::from([VariableId::from_raw(9),VariableId::from_raw(2)]) }];
        let statuses = [AssumptionStatus::Declared, AssumptionStatus::Supported, AssumptionStatus::Contradicted, AssumptionStatus::Untestable];
        let mut set = AssumptionSet::new();
        for (index, assumption) in assumptions.into_iter().enumerate() {
            set.entries.push(AssumptionRecord { assumption, source: sources[index % sources.len()].clone(),
                scope: scopes[index % scopes.len()].clone(), status: statuses[index % statuses.len()] });
        }
        let original = assumptions_to_wire(&set);
        let borrowed = BorrowedAssumptionsWire(&set);
        assert_eq!(serde_json::to_vec(&borrowed).unwrap(), serde_json::to_vec(&original).unwrap());
        let mut old = Vec::new();let mut new = Vec::new();
        ciborium::into_writer(&original,&mut old).unwrap();
        ciborium::into_writer(&borrowed,&mut new).unwrap();
        assert_eq!(new,old);
    }
}
