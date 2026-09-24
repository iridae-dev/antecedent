//! Exact supplied-law records, kept distinct from empirical tables.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{IoError, query_wire::ValueWire};
use antecedent_core::{RegimeId, VariableId};
use antecedent_expr::{
    DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawOrigin, LawTolerance,
};
use serde::{Deserialize, Serialize};

/// Dense complete law. Axis order determines row-major probability order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct ExactLawWire {
    pub population: String,
    pub regime: u32,
    pub interventions: Vec<(u32, ValueWire)>,
    pub axes: Vec<(u32, Vec<ValueWire>)>,
    pub probabilities: Vec<f64>,
    pub snapshot: String,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    #[serde(default)]
    pub origin: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empirical_counts: Option<Vec<u64>>,
}
impl ExactLawWire {
    /// Preserve exact values and declared tolerance without normalization.
    #[must_use]
    pub fn from_law(law: &ExactDiscreteLaw) -> Self {
        let mut wire = Self::metadata(law);
        wire.probabilities = law.probabilities().to_vec();
        wire.empirical_counts = law.empirical_counts().map(<[u64]>::to_vec);
        wire
    }
    /// Read only declared axes and provider identity; never read table masses.
    #[must_use]
    pub fn metadata(law: &ExactDiscreteLaw) -> Self {
        Self {
            population: law.population().into(),
            regime: law.regime().raw(),
            interventions: law
                .interventions()
                .iter()
                .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                .collect(),
            axes: law
                .axes()
                .iter()
                .map(|a| (a.variable.raw(), a.values.iter().map(ValueWire::from_value).collect()))
                .collect(),
            probabilities: Vec::new(),
            empirical_counts: None,
            snapshot: law.snapshot_identity().into(),
            absolute_tolerance: law.tolerance().absolute,
            relative_tolerance: law.tolerance().relative,
            origin: law.origin().as_str().into(),
        }
    }
    /// Validate all table, coverage, domain and normalization invariants on load.
    ///
    /// # Errors
    /// Invalid exact law.
    pub fn to_law(&self) -> Result<ExactDiscreteLaw, IoError> {
        let origin = LawOrigin::parse(&self.origin)
            .ok_or_else(|| IoError::Convert(format!("unknown law origin {}", self.origin)))?;
        let build = match origin {
            LawOrigin::SuppliedExact | LawOrigin::LearnedPlugin => ExactDiscreteLaw::try_new,
            LawOrigin::EmpiricalPlugin => ExactDiscreteLaw::try_empirical,
            LawOrigin::BayesianPosterior => ExactDiscreteLaw::try_bayesian_posterior,
        };
        let law = build(
            self.population.clone(),
            RegimeId::from_raw(self.regime),
            self.interventions
                .iter()
                .map(|(v, x)| InterventionAssignment {
                    variable: VariableId::from_raw(*v),
                    value: x.to_value(),
                })
                .collect::<Vec<_>>(),
            self.axes
                .iter()
                .map(|(v, values)| DiscreteAxis {
                    variable: VariableId::from_raw(*v),
                    values: values.iter().map(ValueWire::to_value).collect(),
                })
                .collect::<Vec<_>>(),
            self.probabilities.clone(),
            self.snapshot.clone(),
            LawTolerance { absolute: self.absolute_tolerance, relative: self.relative_tolerance },
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
        match (origin, &self.empirical_counts) {
            (LawOrigin::LearnedPlugin, Some(counts)) => law
                .with_empirical_counts(counts.clone())
                .map_err(|e| IoError::Convert(e.to_string())),
            (LawOrigin::LearnedPlugin, None) => {
                Err(IoError::Convert("learned law lacks empirical support".into()))
            }
            (_, Some(_)) => Err(IoError::Convert("unexpected learned support metadata".into())),
            (_, None) => Ok(law),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{Value, VariableId};
    use std::sync::Arc;

    #[test]
    fn bayesian_posterior_origin_survives_exact_law_roundtrip() {
        let law = ExactDiscreteLaw::try_bayesian_posterior(
            "target",
            RegimeId::from_raw(7),
            [],
            [DiscreteAxis {
                variable: VariableId::from_raw(2),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            }],
            [0.3, 0.7],
            "snapshot",
            LawTolerance::default(),
        )
        .unwrap();
        let decoded = ExactLawWire::from_law(&law).to_law().unwrap();
        assert_eq!(decoded.origin(), LawOrigin::BayesianPosterior);
        assert_eq!(decoded.probabilities(), law.probabilities());
    }
}
