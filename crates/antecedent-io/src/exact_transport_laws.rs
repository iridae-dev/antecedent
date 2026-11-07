//! Shared canonical ordering of exact transport law wires.

use antecedent_expr::ExactTransportData;

use crate::{IoError, exact_law_wire::ExactLawWire};

pub(crate) fn canonical_laws(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            crate::to_cbor(&(&wire.population, wire.regime, &wire.interventions))
                .map(|key| (key, wire))
        })
        .collect::<Result<Vec<_>, _>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{RegimeId, Value, VariableId};
    use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};

    use super::*;

    #[test]
    fn canonical_laws_preserve_bytes_across_input_order() {
        let law = |population: &str| {
            ExactDiscreteLaw::try_new(
                population,
                RegimeId::from_raw(0),
                [],
                [DiscreteAxis {
                    variable: VariableId::from_raw(0),
                    values: Arc::from([Value::f64(0.0), Value::f64(1.0)]),
                }],
                [0.4, 0.6],
                population,
                LawTolerance::default(),
            )
            .unwrap()
        };
        let first = ExactTransportData::try_new(vec![law("b"), law("a")], 10).unwrap();
        let second = ExactTransportData::try_new(vec![law("a"), law("b")], 10).unwrap();
        let ordered = canonical_laws(&first).unwrap();
        assert_eq!(
            ordered.iter().map(|wire| wire.population.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(
            crate::to_cbor(&ordered).unwrap(),
            crate::to_cbor(&canonical_laws(&second).unwrap()).unwrap()
        );
    }
}
