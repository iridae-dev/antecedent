//! Shared canonical ordering of exact transport law wires.

use antecedent_core::EvidenceCatalog;
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

type CatalogBindingIdentity = (u32, String, Option<String>);
type LawSnapshotIdentity = (String, u32, String);

pub(crate) fn sorted_catalog_bindings(catalog: &EvidenceCatalog) -> Vec<CatalogBindingIdentity> {
    let mut bindings = catalog
        .bindings
        .iter()
        .map(|binding| {
            (
                binding.regime.raw(),
                binding.snapshot_identity.to_string(),
                binding.dataset_identity.as_deref().map(str::to_owned),
            )
        })
        .collect::<Vec<_>>();
    bindings.sort();
    bindings
}

pub(crate) fn sorted_law_snapshots(laws: &[ExactLawWire]) -> Vec<LawSnapshotIdentity> {
    let mut snapshots = laws
        .iter()
        .map(|law| (law.population.clone(), law.regime, law.snapshot.clone()))
        .collect::<Vec<_>>();
    snapshots.sort();
    snapshots
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        DependenceGroup, RegimeBinding, RegimeId, SamplingDesign, Value, VariableId,
    };
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
        let expected =
            vec![("a".to_owned(), 0, "a".to_owned()), ("b".to_owned(), 0, "b".to_owned())];
        assert_eq!(sorted_law_snapshots(&ordered), expected);
        let mut reversed = ordered.clone();
        reversed.reverse();
        assert_eq!(sorted_law_snapshots(&reversed), expected);
    }
    #[test]
    fn catalog_binding_projection_sorts_and_preserves_optional_values() {
        let binding = |regime, snapshot: &str, dataset: Option<&str>| RegimeBinding {
            regime: RegimeId::from_raw(regime),
            snapshot_identity: Arc::from(snapshot),
            dataset_identity: dataset.map(Arc::from),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        };
        let catalog = EvidenceCatalog {
            environments: Arc::from([]),
            regimes: Arc::from([]),
            bindings: Arc::from([
                binding(2, "z", None),
                binding(1, "a", Some("")),
                binding(1, "a", None),
                binding(1, "a", None),
            ]),
            target_sampling: None,
        };
        assert_eq!(
            sorted_catalog_bindings(&catalog),
            vec![
                (1, "a".into(), None),
                (1, "a".into(), None),
                (1, "a".into(), Some("".into())),
                (2, "z".into(), None),
            ]
        );
    }
}
