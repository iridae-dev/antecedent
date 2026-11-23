//! Bounded check of horizon eligibility under the current three-variable identifier.
use antecedent_core::{MediationContrast, MediationQuery, VariableId};
use antecedent_discovery::{mask_is_dag, temporal_dag_from_dbn_masks};
use antecedent_identify::{IdentificationError, TemporalMediationIdentifier};

#[test]
fn three_variable_lag_one_templates_have_horizon_specific_certifiability() {
    let variables = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let query = MediationQuery::binary(
        variables[0],
        variables[2],
        [variables[1]],
        MediationContrast::Mediated,
    );
    let ider = TemporalMediationIdentifier {
        allow_natural_controlled_alias: true,
        ..TemporalMediationIdentifier::new()
    };
    let (mut both, mut neither, mut mixed) = (0, 0, 0);
    for contemporaneous in 0..64 {
        if contemporaneous & 8 == 0 || !mask_is_dag(contemporaneous, 3) {
            continue;
        }
        for lagged in 0..512 {
            if lagged & 2 == 0 {
                continue;
            }
            let Ok(graph) = temporal_dag_from_dbn_masks(contemporaneous, lagged, 3, 1, &variables)
            else {
                continue;
            };
            let one = ider.identify_with_horizon(&graph, &query, 1);
            let two = ider.identify_with_horizon(&graph, &query, 2);
            for result in [&one, &two] {
                assert!(
                    !matches!(
                        result,
                        Err(IdentificationError::Budget { .. } | IdentificationError::Cancelled)
                    ),
                    "operational failure at c={contemporaneous}, l={lagged}: {result:?}"
                );
            }
            if contemporaneous == 8 && lagged == 34 {
                // M(0) -> Y(0), T(-1) -> M(0), and M(-1) -> Y(0).
                // At horizon 2, T(-1) -> M(0) -> Y(1) bypasses the
                // outcome-slice mediator M(1), so it cannot be represented by
                // the current path-product estimand.
                assert!(one.is_ok(), "horizon 1: {one:?}");
                assert!(
                    matches!(
                        &two,
                        Err(IdentificationError::NotCertified { message })
                            if message.contains("earlier slice")
                    ),
                    "horizon 2: {two:?}"
                );
            }
            match (one.is_ok(), two.is_ok()) {
                (true, true) => both += 1,
                (false, false) => neither += 1,
                _ => mixed += 1,
            }
        }
    }
    assert!(both > 0, "the enumeration must contain identified templates");
    // The enumeration also contains templates both horizons refuse.
    assert!(neither > 0, "the enumeration must contain refused templates");
    assert!(mixed > 0, "the enumeration must contain horizon-specific refusals");
}
