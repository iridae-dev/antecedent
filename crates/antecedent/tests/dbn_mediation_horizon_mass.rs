//! Bounded check of horizon eligibility under the current three-variable identifier.
use antecedent_core::{MediationContrast, MediationQuery, VariableId};
use antecedent_discovery::{mask_is_dag, temporal_dag_from_dbn_masks};
use antecedent_identify::TemporalMediationIdentifier;

#[test]
fn three_variable_lag_one_templates_share_horizon_certifiability() {
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
    let mut both = 0;
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
            // This bounded family has horizon-varying adjustment sets but no
            // mixed certification. Synthetic certificate failures exercise
            // the general cache-isolation contract in prepared unit tests.
            assert_eq!(one.is_ok(), two.is_ok(), "c={contemporaneous}, l={lagged}");
            both += usize::from(one.is_ok());
        }
    }
    assert!(both > 0, "the enumeration must contain identified templates");
}
