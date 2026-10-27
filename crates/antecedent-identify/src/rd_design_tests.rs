//! Design-parameter checks of the sharp regression-discontinuity identifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, VariableId};

use super::*;

const T: u32 = 0;
const Y: u32 = 1;
const R: u32 = 2;

fn ate() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y))
}

fn identify_with(cutoff: f64, bandwidth: f64) -> Result<IdentificationResult, IdentificationError> {
    SharpRdIdentifier::new(SharpRdConfig::new(VariableId::from_raw(R), cutoff, bandwidth))
        .identify(CausalQuery::AverageEffect(ate()))
}

#[test]
fn invalid_bandwidth_or_cutoff_is_refused_not_silently_accepted() {
    for (cutoff, bandwidth) in
        [(0.5, 0.0), (0.5, -1.0), (0.5, f64::NAN), (0.5, f64::INFINITY), (f64::NAN, 1.0)]
    {
        assert!(
            matches!(
                identify_with(cutoff, bandwidth),
                Err(IdentificationError::UnsupportedQuery { .. })
            ),
            "cutoff={cutoff} bandwidth={bandwidth}"
        );
    }
    // A negative cutoff is a legitimate design.
    assert!(identify_with(-2.0, 0.5).is_ok());
}

#[test]
fn the_bandwidth_is_carried_with_the_design_but_does_not_change_the_functional() {
    let narrow = identify_with(0.5, 0.1).unwrap();
    let wide = identify_with(0.5, 1.0).unwrap();
    assert_eq!(narrow.estimands[0].rd_design.unwrap().bandwidth.to_bits(), 0.1f64.to_bits());
    assert_eq!(wide.estimands[0].rd_design.unwrap().bandwidth.to_bits(), 1.0f64.to_bits());
    assert_eq!(narrow.status, wide.status);
    assert_eq!(
        narrow.arena.node(narrow.estimands[0].functional),
        wide.arena.node(wide.estimands[0].functional)
    );
}

#[test]
fn the_cutoff_enters_the_functional_so_a_different_cutoff_is_a_different_estimand() {
    let at_half = identify_with(0.5, 1.0).unwrap();
    let at_one = identify_with(1.0, 1.0).unwrap();
    assert_ne!(
        at_half.average_effect().unwrap().target_population,
        at_one.average_effect().unwrap().target_population
    );
    assert_eq!(at_half.estimands[0].rd_design.unwrap().cutoff.to_bits(), 0.5f64.to_bits());
    assert_eq!(at_one.estimands[0].rd_design.unwrap().cutoff.to_bits(), 1.0f64.to_bits());
}
