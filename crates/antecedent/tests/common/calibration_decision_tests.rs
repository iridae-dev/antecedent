use super::*;

fn config() -> Config {
    Config {
        id: ID.into(),
        family_digest: "a".repeat(64),
        validation_domain: VALIDATION_DOMAIN.into(),
        family_alpha: 0.05,
        family_groups: 700,
        family_grid_points: 3,
        family_cells: 700 * 768,
        cell_cap_per_group: 256,
        looks: vec![
            Look { planned_replicates: 400, weight: 0.5 },
            Look { planned_replicates: 2000, weight: 0.25 },
            Look { planned_replicates: 20000, weight: 0.25 },
        ],
    }
}

#[test]
fn registered_domains_keep_distinct_fixed_streams_and_unknown_domains_fail_closed() {
    let mut c = config();
    assert_eq!(c.validation_salt(), Some(0xE1FC_1261_2A1E_C540));
    assert!(c.validate().is_ok());
    c.validation_domain = HC3_VALIDATION_DOMAIN.into();
    assert_eq!(c.validation_salt(), Some(0xDD5E_E217_1F1D_846C));
    assert!(c.validate().is_ok());
    assert_ne!(VALIDATION_SALT, HC3_VALIDATION_SALT);
    c.validation_domain = PAG_VALIDATION_DOMAIN.into();
    assert_eq!(c.validation_salt(), Some(0xDE3D_61A0_4092_7214));
    assert!(c.validate().is_ok());
    assert_ne!(PAG_VALIDATION_SALT, VALIDATION_SALT);
    assert_ne!(PAG_VALIDATION_SALT, HC3_VALIDATION_SALT);
    c.validation_domain = PAG_CONFIDENCE_SET_VALIDATION_DOMAIN.into();
    assert_eq!(c.validation_salt(), Some(0x6921_1267_88F4_5DC5));
    assert!(c.validate().is_ok());
    for previous in [VALIDATION_SALT, HC3_VALIDATION_SALT, PAG_VALIDATION_SALT] {
        assert_ne!(PAG_CONFIDENCE_SET_VALIDATION_SALT, previous);
    }
    for unknown in ["", "release_2_4_1_weighted_hc3_validation_v2", "arbitrary_retry"] {
        c.validation_domain = unknown.into();
        assert_eq!(c.validation_salt(), None);
        assert!(c.validate().is_err());
    }
}

#[test]
fn binomial_tails_match_independent_integer_enumeration() {
    for n in 1..=20_u32 {
        let mut masses = Vec::new();
        for k in 0..=n {
            let choose = (0..k).fold(1_u128, |v, j| v * u128::from(n - j) / u128::from(j + 1));
            // Exact integer numerator for Binomial(n,19/20).
            masses.push(choose * 19_u128.pow(k));
        }
        for k in 0..=n {
            #[allow(clippy::cast_precision_loss, reason = "bounded integer probability oracle")]
            let lower = masses[..=k as usize].iter().sum::<u128>() as f64 / 20_u128.pow(n) as f64;
            #[allow(clippy::cast_precision_loss, reason = "bounded integer probability oracle")]
            let upper = masses[k as usize..].iter().sum::<u128>() as f64 / 20_u128.pow(n) as f64;
            let got = binomial_p_value(n, k, 0.95, true).unwrap();
            assert!((got - lower).abs() < 1e-12, "n={n} k={k} {got} {lower}");
            let got = binomial_p_value(n, k, 0.95, false).unwrap();
            assert!((got - (2.0 * lower.min(upper)).min(1.0)).abs() < 1e-12);
        }
    }
}

#[test]
fn tiny_prefix_misses_are_distinct_from_clear_miscalibration() {
    let c = config();
    assert!(c.evaluate(20000, 20000, 18926, 0.95, false).unwrap().compatible);
    assert!(!c.evaluate(20000, 20000, 17200, 0.95, false).unwrap().compatible);
    assert!(!c.evaluate(20000, 20000, 19900, 0.95, false).unwrap().compatible);
    assert!(c.evaluate(20000, 20000, 19900, 0.95, true).unwrap().compatible);
}

#[test]
fn reported_denominator_does_not_choose_the_spending_look() {
    let d = config().evaluate(2000, 1953, 1829, 0.95, false).unwrap();
    assert_eq!(d.planned_replicates, 2000);
    assert_eq!(d.scored, 1953);
    assert_eq!(d.look_weight.to_bits(), 0.25_f64.to_bits());
    assert!(d.compatible);
    assert!(config().evaluate(1953, 1953, 1829, 0.95, false).is_err());
}

#[test]
fn one_cell_cannot_test_conflicting_prefixes_at_the_same_look() {
    let mut registry = CellRegistry::default();
    let c = config();
    let d = c.evaluate(2000, 2000, 1900, 0.95, false).unwrap();
    assert!(registry.register("fixed cell", &d, 0.95).is_ok());
    assert!(registry.register("fixed cell", &d, 0.95).is_ok());
    let changed = c.evaluate(2000, 1999, 1900, 0.95, false).unwrap();
    assert!(registry.register("fixed cell", &changed, 0.95).is_err());
    assert!(registry.register("fixed cell", &d, 0.9).is_err());
    let later = c.evaluate(20000, 20000, 19000, 0.95, false).unwrap();
    assert!(registry.register("fixed cell", &later, 0.95).is_ok());
    for index in 1..256 {
        assert!(registry.register(&format!("fixed cell {index}"), &d, 0.95).is_ok());
    }
    assert!(registry.register("extra cell", &d, 0.95).is_err());
}

#[test]
fn malformed_families_and_unregistered_looks_fail_closed() {
    let c = config();
    assert!(c.evaluate(3000, 2000, 1900, 0.95, false).is_err());
    assert!(c.evaluate(2000, 2001, 1900, 0.95, false).is_err());
    for field in 0..5 {
        let mut bad = c.clone();
        match field {
            0 => bad.family_cells -= 1,
            1 => bad.cell_cap_per_group = 257,
            2 => bad.looks[0].weight = 1.0,
            3 => bad.family_digest = "x".repeat(64),
            _ => bad.family_alpha = 0.1,
        }
        assert!(bad.validate().is_err());
    }
    assert!(binomial_p_value(0, 0, 0.95, false).is_err());
    assert!(binomial_p_value(20, 21, 0.95, false).is_err());
}
