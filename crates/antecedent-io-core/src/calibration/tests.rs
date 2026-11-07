use super::*;

#[test]
fn calibration_status_wire_text_is_the_four_status_codes() {
    for status in CalibrationStatus::ALL {
        crate::convert::assert_cbor_text_code(status.as_str(), &status);
    }
}

#[test]
#[allow(clippy::float_cmp, reason = "the check requires exact equality of the retained values")]
fn basis_wire_carries_every_field_of_the_basis() {
    let names = ["q", "g", "fixed", "tabular", "Bayesian", "e", "m", "s", "d", "p", "f"];
    let basis = antecedent_core::CalibrationBasis::new(
        names.map(std::sync::Arc::<str>::from),
        0.95,
        std::sync::Arc::from("point"),
        120,
        Some(7),
        Some(9),
        0.25,
    );
    let wire = CalibrationBasisWire::from(&basis);
    let key = &wire.key;
    assert_eq!(
        [
            key.query.as_str(),
            key.graph_class.as_str(),
            key.structure.as_str(),
            key.modality.as_str(),
            key.inference.as_str(),
            key.estimator.as_str(),
            key.interval_method.as_str(),
            key.se_kind.as_str(),
            key.dependence.as_str(),
            key.posterior.as_str(),
            key.functional.as_str(),
        ],
        names
    );
    assert_eq!((key.level, key.identification.as_str()), (0.95, "point"));
    assert_eq!(wire.scope.row_count, 120);
    assert_eq!((wire.scope.replicates_ok, wire.scope.posterior_draws), (Some(7), Some(9)));
    assert_eq!(wire.scope.unidentified_mass, 0.25);
}

#[test]
fn structure_and_inference_spell_their_keys() {
    assert_eq!(CalibrationStructure::from_axis("graph_posterior").as_str(), "graph_posterior");
    assert_eq!(CalibrationStructure::from_axis("anything else").as_str(), "fixed");
    assert_eq!(CalibrationInference::from_label("BAYESIAN"), CalibrationInference::Bayesian);
    assert_eq!(CalibrationInference::from_label("Bayesian").as_str(), "Bayesian");
    assert_eq!(CalibrationInference::from_label("frequentist").as_str(), "Frequentist");
}

fn key_from(record: &CoverageRecord) -> CalibrationKeyWire {
    CalibrationKeyWire {
        validation: if record.validation_schema == 2 {
            record.validation.to_owned()
        } else {
            HISTORICAL_VALIDATION_BINDINGS
                .iter()
                .find(|binding| {
                    binding.record_id == record.id
                        && binding.calibration_sha == record.calibration_sha
                        && binding.test == record.test
                })
                .map_or_else(String::new, |binding| binding.validation.to_owned())
        },
        query: record.query.to_owned(),
        graph_class: record.graph_class.to_owned(),
        structure: if HISTORICAL_VALIDATION_BINDINGS.iter().any(|binding| {
            binding.record_id == record.id
                && binding.calibration_sha == record.calibration_sha
                && binding.test == record.test
                && binding.source_structure == "accepted"
        }) {
            "accepted".to_owned()
        } else {
            record.structure.to_owned()
        },
        modality: record.modality.to_owned(),
        inference: record.inference.to_owned(),
        estimator: record.estimator.to_owned(),
        interval_method: record.interval_method.to_owned(),
        se_kind: record.se_kind.to_owned(),
        dependence: record.dependence.to_owned(),
        posterior: record.posterior.to_owned(),
        functional: record.functional.to_owned(),
        level: record.nominal,
        identification: record.identification.to_owned(),
    }
}

fn matching_ids(key: &CalibrationKeyWire) -> Vec<&'static str> {
    RECORDS.iter().filter(|record| key.same_construction(record)).map(|record| record.id).collect()
}

#[test]
fn versioned_graph_imputation_key_matches_only_its_exact_construction() {
    // Synthetic in-memory record: this does not assert that measurement exists.
    let mut record = RECORDS[0];
    record.id = "test.graph_imputation.v2_validation_none";
    record.validation_schema = 2;
    record.validation = "none";
    record.structure = "graph_posterior";
    record.inference = "GraphImputationCompositeV1";
    record.estimator = "graph_imputation.bic_conjugate";
    record.interval_method = "nearest_rank_95";
    record.posterior = "mar_bic_graph_mixture_conjugate";
    record.functional = "graph_imputation.v1:synthetic";
    let mut key = key_from(&record);
    assert!(key.same_construction(&record));
    for legacy in ["Frequentist", "Bayesian", "Composite", "GraphImputationCompositeV0"] {
        key.inference = legacy.to_owned();
        assert!(!key.same_construction(&record));
    }
    // Historical generic conversion remains untouched.
    assert_eq!(
        CalibrationInference::from_label("GraphImputationCompositeV1"),
        CalibrationInference::Frequentist
    );
}

#[test]
fn prior_predictive_graph_scope_never_borrows_fixed_effect_record() {
    let mut record = RECORDS[0];
    record.id = "test.graph_imputation.v2_validation_none";
    record.validation_schema = 2;
    record.validation = "none";
    record.structure = "graph_posterior";
    record.inference = "GraphImputationCompositeV2";
    record.functional = "graph_imputation.v2:synthetic";
    let mut key = key_from(&record);
    assert!(key.same_construction(&record));
    key.inference = "GraphImputationCompositeV3".to_owned();
    key.functional =
        "prior_predictive.v1:beta_uniform_minus1_plus1:graph_imputation.v2:synthetic".to_owned();
    assert!(!key.same_construction(&record));
    record.inference = "GraphImputationCompositeV3";
    record.functional =
        "prior_predictive.v1:beta_uniform_minus1_plus1:graph_imputation.v2:synthetic";
    assert!(key.same_construction(&record));
    key.functional =
        "prior_predictive.v1:beta_uniform_minus2_plus2:graph_imputation.v2:synthetic".to_owned();
    assert!(!key.same_construction(&record));
}

#[test]
fn construction_index_matches_a_linear_scan() {
    let mut positive_modes = std::collections::BTreeSet::new();
    for record in &RECORDS {
        let key = key_from(record);
        let indexed: Vec<_> = shipped_candidates(&key)
            .iter()
            .filter(|candidate| key.same_construction(candidate))
            .map(|candidate| candidate.id)
            .collect();
        assert_eq!(indexed, matching_ids(&key), "{}", record.id);
        if !key.validation.is_empty() {
            assert!(
                indexed.contains(&record.id),
                "reviewed record must match its own key: {}",
                record.id
            );
            positive_modes.insert(key.validation.clone());
        }
    }
    for mode in ["none", "full", "not_applicable"] {
        assert!(positive_modes.contains(mode), "generated reviewed snapshot must exercise {mode}");
    }
    for alias in &FUNCTIONAL_IDENTITY_ALIASES {
        let Some(record) = RECORDS.iter().find(|record| record.id == alias.record_id) else {
            continue;
        };
        let mut key = key_from(record);
        key.functional = alias.canonical_functional.to_owned();
        let indexed: Vec<_> = shipped_candidates(&key)
            .iter()
            .filter(|candidate| key.same_construction(candidate))
            .map(|candidate| candidate.id)
            .collect();
        assert_eq!(indexed, matching_ids(&key), "{}", alias.record_id);
    }
}

#[test]
fn attesting_index_matches_linear_membership() {
    for record in &RECORDS {
        assert_eq!(
            record_attests_current_code(record),
            ATTESTING_RECORD_IDS.contains(&record.id),
            "{}",
            record.id
        );
    }
    assert!(attesting_ids().binary_search(&"not-a-coverage-record").is_err());
}

fn record(id: &'static str) -> CoverageRecord {
    CoverageRecord {
        id: Box::leak(format!("{id}.v2_validation_none").into_boxed_str()),
        validation_schema: 2,
        validation: "none",
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "fixed",
        modality: "tabular",
        inference: "Frequentist",
        estimator: "aipw",
        interval_method: "bootstrap_se",
        se_kind: "",
        dependence: "iid",
        posterior: "",
        functional: "all_observed.mean",
        identification: "point",
        nominal: 0.95,
        n_min: 500,
        n_max: 500,
        replicates_min: 200,
        posterior_draws_min: 0,
        unidentified_mass_max: 0.0,
        observed: 0.948,
        mcse: 0.011,
        replicates: 400,
        boundary: false,
        grid: &[],
        confirmation: &[],
        confirmed_passing_range: None,
        dgp: "crates/antecedent/tests/x.rs::dgp",
        test: "crates/antecedent/tests/x.rs::test",
        calibration_sha: "0123456789abcdef0123456789abcdef01234567",
    }
}

#[test]
fn legacy_wire_omits_unestablished_validation() {
    let mut original = basis();
    original.key.validation.clear();
    let value = serde_json::to_value(&original).expect("serialize legacy basis");
    assert!(value["key"].get("validation").is_none());
    let restored: CalibrationBasisWire = serde_json::from_value(value).expect("read legacy basis");
    assert_eq!(restored, original);
    assert!(!restored.key.same_construction(&record("fixture")));
}

#[test]
fn graphless_designs_match_only_their_not_applicable_measurement() {
    for query in ["PanelDid", "RandomizedEffect"] {
        let mut actual = basis();
        actual.key.query = query.into();
        actual.key.graph_class = "RandomizedTrial".into();
        actual.key.structure = "design".into();
        actual.key.validation = "not_applicable".into();
        let mut measured = record("fixture");
        measured.id = "cov.fixture.v2_validation_not_applicable";
        measured.validation = "not_applicable";
        measured.query = query;
        measured.graph_class = "RandomizedTrial";
        measured.structure = "design";
        assert_eq!(calibration_slot_in(&actual, &[measured]).status, "calibrated");
        for validation in ["", "none", "cheap", "full"] {
            actual.key.validation = validation.into();
            assert_ne!(
                calibration_slot_in(&actual, &[measured]).status,
                "calibrated",
                "{validation}"
            );
        }
    }
}

#[test]
fn validation_coordinate_refuses_legacy_unknown_and_other_modes() {
    let basis = basis();
    let measured = record("fixture");
    assert!(basis.key.same_construction(&measured));
    for validation in ["", "unknown", "cheap", "full", "not_applicable"] {
        let mut altered = basis.clone();
        altered.key.validation = validation.into();
        assert!(!altered.key.same_construction(&measured));
    }
    let mut legacy = measured;
    legacy.validation_schema = 0;
    legacy.validation = "";
    assert!(!basis.key.same_construction(&legacy));
    assert!(calibration_slot_in(&basis, &[legacy]).reason.is_some());
    let mut forged = measured;
    forged.validation = "cheap";
    let mut cheap = basis.clone();
    cheap.key.validation = "cheap".into();
    assert!(!cheap.key.same_construction(&forged));
    forged.id = "cov.fixture.v2_validation_cheap";
    assert!(cheap.key.same_construction(&forged));
    forged.id = "cov.fixture.v2_validation_cheap.v2_validation_none";
    assert!(!cheap.key.same_construction(&forged));
    let mut accepted = measured;
    accepted.structure = "accepted";
    accepted.id = "cov.fixture.v2_validation_none.v2_structure_accepted";
    let mut accepted_basis = basis.clone();
    accepted_basis.key.structure = "accepted".into();
    assert!(accepted_basis.key.same_construction(&accepted));
    assert!(!basis.key.same_construction(&accepted));
    accepted.id = "cov.fixture.v2_validation_none";
    assert!(!accepted_basis.key.same_construction(&accepted));
}

#[test]
fn reviewed_historical_validation_is_bound_to_exact_measurement() {
    let basis = basis();
    let mut legacy = record("historical");
    legacy.validation_schema = 0;
    legacy.validation = "";
    let proof = HistoricalValidationBinding {
        record_id: legacy.id,
        calibration_sha: legacy.calibration_sha,
        test: legacy.test,
        validation: "none",
        source_structure: "explicit",
    };
    assert!(validation_matches_with_bindings(&basis.key, &legacy, &[proof]));
    let mut changed = legacy;
    changed.calibration_sha = "different";
    assert!(!validation_matches_with_bindings(&basis.key, &changed, &[proof]));
    changed = legacy;
    changed.test = "different";
    assert!(!validation_matches_with_bindings(&basis.key, &changed, &[proof]));
    changed = legacy;
    changed.id = "different";
    assert!(!validation_matches_with_bindings(&basis.key, &changed, &[proof]));
    let mut other = basis;
    other.key.validation = "cheap".into();
    assert!(!validation_matches_with_bindings(&other.key, &legacy, &[proof]));
    let mut explicit = proof;
    explicit.source_structure = "explicit";
    other.key.validation = "none".into();
    other.key.structure = "accepted".into();
    assert!(!validation_matches_with_bindings(&other.key, &legacy, &[explicit]));
    let mut unresolved = proof;
    unresolved.source_structure = "";
    assert!(!validation_matches_with_bindings(&other.key, &legacy, &[unresolved]));
    let mut accepted_proof = proof;
    accepted_proof.source_structure = "accepted";
    assert!(validation_matches_with_bindings(&other.key, &legacy, &[accepted_proof]));
}

fn basis() -> CalibrationBasisWire {
    CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "none".into(),
            query: "AverageEffect".into(),
            graph_class: "Dag".into(),
            structure: "fixed".into(),
            modality: "tabular".into(),
            inference: "Frequentist".into(),
            estimator: "aipw".into(),
            interval_method: "bootstrap_se".into(),
            se_kind: String::new(),
            dependence: "iid".into(),
            posterior: String::new(),
            functional: "all_observed.mean".into(),
            level: 0.95,
            identification: "point".into(),
        },
        scope: CalibrationScopeWire {
            row_count: 500,
            replicates_ok: Some(200),
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    }
}

#[test]
fn functional_alias_is_bound_to_exact_original_record_identity() {
    let measured = record("cov.a");
    let alias = FunctionalIdentityAlias {
        record_id: measured.id,
        measurement_sha: measured.calibration_sha,
        original_functional: measured.functional,
        canonical_functional: "canonical_recovery_functional",
    };
    assert!(functional_matches(&measured, alias.canonical_functional, &[alias]));
    assert!(functional_matches(&measured, measured.functional, &[]));
    assert!(!functional_matches(&measured, "foreign_functional", &[alias]));
    assert!(!functional_matches(&record("cov.foreign"), alias.canonical_functional, &[alias]));
    let mut changed = measured;
    changed.calibration_sha = "foreign_measurement";
    assert!(!functional_matches(&changed, alias.canonical_functional, &[alias]));
    changed = measured;
    changed.functional = "changed_original_functional";
    assert!(!functional_matches(&changed, alias.canonical_functional, &[alias]));
    let mut actual = basis();
    actual.key.functional = alias.canonical_functional.into();
    assert!(actual.key.same_construction_with_aliases(&measured, &[alias]));
    assert!(!actual.key.same_construction_with_aliases(&measured, &[]));
    for axis in 0..10 {
        let mut incompatible = actual.clone();
        match axis {
            0 => incompatible.key.query = "foreign".into(),
            1 => incompatible.key.graph_class = "foreign".into(),
            2 => incompatible.key.structure = "foreign".into(),
            3 => incompatible.key.modality = "foreign".into(),
            4 => incompatible.key.inference = "foreign".into(),
            5 => incompatible.key.estimator = "foreign".into(),
            6 => incompatible.key.interval_method = "foreign".into(),
            7 => incompatible.key.se_kind = "foreign".into(),
            8 => incompatible.key.dependence = "foreign".into(),
            _ => incompatible.key.posterior = "foreign".into(),
        }
        assert!(!incompatible.key.same_construction_with_aliases(&measured, &[alias]));
    }
}

#[test]
fn inside_scope_is_calibrated_against_the_worst_covering_record() {
    let mut low = record("cov.low");
    low.observed = 0.94;
    let slot = calibration_slot_in(&basis(), &[record("cov.high"), low]);
    assert_eq!(slot.status, "calibrated");
    assert_eq!(slot.record_id.as_deref(), Some("cov.low.v2_validation_none"));
    assert_eq!(slot.observed, Some(0.94));
    assert_eq!(slot.reason, None);
}

#[test]
fn too_few_successful_replicates_are_not_calibrated() {
    let mut b = basis();
    b.scope.replicates_ok = Some(2);
    let slot = calibration_slot_in(&b, &[record("cov.a")]);
    assert_eq!(slot.status, "outside_measured_scope");
    assert_eq!(slot.reason.as_deref(), Some(REPLICATES_BELOW_MEASURED));
}

#[test]
fn level_and_identification_mismatches_are_not_measured() {
    let mut b = basis();
    b.key.level = 0.9;
    assert_eq!(
        calibration_slot_in(&b, &[record("cov.a")]).reason.as_deref(),
        Some(LEVEL_NOT_MEASURED)
    );
    let mut b = basis();
    b.key.identification = "partial".into();
    let slot = calibration_slot_in(&b, &[record("cov.a")]);
    assert_eq!(slot.status, "not_measured");
    assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
}

#[test]
fn parent_adjustment_binds_only_to_records_measured_for_it() {
    assert_eq!(identification_key("point", ["backdoor.criterion", "temporal.unfold"]), "point");
    let parent = identification_key(
        "point",
        ["temporal.parent_adjustment", "backdoor.adjustment_set", "temporal.unfold"],
    );
    assert_eq!(parent, "point+temporal.parent_adjustment");

    let unfolding = record("cov.unfolding");
    let mut measured = record("cov.parent");
    measured.identification = "point+temporal.parent_adjustment";

    let mut b = basis();
    b.key.identification = parent;
    let slot = calibration_slot_in(&b, &[unfolding]);
    assert_eq!(slot.status, "not_measured");
    assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
    let slot = calibration_slot_in(&b, &[unfolding, measured]);
    assert_eq!(slot.status, "calibrated");
    assert_eq!(slot.record_id.as_deref(), Some("cov.parent.v2_validation_none"));

    // And an unfolding-identified result never binds to the parent record.
    let slot = calibration_slot_in(&basis(), &[measured]);
    assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
}

#[test]
fn row_count_outside_the_measured_range_is_not_assessed() {
    let mut b = basis();
    b.scope.row_count = 1200;
    let slot = calibration_slot_in(&b, &[record("cov.a")]);
    assert_eq!(slot.status, "outside_measured_scope");
    assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
    assert_eq!(slot.scope_n, Some(500));
}

const GRID: [CoverageGridPoint; 3] = [
    CoverageGridPoint {
        point: 0,
        n_min: 250,
        n_max: 250,
        observed: 0.945,
        mcse: 0.011,
        replicates: 400,
        boundary: false,
    },
    CoverageGridPoint {
        point: 1,
        n_min: 500,
        n_max: 500,
        observed: 0.948,
        mcse: 0.011,
        replicates: 400,
        boundary: false,
    },
    CoverageGridPoint {
        point: 2,
        n_min: 1000,
        n_max: 1000,
        observed: 0.951,
        mcse: 0.011,
        replicates: 400,
        boundary: false,
    },
];
static FAILING_SMALL: [CoverageGridPoint; 3] = {
    let mut grid = GRID;
    grid[0].observed = 0.90;
    grid[0].boundary = true;
    grid
};

fn gridded(id: &'static str, grid: &'static [CoverageGridPoint]) -> CoverageRecord {
    let mut out = record(id);
    out.n_min = 250;
    out.n_max = 1000;
    out.grid = grid;
    out
}

#[test]
fn a_passing_grid_is_calibrated_across_its_whole_measured_range() {
    for n in [250, 251, 500, 777, 1000] {
        let mut b = basis();
        b.scope.row_count = n;
        let slot = calibration_slot_in(&b, &[gridded("cov.grid", &GRID)]);
        assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
        assert_eq!((slot.scope_n, slot.scope_n_max), (Some(250), Some(1000)));
    }
    for n in [249, 1001, 5000] {
        let mut b = basis();
        b.scope.row_count = n;
        let slot = calibration_slot_in(&b, &[gridded("cov.grid", &GRID)]);
        assert_eq!(slot.status, "outside_measured_scope", "n = {n}: {slot:?}");
        assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
    }
}

#[test]
fn the_range_is_the_grid_span_not_a_contradicting_summary() {
    let mut stale = gridded("cov.grid", &GRID);
    stale.n_min = 500;
    stale.n_max = 500;
    let mut b = basis();
    b.scope.row_count = 300;
    assert_eq!(calibration_slot_in(&b, &[stale]).status, "calibrated");
}

#[test]
fn a_grid_that_fails_at_any_point_is_a_boundary_over_the_whole_range() {
    // Even at the passing base point: a failure at 250 rows is not averaged
    // into a pass at 500.
    for n in [250, 500, 1000] {
        let mut b = basis();
        b.scope.row_count = n;
        let slot = calibration_slot_in(&b, &[gridded("cov.grid", &FAILING_SMALL)]);
        assert_eq!(slot.status, "measured_below_nominal", "n = {n}: {slot:?}");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        assert_eq!(slot.observed, Some(0.90), "the failing point's coverage is reported");
    }
}

const CONFIRMED_UPPER: [CoverageGridPoint; 2] = {
    let mut second = GRID[1];
    second.observed = 0.944;
    let mut third = GRID[2];
    third.observed = 0.957;
    [second, third]
};

fn recovered(confirmation: &'static [CoverageGridPoint]) -> CoverageRecord {
    let mut out = gridded("cov.recovered", &FAILING_SMALL);
    out.observed = 0.90;
    out.boundary = true;
    out.confirmation = confirmation;
    out.confirmed_passing_range = Some((500, 1000));
    out
}

#[test]
fn a_confirmed_upper_range_is_calibrated_by_its_worst_passing_point() {
    for n in [500, 777, 1000] {
        let mut b = basis();
        b.scope.row_count = n;
        let slot = calibration_slot_in(&b, &[recovered(&CONFIRMED_UPPER)]);
        assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
        assert_eq!(slot.reason, None);
        assert_eq!((slot.scope_n, slot.scope_n_max), (Some(500), Some(1000)));
        // The measurement's 0.948 and 0.951 and the confirmation's 0.944 and
        // 0.957: the lowest of them, never the failing point's 0.90.
        assert_eq!(slot.observed, Some(0.944));
    }
}

#[test]
fn a_confirmed_range_leaves_the_failing_region_refused() {
    for n in [250, 251, 499] {
        let mut b = basis();
        b.scope.row_count = n;
        let slot = calibration_slot_in(&b, &[recovered(&CONFIRMED_UPPER)]);
        assert_eq!(slot.status, "measured_below_nominal", "n = {n}: {slot:?}");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        assert_eq!(slot.observed, Some(0.90), "the failing point stays on the record");
    }
    let mut b = basis();
    b.scope.row_count = 1001;
    assert_eq!(
        calibration_slot_in(&b, &[recovered(&CONFIRMED_UPPER)]).reason.as_deref(),
        Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE)
    );
}

#[test]
fn a_range_without_a_passing_confirmation_of_every_point_is_not_recovered() {
    static ONE_POINT: [CoverageGridPoint; 1] = [CONFIRMED_UPPER[0]];
    static FAILED_CONFIRMATION: [CoverageGridPoint; 2] = {
        let mut points = CONFIRMED_UPPER;
        points[1].boundary = true;
        points[1].observed = 0.91;
        points
    };
    static SHIFTED_CONFIRMATION: [CoverageGridPoint; 2] = {
        let mut points = CONFIRMED_UPPER;
        points[1].n_min = 900;
        points[1].n_max = 900;
        points
    };
    let cases: [&'static [CoverageGridPoint]; 4] =
        [&[], &ONE_POINT, &FAILED_CONFIRMATION, &SHIFTED_CONFIRMATION];
    for confirmation in cases {
        let mut b = basis();
        b.scope.row_count = 800;
        let slot = calibration_slot_in(&b, &[recovered(confirmation)]);
        assert_eq!(slot.status, "measured_below_nominal", "{confirmation:?}");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
    }
}

#[test]
fn a_confirmed_range_never_covers_a_failing_grid_point() {
    // A range that reaches back over the failing point is not a passing
    // region, whatever the record states.
    let mut forged = recovered(&CONFIRMED_UPPER);
    forged.confirmed_passing_range = Some((250, 1000));
    let mut b = basis();
    b.scope.row_count = 800;
    assert_eq!(calibration_slot_in(&b, &[forged]).reason.as_deref(), Some(BOUNDARY_RECORD));
}

#[test]
fn a_confirmed_range_still_requires_attestation_and_every_other_axis() {
    let record = recovered(&CONFIRMED_UPPER);
    let mut b = basis();
    b.scope.row_count = 800;
    let stale = calibration_slot_with(&b, &[record], |_| false);
    assert_eq!(stale.reason.as_deref(), Some(RECORD_NOT_ATTESTING));
    let mut other_level = b.clone();
    other_level.key.level = 0.9;
    assert_eq!(
        calibration_slot_in(&other_level, &[record]).reason.as_deref(),
        Some(LEVEL_NOT_MEASURED)
    );
    let mut other_dependence = b.clone();
    other_dependence.key.dependence = "panel_cluster".into();
    assert_eq!(
        calibration_slot_in(&other_dependence, &[record]).reason.as_deref(),
        Some(CONSTRUCTION_NOT_MEASURED)
    );
    let mut other_identification = b;
    other_identification.key.identification = "partial".into();
    assert_eq!(
        calibration_slot_in(&other_identification, &[record]).reason.as_deref(),
        Some(IDENTIFICATION_NOT_MEASURED)
    );
}

#[test]
fn a_covering_boundary_record_is_reported_with_its_coverage() {
    let mut boundary = record("cov.boundary");
    boundary.boundary = true;
    boundary.observed = 0.885;
    let slot = calibration_slot_in(&basis(), &[record("cov.a"), boundary]);
    assert_eq!(slot.status, "measured_below_nominal");
    assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
    assert_eq!(slot.record_id.as_deref(), Some("cov.boundary.v2_validation_none"));
    assert_eq!(slot.observed, Some(0.885));
}

#[test]
fn no_interval_and_no_record_are_not_measured_with_codes() {
    let mut b = basis();
    b.key.interval_method = "none".into();
    assert_eq!(
        calibration_slot_in(&b, &[record("cov.a")]).reason.as_deref(),
        Some(NO_INTERVAL_REPORTED)
    );
    assert_eq!(
        calibration_slot_in(&basis(), &[]).reason.as_deref(),
        Some(CONSTRUCTION_NOT_MEASURED)
    );
}

#[test]
fn rederive_reproduces_primary_and_secondary_slots() {
    let mut secondary = basis();
    secondary.key.interval_method = "identified_set".into();
    let slot = calibration_slots(&[basis(), secondary]);
    assert_eq!(slot.secondary.len(), 1);
    assert_eq!(rederive_calibration(&slot), slot);
    let mut forged = slot.clone();
    forged.status = "calibrated".into();
    forged.secondary[0].status = "calibrated".into();
    assert_ne!(rederive_calibration(&forged), forged);
}

#[test]
fn calibrated_requires_a_fresh_attesting_record() {
    let fresh = record("cov.fresh");
    let mut stale = record("cov.stale");
    stale.calibration_sha = "stale00000000000000000000000000000000000";
    let attests = |record: &CoverageRecord| record.id == "cov.fresh.v2_validation_none";

    let fresh_slot = calibration_slot_with(&basis(), &[fresh], attests);
    assert_eq!(fresh_slot.status, "calibrated");
    assert_eq!(fresh_slot.record_id.as_deref(), Some("cov.fresh.v2_validation_none"));
    assert_eq!(fresh_slot.reason, None);

    let stale_slot = calibration_slot_with(&basis(), &[stale], attests);
    assert_eq!(stale_slot.status, "outside_measured_scope");
    assert_eq!(stale_slot.record_id.as_deref(), Some("cov.stale.v2_validation_none"));
    assert_eq!(stale_slot.reason.as_deref(), Some(RECORD_NOT_ATTESTING));

    // A measured-looking record the registry does not list as attesting is not
    // calibrated.
    let drifted = record("cov.drifted");
    assert!(!record_attests_current_code(&drifted));
    let drifted_slot = calibration_slot_with(&basis(), &[drifted], record_attests_current_code);
    assert_ne!(drifted_slot.status, "calibrated");
    assert_eq!(drifted_slot.reason.as_deref(), Some(RECORD_NOT_ATTESTING));
}

#[test]
fn attested_external_estimate_stays_uncalibrated_without_a_basis() {
    let slot = CalibrationSlotWire::not_measured("attested_not_reverifiable");
    assert_eq!(rederive_calibration(&slot), slot);
    assert_eq!(
        rederive_calibration(&CalibrationSlotWire::not_measured("forged")).reason.as_deref(),
        Some(BASIS_MISSING)
    );
}

// -- recovery of passing subranges on a four-point grid ------------------------------------

const fn grid_point(point: u8, n: u64, observed: f64, boundary: bool) -> CoverageGridPoint {
    CoverageGridPoint {
        point,
        n_min: n,
        n_max: n,
        observed,
        mcse: 0.011,
        replicates: 2000,
        boundary,
    }
}

/// Four sample sizes, the smallest failing.
const LOWER_FAILS: [CoverageGridPoint; 4] = [
    grid_point(0, 400, 0.902, true),
    grid_point(1, 800, 0.951, false),
    grid_point(2, 1600, 0.949, false),
    grid_point(3, 3200, 0.955, false),
];
/// The same upper three sizes measured again with fresh seeds, all passing.
const UPPER_CONFIRMED: [CoverageGridPoint; 3] = [
    grid_point(1, 800, 0.944, false),
    grid_point(2, 1600, 0.957, false),
    grid_point(3, 3200, 0.953, false),
];

fn boundary_over(
    grid: &'static [CoverageGridPoint],
    confirmation: &'static [CoverageGridPoint],
    range: Option<(u64, u64)>,
) -> CoverageRecord {
    let mut out = record("cov.subrange");
    out.n_min = grid[0].n_min;
    out.n_max = grid[grid.len() - 1].n_max;
    out.observed = grid.iter().map(|p| p.observed).fold(f64::INFINITY, f64::min);
    out.boundary = true;
    out.grid = grid;
    out.confirmation = confirmation;
    out.confirmed_passing_range = range;
    out
}

fn at(rows: u64) -> CalibrationBasisWire {
    let mut b = basis();
    b.scope.row_count = rows;
    b
}

#[test]
fn independently_confirmed_upper_subrange_recovered() {
    let subrange = boundary_over(&LOWER_FAILS, &UPPER_CONFIRMED, Some((800, 3200)));
    for n in [800, 1200, 1600, 2400, 3200] {
        let slot = calibration_slot_in(&at(n), &[subrange]);
        assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
        assert_eq!(slot.reason, None);
        assert_eq!((slot.scope_n, slot.scope_n_max), (Some(800), Some(3200)));
        // The lowest of the measurement's 0.951, 0.949, 0.955 and the confirmation's 0.944,
        // 0.957, 0.953; never the failed point's 0.902.
        assert_eq!(slot.observed, Some(0.944));
        assert!(matches!(
            resolve_passing_range(&subrange, n),
            RangeResolution::Recovered(region) if (region.n_min, region.n_max) == (800, 3200)
        ));
    }
    // The failed lower point and its evidence stay on the record, and govern its own region.
    assert!(subrange.boundary && subrange.grid[0].boundary);
    assert!((subrange.grid[0].observed - 0.902).abs() < 1e-12);
    assert!(
        subrange.confirmation.iter().all(|p| p.point != 0),
        "the failed point was not confirmed"
    );
    for n in [400, 500, 799] {
        let slot = calibration_slot_in(&at(n), &[subrange]);
        assert_eq!(slot.status, "measured_below_nominal", "n = {n}: {slot:?}");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        assert_eq!(slot.observed, Some(0.902), "the failing point's coverage is reported");
        assert_eq!(
            resolve_passing_range(&subrange, n),
            RangeResolution::FailedPoint { point: 0, n_min: 400, n_max: 400, observed: 0.902 }
        );
    }
    // Outside every measured size nothing is recovered.
    for n in [399, 3201] {
        let slot = calibration_slot_in(&at(n), &[subrange]);
        assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
        assert_eq!(resolve_passing_range(&subrange, n), RangeResolution::OutsideMeasuredRange);
    }
    assert_eq!(resolve_passing_range(&record("cov.plain"), 500), RangeResolution::NotABoundary);
}

#[test]
fn unconfirmed_subrange_not_recovered() {
    static PARTIAL: [CoverageGridPoint; 2] = [UPPER_CONFIRMED[0], UPPER_CONFIRMED[1]];
    static FAILED_CONFIRMATION: [CoverageGridPoint; 3] = {
        let mut points = UPPER_CONFIRMED;
        points[1].boundary = true;
        points[1].observed = 0.912;
        points
    };
    static OTHER_SIZES: [CoverageGridPoint; 3] = {
        let mut points = UPPER_CONFIRMED;
        points[0].n_min = 801;
        points[0].n_max = 801;
        points
    };
    // The upper three points passed the measurement, but nothing independent confirms them:
    // the record is still a boundary, and the reason is the missing confirmation.
    let unconfirmed: [CoverageRecord; 5] = [
        boundary_over(&LOWER_FAILS, &[], None),
        boundary_over(&LOWER_FAILS, &PARTIAL, None),
        boundary_over(&LOWER_FAILS, &FAILED_CONFIRMATION, None),
        boundary_over(&LOWER_FAILS, &OTHER_SIZES, None),
        // A range stated without evidence for every point in it is not honoured.
        boundary_over(&LOWER_FAILS, &PARTIAL, Some((800, 3200))),
    ];
    for (index, record) in unconfirmed.iter().enumerate() {
        for n in [800, 1600, 3200] {
            let slot = calibration_slot_in(&at(n), &[*record]);
            assert_eq!(slot.status, "measured_below_nominal", "record {index} n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
            assert_eq!(
                resolve_passing_range(record, n),
                RangeResolution::Unconfirmed { n_min: 800, n_max: 3200 },
                "record {index} n = {n}"
            );
        }
    }
}

#[test]
fn non_contiguous_passing_points_not_merged() {
    // 400 and 1600 passed, 800 between them failed. Even with both passing points
    // re-measured and a range stated across them, the failed interior point is not merged.
    static MIDDLE_FAILS: [CoverageGridPoint; 3] = [
        grid_point(0, 400, 0.951, false),
        grid_point(1, 800, 0.905, true),
        grid_point(2, 1600, 0.949, false),
    ];
    static ENDS_CONFIRMED: [CoverageGridPoint; 2] =
        [grid_point(0, 400, 0.953, false), grid_point(2, 1600, 0.955, false)];
    for range in [None, Some((400, 1600))] {
        let record = boundary_over(&MIDDLE_FAILS, &ENDS_CONFIRMED, range);
        for n in [400, 600, 800, 1000, 1600] {
            let slot = calibration_slot_in(&at(n), &[record]);
            assert_eq!(slot.status, "measured_below_nominal", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
            assert_eq!(slot.observed, Some(0.905), "the failed interior point is reported");
            assert_eq!(
                resolve_passing_range(&record, n),
                RangeResolution::NotContiguous { failed_point: 1, n_min: 800, n_max: 800 },
                "n = {n}"
            );
        }
    }
}

#[test]
#[allow(clippy::items_after_statements, reason = "each fixture grid is declared next to its case")]
fn a_passing_run_must_reach_an_end_of_the_grid_and_hold_two_points() {
    static ONE_PASSES: [CoverageGridPoint; 3] = [
        grid_point(0, 400, 0.902, true),
        grid_point(1, 800, 0.951, false),
        grid_point(2, 1600, 0.903, true),
    ];
    static MIDDLE_RUN: [CoverageGridPoint; 4] = [
        grid_point(0, 400, 0.902, true),
        grid_point(1, 800, 0.951, false),
        grid_point(2, 1600, 0.949, false),
        grid_point(3, 3200, 0.903, true),
    ];
    let lone = boundary_over(&ONE_PASSES, &[], None);
    assert_eq!(
        resolve_passing_range(&lone, 800),
        RangeResolution::LonePassingPoint { n_min: 800, n_max: 800 }
    );
    let middle = boundary_over(&MIDDLE_RUN, &[], None);
    assert_eq!(
        resolve_passing_range(&middle, 1200),
        RangeResolution::InteriorRun { n_min: 800, n_max: 1600 }
    );
    // The lower run of the mirror image is recovered like the upper one.
    static UPPER_FAILS: [CoverageGridPoint; 4] = [
        grid_point(0, 400, 0.951, false),
        grid_point(1, 800, 0.949, false),
        grid_point(2, 1600, 0.955, false),
        grid_point(3, 3200, 0.902, true),
    ];
    static LOWER_CONFIRMED: [CoverageGridPoint; 3] = [
        grid_point(0, 400, 0.952, false),
        grid_point(1, 800, 0.946, false),
        grid_point(2, 1600, 0.956, false),
    ];
    let lower = boundary_over(&UPPER_FAILS, &LOWER_CONFIRMED, Some((400, 1600)));
    let slot = calibration_slot_in(&at(1000), &[lower]);
    assert_eq!(slot.status, "calibrated", "{slot:?}");
    assert_eq!(
        (slot.scope_n, slot.scope_n_max, slot.observed),
        (Some(400), Some(1600), Some(0.946))
    );
    assert_eq!(calibration_slot_in(&at(3200), &[lower]).reason.as_deref(), Some(BOUNDARY_RECORD));
}

/// A named change to one field of a calibration key.
type KeyMutation = fn(&mut CalibrationKeyWire);

#[test]
fn a_confirmed_range_binds_only_to_the_exact_construction_estimand_level_dependence_identification_and_attestation()
 {
    let record = boundary_over(&LOWER_FAILS, &UPPER_CONFIRMED, Some((800, 3200)));
    let positive = calibration_slot_in(&at(1600), &[record]);
    assert_eq!(positive.status, "calibrated");
    // Construction (estimator, interval method), estimand (functional) and dependence differ:
    // no record measured this construction, so nothing is combined.
    let variants: [(&str, KeyMutation); 5] = [
        ("estimator", |k| k.estimator = "foreign".into()),
        ("interval method", |k| k.interval_method = "foreign".into()),
        ("estimand", |k| k.functional = "foreign.contrast".into()),
        ("dependence", |k| k.dependence = "panel_cluster".into()),
        ("graph class", |k| k.graph_class = "foreign".into()),
    ];
    for (what, edit) in variants {
        let mut b = at(1600);
        edit(&mut b.key);
        let slot = calibration_slot_in(&b, &[record]);
        assert_eq!(slot.status, "not_measured", "{what}: {slot:?}");
        assert_eq!(slot.reason.as_deref(), Some(CONSTRUCTION_NOT_MEASURED), "{what}");
    }
    // The nominal level and the identification are exact too.
    let mut level = at(1600);
    level.key.level = 0.9;
    assert_eq!(calibration_slot_in(&level, &[record]).reason.as_deref(), Some(LEVEL_NOT_MEASURED));
    let mut identification = at(1600);
    identification.key.identification = "partial".into();
    assert_eq!(
        calibration_slot_in(&identification, &[record]).reason.as_deref(),
        Some(IDENTIFICATION_NOT_MEASURED)
    );
    // And the record must still attest the current code.
    let stale = calibration_slot_with(&at(1600), &[record], |_| false);
    assert_eq!(stale.status, "outside_measured_scope");
    assert_eq!(stale.reason.as_deref(), Some(RECORD_NOT_ATTESTING));
}
