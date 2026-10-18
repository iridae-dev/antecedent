//! A numerically changed matching construction carries its own identity and its own evidence.
//!
//! Matching beyond the former donor limit changed no number: the sorted one-dimensional
//! search and the k-d tree return the linear scan's matches bit for bit, so those searches
//! keep the identities of the estimators they serve. Averaging several nearest donors does
//! change the point estimate, its bias and its variance, so it is a construction of its own:
//! its own estimator type and identity (with the neighbour count), its own provenance record
//! naming where it departs from one-nearest-neighbour matching, and its own calibration
//! design registered with the gate. This test reads those registrations.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_estimate::{KNN_DISTANCE_MATCHING_CONSTRUCTION, KnnDistanceMatching};

fn repository_file(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

#[test]
fn matching_changed_numerical_construction_has_own_identity() {
    // Identity: its own name, and the neighbour count is part of it.
    assert_eq!(KNN_DISTANCE_MATCHING_CONSTRUCTION, "estimate.knn_distance_matching");
    let single = KnnDistanceMatching::new(1).unwrap().construction_id();
    let triple = KnnDistanceMatching::new(3).unwrap().construction_id();
    let quintuple = KnnDistanceMatching::new(5).unwrap().construction_id();
    assert_ne!(single, triple);
    assert_ne!(triple, quintuple);
    for id in [&single, &triple, &quintuple] {
        assert!(id.starts_with(KNN_DISTANCE_MATCHING_CONSTRUCTION), "{id}");
        assert!(
            !["estimate.distance_matching", "estimate.propensity_matching"].contains(&id.as_str())
        );
    }

    // Provenance: a record of its own, distinct from the single-neighbour records, that
    // says it is a separate numerical construction and names its evidence.
    let own = repository_file("provenance/estimate.knn_distance_matching.toml");
    assert!(own.contains("feature_id = \"estimate.knn_distance_matching\""));
    assert!(own.contains("separate numerical construction"));
    assert!(own.contains("no coverage measured for estimate.distance_matching applies"));
    assert!(own.contains("crates/antecedent-estimate/tests/knn_matching_calibration.rs"));
    assert!(own.contains("crates/antecedent-estimate/src/propensity/knn.rs"));
    for neighbour in ["estimate.distance_matching", "estimate.propensity_matching"] {
        let record = repository_file(&format!("provenance/{neighbour}.toml"));
        assert!(record.contains(&format!("feature_id = \"{neighbour}\"")));
        assert!(!record.contains("estimate.knn_distance_matching"));
    }

    // Evidence: the calibration design exists, is ignored from the default run and is
    // registered with the gate under its own name.
    let design = repository_file("crates/antecedent-estimate/tests/knn_matching_calibration.rs");
    assert!(design.contains("fn knn_matching_interval_ci_coverage()"));
    assert!(design.contains("#[ignore = \"calibration: run via scripts/gate_calibration.sh\"]"));
    let gate = repository_file("scripts/gate_calibration.sh");
    assert!(
        gate.lines().any(|line| line.contains("--test knn_matching_calibration")
            || line.contains("run_knn_matching_calibration knn_matching_interval_ci_coverage")),
        "scripts/gate_calibration.sh does not register the knn matching coverage design"
    );
}
