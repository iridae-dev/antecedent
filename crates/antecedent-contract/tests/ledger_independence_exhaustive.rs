//! Declared independence of two fitted mechanisms, decided from the evidence ledger's rows,
//! against a set-intersection oracle over every ordered pair of sources.
//!
//! Five ledger sources: four with declared observation rows (`{1,2,3,4}`, `{4,5}`, `{6,7}`,
//! `{8}`) and one with no declared rows. For each ordered pair `(a, b)` of sources (including a
//! source with itself) the expected decision is computed here by plain set algebra: undeclared
//! rows make the overlap unknown; a source paired with itself shares its evidence by identity;
//! otherwise a nonempty intersection is shared evidence declared independent; otherwise the
//! declaration is accepted. The two mechanisms are otherwise
//! identical across all 25 declarations.

use std::collections::BTreeSet;

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_contract::{
    DependenceDeclaration, MechanismCompositionDeclaration, MechanismDeclaration,
    MechanismOriginKind, QuantityRole, ScientificQuantity,
    check_declared_independence_against_ledger,
};

fn q(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn observation(row: u8) -> ObservationId {
    ObservationId::from_key("trial", &[row])
}

fn mechanism(evidence: SourceId, output: &str) -> MechanismDeclaration {
    MechanismDeclaration {
        output: q(output, "adults"),
        parents: Vec::new(),
        conditioning_variable_ids: Vec::new(),
        origin: MechanismOriginKind::Fitted,
        artifact_id: format!("fit-{output}"),
        evidence_id: evidence.to_hex(),
        latent_causes: Vec::new(),
    }
}

fn declaration(first: SourceId, second: SourceId) -> MechanismCompositionDeclaration {
    MechanismCompositionDeclaration {
        target_population_id: "adults".into(),
        target_regime_id: "observational".into(),
        mechanisms: vec![mechanism(first, "schema:y"), mechanism(second, "schema:x")],
        edges: Vec::new(),
        admissions: Vec::new(),
        dependence: DependenceDeclaration::IndependentByDesign {
            justification: "separate trials".into(),
        },
        admit_declared_mechanisms: false,
    }
}

#[test]
fn declared_independence_matches_set_intersection_for_every_ordered_pair_of_sources() {
    let row_sets: [Option<Vec<u8>>; 5] =
        [Some(vec![1, 2, 3, 4]), Some(vec![4, 5]), Some(vec![6, 7]), Some(vec![8]), None];
    let mut ledger = EvidenceLedger::new();
    let ids: Vec<SourceId> = row_sets
        .iter()
        .enumerate()
        .map(|(i, rows)| {
            let declaration = SourceDeclaration::new(format!("fit-{i}"), SourceKind::ObservationSet);
            let declaration = match rows {
                Some(rows) => declaration.with_observations(rows.iter().copied().map(observation)),
                None => declaration,
            };
            ledger.add_source(declaration).expect("source registers")
        })
        .collect();
    let (mut accepted, mut shared, mut unknown) = (0, 0, 0);
    for (a, rows_a) in row_sets.iter().enumerate() {
        for (b, rows_b) in row_sets.iter().enumerate() {
            let result = check_declared_independence_against_ledger(
                &declaration(ids[a], ids[b]),
                &ledger,
            );
            match (rows_a, rows_b) {
                // A source paired with itself shares all of its evidence by identity,
                // whether or not it declares rows.
                _ if a == b => {
                    let refusals = result.expect_err("a source shares its own evidence");
                    let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
                    assert_eq!(
                        details,
                        vec!["mechanism_composition.posterior_dependence_shared_evidence_declared_independent"],
                        "pair ({a}, {b})"
                    );
                    shared += 1;
                }
                (Some(x), Some(y)) => {
                    let x: BTreeSet<u8> = x.iter().copied().collect();
                    let y: BTreeSet<u8> = y.iter().copied().collect();
                    if x.intersection(&y).next().is_some() {
                        let refusals = result.expect_err("a shared row is declared independent");
                        let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
                        assert_eq!(
                            details,
                            vec!["mechanism_composition.posterior_dependence_shared_evidence_declared_independent"],
                            "pair ({a}, {b})"
                        );
                        shared += 1;
                    } else {
                        result.expect("disjoint declared rows are accepted");
                        accepted += 1;
                    }
                }
                _ => {
                    let refusals = result.expect_err("an undeclared source's overlap is unknown");
                    let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
                    assert_eq!(
                        details,
                        vec!["mechanism_composition.posterior_dependence_evidence_overlap_unknown"],
                        "pair ({a}, {b})"
                    );
                    unknown += 1;
                }
            }
        }
    }
    // By hand: the 5 diagonal pairs share their own evidence by identity, and so do (0,1)
    // and (1,0) (row 4): 7 shared. Among the other 12 distinct pairs of declared sources,
    // 10 are disjoint; the 8 ordered pairs of a declared with the undeclared source are
    // unknown. 10 + 7 + 8 = 25.
    assert_eq!((accepted, shared, unknown), (10, 7, 8));
}
