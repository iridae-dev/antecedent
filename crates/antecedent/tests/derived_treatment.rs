//! 2.2 E5 derived treatments: the declaration classifies each source column, exclusions need a
//! declared causal rule, leakage, duplicated constituents, rank failure and illegal values are
//! refused with the columns named (nothing is dropped), and a declared joint cell runs through
//! the factorized estimator with an unsupported cell refused alone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test fixtures build small synthetic tables element by element"
)]

use antecedent::{
    CausalError, ConstituentRole, DeclaredExclusion, DerivedTreatmentDeclaration, ExclusionRule,
    InterventionMeaning, SourceColumn, TemporalPosition, Transformation, check_derived_treatment,
    estimate_derived_joint_cells,
};
use antecedent_core::{CausalRng, ExecutionContext, StreamDomain};
use antecedent_data::TabularData;
use antecedent_estimate::{
    CellStatus, FactorizedJointConfig, JointContrast, RidgeTuning, orderings_for,
};
use antecedent_kernels::standard_normal;

fn stream(seed: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xE5)
}

fn named(columns: &[(&str, &Vec<f64>)]) -> TabularData {
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).unwrap()
}

fn coin(n: usize, rng: &mut CausalRng) -> Vec<f64> {
    (0..n).map(|_| f64::from(u8::from(rng.next_f64() < 0.5))).collect()
}

fn normals(n: usize, rng: &mut CausalRng) -> Vec<f64> {
    (0..n).map(|_| standard_normal(rng)).collect()
}

fn source(name: &str, role: ConstituentRole, when: TemporalPosition) -> SourceColumn {
    SourceColumn { name: name.to_string(), role, when }
}

fn component(name: &str) -> SourceColumn {
    source(name, ConstituentRole::TreatmentConstruction, TemporalPosition::AtTreatment)
}

/// A joint cell of `t0`, `t1` with `age` an admissible pre-treatment source.
fn joint() -> DerivedTreatmentDeclaration {
    DerivedTreatmentDeclaration {
        name: "t0_and_t1".to_string(),
        sources: vec![
            component("t0"),
            component("t1"),
            source(
                "age",
                ConstituentRole::AdmissiblePreTreatmentCovariate,
                TemporalPosition::PreTreatment,
            ),
            source("post", ConstituentRole::ForbiddenDescendant, TemporalPosition::PostTreatment),
        ],
        transformation: Transformation::JointCell,
        legal_values: vec![0.0, 1.0, 2.0, 3.0],
        intervention: InterventionMeaning::JointComponents,
        exclusions: Vec::new(),
    }
}

fn exclusion(column: &str, rule: ExclusionRule, why: &str) -> DeclaredExclusion {
    DeclaredExclusion { column: column.to_string(), rule, justification: why.to_string() }
}

struct Table {
    data: TabularData,
    t0: Vec<f64>,
    t1: Vec<f64>,
}

/// `t0`, `t1` independent coins, `age` and `z` covariates, `post` a descendant of `t0`.
fn table(n: usize, seed: u64) -> Table {
    let mut rng = stream(seed);
    let t0 = coin(n, &mut rng);
    let t1 = coin(n, &mut rng);
    let age = normals(n, &mut rng);
    let z = normals(n, &mut rng);
    let noise = normals(n, &mut rng);
    let post: Vec<f64> = (0..n).map(|i| t0[i] + 0.3 * noise[i]).collect();
    let y: Vec<f64> = (0..n).map(|i| t0[i] + t1[i] + age[i] + 0.3 * noise[i]).collect();
    let data =
        named(&[("t0", &t0), ("t1", &t1), ("age", &age), ("z", &z), ("post", &post), ("y", &y)]);
    Table { data, t0, t1 }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn names(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

#[track_caller]
fn refusal(error: &CausalError, code: &str, detail: &str) {
    assert_eq!(error.reason_code(), Some(code), "{error}");
    assert!(error.to_string().contains(detail), "{error}");
}

fn implicated(error: &CausalError) -> Vec<String> {
    error.refusal_fields().expect("structured refusal fields").implicated_columns.clone()
}

/// The accepted plan names the components, keeps an admissible pre-treatment source, and
/// observes exactly the derived levels an independent count finds.
#[test]
fn a_valid_declaration_reports_its_construction() {
    let t = table(500, 1);
    let plan =
        check_derived_treatment(&t.data, &joint(), "y", &names(&["age", "z"]), &ctx()).unwrap();
    assert_eq!(plan.treatment_columns, names(&["t0", "t1"]));
    assert_eq!(plan.adjustment, names(&["age", "z"]));
    assert_eq!(plan.retained_covariates, names(&["age"]));
    assert!(plan.exclusions.is_empty());
    assert_eq!(plan.rows_complete, 500);
    let mut by_hand: Vec<f64> = t.t0.iter().zip(&t.t1).map(|(a, b)| a + 2.0 * b).collect();
    by_hand.sort_by(f64::total_cmp);
    by_hand.dedup();
    assert_eq!(plan.observed_levels, by_hand);
    assert_eq!(plan.observed_levels, vec![0.0, 1.0, 2.0, 3.0]);
}

/// A constituent column or a descendant in the adjustment set is refused with the column named;
/// only a declared exclusion under the matching causal rule removes it, and it is recorded.
#[test]
fn constituent_and_descendant_leakage_needs_a_declared_exclusion() {
    let t = table(500, 2);
    let leaking = names(&["age", "t1"]);
    let error = check_derived_treatment(&t.data, &joint(), "y", &leaking, &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_constituent_in_adjustment");
    assert_eq!(implicated(&error), names(&["t1"]));

    let mut declared = joint();
    declared.exclusions.push(exclusion(
        "t1",
        ExclusionRule::ConstituentOfTreatment,
        "t1 is a component of the treatment",
    ));
    let plan = check_derived_treatment(&t.data, &declared, "y", &leaking, &ctx()).unwrap();
    assert_eq!(plan.adjustment, names(&["age"]));
    assert_eq!(plan.exclusions, declared.exclusions);

    let descendant = names(&["age", "post"]);
    let error = check_derived_treatment(&t.data, &joint(), "y", &descendant, &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_descendant_in_adjustment");
    assert_eq!(implicated(&error), names(&["post"]));
    let mut declared = joint();
    declared.exclusions.push(exclusion(
        "post",
        ExclusionRule::PostTreatmentDescendant,
        "post is measured after the treatment",
    ));
    let plan = check_derived_treatment(&t.data, &declared, "y", &descendant, &ctx()).unwrap();
    assert_eq!(plan.adjustment, names(&["age"]));

    let error =
        check_derived_treatment(&t.data, &joint(), "y", &names(&["age", "y"]), &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_outcome_in_adjustment");
}

/// An exclusion needs a rule that matches the column's role and a justification; none excludes an
/// admissible covariate, and a stale exclusion is refused rather than ignored.
#[test]
fn exclusions_are_refused_unless_they_carry_a_matching_causal_rule() {
    let t = table(300, 3);
    let adjustment = names(&["age", "t1", "post"]);
    let attempts = [
        exclusion("t1", ExclusionRule::PostTreatmentDescendant, "wrong rule for a component"),
        exclusion("post", ExclusionRule::ConstituentOfTreatment, "wrong rule for a descendant"),
        exclusion("age", ExclusionRule::ConstituentOfTreatment, "an admissible covariate"),
        exclusion("age", ExclusionRule::PostTreatmentDescendant, "an admissible covariate"),
        exclusion("nowhere", ExclusionRule::ConstituentOfTreatment, "not a source"),
    ];
    for attempt in attempts {
        let mut declared = joint();
        declared.exclusions.push(attempt);
        let error =
            check_derived_treatment(&t.data, &declared, "y", &adjustment, &ctx()).unwrap_err();
        refusal(&error, "derived_treatment_invalid", "joint_cells.derived_exclusion_rule_mismatch");
    }
    let mut blank = joint();
    blank.exclusions.push(exclusion("t1", ExclusionRule::ConstituentOfTreatment, "  "));
    let error = check_derived_treatment(&t.data, &blank, "y", &adjustment, &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_declaration_invalid");

    let mut twice = joint();
    for _ in 0..2 {
        twice.exclusions.push(exclusion("t1", ExclusionRule::ConstituentOfTreatment, "component"));
    }
    let error = check_derived_treatment(&t.data, &twice, "y", &adjustment, &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_exclusion_rule_mismatch");

    let mut stale = joint();
    stale.exclusions.push(exclusion("post", ExclusionRule::PostTreatmentDescendant, "descendant"));
    let error =
        check_derived_treatment(&t.data, &stale, "y", &names(&["age"]), &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_exclusion_not_in_adjustment");
}

/// Declarations that contradict themselves are refused before any data is read.
#[test]
fn contradictory_declarations_are_refused() {
    let t = table(100, 4);
    let attempt = |edit: &dyn Fn(&mut DerivedTreatmentDeclaration)| {
        let mut declared = joint();
        edit(&mut declared);
        check_derived_treatment(&t.data, &declared, "y", &names(&["age"]), &ctx()).unwrap_err()
    };
    for error in [
        attempt(&|d| d.name = " ".to_string()),
        attempt(&|d| d.sources[2].when = TemporalPosition::PostTreatment),
        attempt(&|d| d.sources[3].when = TemporalPosition::PreTreatment),
        attempt(&|d| d.sources[0].when = TemporalPosition::PostTreatment),
        attempt(&|d| d.sources.push(component("t0"))),
        attempt(&|d| d.sources.retain(|s| s.role != ConstituentRole::TreatmentConstruction)),
        attempt(&|d| d.legal_values = vec![0.0]),
        attempt(&|d| d.legal_values = vec![0.0, 0.0, 1.0]),
        attempt(&|d| d.legal_values = vec![0.0, 1.5]),
        attempt(&|d| d.legal_values = vec![0.0, 4.0]),
        attempt(&|d| d.legal_values = vec![0.0, f64::NAN]),
    ] {
        refusal(&error, "derived_treatment_invalid", "joint_cells.derived_declaration_invalid");
    }
    let error = attempt(&|d| d.transformation = Transformation::Product);
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_intervention_not_defined");
    let error = attempt(&|d| {
        d.sources.push(component("z"));
        d.sources.push(component("age2"));
        d.sources.push(component("age3"));
    });
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_declaration_invalid");
}

/// Exact duplicate treatment components, an undeclared exact copy of a component in the
/// adjustment set, and a covariate that tracks a component are each refused with the columns
/// named; nothing is dropped.
#[test]
fn duplicated_and_tracking_constituents_are_refused() {
    let n = 600;
    let mut rng = stream(6);
    let t0 = coin(n, &mut rng);
    let age = normals(n, &mut rng);
    let noise = normals(n, &mut rng);
    let y: Vec<f64> = (0..n).map(|i| t0[i] + age[i] + 0.3 * noise[i]).collect();

    let data = named(&[("t0", &t0), ("t1", &t0), ("age", &age), ("y", &y)]);
    let error =
        check_derived_treatment(&data, &joint(), "y", &names(&["age"]), &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_duplicate_constituents");
    assert_eq!(implicated(&error), names(&["t0", "t1"]));

    let t1 = coin(n, &mut rng);
    let data = named(&[("t0", &t0), ("t1", &t1), ("t0_copy", &t0), ("age", &age), ("y", &y)]);
    let error = check_derived_treatment(&data, &joint(), "y", &names(&["age", "t0_copy"]), &ctx())
        .unwrap_err();
    refusal(
        &error,
        "derived_treatment_invalid",
        "joint_cells.derived_constituent_copy_in_adjustment",
    );
    assert!(implicated(&error).contains(&"t0_copy".to_string()));

    let tracker: Vec<f64> = (0..n).map(|i| t0[i] + 0.005 * noise[i]).collect();
    let data = named(&[("t0", &t0), ("t1", &t1), ("tracker", &tracker), ("age", &age), ("y", &y)]);
    let error = check_derived_treatment(&data, &joint(), "y", &names(&["age", "tracker"]), &ctx())
        .unwrap_err();
    refusal(
        &error,
        "derived_treatment_invalid",
        "joint_cells.derived_treatment_tracked_by_covariates",
    );
    assert!(implicated(&error).contains(&"t0".to_string()));
}

/// A rank-deficient adjustment design is detected by the preflight rank machinery and refused
/// with the dependent column named and the rank reported; the check never drops it.
#[test]
fn a_rank_deficient_adjustment_design_is_refused_not_repaired() {
    let n = 400;
    let mut rng = stream(7);
    let t0 = coin(n, &mut rng);
    let t1 = coin(n, &mut rng);
    let z1 = normals(n, &mut rng);
    let z2 = normals(n, &mut rng);
    let z3: Vec<f64> = (0..n).map(|i| z1[i] + z2[i]).collect();
    let z1_copy = z1.clone();
    let y: Vec<f64> = (0..n).map(|i| t0[i] + z1[i]).collect();
    let data = named(&[
        ("t0", &t0),
        ("t1", &t1),
        ("z1", &z1),
        ("z2", &z2),
        ("z3", &z3),
        ("z1_copy", &z1_copy),
        ("y", &y),
    ]);

    let error = check_derived_treatment(&data, &joint(), "y", &names(&["z1", "z2", "z3"]), &ctx())
        .unwrap_err();
    refusal(&error, "design_rank_deficient", "joint_cells.derived_rank_deficient");
    assert_eq!(implicated(&error), names(&["z3"]));
    let fields = error.refusal_fields().unwrap();
    assert_eq!((fields.numerical_rank, fields.design_columns), (Some(3), Some(4)));
    assert_eq!(fields.stage.as_deref(), Some("derived_treatment"));

    let error = check_derived_treatment(&data, &joint(), "y", &names(&["z1", "z1_copy"]), &ctx())
        .unwrap_err();
    refusal(&error, "design_rank_deficient", "joint_cells.derived_rank_deficient");
    assert_eq!(implicated(&error), names(&["z1_copy"]));

    // The same design without the dependent column is accepted unchanged.
    let plan =
        check_derived_treatment(&data, &joint(), "y", &names(&["z1", "z2"]), &ctx()).unwrap();
    assert_eq!(plan.adjustment, names(&["z1", "z2"]));
}

/// A component value or derived value outside the declared legal set is refused, an unknown
/// column is refused, and the check observes cancellation.
#[test]
fn illegal_values_unknown_columns_and_cancellation_are_refused() {
    let n = 200;
    let mut rng = stream(8);
    let t0 = coin(n, &mut rng);
    let mut t1 = coin(n, &mut rng);
    let age = normals(n, &mut rng);
    let y = normals(n, &mut rng);
    let data = named(&[("t0", &t0), ("t1", &t1), ("age", &age), ("y", &y)]);

    let mut narrow = joint();
    narrow.legal_values = vec![0.0, 1.0, 2.0];
    let error = check_derived_treatment(&data, &narrow, "y", &names(&["age"]), &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_illegal_value");
    assert!(error.to_string().contains("derived value 3"), "{error}");

    t1[3] = 2.0;
    let data = named(&[("t0", &t0), ("t1", &t1), ("age", &age), ("y", &y)]);
    let error =
        check_derived_treatment(&data, &joint(), "y", &names(&["age"]), &ctx()).unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_illegal_value");
    assert!(error.to_string().contains("component value 2"), "{error}");

    let error =
        check_derived_treatment(&data, &joint(), "y", &names(&["missing"]), &ctx()).unwrap_err();
    refusal(&error, "invalid_argument", "joint_cells.derived_unknown_column");

    let cancelled = ExecutionContext::for_tests(5);
    cancelled.cancellation.cancel();
    let t = table(100, 9);
    let result = check_derived_treatment(&t.data, &joint(), "y", &names(&["age"]), &cancelled);
    assert!(matches!(result, Err(CausalError::Cancelled { .. })));
}

/// A sum of binary components is a many-to-one derived level: it is declarable with a
/// derived-level intervention, observes its levels, and is refused by the joint-cell estimator.
#[test]
fn a_sum_is_a_derived_level_and_is_not_a_joint_cell() {
    let t = table(300, 10);
    let mut sum = joint();
    sum.transformation = Transformation::Sum;
    sum.intervention = InterventionMeaning::DerivedLevel;
    sum.legal_values = vec![0.0, 1.0, 2.0];
    let plan = check_derived_treatment(&t.data, &sum, "y", &names(&["age"]), &ctx()).unwrap();
    assert_eq!(plan.observed_levels, vec![0.0, 1.0, 2.0]);
    let config = FactorizedJointConfig::new(RidgeTuning::default());
    let error = estimate_derived_joint_cells(
        &t.data,
        &sum,
        "y",
        &names(&["age"]),
        &orderings_for(2, &[0, 1], false).unwrap(),
        &config,
        &ctx(),
    )
    .unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_not_a_joint_cell");
}

fn law_sample(n: usize, seed: u64, law: &[Vec<f64>]) -> TabularData {
    let mut rng = stream(seed);
    let (mut t0, mut t1, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = rng.next_f64() < 0.5;
        let probs = &law[usize::from(zi)];
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        let mut cell = probs.iter().rposition(|&p| p > 0.0).unwrap();
        for (c, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                cell = c;
                break;
            }
        }
        let (a, b) = (cell & 1, cell >> 1);
        t0.push(a as f64);
        t1.push(b as f64);
        z.push(f64::from(u8::from(zi)));
        y.push(
            0.4 * cell as f64
                + f64::from(u8::from(cell == 3))
                + 0.5 * f64::from(u8::from(zi))
                + 0.3 * standard_normal(&mut rng),
        );
    }
    named(&[("t0", &t0), ("t1", &t1), ("z", &z), ("y", &y)])
}

fn simple_declaration() -> DerivedTreatmentDeclaration {
    DerivedTreatmentDeclaration { sources: vec![component("t0"), component("t1")], ..joint() }
}

/// End to end on a known law: the declaration is checked, then the four cells are estimated
/// against `E[Y^{do(cell)}] = 0.4 cell + 1{cell = 3} + 0.25` and the interaction is 1. A
/// constituent left in the adjustment set is refused before any fit.
#[test]
fn a_declared_joint_cell_is_estimated_on_a_known_law() {
    let law = vec![vec![0.4, 0.2, 0.2, 0.2], vec![0.1, 0.3, 0.2, 0.4]];
    let data = law_sample(8_000, 21, &law);
    let mut config = FactorizedJointConfig::new(RidgeTuning::default());
    config.seed = 4;
    let orderings = orderings_for(2, &[0, 1], true).unwrap();
    let (plan, fit) = estimate_derived_joint_cells(
        &data,
        &simple_declaration(),
        "y",
        &names(&["z"]),
        &orderings,
        &config,
        &ctx(),
    )
    .unwrap();
    assert_eq!(plan.observed_levels, vec![0.0, 1.0, 2.0, 3.0]);
    for (cell, report) in fit.cells.iter().enumerate() {
        let CellStatus::Supported(estimate) = &report.status else {
            panic!("cell {cell} unsupported")
        };
        let truth = 0.4 * cell as f64 + f64::from(u8::from(cell == 3)) + 0.25;
        assert!((estimate.estimate - truth).abs() < 0.06, "cell {cell}");
    }
    let interaction = fit.contrast_point(JointContrast::Interaction).unwrap();
    assert!((interaction - 1.0).abs() < 0.1, "{interaction}");

    let error = estimate_derived_joint_cells(
        &data,
        &simple_declaration(),
        "y",
        &names(&["z", "t1"]),
        &orderings,
        &config,
        &ctx(),
    )
    .unwrap_err();
    refusal(&error, "derived_treatment_invalid", "joint_cells.derived_constituent_in_adjustment");
}

/// A family with one unsupported cell keeps the rest: with joint cell (1, 1) never observed the
/// declared level 3 stays legal, the cell is refused alone, and the other three are estimated.
#[test]
fn a_family_with_one_unsupported_cell_keeps_the_rest() {
    let law = vec![vec![0.4, 0.3, 0.3, 0.0], vec![0.2, 0.4, 0.4, 0.0]];
    let data = law_sample(6_000, 22, &law);
    let config = FactorizedJointConfig::new(RidgeTuning::default());
    let (plan, fit) = estimate_derived_joint_cells(
        &data,
        &simple_declaration(),
        "y",
        &names(&["z"]),
        &orderings_for(2, &[1, 0], true).unwrap(),
        &config,
        &ctx(),
    )
    .unwrap();
    assert_eq!(plan.observed_levels, vec![0.0, 1.0, 2.0]);
    let supported: Vec<bool> =
        fit.cells.iter().map(|c| matches!(c.status, CellStatus::Supported(_))).collect();
    assert_eq!(supported, vec![true, true, true, false]);
    let CellStatus::Unsupported(refused) = &fit.cells[3].status else { unreachable!() };
    assert_eq!(refused.code, "arm_not_populated");
    assert_eq!(fit.scores.n_columns(), 3);
}
