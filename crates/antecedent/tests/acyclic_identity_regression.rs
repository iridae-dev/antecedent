//! Acyclic identity regression: golden target, program, lowered-program and claim
//! digests of representative acyclic cells.
//!
//! The feedback vocabulary (a cyclic graph family; equilibrium, continuous-time
//! and stationary frames; structural actions) is append-only, so every target
//! stated in the acyclic vocabulary keeps its canonical bytes and its identity.
//! These goldens pin that: the static linear Gaussian average effect (scientific
//! target, layered contract identities, lowered program and claim) and scientific
//! targets of the static, temporal and mixed-graph vocabularies.
//!
//! The golden lives at `tests/fixtures/acyclic_identity_goldens.txt`, one
//! `name=hex` line per digest. A missing golden fails with the command that
//! writes it. A changed digest is a compatibility decision. The claim digest binds
//! the execution identity, which records the library version.
//!
//! Generate or refresh with
//! `cargo test -p antecedent --test acyclic_identity_regression -- --ignored write_`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;

use antecedent::{Dag, RefuteSuite, Study};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, Conditioning, ExecutionContext,
    GraphFamily, HistoryWindow, InferenceSelection, InitialLaw, Intervention, InterventionContrast,
    InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits,
    ScientificTarget, ScientificWorldId, TargetForm, TargetPopulation, TemporalPolicy, TimeFrame,
    Value, VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_data::TabularData;

const GOLDEN: &str = "acyclic_identity_goldens.txt";
const GENERATOR: &str =
    "cargo test -p antecedent --test acyclic_identity_regression -- --ignored write_";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn binding(outcome: u32) -> WorldBinding {
    WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("subject").unwrap(),
        OutcomeQuantity::mean(VariableId::from_raw(outcome), QuantityUnits::dimensionless())
            .unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap()
}

fn set(variable: u32, level: f64) -> Intervention {
    Intervention::set(VariableId::from_raw(variable), Value::f64(level))
}

fn contrast(action: impl Fn(f64) -> WorldAction, time: &TimeFrame, graph: GraphFamily) -> String {
    let world = |id: u32, level: f64, reference: bool| {
        InterventionWorld::new(
            ScientificWorldId::from_raw(id),
            action(level),
            reference,
            binding(1),
            time.clone(),
        )
        .unwrap()
    };
    let worlds = WorldCollection::new([world(0, 0.0, true), world(1, 1.0, false)]).unwrap();
    let id = ScientificWorldId::from_raw;
    let contrast = InterventionContrast::difference(worlds, id(1), id(0), id(0)).unwrap();
    ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(graph, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    )
    .identity()
    .to_hex()
}

/// Scientific targets of the acyclic vocabulary.
fn vocabulary_targets() -> Vec<(String, String)> {
    let at_decision = |level: f64| WorldAction::at_decision([set(0, level)]).unwrap();
    let temporal = TimeFrame::temporal(
        2,
        HistoryWindow::planner_default(),
        InitialLaw::contemporaneous_observed(),
    )
    .unwrap();
    let pulse =
        |level: f64| WorldAction::schedule(set(0, level), TemporalPolicy::pulse(0)).unwrap();
    let sustained =
        |level: f64| WorldAction::schedule(set(0, level), TemporalPolicy::sustained(0, 1)).unwrap();
    vec![
        (
            "target.static_contrast.dag".into(),
            contrast(at_decision, &TimeFrame::static_decision(), GraphFamily::Dag),
        ),
        (
            "target.static_contrast.admg".into(),
            contrast(at_decision, &TimeFrame::static_decision(), GraphFamily::Admg),
        ),
        (
            "target.static_contrast.unspecified".into(),
            contrast(at_decision, &TimeFrame::static_decision(), GraphFamily::Unspecified),
        ),
        (
            "target.temporal_pulse.temporal_dag".into(),
            contrast(pulse, &temporal, GraphFamily::TemporalDag),
        ),
        (
            "target.temporal_sustained.temporal_dag".into(),
            contrast(sustained, &temporal, GraphFamily::TemporalDag),
        ),
    ]
}

fn linear_ate() -> (Study, TabularData) {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.1, 0.9, 2.9, 1.0, 3.0, 1.1, 3.2];
    let columns: [(&str, &[f64]); 2] = [("t", &treatment), ("y", &outcome)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    (study, data)
}

/// The static linear Gaussian average effect: lowered target and program, layered
/// contract identities, and the claim of one execution.
fn linear_ate_identities() -> Vec<(String, String)> {
    let ctx = ExecutionContext::for_tests(3);
    let (study, data) = linear_ate();
    let prepared = study.prepare(&ctx).unwrap();
    let lowered = prepared.plan().lowered().expect("compile attaches a lowered program");
    let mut out =
        vec![("static_linear_ate.lowered_program".to_owned(), lowered.identity().to_hex())];
    if let antecedent::planner::lowered::ProblemTarget::Target(target) =
        lowered.scientific().target()
    {
        out.push(("static_linear_ate.lowered_target".into(), target.identity().to_hex()));
    }
    out.push((
        "static_linear_ate.lowered_statistical".into(),
        lowered.statistical().identity().to_hex(),
    ));
    let contract = prepared.contract().unwrap();
    let identities = contract.identities;
    out.push(("static_linear_ate.target".into(), identities.target.to_hex()));
    out.push(("static_linear_ate.identification".into(), identities.identification.to_hex()));
    if let Some(product) = identities.identification_product {
        out.push(("static_linear_ate.identification_product".into(), product.to_hex()));
    }
    if let Some(program) = identities.program {
        out.push(("static_linear_ate.program".into(), program.to_hex()));
    }
    out.push(("static_linear_ate.inference_binding".into(), identities.inference_binding.to_hex()));
    out.push(("static_linear_ate.observation".into(), identities.observation.to_hex()));
    out.push(("static_linear_ate.data_snapshot".into(), identities.data_snapshot.to_hex()));
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = result.claim(&contract, &ctx).unwrap();
    out.push(("static_linear_ate.claim".into(), claim.claim_id.to_hex()));
    out
}

fn current() -> BTreeMap<String, String> {
    vocabulary_targets().into_iter().chain(linear_ate_identities()).collect()
}

fn render(digests: &BTreeMap<String, String>) -> String {
    digests.iter().map(|(name, hex)| format!("{name}={hex}\n")).collect()
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_acyclic_identity_goldens() {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(GOLDEN), render(&current())).expect("write the golden");
}

#[test]
fn acyclic_identities_are_byte_identical_to_their_goldens() {
    let bytes = antecedent_test_support::read_fixture_bytes(
        &fixture_dir(),
        GOLDEN,
        &format!("generate it with `{GENERATOR}`"),
    );
    let text = String::from_utf8(bytes).expect("the golden is text");
    let golden: BTreeMap<String, String> = text
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(name, hex)| (name.to_owned(), hex.to_owned()))
        .collect();
    let now = current();
    let changed: Vec<&String> = golden
        .iter()
        .filter(|(name, hex)| now.get(*name) != Some(*hex))
        .map(|(name, _)| name)
        .collect();
    assert!(changed.is_empty(), "acyclic identities changed: {changed:?}");
    assert_eq!(golden.len(), now.len(), "the golden names every pinned identity");
}
