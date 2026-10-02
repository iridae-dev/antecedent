//! Pinned external parity of mixed-source proof search.
//!
//! The cases in `conformance/identify/mixed_source_external/expected.json` record
//! the verbatim output of executing the pinned oracles (Ananke `OneLine*` 0.5.0 and
//! dosearch 1.0.12) on the documented Ananke (GID/AID surrogate-experiment) and
//! dosearch (mixed-distribution) graphs and inputs, in a temporary environment that
//! was deleted afterwards. This test runs the same graphs and inputs through the
//! search and requires the recorded outcome and agreement, and that every
//! disagreement or out-of-scope case is recorded rather than reshaped.
//!
//! The recorded `oracle_output_verbatim` is parsed rather than trusted: the
//! oracle's verdict (Ananke's `identified` flag; dosearch's presence of a
//! formula) must equal the summary `oracle_result`, and the distributions the
//! oracle's formula cites (each `p(vars | do(..))` term of Ananke, each `p(..|..)`
//! term of dosearch) must equal the set of source distributions the recorded
//! proof cites, as `(intervened set, measured set)` pairs. Where Antecedent
//! returns a theorem-scoped named route instead of a proof, the agreement class is
//! `agree_named_route`: there is no proof to compare with, and the check is that
//! every term of the oracle's formula is a margin or conditional of the one input
//! distribution. Cases whose formula is not expressible here (selection and
//! transportability nodes) are recorded out of scope and are not compared. The
//! generator scripts were deleted after the import, per convention: their hashes
//! are provenance only and the outputs are not reproducible from this repository.
//!
//! Citing the same distributions is not the same as computing the same number, so
//! every proof this route returns on an agreed case is also executed: its
//! expression is evaluated by the exact provider on laws enumerated from random
//! binary structural models of the case's graph and must equal the model's
//! `P(outcomes | do(treatments))` at every level of the treatments.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    MIXED_SOURCE_DEFAULT_LIMITS, MixedSourceDecision, MixedSourceDerivation, MixedSourceQuery,
    decide_mixed_source, verify_mixed_source_derivation,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/identify/mixed_source_external/expected.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_owned()).collect()
}

/// `(intervened, measured)` name sets of one distribution a formula cites.
type Cited = (BTreeSet<String>, BTreeSet<String>);

/// The oracle's verdict, read from its verbatim output: Ananke's dictionary
/// carries an `identified` flag; dosearch's output is the formula itself.
fn verdict_of(verbatim: &Value) -> Result<&'static str, String> {
    let verdict = match verbatim {
        Value::Object(map) => match map.get("identified").and_then(Value::as_bool) {
            Some(true) => Some("identified"),
            Some(false) => Some("not_identified"),
            None => None,
        },
        Value::String(text) if text.contains("p(") => Some("identified"),
        Value::String(text) if text.trim().is_empty() => Some("not_identified"),
        _ => None,
    };
    verdict.ok_or_else(|| format!("unreadable verbatim output {verbatim}"))
}

/// The formula text of a verbatim output.
fn formula_of(verbatim: &Value) -> Option<&str> {
    match verbatim {
        Value::Object(map) => map.get("functional").and_then(Value::as_str),
        Value::String(text) => Some(text),
        _ => None,
    }
}

/// Split at top-level commas.
fn top_level(text: &str) -> Vec<&str> {
    let (mut depth, mut start, mut out) = (0usize, 0usize, Vec::new());
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(text[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(text[start..].trim());
    out.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Every `p(left | right)` term of a formula: the intervened set (`do(..)` entries)
/// and the measured set (left side and the non-`do` conditions). `None` when a
/// term names a selection or transportability node (`s`, `t`), which this route
/// cannot express.
fn parse_terms(formula: &str, transport_nodes: &[&str]) -> Option<Vec<Cited>> {
    let bytes = formula.as_bytes();
    let mut terms = Vec::new();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        let at_term = bytes[i] == b'p'
            && bytes[i + 1] == b'('
            && (i == 0 || !(bytes[i - 1] as char).is_ascii_alphanumeric());
        if !at_term {
            i += 1;
            continue;
        }
        let (mut depth, mut end) = (0usize, i + 1);
        for (offset, c) in formula[i + 1..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1 + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        let inner = &formula[i + 2..end];
        let (left, right) = inner.split_once('|').unwrap_or((inner, ""));
        let (mut intervened, mut measured) = (BTreeSet::new(), BTreeSet::new());
        for name in top_level(left) {
            measured.insert(name.to_owned());
        }
        for condition in top_level(right) {
            if let Some(list) = condition.strip_prefix("do(").and_then(|c| c.strip_suffix(')')) {
                intervened.extend(top_level(list).into_iter().map(str::to_owned));
            } else {
                measured.insert(condition.to_owned());
            }
        }
        if intervened.iter().chain(&measured).any(|n| transport_nodes.contains(&n.as_str())) {
            return None;
        }
        terms.push((intervened, measured));
        i = end + 1;
    }
    Some(terms)
}

/// Parse the verbatim output and require it to agree with the recorded summary
/// fields (`oracle_result`, `oracle_formula`); returns the distributions the
/// oracle's formula cites, when it names any and they are expressible here.
fn read_oracle(case: &Value) -> Result<Option<Vec<Cited>>, String> {
    let id = case["id"].as_str().unwrap_or("?");
    let verbatim = case.get("oracle_output_verbatim").ok_or(format!("{id}: no verbatim output"))?;
    let verdict = verdict_of(verbatim).map_err(|e| format!("{id}: {e}"))?;
    let recorded = case["oracle_result"].as_str().unwrap_or("");
    if verdict != recorded {
        return Err(format!("{id}: verbatim output says {verdict}, oracle_result says {recorded}"));
    }
    match (formula_of(verbatim), case["oracle_formula"].as_str()) {
        (Some(text), Some(recorded)) if text == recorded => {}
        (None, None) => {}
        (found, recorded) => {
            return Err(format!(
                "{id}: formula {found:?} differs from oracle_formula {recorded:?}"
            ));
        }
    }
    let Some(formula) = formula_of(verbatim) else { return Ok(None) };
    let terms = parse_terms(formula, &["s", "t"]);
    if verdict == "identified" && terms.as_ref().is_some_and(Vec::is_empty) {
        return Err(format!("{id}: an identified formula names no distribution"));
    }
    Ok(terms)
}

/// Compare the oracle's cited distributions with the ones our proof cites.
fn cited_by_proof(
    nodes: &[String],
    derivation: &antecedent_identify::MixedSourceDerivation,
) -> BTreeSet<Cited> {
    let names = |variables: &[VariableId]| {
        variables.iter().map(|v| nodes[v.as_usize()].clone()).collect::<BTreeSet<_>>()
    };
    derivation
        .steps()
        .iter()
        .filter(|step| step.source.is_some())
        .map(|step| (names(&step.intervened), names(&step.y)))
        .collect()
}

/// One case's graph, inputs and query: `(regime id, intervened, measured)` of
/// every input in catalog order, with the variable names.
struct Case {
    nodes: Vec<String>,
    graph: Admg,
    inputs: Vec<(RegimeId, Vec<usize>, Vec<usize>)>,
    catalog: EvidenceCatalog,
    query: MixedSourceQuery,
}

/// `Some` for a case this route can express; `None` when its inputs (selection
/// or transportability nodes) are outside the route.
fn build_case(case: &Value) -> Option<Case> {
    let nodes = strings(&case["nodes"]);
    let id = |name: &str| {
        VariableId::from_raw(u32::try_from(nodes.iter().position(|n| n == name).unwrap()).unwrap())
    };
    let dense = |name: &str| DenseNodeId::from_raw(id(name).raw());
    let mut graph = Admg::with_variables(u32::try_from(nodes.len()).unwrap());
    for edge in case["directed"].as_array().unwrap() {
        graph
            .insert_directed(dense(edge[0].as_str().unwrap()), dense(edge[1].as_str().unwrap()))
            .unwrap();
    }
    for edge in case["bidirected"].as_array().unwrap() {
        graph
            .insert_bidirected(dense(edge[0].as_str().unwrap()), dense(edge[1].as_str().unwrap()))
            .unwrap();
    }
    let mut regimes = Vec::new();
    let mut inputs = Vec::new();
    for (k, input) in case["inputs"].as_array()?.iter().enumerate() {
        let regime_id = RegimeId::from_raw(u32::try_from(k + 1).unwrap());
        let on = strings(&input["do"]).iter().map(|n| id(n)).collect::<Vec<_>>();
        let measured = strings(&input["measured"]).iter().map(|n| id(n)).collect::<Vec<_>>();
        let mut regime = EvidenceRegime::try_new(
            regime_id,
            if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            on.clone(),
            [],
            measured.clone(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        regime.study = Some(Arc::from(input["study"].as_str().unwrap()));
        regimes.push(regime);
        inputs.push((
            regime_id,
            on.iter().map(|v| v.as_usize()).collect(),
            measured.iter().map(|v| v.as_usize()).collect(),
        ));
    }
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    let catalog = EvidenceCatalog::try_new([], regimes, bindings, None).unwrap();
    let query = MixedSourceQuery {
        outcomes: strings(&case["query"]["outcomes"])
            .iter()
            .map(|n| id(n))
            .collect::<Vec<_>>()
            .into(),
        treatments: strings(&case["query"]["treatments"])
            .iter()
            .map(|n| id(n))
            .collect::<Vec<_>>()
            .into(),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    Some(Case { nodes, graph, inputs, catalog, query })
}

/// The case's decision: its outcome, the studies and distributions a proof cites.
fn run(case: &Value) -> Option<(String, Vec<String>, BTreeSet<Cited>)> {
    let Case { nodes, graph, catalog, query, .. } = build_case(case)?;
    let ctx = ExecutionContext::for_tests(1);
    Some(
        match decide_mixed_source(&graph, &query, &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
            .unwrap()
        {
            MixedSourceDecision::Identified { derivation, .. } => {
                verify_mixed_source_derivation(&graph, &derivation, &catalog).unwrap();
                let studies =
                    derivation.leaves().iter().map(|(_, l)| l.study.to_string()).collect();
                ("identified".into(), studies, cited_by_proof(&nodes, &derivation))
            }
            MixedSourceDecision::NamedRoute { .. } => {
                ("named_route".into(), Vec::new(), BTreeSet::new())
            }
            MixedSourceDecision::NotCertified(_) => {
                ("not_certified".into(), Vec::new(), BTreeSet::new())
            }
            other => panic!("unexpected {other:?}"),
        },
    )
}

/// The agreement class of one case, derived from the oracle's parsed verbatim
/// output, our outcome and (for a proof) the compared cited distributions.
fn derive_agreement(case: &Value, outcome: &str, proof_cited: &BTreeSet<Cited>) -> String {
    let id = case["id"].as_str().unwrap();
    let oracle_cited = read_oracle(case).unwrap_or_else(|error| panic!("{error}"));
    let oracle = case["oracle_result"].as_str().unwrap();
    match (oracle, outcome) {
        ("identified", "identified") => {
            // The formula's distributions are the proof's leaves, as a set.
            let terms = oracle_cited.unwrap_or_else(|| panic!("{id}: formula not parseable"));
            let expected = terms.into_iter().collect::<BTreeSet<_>>();
            assert_eq!(&expected, proof_cited, "{id}: cited distributions differ");
            "agree".into()
        }
        ("identified", "named_route") => {
            // No proof: every term is a margin or conditional of the one input.
            let terms = oracle_cited.unwrap_or_else(|| panic!("{id}: formula not parseable"));
            let input = case["inputs"].as_array().unwrap();
            assert_eq!(input.len(), 1, "{id}: a named route here has one input distribution");
            let measured = strings(&input[0]["measured"]).into_iter().collect::<BTreeSet<_>>();
            for (do_, vars) in terms {
                assert!(do_.is_empty() && vars.is_subset(&measured), "{id}: {vars:?}");
            }
            "agree_named_route".into()
        }
        ("not_identified", "not_certified") => "consistent".into(),
        ("not_executed", _) => "unverified".into(),
        _ => "disagree".into(),
    }
}

#[test]
fn ananke_gid_and_aid_cases_agree_and_every_disagreement_is_recorded() {
    let fixture = fixture();
    assert_eq!(fixture["oracle"]["executed_here"], true);
    assert_eq!(fixture["oracle"]["generation"]["generator_retained"], false);
    assert_eq!(fixture["oracle"]["generation"]["reproducible_from_repo"], false);
    let (mut agreed, mut named) = (0, 0);
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let expected = case["expected_here"].as_str().unwrap();
        let agreement = case["agreement"].as_str().unwrap();
        let Some((outcome, studies, cited)) = run(case) else {
            assert_eq!(agreement, "out_of_scope", "{id}");
            // The oracle's own output is still read and must match its summary; its
            // formula cites selection or transportability nodes, so it is not compared.
            assert!(read_oracle(case).unwrap().is_none(), "{id}");
            continue;
        };
        assert_eq!(outcome, expected, "{id}");
        for name in case["must_cite_studies"].as_array().into_iter().flatten() {
            assert!(studies.iter().any(|s| s == name.as_str().unwrap()), "{id}: {studies:?}");
        }
        // A recorded agreement is derived from the parsed verbatim output, never taken from
        // the summary fields, and a disagreement is never silently coded as agreement.
        assert_eq!(agreement, derive_agreement(case, &outcome, &cited), "{id}");
        agreed += usize::from(agreement == "agree");
        named += usize::from(agreement == "agree_named_route");
        if agreement == "agree_named_route" {
            assert!(
                case["note"].as_str().unwrap().contains("agree (named route)"),
                "{id}: the note says the agreement is a named route, not a proof"
            );
        }
    }
    assert!(agreed >= 3 && named == 2, "{agreed} proofs agree, {named} named routes");
}

/// A binary structural model consistent with an ADMG: every node is a random
/// function of its parents and the latent bits of its bidirected edges, flipped by
/// independent noise so every configuration has positive mass.
struct Model {
    n: usize,
    exo_p: Vec<f64>,
    tables: Vec<(Vec<usize>, Vec<usize>, Vec<u8>)>,
}

impl Model {
    fn random(seed: u64, graph: &Admg) -> Self {
        let n = graph.node_count();
        let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap());
        // SplitMix64: consecutive seeds give unrelated streams.
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            usize::try_from((z ^ (z >> 31)) >> 8).unwrap()
        };
        let mut bidirected = Vec::new();
        for a in 0..n {
            for b in graph.bidirected_neighbors(dense(a)) {
                if a < b.as_usize() {
                    bidirected.push((a, b.as_usize()));
                }
            }
        }
        let mut exo_p =
            (0..n).map(|_| 0.15 + 0.2 * (next() % 1000) as f64 / 1000.0).collect::<Vec<_>>();
        exo_p.extend(bidirected.iter().map(|_| 0.3 + 0.4 * (next() % 1000) as f64 / 1000.0));
        let tables = (0..n)
            .map(|i| {
                let parents =
                    graph.parents(dense(i)).iter().map(|p| p.as_usize()).collect::<Vec<_>>();
                let latents = bidirected
                    .iter()
                    .enumerate()
                    .filter(|(_, (a, b))| *a == i || *b == i)
                    .map(|(k, _)| n + k)
                    .collect::<Vec<_>>();
                let width = parents.len() + latents.len();
                let table = (0..1usize << width).map(|_| u8::from(next() % 2 == 1)).collect();
                (parents, latents, table)
            })
            .collect();
        Self { n, exo_p, tables }
    }

    /// Exact joint law over `measured` (first variable most significant) under `do_`.
    fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; 1 << measured.len()];
        let m = self.exo_p.len();
        for mask in 0..(1usize << m) {
            let exo = (0..m).map(|bit| u8::from((mask >> bit) & 1 == 1)).collect::<Vec<_>>();
            let weight = self
                .exo_p
                .iter()
                .enumerate()
                .map(|(bit, p)| if exo[bit] == 1 { *p } else { 1.0 - *p })
                .product::<f64>();
            let mut values = vec![0u8; self.n];
            for i in 0..self.n {
                values[i] = if let Some((_, level)) = do_.iter().find(|(v, _)| *v == i) {
                    *level
                } else {
                    let (parents, latents, table) = &self.tables[i];
                    let key = parents
                        .iter()
                        .map(|p| values[*p])
                        .chain(latents.iter().map(|l| exo[*l]))
                        .fold(0usize, |acc, bit| (acc << 1) | usize::from(bit));
                    table[key] ^ exo[i]
                };
            }
            let index = measured.iter().fold(0usize, |acc, v| (acc << 1) | usize::from(values[*v]));
            out[index] += weight;
        }
        out
    }
}

/// Exact laws of every input of `case`, enumerated from `model`, one per level of
/// each input's intervention set.
fn enumerated_laws(case: &Case, model: &Model) -> ExactTransportData {
    let axis = |v: usize| DiscreteAxis {
        variable: VariableId::from_raw(u32::try_from(v).unwrap()),
        values: Arc::from([
            antecedent_core::Value::Bool(false),
            antecedent_core::Value::Bool(true),
        ]),
    };
    let mut laws = Vec::new();
    for (regime, on, measured) in &case.inputs {
        for levels in 0..(1usize << on.len()) {
            let do_ = on
                .iter()
                .enumerate()
                .map(|(bit, v)| (*v, u8::from((levels >> bit) & 1 == 1)))
                .collect::<Vec<_>>();
            laws.push(
                ExactDiscreteLaw::try_new(
                    "target",
                    *regime,
                    do_.iter()
                        .map(|(v, level)| {
                            InterventionAssignment::concrete(
                                VariableId::from_raw(u32::try_from(*v).unwrap()),
                                antecedent_core::Value::Bool(*level == 1),
                            )
                        })
                        .collect::<Vec<_>>(),
                    measured.iter().map(|v| axis(*v)).collect::<Vec<_>>(),
                    model.law(&do_, measured),
                    format!("snapshot-{}", regime.raw()),
                    LawTolerance::default(),
                )
                .unwrap(),
            );
        }
    }
    ExactTransportData::try_new(laws, 4096).unwrap()
}

/// Evaluate `derivation` on the model's laws at every treatment level and compare
/// with the model's own `P(outcomes | do(treatments))`.
fn assert_proof_computes_the_truth(
    case: &Case,
    model: &Model,
    derivation: &MixedSourceDerivation,
    label: &str,
) {
    let ctx = ExecutionContext::for_tests(1);
    let data = enumerated_laws(case, model).with_world_bound_leaves(derivation.cited_regimes());
    let treatments = case.query.treatments.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
    let outcomes = case.query.outcomes.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
    for levels in 0..(1usize << treatments.len()) {
        let do_ = treatments
            .iter()
            .enumerate()
            .map(|(bit, v)| (*v, u8::from((levels >> bit) & 1 == 1)))
            .collect::<Vec<_>>();
        let request = Assignment::from_pairs(do_.iter().map(|(v, level)| {
            (
                VariableId::from_raw(u32::try_from(*v).unwrap()),
                antecedent_core::Value::Bool(*level == 1),
            )
        }));
        let plan = ExactEvaluationPlan::compile(
            derivation.arena(),
            derivation.root(),
            data.clone(),
            case.query.outcomes.to_vec(),
            request,
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let got = plan.evaluate(&ctx).unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let truth = model.law(&do_, &outcomes);
        assert_eq!(got.outcomes.as_ref(), case.query.outcomes.as_ref(), "{label}");
        for (atom, p) in got.atoms.iter().zip(got.probabilities.iter()) {
            let index = atom.iter().fold(0usize, |acc, value| {
                (acc << 1) | usize::from(*value == antecedent_core::Value::Bool(true))
            });
            assert!(
                (p - truth[index]).abs() < 1e-9,
                "{label}: do{do_:?} atom {atom:?}: formula {p}, truth {}",
                truth[index]
            );
        }
    }
}

/// Agreement on the cited distributions is backed by the number: every proof this
/// route returns on an agreed case computes the enumerated structural model's
/// interventional truth at every treatment level, over several random models of
/// the case's graph, and so does every alternative derivation.
#[test]
fn every_agreed_proof_computes_the_enumerated_truth_of_its_case() {
    let fixture = fixture();
    let ctx = ExecutionContext::for_tests(1);
    let mut proofs = 0usize;
    for case_value in fixture["cases"].as_array().unwrap() {
        if case_value["agreement"] != "agree" {
            continue;
        }
        let id = case_value["id"].as_str().unwrap();
        let case =
            build_case(case_value).unwrap_or_else(|| panic!("{id}: an agreed case is expressible"));
        let MixedSourceDecision::Identified { derivation, alternatives, .. } = decide_mixed_source(
            &case.graph,
            &case.query,
            &case.catalog,
            MIXED_SOURCE_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap() else {
            panic!("{id}: an agreed case is identified");
        };
        let treatments = case.query.treatments.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
        let outcomes = case.query.outcomes.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
        let mut with_effect = 0usize;
        for seed in 0..8u64 {
            let model = Model::random(0x9A71_7E57 + seed, &case.graph);
            for (k, candidate) in std::iter::once(&*derivation).chain(&alternatives).enumerate() {
                assert_proof_computes_the_truth(
                    &case,
                    &model,
                    candidate,
                    &format!("{id} proof {k} model {seed}"),
                );
                proofs += 1;
            }
            // The comparison is not trivial: in some model the interventional truth
            // is not constant in the treatments.
            let at = |level: u8| {
                model.law(&treatments.iter().map(|t| (*t, level)).collect::<Vec<_>>(), &outcomes)
            };
            with_effect += usize::from((at(0)[1] - at(1)[1]).abs() > 1e-3);
        }
        assert!(with_effect >= 1, "{id}: no model of 8 has an effect");
    }
    assert!(proofs >= 24, "{proofs} proofs executed");
}

/// The comparison is not vacuous: tampered verbatim output, an altered formula
/// term, or a wrong verdict each fail.
#[test]
fn the_verbatim_output_is_read_not_trusted() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().unwrap();
    let case = |id: &str| cases.iter().find(|c| c["id"] == id).unwrap().clone();
    // The recorded summaries agree with the parsed verbatim output.
    for c in cases {
        read_oracle(c).unwrap();
    }
    let mut flipped = case("ananke_gid_two_experiments");
    flipped["oracle_output_verbatim"]["identified"] = Value::Bool(false);
    assert!(read_oracle(&flipped).unwrap_err().contains("verbatim output says not_identified"));
    let mut renamed = case("dosearch_mixed_distributions_two_studies");
    renamed["oracle_formula"] = Value::String("p(y|z)".into());
    assert!(read_oracle(&renamed).is_err());
    let mut not_identified = case("ananke_observational_only");
    not_identified["oracle_result"] = Value::String("identified".into());
    assert!(read_oracle(&not_identified).is_err());
    // A formula that cites a different distribution than the proof does is caught.
    let mut moved = case("dosearch_mixed_distributions_two_studies");
    let formula = "\\sum_{z}\\left(p(y|z)p(z|x,y)\\right)";
    moved["oracle_formula"] = Value::String(formula.into());
    moved["oracle_output_verbatim"] = Value::String(formula.into());
    let (outcome, _, cited) = run(&moved).unwrap();
    let caught = std::panic::catch_unwind(|| derive_agreement(&moved, &outcome, &cited));
    assert!(caught.is_err(), "a different cited distribution must not read as agreement");
    // The parser reads terms, do-sets and conditions.
    let terms = parse_terms("ΣW ΦX2,Y p(W,X2,Y | do(X1))ΦX1,W p(W | do(X2, X3), A)", &[]).unwrap();
    let names = |xs: &[&str]| xs.iter().map(|x| (*x).to_owned()).collect::<BTreeSet<_>>();
    assert_eq!(terms[0], (names(&["X1"]), names(&["W", "X2", "Y"])));
    assert_eq!(terms[1], (names(&["X2", "X3"]), names(&["W", "A"])));
    assert!(parse_terms("p(y|do(x),z,t)", &["t"]).is_none());
}

#[test]
fn cases_outside_the_route_are_recorded_out_of_scope_not_shaped() {
    let fixture = fixture();
    let out = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["agreement"] == "out_of_scope")
        .collect::<Vec<_>>();
    assert_eq!(out.len(), 1);
    assert!(
        out[0].get("inputs").is_none(),
        "selection and transportability inputs are not expressible"
    );
    assert!(out[0]["note"].as_str().unwrap().contains("Recorded, not shaped"));
}
