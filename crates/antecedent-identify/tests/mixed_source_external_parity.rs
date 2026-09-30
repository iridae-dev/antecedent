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

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    MIXED_SOURCE_DEFAULT_LIMITS, MixedSourceDecision, MixedSourceQuery, decide_mixed_source,
    verify_mixed_source_derivation,
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

/// `Some(outcome)` for a case this route can express; `None` when its inputs
/// (selection or transportability nodes) are outside the route.
fn run(case: &Value) -> Option<(String, Vec<String>, BTreeSet<Cited>)> {
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
    let inputs = case["inputs"].as_array()?;
    let mut regimes = Vec::new();
    for (k, input) in inputs.iter().enumerate() {
        let on = strings(&input["do"]).iter().map(|n| id(n)).collect::<Vec<_>>();
        let mut regime = EvidenceRegime::try_new(
            RegimeId::from_raw(u32::try_from(k + 1).unwrap()),
            if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            on,
            [],
            strings(&input["measured"]).iter().map(|n| id(n)).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        regime.study = Some(Arc::from(input["study"].as_str().unwrap()));
        regimes.push(regime);
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
