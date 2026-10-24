//! The readable report rendered from a contract, a claim and a plan view.
//!
//! Each scenario is a real compiled study: a licensed static effect, a route that
//! executes without a licence, a cell the support matrix refuses, a Bayesian
//! effect and a temporal effect. The Markdown and HTML of each are compared with
//! the committed golden files; regenerate them deliberately with
//! `ANTECEDENT_REGENERATE_REPORTS=1 cargo test -p antecedent --test report` and
//! review the diff. The report is deterministic, escapes every value it takes from
//! an input, and prints the values its inputs hold.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use antecedent::analysis::CausalContract;
use antecedent::planner::PlanView;
use antecedent::report::{ReportFormat, ReportInput, render_report};
use antecedent::{
    BayesianConfig, CausalError, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_contract::{ClaimEnvelope, SlotAvailability};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, PathSpecificEffectQuery, SequentialContrastQuery,
    TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Admg, DenseNodeId, TemporalDag, ensure_lagged};

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

/// One report subject: what `render_report` reads.
struct Scenario {
    name: &'static str,
    contract: CausalContract,
    claim: Option<ClaimEnvelope>,
    plan: Option<PlanView>,
    names: Vec<Arc<str>>,
}

impl Scenario {
    fn render(&self, format: ReportFormat) -> String {
        self.render_named(&self.names, format)
    }

    fn render_named(&self, names: &[Arc<str>], format: ReportFormat) -> String {
        let mut input = ReportInput::new(&self.contract).with_names(names);
        if let Some(claim) = &self.claim {
            input = input.with_claim(claim);
        }
        if let Some(plan) = &self.plan {
            input = input.with_plan(plan);
        }
        render_report(&input, format)
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn names_of(prepared: &antecedent::PreparedStudy) -> Vec<Arc<str>> {
    prepared.schema().variables().iter().map(|variable| Arc::clone(&variable.name)).collect()
}

fn executed(
    name: &'static str,
    prepared: &antecedent::PreparedStudy,
    result: &antecedent::StudyResult,
) -> Scenario {
    let (contract, claim) = prepared.execution_contract(result, &ctx()).unwrap();
    Scenario {
        name,
        contract,
        claim: Some(claim),
        plan: Some(prepared.plan_view().unwrap()),
        names: names_of(prepared),
    }
}

fn licensed_static_effect() -> Scenario {
    let (data, dag, query) = confounded_scm(128, 41);
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    executed("licensed_static_effect", &prepared, &result)
}

const SUBJECTS: usize = 80;

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / 9_007_199_254_740_992.0
}

/// Columns l0, a0, l1, a1, y of a two-period sequentially randomized panel.
fn sequential_panel() -> TabularData {
    let mut state = 12_345_u64;
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..SUBJECTS {
        let u1 = uniform(&mut state);
        let u2 = uniform(&mut state);
        let u3 = uniform(&mut state);
        let u4 = uniform(&mut state);
        let u5 = uniform(&mut state);
        let l0 = 2.0 * u1 - 1.0;
        let a0 = if u2 < 0.5 + 0.2 * l0 { 1.0 } else { 0.0 };
        let l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u3 - 1.0) * 0.5;
        let a1 = if u4 < 0.5 + 0.1 * (l1 - 0.4) { 1.0 } else { 0.0 };
        ys.push(1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u5 - 1.0));
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .unwrap()
}

fn unlicensed_executing_route() -> Scenario {
    let data = sequential_panel();
    let query = SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..SUBJECTS).map(|row| u32::from(matches!(row % 4, 1 | 2))).collect(),
        fold_count: 2,
        min_probability: 0.05,
    };
    let study =
        Study::tabular(data.clone()).query(CausalQuery::SequentialContrast(query)).build().unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    executed("unlicensed_executing_route", &prepared, &result)
}

/// A path-specific effect on an explicit mixed graph: a cell the support matrix
/// refuses, so the study cannot be built and only its contract can be read.
fn refused_cell() -> Scenario {
    let mut t = Vec::new();
    let mut m = Vec::new();
    let mut y = Vec::new();
    for (ti, mi, yi, count) in [
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
    ] {
        for _ in 0..count {
            t.push(ti);
            m.push(mi);
            y.push(yi);
        }
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
        .with_path_nodes([VariableId::from_raw(1)]);
    let builder = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::PathSpecific(query))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .bootstrap_replicates(0);
    assert!(matches!(builder.clone().build(), Err(CausalError::Support { .. })));
    Scenario {
        name: "refused_cell",
        contract: builder.inspect().unwrap(),
        claim: None,
        plan: None,
        names: ["t", "m", "y"].map(Arc::from).to_vec(),
    }
}

fn bayesian_effect() -> Scenario {
    let (data, graph, query) = confounded_scm(512, 271);
    let config = BayesianConfig::conjugate().n_draws(96).prior_scale(30.0);
    let study = Study::tabular(data.clone())
        .query(query)
        .estimator(EstimatorId::BayesianGcomp)
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .graph(graph)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    executed("bayesian_effect", &prepared, &result)
}

fn lagged_series() -> TimeSeriesData {
    let n = 320usize;
    let t = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn temporal_effect() -> Scenario {
    let data = lagged_series();
    let mut graph = TemporalDag::empty();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let t1 = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, t, Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    let query = TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::pulse(-1));
    let study = Study::series(data.clone())
        .graph(graph)
        .query(CausalQuery::TemporalEffect(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate_series(&data, &ctx()).unwrap();
    executed("temporal_effect", &prepared, &result)
}

fn scenarios() -> Vec<Scenario> {
    vec![
        licensed_static_effect(),
        unlicensed_executing_route(),
        refused_cell(),
        bayesian_effect(),
        temporal_effect(),
    ]
}

fn golden_path(name: &str, extension: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("report")
        .join(format!("{name}.{extension}"))
}

fn check_golden(name: &str, extension: &str, actual: &str) {
    let path = golden_path(name, extension);
    if std::env::var_os("ANTECEDENT_REGENERATE_REPORTS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "golden report {} is missing ({error}); generate it with \
             ANTECEDENT_REGENERATE_REPORTS=1 cargo test -p antecedent --test report",
            path.display()
        )
    });
    assert!(
        expected == actual,
        "report {name}.{extension} differs from its golden file; regenerate deliberately with \
         ANTECEDENT_REGENERATE_REPORTS=1 and review the diff"
    );
}

#[test]
fn every_scenario_matches_its_golden_markdown_and_html() {
    for scenario in scenarios() {
        check_golden(scenario.name, "md", &scenario.render(ReportFormat::Markdown));
        check_golden(scenario.name, "html", &scenario.render(ReportFormat::Html));
    }
}

#[test]
fn rendering_is_deterministic_from_the_inputs_to_the_bytes() {
    let first = scenarios();
    let second = scenarios();
    for (one, two) in first.iter().zip(&second) {
        for format in [ReportFormat::Markdown, ReportFormat::Html] {
            let rendered = one.render(format);
            assert!(rendered == one.render(format), "{}: rendering twice differs", one.name);
            assert!(
                rendered == two.render(format),
                "{}: rebuilding the study changed the {} report",
                one.name,
                format.as_str()
            );
        }
    }
}

#[test]
fn the_prepared_handle_reports_what_the_inputs_render() {
    let (data, dag, query) = confounded_scm(128, 41);
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    let scenario = executed("licensed_static_effect", &prepared, &result);
    for format in [ReportFormat::Markdown, ReportFormat::Html] {
        assert!(
            prepared.report(Some(&result), &ctx(), format).unwrap() == scenario.render(format),
            "{}",
            format.as_str()
        );
    }
    // Before any execution the handle reports its contract and plan without a claim.
    let before = prepared.report(None, &ctx(), ReportFormat::Markdown).unwrap();
    assert!(before.contains("a contract before execution"), "{before}");
    assert!(!before.contains("## Claim"), "{before}");
}

/// The text after `- **<key>:** ` on the first line of `markdown` that has it.
fn fact(markdown: &str, key: &str) -> String {
    let prefix = format!("- **{key}:** ");
    markdown
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("no fact {key:?} in\n{markdown}"))
        .to_owned()
}

#[test]
fn the_numbers_are_the_values_the_inputs_hold() {
    for scenario in scenarios() {
        let markdown = scenario.render(ReportFormat::Markdown);
        if let Some(claim) = &scenario.claim {
            match claim.value {
                Some(value) => {
                    let printed: f64 = fact(&markdown, "Value").parse().unwrap();
                    assert!(
                        (printed - value).abs() <= 1.0e-11 * value.abs().max(1.0),
                        "{}: value {printed} against {value}",
                        scenario.name
                    );
                }
                None => assert_eq!(fact(&markdown, "Value"), "none", "{}", scenario.name),
            }
        }
        let reasoning =
            scenario.claim.as_ref().map_or(&scenario.contract.reasoning, |c| &c.reasoning);
        if let SlotAvailability::Available(slot) = &reasoning.identification {
            for (key, value) in [
                ("Identified mass", slot.identified_mass),
                ("Unidentified mass", slot.unidentified_mass),
                ("Unevaluable mass", slot.unevaluable_mass),
                ("Incomplete-search mass", slot.incomplete_search_mass),
            ] {
                let printed: f64 = fact(&markdown, key).parse().unwrap();
                assert!(
                    (printed - value).abs() <= 1.0e-11,
                    "{}: {key} {printed} against {value}",
                    scenario.name
                );
            }
        }
    }
}

#[test]
fn a_refusal_names_its_blockers_next_actions_and_no_value() {
    let scenarios = scenarios();
    let refused = scenarios.iter().find(|s| s.name == "refused_cell").unwrap();
    let markdown = refused.render(ReportFormat::Markdown);
    assert!(markdown.contains("support.refused"), "{markdown}");
    assert!(markdown.contains("No claim is attached"), "{markdown}");
    assert!(markdown.contains("## Refusals and limits"), "{markdown}");
    assert!(!markdown.contains("- **Value:** "), "{markdown}");
    assert!(!markdown.contains("## Claim"), "{markdown}");
    assert_eq!(fact(&markdown, "Support status"), "refused");
}

#[test]
fn a_noncausal_claim_says_so_and_never_reads_as_an_effect() {
    use antecedent_contract::{ClaimDomains, ClaimKind, DomainStatus, IDENTIFICATION_NOT_CLAIMED};
    let mut scenario = licensed_static_effect();
    scenario.plan = None;
    let claim = scenario.claim.as_mut().unwrap();
    claim.kind = ClaimKind::Associational;
    claim.value = None;
    claim.reasoning.identification = SlotAvailability::unavailable(IDENTIFICATION_NOT_CLAIMED);
    claim.domains =
        ClaimDomains::new(DomainStatus::NotClaimed, DomainStatus::Unknown, DomainStatus::Evaluated);
    for format in [ReportFormat::Markdown, ReportFormat::Html] {
        let rendered = scenario.render(format);
        assert!(rendered.contains("noncausal association"), "{rendered}");
        assert!(rendered.contains("Identification is not claimed"), "{rendered}");
        // Markdown escapes the underscore of a value it prints; HTML prints it raw.
        let printed = match format {
            ReportFormat::Markdown => "not\\_claimed",
            _ => "not_claimed",
        };
        assert!(rendered.contains(printed), "{rendered}");
        assert!(!rendered.contains("publishes a value of kind"), "{rendered}");
        assert!(!rendered.contains("Identification status is"), "{rendered}");
    }
    let markdown = scenario.render(ReportFormat::Markdown);
    assert_eq!(fact(&markdown, "Claim kind"), "associational");
    assert_eq!(fact(&markdown, "Identification domain"), "not\\_claimed");
    assert!(fact(&markdown, "Causal reading").starts_with("noncausal"), "{markdown}");
}

#[test]
fn status_and_inference_are_stated_for_each_scenario() {
    for scenario in scenarios() {
        let markdown = scenario.render(ReportFormat::Markdown);
        let status = fact(&markdown, "Support status");
        match scenario.name {
            "licensed_static_effect" | "bayesian_effect" | "temporal_effect" => {
                assert_eq!(status, "licensed", "{}", scenario.name);
            }
            "unlicensed_executing_route" => {
                assert_ne!(status, "licensed", "{}: it executes without a licence", scenario.name);
            }
            _ => assert_eq!(status, "refused", "{}", scenario.name),
        }
        let inference = fact(&markdown, "Inference").to_lowercase();
        let want = if scenario.name == "bayesian_effect" { "bayesian" } else { "frequentist" };
        assert_eq!(inference, want, "{}", scenario.name);
    }
}

#[test]
fn hostile_names_are_escaped_in_html_and_markdown() {
    let scenarios = scenarios();
    let scenario = scenarios.iter().find(|s| s.name == "licensed_static_effect").unwrap();
    let hostile: Vec<Arc<str>> = vec![
        Arc::from("a\"onload=\"x"),
        Arc::from("&amp;'<img src=x>"),
        Arc::from("<script>alert(1)</script>|`z`*_[l](u)"),
    ];
    let html = scenario.render_named(&hostile, ReportFormat::Html);
    assert!(!html.contains("<script>alert"), "{html}");
    assert!(!html.contains("<img"), "{html}");
    assert!(
        html.contains("&lt;script&gt;alert(1)&lt;/script&gt;|`z`*_[l](u)"),
        "the adjustment variable is written escaped"
    );
    assert!(html.contains("&amp;amp;&#39;&lt;img src=x&gt;"), "ampersand and quote are escaped");
    assert!(html.contains("a&quot;onload=&quot;x"), "a double quote cannot leave an attribute");
    // The only script-like content is none: the document has no script element.
    assert!(!html.contains("<script"), "{html}");

    let markdown = scenario.render_named(&hostile, ReportFormat::Markdown);
    // Every angle bracket in the document is backslash-escaped, so no tag can open.
    let bytes = markdown.as_bytes();
    assert!(
        bytes.iter().enumerate().all(|(at, &byte)| {
            !matches!(byte, b'<' | b'>') || (at > 0 && bytes[at - 1] == b'\\')
        }),
        "{markdown}"
    );
    assert!(
        markdown.contains("\\<script\\>alert(1)\\</script\\>\\|\\`z\\`\\*\\_\\[l\\](u)"),
        "{markdown}"
    );
    assert!(markdown.contains("\\&amp;'\\<img src=x\\>"), "{markdown}");
}

#[test]
fn the_html_document_is_inert_and_complete() {
    for scenario in scenarios() {
        let html = scenario.render(ReportFormat::Html);
        assert!(html.starts_with("<!doctype html>\n<html lang=\"en\">"), "{}", scenario.name);
        assert!(html.ends_with("</html>\n"), "{}", scenario.name);
        assert!(!html.contains("<script"), "{}", scenario.name);
        assert!(!html.contains("http://") && !html.contains("https://"), "{}", scenario.name);
        assert!(html.contains("default-src 'none'"), "{}", scenario.name);
        assert_eq!(html.matches("<section").count(), html.matches("</section>").count());
        assert_eq!(html.matches("<table>").count(), html.matches("</table>").count());
    }
}

#[test]
fn format_names_round_trip() {
    for format in [ReportFormat::Markdown, ReportFormat::Html] {
        assert_eq!(ReportFormat::from_name(format.as_str()), Some(format));
    }
    assert_eq!(ReportFormat::from_name("md"), Some(ReportFormat::Markdown));
    assert_eq!(ReportFormat::from_name("pdf"), None);
}

#[test]
fn the_sections_are_the_ones_the_python_report_pins() {
    let scenarios = scenarios();
    let licensed = scenarios.iter().find(|s| s.name == "licensed_static_effect").unwrap();
    let markdown = licensed.render(ReportFormat::Markdown);
    let headings: Vec<&str> =
        markdown.lines().filter_map(|line| line.strip_prefix("## ")).collect();
    assert_eq!(
        headings,
        [
            "Summary",
            "Claim",
            "Identification",
            "Assumptions",
            "Diagnostics and evidence",
            "Uncertainty",
            "Support and licence",
            "Refusals and limits",
            "Program plan",
            "Identities",
        ]
    );
}
