//! A readable report rendered from a contract.
//!
//! [`render_report`] turns a [`CausalContract`], optionally the
//! [`ClaimEnvelope`] of one execution and the [`PlanView`] of the program, into
//! one Markdown or HTML document. The report restates what those records hold:
//! the claim and its domains, the identification, the assumptions and
//! obligations, the diagnostics and evidence, the uncertainty axes and the
//! calibration slot, the support cell, the refusals and limits the operation
//! report names, and the program plan.
//!
//! The report computes nothing. It identifies, fits, scans and samples
//! nothing, derives no number, and states no verdict the records do not state:
//! every figure it prints is a value read from one of its inputs, written to at
//! most twelve significant digits. The wording is fixed text selected by the
//! records' own status tags. The same inputs render to the same bytes: there
//! is no clock, no random source, no unordered iteration and no environment
//! read.
//!
//! The HTML document is self-contained and inert. It carries no script and
//! loads no resource, and every value that comes from an input is escaped; the
//! Markdown writes the same values with every markup character escaped.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::sync::Arc;

use antecedent_contract::{
    AttestedEvidence, CalibrationView, ClaimEnvelope, ClaimKind, IDENTIFICATION_NOT_CLAIMED,
    IdentificationSlot, ObligationRecord, ObligationScope, ReasoningView, SemanticLayer,
    SlotAvailability,
};
use antecedent_core::{AssumptionSource, ExecutionContext, VariableId};

use crate::PreparedStudy;
use crate::analysis::CausalContract;
use crate::analysis::association::ASSOCIATION_CALIBRATION;
use crate::analysis::fallback::{AnalysisWithFallback, NONCAUSAL_NOTICE, RankedPlan};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::fact::Fact;
use crate::result::StudyResult;
use crate::support::CellStatus;

/// The document format of a report.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ReportFormat {
    /// `CommonMark` with pipe tables.
    Markdown,
    /// One self-contained HTML document with inline style and no script.
    Html,
}

impl ReportFormat {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Html => "html",
        }
    }

    /// The format a stable name (`markdown`, `md` or `html`) selects.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "markdown" | "md" => Some(Self::Markdown),
            "html" => Some(Self::Html),
            _ => None,
        }
    }
}

/// What a report is read from.
///
/// A contract alone reads a program before execution; a claim adds the
/// execution's published reasoning and calibration; a plan view adds the
/// lowered program. `names` renders dense variable ids as the schema's names.
#[derive(Clone, Copy, Debug)]
pub struct ReportInput<'a> {
    contract: &'a CausalContract,
    claim: Option<&'a ClaimEnvelope>,
    plan: Option<&'a PlanView>,
    names: &'a [Arc<str>],
}

impl<'a> ReportInput<'a> {
    /// A report of `contract` alone.
    #[must_use]
    pub const fn new(contract: &'a CausalContract) -> Self {
        Self { contract, claim: None, plan: None, names: &[] }
    }

    /// Add the claim of one execution under this contract.
    #[must_use]
    pub const fn with_claim(mut self, claim: &'a ClaimEnvelope) -> Self {
        self.claim = Some(claim);
        self
    }

    /// Add the plan view of the program.
    #[must_use]
    pub const fn with_plan(mut self, plan: &'a PlanView) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Name variables by schema position; an id past the end reads `v<id>`.
    #[must_use]
    pub const fn with_names(mut self, names: &'a [Arc<str>]) -> Self {
        self.names = names;
        self
    }
}

/// Render the report of `input` in `format`.
///
/// The output is a pure function of the input: rendering twice gives the same
/// bytes, and no number in it is computed here.
#[must_use]
pub fn render_report(input: &ReportInput<'_>, format: ReportFormat) -> String {
    let report = Report::of(input);
    match format {
        ReportFormat::Markdown => report.markdown(),
        ReportFormat::Html => report.html(),
    }
}

/// Render the report of a descriptive fallback in `format`.
///
/// The causal refusal comes first, unchanged. Then the noncausal description (or the typed reason
/// there is none) and the evidence that would resolve the refusal, with the sentence that the
/// associational result is not an answer to the causal question. Like [`render_report`] it
/// computes nothing: every figure is read from the fallback.
#[must_use]
pub fn render_fallback_report(fallback: &AnalysisWithFallback, format: ReportFormat) -> String {
    let report = Report {
        title: "Refusal with descriptive fallback",
        sections: fallback_sections(fallback),
    };
    match format {
        ReportFormat::Markdown => report.markdown(),
        ReportFormat::Html => report.html(),
    }
}

fn fallback_sections(fallback: &AnalysisWithFallback) -> Vec<Section> {
    let mut refusal = Vec::new();
    fact(&mut refusal, "Status", "unsupported");
    fact(&mut refusal, "Reason code", opt_str(fallback.refusal.reason_code()));
    fact(&mut refusal, "Message", clean(&fallback.refusal.to_string()));
    fact(&mut refusal, "Remedy", opt_str(fallback.refusal.remedy()));
    let mut sections = vec![
        Section {
            title: "Causal question",
            blocks: vec![
                Block::Text(
                    "The causal query is refused. The refusal below is the answer to the causal \
                     question and is unchanged by anything else in this report."
                        .to_owned(),
                ),
                Block::Facts(refusal),
            ],
        },
        Section {
            title: "Noncausal description",
            blocks: vec![Block::Text(NONCAUSAL_NOTICE.to_owned()), associational_block(fallback)],
        },
    ];
    sections.push(Section {
        title: "Evidence that would resolve the refusal",
        blocks: evidence_blocks(fallback),
    });
    sections
}

fn associational_block(fallback: &AnalysisWithFallback) -> Block {
    match &fallback.associational {
        Err(reason) => unavailable_text("The associational description", &reason.to_string()),
        Ok(part) => {
            let mut facts = Vec::new();
            fact(&mut facts, "Claim kind", part.claim.kind.as_str());
            fact(&mut facts, "Identification", part.claim.domains.identification.as_str());
            fact(&mut facts, "Selection rule", part.roles.rule);
            if let Some(payload) = part.result.association() {
                fact(&mut facts, "Analysis", payload.estimate.selection.analysis.as_str());
                fact(&mut facts, "Rows", payload.estimate.rows.to_string());
                fact(&mut facts, "Uncertainty", payload.uncertainty_label());
                fact(&mut facts, "Calibration", ASSOCIATION_CALIBRATION);
                fact(&mut facts, "Refusal retained", opt_str(payload.refused_after.as_deref()));
                for term in &payload.estimate.terms {
                    fact(&mut facts, &term.label, num(term.estimate));
                }
            }
            fact(&mut facts, "Ranks actions", "no");
            fact(&mut facts, "Is an intervention effect", "no");
            Block::Facts(facts)
        }
    }
}

fn evidence_blocks(fallback: &AnalysisWithFallback) -> Vec<Block> {
    let needs = match &fallback.evidence {
        Err(reason) => return vec![unavailable_text("The evidence plan", &reason.to_string())],
        Ok(needs) => needs,
    };
    let mut blocks = Vec::new();
    let rows = needs
        .obligations
        .iter()
        .map(|o| {
            vec![
                clean(&o.id),
                o.kind.as_str().to_owned(),
                o.provenance.family.to_string(),
                o.provenance.proof_step.as_deref().unwrap_or("unrecorded").to_owned(),
                yes_no(o.satisfiable_by_study()).to_owned(),
            ]
        })
        .collect();
    blocks.push(Block::Table {
        head: &["Obligation", "Kind", "Route", "Proof step", "A study can satisfy it"],
        rows,
    });
    if needs.next_actions.is_empty() {
        blocks.push(Block::Text("No evidence-gathering next action follows.".to_owned()));
    } else {
        blocks.push(Block::Bullets(
            needs.next_actions.iter().map(|a| format!("{}: {}", a.id, a.description)).collect(),
        ));
    }
    blocks.push(match &needs.ranking {
        RankedPlan::NotRequested => Block::Text(
            "No candidate studies were declared, so the options are not ranked.".to_owned(),
        ),
        RankedPlan::Unavailable(reason) => unavailable_text("The ranked plan", &reason.to_string()),
        RankedPlan::Ranked(result) => Block::Table {
            head: &["Rank", "Route", "Studies", "Gate", "Cost"],
            rows: result
                .plan
                .options
                .iter()
                .map(|option| {
                    vec![
                        option.rank.to_string(),
                        option.route.to_owned(),
                        option
                            .labels
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", "),
                        option.gate.as_str().to_owned(),
                        format!("{} {}", option.cost.units, option.cost.unit_label),
                    ]
                })
                .collect(),
        },
    });
    blocks
}

impl PreparedStudy {
    /// The report of this handle: the executed claim of `result` when one is
    /// given, otherwise the prepared contract, with the plan view of the
    /// program. Nothing is identified, fitted or scanned.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::execution_contract`], [`Self::contract`] and
    /// [`Self::plan_view`]; `route_not_lowered` when the plan carries no lowered program.
    pub fn report(
        &self,
        result: Option<&StudyResult>,
        ctx: &ExecutionContext,
        format: ReportFormat,
    ) -> Result<String, CausalError> {
        // A plan with no lowered program is refused (`route_not_lowered`), not reported
        // without its plan section.
        let plan = self.plan_view()?;
        let names: Vec<Arc<str>> =
            self.schema().variables().iter().map(|variable| Arc::clone(&variable.name)).collect();
        let (contract, claim) = match result {
            Some(result) => {
                let (contract, claim) = self.execution_contract(result, ctx)?;
                (contract, Some(claim))
            }
            None => (self.contract()?, None),
        };
        let mut input = ReportInput::new(&contract).with_names(&names).with_plan(&plan);
        if let Some(claim) = &claim {
            input = input.with_claim(claim);
        }
        Ok(render_report(&input, format))
    }
}

// --- document model -------------------------------------------------------

/// One block of a section.
enum Block {
    /// Key and value pairs.
    Facts(Vec<(String, String)>),
    /// A paragraph of fixed or input text.
    Text(String),
    /// A bulleted list.
    Bullets(Vec<String>),
    /// A table with a header row.
    Table { head: &'static [&'static str], rows: Vec<Vec<String>> },
    /// A third-level heading.
    Subheading(String),
}

/// One titled section.
struct Section {
    title: &'static str,
    blocks: Vec<Block>,
}

/// The whole document.
struct Report {
    title: &'static str,
    sections: Vec<Section>,
}

/// Dense variable ids rendered as schema names.
#[derive(Clone, Copy)]
struct Names<'a>(&'a [Arc<str>]);

impl Names<'_> {
    fn of(self, id: VariableId) -> String {
        self.0.get(id.as_usize()).map_or_else(|| format!("v{}", id.raw()), ToString::to_string)
    }

    fn list(self, ids: &[VariableId]) -> String {
        if ids.is_empty() {
            return "none".to_owned();
        }
        ids.iter().map(|id| self.of(*id)).collect::<Vec<_>>().join(", ")
    }
}

// --- number and text formatting ------------------------------------------

/// A value as the input holds it, to at most twelve significant digits.
fn num(value: f64) -> String {
    use std::fmt::Write as _;
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let mut text = String::with_capacity(24);
    // Writing into a `String` cannot fail.
    let _ = write!(text, "{value:.11e}");
    let rounded: f64 = text.parse().unwrap_or(value);
    text.clear();
    // Adding zero turns a negative zero into zero.
    let _ = write!(text, "{}", rounded + 0.0);
    text
}

fn opt_num(value: Option<f64>) -> Cow<'static, str> {
    value.map_or(Cow::Borrowed("none"), |value| Cow::Owned(num(value)))
}

fn opt_str(value: Option<&str>) -> Cow<'_, str> {
    Cow::Borrowed(value.unwrap_or("none"))
}

const fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

/// One line of plain text: control characters become spaces.
fn clean(text: &str) -> String {
    text.chars().map(|c| if c.is_control() { ' ' } else { c }).collect()
}

/// Text safe inside Markdown: every markup character is backslash-escaped.
fn md_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in clean(text).chars() {
        if matches!(c, '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '|' | '~' | '&' | '#') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Text safe inside HTML text and quoted attribute values.
fn html_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in clean(text).chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

fn slug(title: &str) -> String {
    title
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect()
}

// --- fact flattening ------------------------------------------------------

/// A scalar fact as text; a variable is its schema name.
fn scalar(value: &Fact, names: Names<'_>) -> String {
    match value {
        Fact::Null => "none".to_owned(),
        Fact::Bool(flag) => yes_no(*flag).to_owned(),
        Fact::Int(integer) => integer.to_string(),
        Fact::Uint(integer) => integer.to_string(),
        Fact::Real(real) => num(*real),
        Fact::Text(text) => text.clone(),
        Fact::Variable(variable) => names.of(*variable),
        Fact::List(_) | Fact::Map(_) => String::new(),
    }
}

/// Key and value pairs for every leaf of `value`, in key order. Nulls
/// are left out: an absent layer is not a statement.
fn flatten(prefix: &str, value: &Fact, names: Names<'_>, out: &mut Vec<(String, String)>) {
    match value {
        Fact::Null => {}
        Fact::Map(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() { key.clone() } else { format!("{prefix}.{key}") };
                flatten(&path, child, names, out);
            }
        }
        Fact::List(items) => {
            if items.is_empty() {
                out.push((prefix.to_owned(), "none".to_owned()));
            } else if items.iter().all(|item| !matches!(item, Fact::List(_) | Fact::Map(_))) {
                out.push((
                    prefix.to_owned(),
                    items.iter().map(|item| scalar(item, names)).collect::<Vec<_>>().join(", "),
                ));
            } else {
                for (index, item) in items.iter().enumerate() {
                    flatten(&format!("{prefix}[{index}]"), item, names, out);
                }
            }
        }
        scalar_value => out.push((prefix.to_owned(), scalar(scalar_value, names))),
    }
}

fn facts_of(prefix: &str, value: &Fact, names: Names<'_>) -> Block {
    let mut facts = Vec::new();
    flatten(prefix, value, names, &mut facts);
    Block::Facts(facts)
}

fn fact(facts: &mut Vec<(String, String)>, key: &str, value: impl Into<String>) {
    facts.push((key.to_owned(), value.into()));
}

// --- record renderings ----------------------------------------------------

fn source_text(source: &AssumptionSource) -> String {
    match source {
        AssumptionSource::UserDeclared => "user_declared".to_owned(),
        AssumptionSource::AlgorithmDefault { algorithm } => {
            format!("algorithm_default:{algorithm}")
        }
        AssumptionSource::Artifact => "artifact".to_owned(),
        AssumptionSource::Derived { from } => format!("derived:{from}"),
    }
}

fn scope_text(scope: ObligationScope) -> String {
    match scope {
        ObligationScope::Horizon { horizon } => format!("horizon:{horizon}"),
        other => other.as_str().to_owned(),
    }
}

fn obligation_row(record: &ObligationRecord) -> Vec<String> {
    vec![
        record.id.to_string(),
        scope_text(record.scope),
        record.kind.as_str().to_owned(),
        record.status.as_str().to_owned(),
        source_text(&record.source),
        opt_str(record.required_check.as_deref()).into_owned(),
        record.description.to_string(),
    ]
}

const OBLIGATION_HEAD: &[&str] =
    &["Id", "Scope", "Kind", "Status", "Source", "Required check", "Description"];

fn attested_row(evidence: &AttestedEvidence) -> Vec<String> {
    vec![
        evidence.name.to_string(),
        evidence.kind.to_string(),
        yes_no(evidence.passed).to_owned(),
        yes_no(evidence.informative).to_owned(),
        opt_num(evidence.refuted_ate).into_owned(),
        opt_num(evidence.comparison).into_owned(),
        opt_str(evidence.failure_condition.as_deref()).into_owned(),
        yes_no(evidence.reverifiable).to_owned(),
        opt_str(evidence.config_digest.as_deref()).into_owned(),
        opt_str(evidence.payload_digest.as_deref()).into_owned(),
    ]
}

fn calibration_facts(calibration: &CalibrationView) -> Vec<(String, String)> {
    let mut facts = Vec::new();
    fact(&mut facts, "Calibration status", calibration.status.to_string());
    fact(&mut facts, "Governing record", opt_str(calibration.record_id.as_deref()));
    fact(&mut facts, "Reason", opt_str(calibration.reason.as_deref()));
    fact(
        &mut facts,
        "Measured rows (smallest)",
        calibration.scope_n.map_or_else(|| "none".to_owned(), |n| n.to_string()),
    );
    fact(
        &mut facts,
        "Measured rows (largest)",
        calibration.scope_n_max.map_or_else(|| "none".to_owned(), |n| n.to_string()),
    );
    fact(&mut facts, "Record dependence", opt_str(calibration.scope_dependence.as_deref()));
    fact(&mut facts, "Measured at", opt_str(calibration.calibration_sha.as_deref()));
    fact(&mut facts, "Nominal level", opt_num(calibration.nominal));
    fact(&mut facts, "Observed coverage", opt_num(calibration.observed));
    if let Some(basis) = &calibration.basis {
        fact(&mut facts, "Basis query", basis.query.to_string());
        fact(&mut facts, "Basis graph class", basis.graph_class.to_string());
        fact(&mut facts, "Basis structure", basis.structure.to_string());
        fact(&mut facts, "Basis modality", basis.modality.to_string());
        fact(&mut facts, "Basis inference", basis.inference.to_string());
        fact(&mut facts, "Basis estimator", basis.estimator.to_string());
        fact(&mut facts, "Basis interval method", basis.interval_method.to_string());
        fact(&mut facts, "Basis standard error", basis.se_kind.to_string());
        fact(&mut facts, "Basis dependence", basis.dependence.to_string());
        fact(&mut facts, "Basis posterior", basis.posterior.to_string());
        fact(&mut facts, "Basis functional", basis.functional.to_string());
        fact(&mut facts, "Basis level", num(basis.level));
        fact(&mut facts, "Basis identification", basis.identification.to_string());
        fact(&mut facts, "Basis rows", basis.row_count.to_string());
        fact(
            &mut facts,
            "Basis successful replicates",
            basis.replicates_ok.map_or_else(|| "none".to_owned(), |n| n.to_string()),
        );
    }
    facts
}

fn identification_facts(slot: &IdentificationSlot) -> Vec<(String, String)> {
    let mut facts = Vec::new();
    fact(&mut facts, "Status", slot.status.as_str());
    fact(&mut facts, "Identified mass", num(slot.identified_mass));
    fact(&mut facts, "Unidentified mass", num(slot.unidentified_mass));
    fact(&mut facts, "Unevaluable mass", num(slot.unevaluable_mass));
    fact(&mut facts, "Incomplete-search mass", num(slot.incomplete_search_mass));
    fact(&mut facts, "Mass covers the full class", yes_no(slot.full_mass_scope));
    fact(&mut facts, "Weight basis", opt_str(slot.weight_basis.as_deref()));
    fact(&mut facts, "Search capped", yes_no(slot.search_capped));
    facts
}

fn unavailable_text(what: &str, reason: &str) -> Block {
    Block::Text(format!("{what} is unavailable: {reason}."))
}

// --- sections -------------------------------------------------------------

impl Report {
    fn of(input: &ReportInput<'_>) -> Self {
        let names = Names(input.names);
        let mut sections = vec![summary_section(input)];
        if let Some(claim) = input.claim {
            sections.push(claim_section(claim));
        }
        sections.push(identification_section(input));
        sections.push(assumptions_section(input));
        sections.push(diagnostics_section(input));
        sections.push(uncertainty_section(input, names));
        sections.push(support_section(input));
        sections.push(limits_section(input));
        if let Some(plan) = input.plan {
            sections.push(plan_section(plan, names));
        }
        sections.push(identities_section(input));
        Self { title: "Causal analysis report", sections }
    }

    fn markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# ");
        out.push_str(&md_escape(self.title));
        out.push_str("\n\n");
        for section in &self.sections {
            out.push_str("## ");
            out.push_str(&md_escape(section.title));
            out.push_str("\n\n");
            for block in &section.blocks {
                md_block(&mut out, block);
            }
        }
        out
    }

    fn html(&self) -> String {
        let mut out = String::new();
        out.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        out.push_str(
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; \
             style-src 'unsafe-inline'\">\n",
        );
        out.push_str("<title>");
        out.push_str(&html_escape(self.title));
        out.push_str("</title>\n<style>\n");
        out.push_str(HTML_STYLE);
        out.push_str("</style>\n</head>\n<body>\n<main>\n<h1>");
        out.push_str(&html_escape(self.title));
        out.push_str("</h1>\n");
        for section in &self.sections {
            out.push_str("<section id=\"");
            out.push_str(&html_escape(&slug(section.title)));
            out.push_str("\">\n<h2>");
            out.push_str(&html_escape(section.title));
            out.push_str("</h2>\n");
            for block in &section.blocks {
                html_block(&mut out, block);
            }
            out.push_str("</section>\n");
        }
        out.push_str("</main>\n</body>\n</html>\n");
        out
    }
}

const HTML_STYLE: &str = "\
:root { color-scheme: light dark; --fg: #1b1f24; --bg: #ffffff; --rule: #c9d1d9; --muted: #57606a; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6edf3; --bg: #0d1117; --rule: #30363d; --muted: #8b949e; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 64rem; margin: 0 auto; padding: 1rem; }
h1, h2, h3 { line-height: 1.25; }
h2 { border-bottom: 1px solid var(--rule); padding-bottom: 0.25rem; margin-top: 2rem; }
dl.facts { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; }
dl.facts dt { font-weight: 600; color: var(--muted); }
dl.facts dd { margin: 0; overflow-wrap: anywhere; }
table { border-collapse: collapse; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: 0.25rem 0.5rem; text-align: left; vertical-align: top; }
";

fn md_block(out: &mut String, block: &Block) {
    match block {
        Block::Facts(facts) => {
            for (key, value) in facts {
                out.push_str("- **");
                out.push_str(&md_escape(key));
                out.push_str(":** ");
                out.push_str(&md_escape(value));
                out.push('\n');
            }
            out.push('\n');
        }
        Block::Text(text) => {
            out.push_str(&md_escape(text));
            out.push_str("\n\n");
        }
        Block::Bullets(items) => {
            for item in items {
                out.push_str("- ");
                out.push_str(&md_escape(item));
                out.push('\n');
            }
            out.push('\n');
        }
        Block::Table { head, rows } => {
            out.push('|');
            for cell in *head {
                out.push(' ');
                out.push_str(&md_escape(cell));
                out.push_str(" |");
            }
            out.push_str("\n|");
            for _ in *head {
                out.push_str(" --- |");
            }
            out.push('\n');
            for row in rows {
                out.push('|');
                for cell in row {
                    out.push(' ');
                    out.push_str(&md_escape(cell));
                    out.push_str(" |");
                }
                out.push('\n');
            }
            out.push('\n');
        }
        Block::Subheading(title) => {
            out.push_str("### ");
            out.push_str(&md_escape(title));
            out.push_str("\n\n");
        }
    }
}

fn html_block(out: &mut String, block: &Block) {
    match block {
        Block::Facts(facts) => {
            out.push_str("<dl class=\"facts\">\n");
            for (key, value) in facts {
                out.push_str("<dt>");
                out.push_str(&html_escape(key));
                out.push_str("</dt><dd>");
                out.push_str(&html_escape(value));
                out.push_str("</dd>\n");
            }
            out.push_str("</dl>\n");
        }
        Block::Text(text) => {
            out.push_str("<p>");
            out.push_str(&html_escape(text));
            out.push_str("</p>\n");
        }
        Block::Bullets(items) => {
            out.push_str("<ul>\n");
            for item in items {
                out.push_str("<li>");
                out.push_str(&html_escape(item));
                out.push_str("</li>\n");
            }
            out.push_str("</ul>\n");
        }
        Block::Table { head, rows } => {
            out.push_str("<table>\n<thead><tr>");
            for cell in *head {
                out.push_str("<th>");
                out.push_str(&html_escape(cell));
                out.push_str("</th>");
            }
            out.push_str("</tr></thead>\n<tbody>\n");
            for row in rows {
                out.push_str("<tr>");
                for cell in row {
                    out.push_str("<td>");
                    out.push_str(&html_escape(cell));
                    out.push_str("</td>");
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</tbody>\n</table>\n");
        }
        Block::Subheading(title) => {
            out.push_str("<h3>");
            out.push_str(&html_escape(title));
            out.push_str("</h3>\n");
        }
    }
}

/// The reasoning slots the report reads: the claim's when one is attached.
fn reasoning<'a>(input: &'a ReportInput<'_>) -> &'a ReasoningView {
    input.claim.map_or(&input.contract.reasoning, |claim| &claim.reasoning)
}

fn summary_section(input: &ReportInput<'_>) -> Section {
    let contract = input.contract;
    let mut facts = Vec::new();
    fact(
        &mut facts,
        "Reading",
        if input.claim.is_some() { "an executed claim" } else { "a contract before execution" },
    );
    fact(&mut facts, "Query", contract.query_kind.to_string());
    fact(&mut facts, "Inference", contract.inference.to_string());
    fact(&mut facts, "Identifier", opt_str(contract.identifier.as_deref()));
    fact(
        &mut facts,
        "Estimator",
        opt_str(contract.resolved_estimator.as_deref().or(contract.estimator.as_deref())),
    );
    if let Some(plan) = input.plan {
        fact(&mut facts, "Route", plan.route().as_str());
    }
    if let Some(claim) = input.claim {
        fact(&mut facts, "Claim kind", claim.kind.as_str());
        if !claim.kind.is_causal() {
            fact(&mut facts, "Causal reading", "noncausal: identification is not claimed");
        }
        if let Some(value) = claim.value {
            fact(&mut facts, "Value", num(value));
        }
    }
    fact(
        &mut facts,
        "Support status",
        contract.support_status.map_or("off_axis", CellStatus::as_str),
    );

    let mut readings = Vec::new();
    match input.claim {
        None => readings.push(
            "No claim is attached: this report reads the contract before any execution, and \
             no estimate is stated."
                .to_owned(),
        ),
        Some(claim) => match claim.kind {
            ClaimKind::Refusal => {
                readings.push("The analysis refused: it publishes no estimate.".to_owned())
            }
            ClaimKind::Incomplete => readings.push(
                "The analysis is incomplete: it publishes no estimate while its obligations \
                 remain."
                    .to_owned(),
            ),
            ClaimKind::Associational => readings.push(
                "The claim is a noncausal association: identification is not claimed, so it is \
                 not an effect of any intervention and ranks no action."
                    .to_owned(),
            ),
            kind => readings.push(format!(
                "The claim publishes a value of kind {}; the sections below state what it rests on \
                 and where it stops.",
                kind.as_str()
            )),
        },
    }
    readings.push(
        match contract.support_status {
            Some(CellStatus::Licensed) => "The route is a licensed support-matrix cell.",
            Some(CellStatus::Allowlisted { .. }) => {
                "The route runs without a licence (a retained compatibility status)."
            }
            Some(CellStatus::NotApplicable { .. }) => {
                "The combination is typed impossible: the support matrix marks it not applicable."
            }
            Some(CellStatus::Refused) => {
                "The cell is in the product but is not licensed, and the matrix refuses it."
            }
            None => "The query is off the support-matrix axis; no matrix licence applies to it.",
        }
        .to_owned(),
    );
    if let Some(claim) = input.claim {
        if claim.calibration.status.as_ref() != "calibrated" {
            readings.push(format!(
                "Calibration status is {}: the reported uncertainty carries no measured-coverage \
                 record.",
                claim.calibration.status
            ));
        }
    }
    if let SlotAvailability::Available(slot) = &reasoning(input).identification {
        readings.push(format!("Identification status is {}.", slot.status.as_str()));
    }
    Section { title: "Summary", blocks: vec![Block::Facts(facts), Block::Bullets(readings)] }
}

fn claim_section(claim: &ClaimEnvelope) -> Section {
    let mut facts = Vec::new();
    fact(&mut facts, "Claim id", claim.claim_id.to_hex());
    fact(&mut facts, "Kind", claim.kind.as_str());
    fact(&mut facts, "Value", opt_num(claim.value));
    fact(&mut facts, "Outcome units", opt_str(claim.outcome_units.as_deref()));
    fact(&mut facts, "Identification domain", claim.domains.identification.as_str());
    fact(&mut facts, "Support domain", claim.domains.support.as_str());
    fact(&mut facts, "Evaluated domain", claim.domains.evaluated.as_str());
    fact(
        &mut facts,
        "Execution",
        claim.execution.map_or_else(|| "none".to_owned(), |digest| digest.to_hex()),
    );
    Section { title: "Claim", blocks: vec![Block::Facts(facts)] }
}

fn identification_section(input: &ReportInput<'_>) -> Section {
    let contract = input.contract;
    let mut program = Vec::new();
    fact(&mut program, "Graph class", contract.graph_class.as_str());
    fact(&mut program, "Structure source", contract.structure_source.as_str());
    fact(&mut program, "Identifier", opt_str(contract.identifier.as_deref()));
    fact(&mut program, "Estimator", opt_str(contract.estimator.as_deref()));
    fact(&mut program, "Resolved estimator", opt_str(contract.resolved_estimator.as_deref()));
    fact(&mut program, "Discovery algorithm", opt_str(contract.discovery_algorithm.as_deref()));
    let mut blocks = vec![Block::Facts(program)];
    match &reasoning(input).identification {
        SlotAvailability::Available(slot) => blocks.push(Block::Facts(identification_facts(slot))),
        SlotAvailability::Unavailable { reason } if &**reason == IDENTIFICATION_NOT_CLAIMED => {
            blocks.push(Block::Text(
                "Identification is not claimed: the result is a noncausal association of the \
                 observed data, so there is nothing to identify and it is not read as the effect \
                 of an intervention."
                    .to_owned(),
            ));
        }
        SlotAvailability::Unavailable { reason } => {
            blocks.push(unavailable_text("The identification slot", reason));
        }
        _ => blocks.push(unavailable_text("The identification slot", "unrecognised availability")),
    }
    Section { title: "Identification", blocks }
}

fn assumptions_section(input: &ReportInput<'_>) -> Section {
    let mut blocks = Vec::new();
    match &reasoning(input).assumptions {
        SlotAvailability::Available(slot) => {
            let obligations = &slot.obligations;
            let unresolved = obligations.iter().filter(|record| record.is_unresolved()).count();
            blocks.push(Block::Text(format!(
                "{} obligations are recorded; {unresolved} remain unresolved.",
                obligations.len()
            )));
            if !obligations.is_empty() {
                blocks.push(Block::Table {
                    head: OBLIGATION_HEAD,
                    rows: obligations.iter().map(obligation_row).collect(),
                });
            }
        }
        SlotAvailability::Unavailable { reason } => {
            blocks.push(unavailable_text("The assumption slot", reason));
        }
        _ => blocks.push(unavailable_text("The assumption slot", "unrecognised availability")),
    }
    if let Some(plan) = input.plan {
        let assumptions = plan.identification().assumptions();
        blocks.push(Block::Subheading("Assumptions the identification relies on".to_owned()));
        if assumptions.is_empty() {
            blocks.push(Block::Text("The plan records none.".to_owned()));
        } else {
            blocks.push(Block::Table {
                head: &["Assumption", "Status"],
                rows: assumptions
                    .iter()
                    .map(|assumption| {
                        vec![assumption.label().to_owned(), assumption.status().as_str().to_owned()]
                    })
                    .collect(),
            });
        }
    }
    Section { title: "Assumptions", blocks }
}

fn diagnostics_section(input: &ReportInput<'_>) -> Section {
    let mut blocks = Vec::new();
    match input.claim {
        None => blocks.push(Block::Text(
            "No claim is attached, so no execution evidence or diagnostics are recorded."
                .to_owned(),
        )),
        Some(claim) => {
            if claim.evidence.is_empty() {
                blocks.push(Block::Text("The claim references no evidence.".to_owned()));
            } else {
                blocks.push(Block::Subheading("Evidence the claim references".to_owned()));
                blocks
                    .push(Block::Bullets(claim.evidence.iter().map(ToString::to_string).collect()));
            }
            if !claim.attested.is_empty() {
                blocks.push(Block::Subheading("Attested evidence".to_owned()));
                blocks.push(Block::Text(
                    "Attested evidence is supplied by the caller and cannot be re-verified here."
                        .to_owned(),
                ));
                blocks.push(Block::Table {
                    head: &[
                        "Name",
                        "Kind",
                        "Passed",
                        "Informative",
                        "Refuted ATE",
                        "Comparison",
                        "Failure condition",
                        "Re-verifiable",
                        "Config digest",
                        "Payload digest",
                    ],
                    rows: claim.attested.iter().map(attested_row).collect(),
                });
            }
        }
    }
    let failed: Vec<String> = match &reasoning(input).assumptions {
        SlotAvailability::Available(slot) => slot
            .obligations
            .iter()
            .filter(|record| record.kind.is_unresolved())
            .map(|record| {
                format!("{} ({}): {}", record.id, record.kind.as_str(), record.description)
            })
            .collect(),
        _ => Vec::new(),
    };
    if !failed.is_empty() {
        blocks.push(Block::Subheading("Checks that failed or have not run".to_owned()));
        blocks.push(Block::Bullets(failed));
    }
    if let Some(plan) = input.plan {
        let unresolved = plan.identification().unresolved_obligations();
        if !unresolved.is_empty() {
            blocks.push(Block::Subheading("Obligations identification left unresolved".to_owned()));
            blocks.push(Block::Table {
                head: &["Code", "Detail"],
                rows: unresolved
                    .iter()
                    .map(|obligation| {
                        vec![obligation.code().to_owned(), obligation.detail().to_owned()]
                    })
                    .collect(),
            });
        }
    }
    Section { title: "Diagnostics and evidence", blocks }
}

fn uncertainty_section(input: &ReportInput<'_>, names: Names<'_>) -> Section {
    let mut blocks = Vec::new();
    match &reasoning(input).uncertainty {
        SlotAvailability::Available(slot) => {
            let components = &slot.components;
            if components.is_empty() {
                blocks.push(Block::Text("The slot declares no components.".to_owned()));
            } else {
                blocks.push(Block::Table {
                    head: &["Source", "Reported target", "Omitted"],
                    rows: components
                        .iter()
                        .map(|component| {
                            vec![
                                component.source.as_str().to_owned(),
                                component.target.to_string(),
                                yes_no(component.omitted).to_owned(),
                            ]
                        })
                        .collect(),
                });
            }
        }
        SlotAvailability::Unavailable { reason } => {
            blocks.push(unavailable_text("The uncertainty slot", reason));
        }
        _ => blocks.push(unavailable_text("The uncertainty slot", "unrecognised availability")),
    }
    if let Some(plan) = input.plan {
        blocks.push(Block::Subheading("Uncertainty the program must publish".to_owned()));
        blocks.push(facts_of("", plan.facts().get("uncertainty"), names));
    }
    blocks.push(Block::Subheading("Calibration".to_owned()));
    match input.claim {
        None => blocks
            .push(Block::Text("Calibration is read from a claim; none is attached.".to_owned())),
        Some(claim) => {
            blocks.push(Block::Facts(calibration_facts(&claim.calibration)));
            if !claim.calibration.secondary.is_empty() {
                blocks.push(Block::Text(
                    "Further intervals reported beside the primary one carry their own slots."
                        .to_owned(),
                ));
                blocks.push(Block::Table {
                    head: &[
                        "Status",
                        "Governing record",
                        "Reason",
                        "Nominal level",
                        "Observed coverage",
                    ],
                    rows: claim
                        .calibration
                        .secondary
                        .iter()
                        .map(|slot| {
                            vec![
                                slot.status.to_string(),
                                opt_str(slot.record_id.as_deref()).into_owned(),
                                opt_str(slot.reason.as_deref()).into_owned(),
                                opt_num(slot.nominal).into_owned(),
                                opt_num(slot.observed).into_owned(),
                            ]
                        })
                        .collect(),
                });
            }
        }
    }
    Section { title: "Uncertainty", blocks }
}

fn support_section(input: &ReportInput<'_>) -> Section {
    let contract = input.contract;
    let mut facts = Vec::new();
    fact(
        &mut facts,
        "Support status",
        contract.support_status.map_or("off_axis", CellStatus::as_str),
    );
    fact(&mut facts, "Modality", contract.modality.to_string());
    fact(&mut facts, "Inference", contract.inference.to_string());
    fact(&mut facts, "Query kind", contract.query_kind.to_string());
    fact(&mut facts, "Rows", contract.row_count.to_string());
    fact(&mut facts, "Functional", contract.functional.to_string());
    if !contract.posterior.is_empty() {
        fact(&mut facts, "Posterior construction", contract.posterior.to_string());
    }
    if let Some(draws) = contract.posterior_draws {
        fact(&mut facts, "Posterior draws requested", draws.to_string());
    }
    let mut blocks = vec![Block::Facts(facts)];
    match &reasoning(input).support {
        SlotAvailability::Available(slot) => {
            let mut facts = Vec::new();
            fact(&mut facts, "Matrix status", slot.matrix_status.to_string());
            fact(&mut facts, "Matrix coordinate", opt_str(slot.matrix_coordinate.as_deref()));
            fact(&mut facts, "Empirical support", slot.empirical.label(ToString::to_string));
            blocks.push(Block::Subheading("Support slot".to_owned()));
            blocks.push(Block::Facts(facts));
        }
        SlotAvailability::Unavailable { reason } => {
            blocks.push(unavailable_text("The support slot", reason));
        }
        _ => blocks.push(unavailable_text("The support slot", "unrecognised availability")),
    }
    if let Some(plan) = input.plan {
        let support = plan.support();
        let mut cell = Vec::new();
        fact(&mut cell, "Coordinate", opt_str(support.coordinate()));
        fact(&mut cell, "Licensed", yes_no(support.licensed()));
        fact(&mut cell, "Status", opt_str(support.status().map(CellStatus::as_str)));
        blocks.push(Block::Subheading("Support cell of the program".to_owned()));
        blocks.push(Block::Facts(cell));
    }
    Section { title: "Support and licence", blocks }
}

/// The mixture over structures a plan carries, when it carries one.
fn mixture_blocks(plan: &PlanView, blocks: &mut Vec<Block>) {
    if let Some(mixture) = plan.mixture() {
        blocks.push(Block::Subheading("Mixture over structures".to_owned()));
        let mut facts = Vec::new();
        fact(&mut facts, "Atom kind", opt_str(mixture.atom_kind()));
        fact(&mut facts, "Total weight", num(mixture.total_weight()));
        fact(&mut facts, "Identified mass", opt_num(mixture.identified_mass()));
        fact(&mut facts, "Unidentified mass", opt_num(mixture.unidentified_mass()));
        fact(&mut facts, "Identification retained", yes_no(mixture.identification_retained()));
        blocks.push(Block::Facts(facts));
        if !mixture.atoms().is_empty() {
            blocks.push(Block::Table {
                head: &["Atom", "Weight", "Samples", "Identified", "Status", "Completions"],
                rows: mixture
                    .atoms()
                    .iter()
                    .map(|atom| {
                        vec![
                            atom.key().to_string(),
                            num(atom.weight()),
                            atom.samples().to_string(),
                            atom.identified()
                                .map_or_else(|| "none".to_owned(), |flag| yes_no(flag).to_owned()),
                            opt_str(
                                atom.status().map(antecedent_core::IdentificationStatus::as_str),
                            ).into_owned(),
                            atom.completions().to_string(),
                        ]
                    })
                    .collect(),
            });
        }
    }
}

/// The table of the identification products a plan retains.
fn retained_blocks(plan: &PlanView, names: Names<'_>, blocks: &mut Vec<Block>) {
    let identification = plan.identification();
    if !identification.retained().is_empty() {
        blocks.push(Block::Table {
            head: &[
                "Retained product",
                "Status",
                "Estimands",
                "Completions",
                "Truncated",
                "Atoms",
                "Identified mass",
                "Unidentified mass",
            ],
            rows: identification
                .retained()
                .iter()
                .map(|retained| {
                    vec![
                        retained.kind().to_owned(),
                        opt_str(
                            retained.status().map(antecedent_core::IdentificationStatus::as_str),
                        ).into_owned(),
                        retained
                            .estimands()
                            .iter()
                            .map(|estimand| {
                                format!(
                                    "{} [{}]",
                                    estimand.method(),
                                    names.list(estimand.adjustment_set())
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("; "),
                        retained.completions().to_string(),
                        retained.truncated_completions().to_string(),
                        retained.atoms().to_string(),
                        opt_num(retained.identified_mass()).into_owned(),
                        opt_num(retained.unidentified_mass()).into_owned(),
                    ]
                })
                .collect(),
        });
    }
}

/// The limits the claim, the identification and the plan record.
fn recorded_limits(input: &ReportInput<'_>) -> Vec<String> {
    let mut limits: Vec<String> = Vec::new();
    if let Some(claim) = input.claim {
        match claim.kind {
            ClaimKind::Refusal => limits.push("The claim is a refusal.".to_owned()),
            ClaimKind::Incomplete => limits.push("The claim is incomplete.".to_owned()),
            ClaimKind::Associational => limits.push(
                "The claim is a noncausal association: identification is not claimed, so it is \
                 not an effect of any intervention."
                    .to_owned(),
            ),
            _ => {}
        }
        if claim.value.is_none()
            && !matches!(claim.kind, ClaimKind::Refusal | ClaimKind::Incomplete)
        {
            limits.push("The claim states no scalar value.".to_owned());
        }
    }
    if let SlotAvailability::Available(slot) = &reasoning(input).identification {
        if slot.search_capped {
            limits.push("The identification search was capped before a determination.".to_owned());
        }
        if !slot.full_mass_scope {
            limits.push("The reported mass does not cover the full structure class.".to_owned());
        }
        if slot.unidentified_mass > 0.0 {
            limits.push(format!(
                "Mass {} of the structure class is not identified.",
                num(slot.unidentified_mass)
            ));
        }
        if slot.unevaluable_mass > 0.0 {
            limits.push(format!(
                "Mass {} is identified but not evaluable by the selected estimator.",
                num(slot.unevaluable_mass)
            ));
        }
        if slot.incomplete_search_mass > 0.0 {
            limits.push(format!(
                "Mass {} was left out of a capped search.",
                num(slot.incomplete_search_mass)
            ));
        }
    }
    if let SlotAvailability::Available(slot) = &reasoning(input).uncertainty {
        for component in slot.components.iter().filter(|component| component.omitted) {
            limits.push(format!(
                "Uncertainty from {} ({}) is omitted.",
                component.source.as_str(),
                component.target
            ));
        }
    }
    if let Some(claim) = input.claim {
        if claim.calibration.status.as_ref() != "calibrated" {
            limits.push(format!(
                "Calibration is {}{}.",
                claim.calibration.status,
                claim
                    .calibration
                    .reason
                    .as_deref()
                    .map_or_else(String::new, |reason| format!(" ({reason})"))
            ));
        }
    }
    if let Some(plan) = input.plan {
        let identification = plan.identification();
        if let Some(reason) = identification.reason() {
            limits.push(format!(
                "Identification is {} for this program: {reason}.",
                identification.kind().as_str()
            ));
        }
        if let Some(model) = identification.non_graphical() {
            limits.push(format!("Identification is not graphical: {model}."));
        }
        if identification.unresolved_directions() > 0 {
            limits.push(format!(
                "{} directions are left unresolved by the recovery.",
                identification.unresolved_directions()
            ));
        }
        for retained in identification.retained() {
            if retained.truncated_completions() > 0 {
                limits.push(format!(
                    "{} completions of the {} product were truncated.",
                    retained.truncated_completions(),
                    retained.kind()
                ));
            }
        }
    }
    limits
}

fn capability_head(input: &ReportInput<'_>) -> Vec<(String, String)> {
    let capability = input.contract.capability();
    let mut head = Vec::new();
    fact(&mut head, "Operation", capability.operation.as_str());
    fact(&mut head, "Applicability", capability.applicability.as_str());
    fact(
        &mut head,
        "Readiness",
        opt_str(capability.readiness.map(antecedent_contract::OperationReadiness::as_str)),
    );
    fact(&mut head, "Coordinate", opt_str(capability.coordinate.as_deref()));
    head
}

fn limits_section(input: &ReportInput<'_>) -> Section {
    let capability = input.contract.capability();
    let mut blocks = vec![Block::Facts(capability_head(input))];

    let limits = recorded_limits(input);
    if limits.is_empty() {
        blocks.push(Block::Text("No limit is recorded beyond the blockers below.".to_owned()));
    } else {
        blocks.push(Block::Subheading("Limits".to_owned()));
        blocks.push(Block::Bullets(limits));
    }

    blocks.push(Block::Subheading("Blockers and refusals".to_owned()));
    if capability.blockers.is_empty() {
        blocks.push(Block::Text("The operation report records no blocker.".to_owned()));
    } else {
        blocks.push(Block::Table {
            head: &["Id", "Scientific impossibility", "Reason"],
            rows: capability
                .blockers
                .iter()
                .map(|blocker| {
                    vec![
                        blocker.id.to_string(),
                        yes_no(blocker.scientific).to_owned(),
                        blocker.reason.to_string(),
                    ]
                })
                .collect(),
        });
    }
    if !capability.unmet_obligations.is_empty() {
        blocks.push(Block::Subheading("Unmet obligations".to_owned()));
        blocks.push(Block::Table {
            head: OBLIGATION_HEAD,
            rows: capability.unmet_obligations.iter().map(obligation_row).collect(),
        });
    }
    if !capability.next_actions.is_empty() {
        blocks.push(Block::Subheading("Next actions".to_owned()));
        blocks.push(Block::Text(
            "Next actions are suggestions for the caller; none is taken automatically.".to_owned(),
        ));
        blocks.push(Block::Table {
            head: &["Action", "Description"],
            rows: capability
                .next_actions
                .iter()
                .map(|action| vec![action.id.to_string(), action.description.to_string()])
                .collect(),
        });
    }
    if !capability.neighbors.is_empty() {
        blocks.push(Block::Subheading("Licensed neighbours".to_owned()));
        blocks.push(Block::Text(
            "A neighbour changes the stated premises; it is not a fallback.".to_owned(),
        ));
        blocks.push(Block::Table {
            head: &["Coordinate", "Changed premises", "Required action"],
            rows: capability
                .neighbors
                .iter()
                .map(|neighbor| {
                    vec![
                        neighbor.coordinate.to_string(),
                        neighbor
                            .changed
                            .iter()
                            .map(|change| {
                                format!("{}: {} to {}", change.axis, change.from, change.to)
                            })
                            .collect::<Vec<_>>()
                            .join("; "),
                        neighbor.required_action.to_string(),
                    ]
                })
                .collect(),
        });
    }
    let layers = |layers: &[SemanticLayer]| {
        if layers.is_empty() {
            "none".to_owned()
        } else {
            layers.iter().map(|layer| layer.as_str()).collect::<Vec<_>>().join(", ")
        }
    };
    blocks.push(Block::Facts(vec![
        ("Layers the premises preserve".to_owned(), layers(&capability.preserved_layers[..])),
        ("Layers the premises invalidate".to_owned(), layers(&capability.invalidated_layers[..])),
    ]));
    blocks.push(Block::Text(
        "This report restates the contract. It computes no estimate, grants no licence, and a \
         status recorded here is the status the records carry."
            .to_owned(),
    ));
    Section { title: "Refusals and limits", blocks }
}

fn plan_section(plan: &PlanView, names: Names<'_>) -> Section {
    let facts = plan.facts();
    let mut head = Vec::new();
    fact(&mut head, "Route", plan.route().as_str());
    fact(&mut head, "Family", plan.family().as_str());
    fact(&mut head, "Stage", plan.stage().as_str());
    fact(&mut head, "Depth", plan.depth().as_str());
    fact(&mut head, "Program identity", plan.identity().to_hex());
    fact(&mut head, "Sealed operation", opt_str(plan.operation()));
    if let Some(family) = facts.get("cyclic_family").as_text() {
        fact(&mut head, "Cyclic family", family);
    }
    if let Some(design) = facts.get("planned_design").as_text() {
        fact(&mut head, "Planned design", design);
    }
    let mut blocks = vec![Block::Facts(head)];

    blocks.push(Block::Subheading("Scientific problem".to_owned()));
    blocks.push(facts_of("", facts.get("scientific"), names));

    let identification = plan.identification();
    blocks.push(Block::Subheading("Identification product".to_owned()));
    let mut product = Vec::new();
    fact(&mut product, "Carried", identification.kind().as_str());
    fact(
        &mut product,
        "Status",
        opt_str(identification.status().map(antecedent_core::IdentificationStatus::as_str)),
    );
    fact(&mut product, "Installed at prepare", yes_no(identification.installed_at_prepare()));
    let product_facts = facts.get("identification").get("product");
    if let Some(view) = identification.product() {
        fact(&mut product, "Product identity", view.identity().to_hex());
        fact(
            &mut product,
            "Outcome",
            opt_str(product_facts.get("outcome").as_text()),
        );
        fact(
            &mut product,
            "Claim class",
            opt_str(product_facts.get("claim_class").as_text()),
        );
    }
    blocks.push(Block::Facts(product));
    if let Some(view) = identification.product() {
        if !view.estimands().is_empty() {
            blocks.push(Block::Table {
                head: &["Estimand", "Adjustment set", "Instruments", "Mediators"],
                rows: view
                    .estimands()
                    .iter()
                    .map(|estimand| {
                        vec![
                            estimand.method().to_owned(),
                            names.list(estimand.adjustment_set()),
                            names.list(estimand.instruments()),
                            names.list(estimand.mediators()),
                        ]
                    })
                    .collect(),
            });
        }
        if !view.required_factors().is_empty() {
            blocks.push(Block::Table {
                head: &["Required factor", "Variables"],
                rows: view
                    .required_factors()
                    .iter()
                    .map(|(name, variables)| vec![name.to_string(), names.list(variables)])
                    .collect(),
            });
        }
    }
    retained_blocks(plan, names, &mut blocks);

    blocks.push(Block::Subheading("Statistical program".to_owned()));
    blocks.push(facts_of("", facts.get("statistical"), names));
    blocks.push(Block::Subheading("Evidence and decision program".to_owned()));
    blocks.push(facts_of("", facts.get("evidence"), names));

    blocks.push(Block::Subheading("Physical plan".to_owned()));
    let mut physical = facts.get("physical").clone();
    // The sized record and the kernel policy describe this machine, not the program.
    physical.remove("record");
    physical.remove("kernel_policy");
    blocks.push(facts_of("", &physical, names));

    mixture_blocks(plan, &mut blocks);
    Section { title: "Program plan", blocks }
}

fn identities_section(input: &ReportInput<'_>) -> Section {
    let identities = input.claim.map_or(&input.contract.identities, |claim| &claim.identities);
    let hex = |digest: Option<antecedent_core::SemanticDigest>| {
        digest.map_or_else(|| "none".to_owned(), |digest| digest.to_hex())
    };
    let mut facts = Vec::new();
    fact(&mut facts, "Target", identities.target.to_hex());
    fact(&mut facts, "Identification", identities.identification.to_hex());
    fact(&mut facts, "Identification product", hex(identities.identification_product));
    fact(&mut facts, "Program", hex(identities.program));
    fact(&mut facts, "Inference binding", identities.inference_binding.to_hex());
    fact(&mut facts, "Observation", identities.observation.to_hex());
    fact(&mut facts, "Data snapshot", identities.data_snapshot.to_hex());
    fact(&mut facts, "Score reuse", hex(input.contract.score_reuse));
    fact(&mut facts, "Target weights", hex(input.contract.target_weights));
    Section { title: "Identities", blocks: vec![Block::Facts(facts)] }
}

#[cfg(test)]
mod tests {
    use super::{clean, html_escape, md_escape, num, slug};

    #[test]
    fn html_escapes_every_markup_character_and_controls() {
        assert_eq!(
            html_escape("<a href=\"x\">&'</a>"),
            "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;"
        );
        assert_eq!(html_escape("a\nb\u{0}c\r"), "a b c ");
        assert_eq!(html_escape("plain text 1.5"), "plain text 1.5");
    }

    #[test]
    fn markdown_escapes_every_markup_character() {
        assert_eq!(md_escape("a|b"), "a\\|b");
        assert_eq!(md_escape("<b>*_`[x]~&#"), "\\<b\\>\\*\\_\\`\\[x\\]\\~\\&\\#");
        assert_eq!(md_escape("back\\slash"), "back\\\\slash");
        assert_eq!(md_escape("line\none"), "line one");
        assert_eq!(clean("tab\tend"), "tab end");
    }

    #[test]
    fn numbers_print_the_value_the_input_holds() {
        assert_eq!(num(2.0), "2");
        assert_eq!(num(2.000_000_000_000_000_4), "2");
        assert_eq!(num(0.5), "0.5");
        assert_eq!(num(-0.0), "0");
        assert_eq!(num(0.1 + 0.2), "0.3");
        assert_eq!(num(1_234.567_890_123_456), "1234.56789012");
        assert_eq!(num(f64::NAN), "NaN");
        assert_eq!(num(f64::INFINITY), "inf");
        assert_eq!(num(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn section_anchors_are_plain_ascii() {
        assert_eq!(slug("Refusals and limits"), "refusals-and-limits");
    }
}
