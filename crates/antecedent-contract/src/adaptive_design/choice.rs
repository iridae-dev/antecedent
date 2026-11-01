//! The analysis contract of a choice study: respondents making repeated categorical choices among
//! a declared option set, evidence retained from earlier studies, interim looks over respondents
//! and a stopping rule.
//!
//! The assigned-arm contract ([`super::AdaptiveDesignContract`]) states a policy that allocates
//! arms to units. A choice study has no such policy: every respondent faces the same declared
//! choice set and the outcome is the option chosen, so the elements that contract requires
//! (assignment probabilities, exposure) do not exist for it and are not invented. What a choice
//! posterior needs to be a licensed statement is different, and each element is its own typed
//! refusal here:
//!
//! 1. what a choice means, and the population the respondents are drawn from;
//! 2. the ordered option set, which parameterizes every count and every prior;
//! 3. the row-level identity of every observation, so the evidence ledger can refuse a retained
//!    source that rests on rows the respondents contribute (double counting) or on rows another
//!    source contributes;
//! 4. for every retained source, its option set, meaning and population against the target's,
//!    its discount and, when its population differs, the declared reason it transfers;
//! 5. the looks over respondents and the stopping rule, whose kind and parameters must agree with
//!    the procedure cell.
//!
//! The matrix ([`super::procedure_status`]) is the one authority on what is licensed for the
//! cell, through the same gate as the assigned-arm contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{CanonicalWriter, SemanticDigest};

use super::declaration::{
    RouteAuthorization, StoppingRuleDeclaration, StoppingSpec, authorize_cell,
};
use super::matrix::{
    InferenceParadigm, MonitoringStrategy, OutcomeStructure, ProcedureCell, StoppingRuleKind,
};
use crate::ExternalRefusal;
use crate::evidence_ledger::{
    EvidenceLedger, ObservationId, Overlap, SourceDeclaration, SourceId, SourceKind,
};

/// Version word of the canonical choice-contract encoding.
pub const CHOICE_DESIGN_VERSION: u64 = 1;
/// Most options a choice contract declares (the bound of the Dirichlet family).
pub const MAX_CHOICE_CONTRACT_OPTIONS: usize = 64;
/// Most retained sources a prior may rest on.
pub const MAX_CHOICE_SOURCES: usize = 16;
/// Most respondents a contract declares.
pub const MAX_CHOICE_RESPONDENTS: usize = 1_000_000;
/// Most choices one respondent contributes.
pub const MAX_CHOICES_PER_RESPONDENT: u32 = 100_000;

const STAGE_DECLARE: &str = "declare";
const STAGE_BIND: &str = "bind";
const ROW_DOMAIN: &str = "antecedent.choice_design.row";

/// One respondent and the choices they made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChoiceRespondent {
    /// Respondent identity; distinct within a contract. The rows are `{id}#0`, `{id}#1`, ...
    pub id: String,
    /// Choices of each option, in the declared option order.
    pub counts: Vec<u32>,
}

/// Evidence retained from an earlier study, as the counts it contributed.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedChoiceSource {
    /// Source label; distinct within a contract.
    pub label: String,
    /// The source's ordered options.
    pub option_labels: Vec<String>,
    /// What a choice meant in the source.
    pub meaning: String,
    /// The population the source describes.
    pub population: String,
    /// Choices of each option in the source.
    pub counts: Vec<f64>,
    /// Row-level identities of the observations behind the counts.
    pub observation_ids: Vec<String>,
    /// Fraction of the source's information used, in `[0, 1]`.
    pub discount: f64,
    /// Why a source from another population transfers; required exactly when it differs.
    pub transfer_rationale: Option<String>,
}

/// One interim analysis over respondents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChoiceLook {
    /// Respondents enrolled when the analysis was performed (a prefix of the enrolment order).
    pub respondents: u32,
}

/// Everything a producer states about one choice study.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceDesignDeclaration {
    /// Human label; part of the identity.
    pub label: String,
    /// The procedure cell the study ran under.
    pub cell: ProcedureCell,
    /// The ordered option set.
    pub option_labels: Vec<String>,
    /// What a choice means.
    pub meaning: String,
    /// The population the respondents are drawn from.
    pub population: String,
    /// Respondents in enrolment order.
    pub respondents: Vec<ChoiceRespondent>,
    /// Dirichlet shapes before any retained source, one per option.
    pub initial_prior: Vec<f64>,
    /// Evidence retained from earlier studies.
    pub retained_sources: Vec<RetainedChoiceSource>,
    /// The interim analyses in order; the last is the final analysis.
    pub looks: Option<Vec<ChoiceLook>>,
    /// The stopping rule.
    pub stopping_rule: Option<StoppingRuleDeclaration>,
}

/// A contract that was refused: one typed refusal per defect.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceContractError {
    /// The refusals, in the order the defects were found.
    pub refusals: Vec<ExternalRefusal>,
}

impl ChoiceContractError {
    /// The namespaced detail of each refusal.
    #[must_use]
    pub fn details(&self) -> Vec<&str> {
        self.refusals.iter().map(|r| r.detail.as_str()).collect()
    }
}

impl std::fmt::Display for ChoiceContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "choice design contract refused: {}", self.details().join(", "))
    }
}

impl std::error::Error for ChoiceContractError {}

fn refusal(code: &'static str, detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(code, STAGE_DECLARE, detail).with_remedy(remedy)
}

fn invalid(detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    refusal(antecedent_core::reason_code!("choice_evidence_invalid"), detail, remedy)
}

fn scope(detail: &'static str, remedy: &'static str) -> ExternalRefusal {
    refusal(antecedent_core::reason_code!("choice_evidence_scope_mismatch"), detail, remedy)
}

fn distinct_labels(labels: &[String]) -> bool {
    let set: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
    set.len() == labels.len() && labels.iter().all(|l| !l.trim().is_empty())
}

/// The rows a respondent contributes, as the strings the sources declare theirs in.
fn respondent_rows(respondent: &ChoiceRespondent) -> impl Iterator<Item = String> + '_ {
    let total: u64 = respondent.counts.iter().map(|c| u64::from(*c)).sum();
    (0..total).map(|k| format!("{}#{k}", respondent.id))
}

fn row_id(row: &str) -> ObservationId {
    ObservationId::from_key(ROW_DOMAIN, row.as_bytes())
}

fn check_frame(d: &ChoiceDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let options = d.option_labels.len();
    if !(2..=MAX_CHOICE_CONTRACT_OPTIONS).contains(&options) || !distinct_labels(&d.option_labels) {
        refusals.push(invalid(
            "choice_inference.options_invalid",
            "declare 2 to 64 distinct, nonempty option labels",
        ));
    }
    if d.meaning.trim().is_empty() || d.population.trim().is_empty() {
        refusals.push(invalid(
            "choice_inference.frame_incomplete",
            "state what a choice means and the population the respondents are drawn from",
        ));
    }
    if !matches!(
        d.cell.outcome,
        OutcomeStructure::CategoricalChoice | OutcomeStructure::RepeatedChoice
    ) {
        refusals.push(invalid(
            "choice_inference.cell_not_choice",
            "a choice contract is declared for a categorical-choice or repeated-choice outcome",
        ));
    }
    if d.initial_prior.len() != options
        || d.initial_prior.iter().any(|a| !(a.is_finite() && *a > 0.0))
    {
        refusals.push(invalid(
            "choice_inference.prior_invalid",
            "declare one positive Dirichlet shape per option",
        ));
    }
}

fn check_respondents(d: &ChoiceDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let options = d.option_labels.len();
    let ids: BTreeSet<&str> = d.respondents.iter().map(|r| r.id.as_str()).collect();
    if d.respondents.is_empty()
        || d.respondents.len() > MAX_CHOICE_RESPONDENTS
        || ids.len() != d.respondents.len()
        || d.respondents.iter().any(|r| {
            r.id.trim().is_empty()
                || r.id.contains('#')
                || r.counts.len() != options
                || r.counts.iter().map(|c| u64::from(*c)).sum::<u64>()
                    > u64::from(MAX_CHOICES_PER_RESPONDENT)
        })
    {
        refusals.push(invalid(
            "choice_inference.respondents_invalid",
            "declare distinct respondent identities without `#`, one count per option, at most 100000 choices each",
        ));
        return;
    }
    if d.cell.outcome == OutcomeStructure::RepeatedChoice
        && !d.respondents.iter().any(|r| r.counts.iter().map(|c| u64::from(*c)).sum::<u64>() >= 2)
    {
        refusals.push(invalid(
            "choice_inference.heterogeneity_not_identified",
            "respondent-level heterogeneity is identified only by respondents who choose more than once; no respondent does",
        ));
    }
}

fn check_sources(d: &ChoiceDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    if d.retained_sources.len() > MAX_CHOICE_SOURCES {
        refusals.push(invalid(
            "choice_inference.too_many_sources",
            "retain at most 16 sources in a prior",
        ));
        return;
    }
    let labels: Vec<String> = d.retained_sources.iter().map(|s| s.label.clone()).collect();
    if !distinct_labels(&labels) {
        refusals.push(invalid(
            "choice_inference.source_label_invalid",
            "label each retained source distinctly",
        ));
    }
    for s in &d.retained_sources {
        if s.option_labels != d.option_labels || s.counts.len() != d.option_labels.len() {
            refusals.push(scope(
                "choice_inference.source_parameterization_mismatch",
                "a source is used only for the same ordered options; re-express the source in the target's options before declaring it",
            ));
        }
        if s.meaning != d.meaning {
            refusals.push(scope(
                "choice_inference.source_meaning_mismatch",
                "a source whose choices mean something else (stated preference, behaviour, another task) does not inform this posterior",
            ));
        }
        let transferred = s.population != d.population;
        let rationale = s.transfer_rationale.as_deref().is_some_and(|r| !r.trim().is_empty());
        if transferred && !rationale {
            refusals.push(scope(
                "choice_inference.source_population_mismatch",
                "declare why a source from another population transfers, or leave it out; no transfer is assumed",
            ));
        }
        if !transferred && s.transfer_rationale.is_some() {
            refusals.push(invalid(
                "choice_inference.transfer_rationale_unneeded",
                "a transfer rationale is declared only for a source from another population",
            ));
        }
        if !(s.discount.is_finite() && (0.0..=1.0).contains(&s.discount))
            || s.counts.iter().any(|c| !(c.is_finite() && *c >= 0.0))
            || s.observation_ids.iter().any(|id| id.trim().is_empty())
            || (s.observation_ids.is_empty() && s.counts.iter().any(|c| *c > 0.0))
        {
            refusals.push(invalid(
                "choice_inference.source_invalid",
                "a source has a discount in [0, 1], nonnegative counts and row-level identities for any counts",
            ));
        }
    }
}

fn check_monitoring_and_rule(d: &ChoiceDesignDeclaration, refusals: &mut Vec<ExternalRefusal>) {
    let units = d.respondents.len();
    match &d.looks {
        None => refusals.push(refusal(
            antecedent_core::reason_code!("adaptive_contract_incomplete"),
            "choice_inference.looks_missing",
            "retain the analyses performed: respondents enrolled at each look",
        )),
        Some(looks) => {
            let ordered = !looks.is_empty()
                && looks.windows(2).all(|w| w[0].respondents < w[1].respondents)
                && looks.first().is_some_and(|l| l.respondents > 0)
                && looks.last().is_some_and(|l| l.respondents as usize == units);
            if !ordered {
                refusals.push(invalid(
                    "choice_inference.looks_not_ordered",
                    "looks must increase strictly in respondents and the last must include every respondent",
                ));
            } else if (looks.len() == 1)
                != (d.cell.monitoring == MonitoringStrategy::FinalAnalysisOnly)
            {
                refusals.push(invalid(
                    "choice_inference.rule_mismatch",
                    "the number of looks disagrees with the declared monitoring strategy",
                ));
            }
        }
    }
    let Some(rule) = &d.stopping_rule else {
        refusals.push(refusal(
            antecedent_core::reason_code!("adaptive_contract_incomplete"),
            "choice_inference.stopping_rule_missing",
            "retain the stopping rule, its parameters and whether it ended the study",
        ));
        return;
    };
    let spec_ok = match (rule.kind, rule.spec) {
        (StoppingRuleKind::NoStopping, StoppingSpec::NoStopping) => true,
        (StoppingRuleKind::Efficacy, StoppingSpec::PosteriorProbabilityBest { threshold }) => {
            threshold > 0.5 && threshold < 1.0
        }
        (
            StoppingRuleKind::DecisionTheoretic,
            StoppingSpec::PosteriorExpectedLoss { loss_threshold },
        ) => loss_threshold.is_finite() && loss_threshold > 0.0 && loss_threshold < 1.0,
        _ => false,
    };
    if rule.kind != d.cell.stopping || !spec_ok {
        refusals.push(invalid(
            "choice_inference.rule_mismatch",
            "the stopping rule disagrees with the procedure cell: its kind and parameters must agree",
        ));
    }
}

/// The ledger of the observations behind a choice contract: the respondents' rows and each
/// retained source's, with the overlap of every pair resolved by the evidence ledger.
#[derive(Clone, Debug)]
pub struct ChoiceEvidence {
    /// The ledger.
    pub ledger: EvidenceLedger,
    /// The source holding every respondent row.
    pub respondents: SourceId,
    /// The source of each retained source, in declaration order.
    pub sources: Vec<SourceId>,
}

fn account(d: &ChoiceDesignDeclaration) -> Result<ChoiceEvidence, ExternalRefusal> {
    let limit = |detail: &'static str| {
        ExternalRefusal::new(antecedent_core::reason_code!("choice_evidence_invalid"), STAGE_DECLARE, detail)
            .with_remedy("reduce the number of respondents, choices or sources so the evidence ledger can account for them")
    };
    let mut ledger = EvidenceLedger::new();
    let mut rows: Vec<ObservationId> =
        d.respondents.iter().flat_map(respondent_rows).map(|r| row_id(&r)).collect();
    rows.sort_unstable();
    rows.dedup();
    let respondents = ledger
        .add_source(
            SourceDeclaration::new("respondents", SourceKind::ObservationSet)
                .with_observations(rows),
        )
        .map_err(|_| limit("choice_inference.ledger_limit"))?;
    let mut sources = Vec::with_capacity(d.retained_sources.len());
    for s in &d.retained_sources {
        let mut ids: Vec<ObservationId> = s.observation_ids.iter().map(|r| row_id(r)).collect();
        ids.sort_unstable();
        ids.dedup();
        sources.push(
            ledger
                .add_source(
                    SourceDeclaration::new(s.label.clone(), SourceKind::Prior)
                        .with_observations(ids),
                )
                .map_err(|_| limit("choice_inference.ledger_limit"))?,
        );
    }
    Ok(ChoiceEvidence { ledger, respondents, sources })
}

fn check_overlap(evidence: &ChoiceEvidence, refusals: &mut Vec<ExternalRefusal>) {
    let double_counted = |detail: &'static str| {
        ExternalRefusal::new(antecedent_core::reason_code!("evidence_double_counted"), STAGE_DECLARE, detail)
            .with_remedy("remove the shared rows from one of the two, or leave the source out: evidence is counted once")
    };
    for (i, a) in evidence.sources.iter().enumerate() {
        match evidence.ledger.overlap(*a, evidence.respondents) {
            Ok(Overlap::Independent) => {}
            _ => refusals.push(double_counted("choice_inference.source_overlaps_respondents")),
        }
        for b in &evidence.sources[i + 1..] {
            match evidence.ledger.overlap(*a, *b) {
                Ok(Overlap::Independent) => {}
                _ => refusals.push(double_counted("choice_inference.sources_overlap")),
            }
        }
    }
}

/// A validated choice contract.
#[derive(Clone, Debug)]
pub struct ChoiceDesignContract {
    declaration: ChoiceDesignDeclaration,
    identity: SemanticDigest,
    evidence: ChoiceEvidence,
}

impl ChoiceDesignContract {
    /// Validate a declaration.
    ///
    /// # Errors
    /// One refusal per missing element and per inconsistency; nothing is defaulted.
    pub fn new(declaration: ChoiceDesignDeclaration) -> Result<Self, ChoiceContractError> {
        let mut refusals = Vec::new();
        check_frame(&declaration, &mut refusals);
        check_respondents(&declaration, &mut refusals);
        check_sources(&declaration, &mut refusals);
        check_monitoring_and_rule(&declaration, &mut refusals);
        if !refusals.is_empty() {
            return Err(ChoiceContractError { refusals });
        }
        let evidence = match account(&declaration) {
            Ok(evidence) => evidence,
            Err(refused) => return Err(ChoiceContractError { refusals: vec![refused] }),
        };
        check_overlap(&evidence, &mut refusals);
        if !refusals.is_empty() {
            return Err(ChoiceContractError { refusals });
        }
        let identity = identity_of(&declaration);
        Ok(Self { declaration, identity, evidence })
    }

    /// The validated declaration.
    #[must_use]
    pub const fn declaration(&self) -> &ChoiceDesignDeclaration {
        &self.declaration
    }

    /// Canonical identity.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// The procedure cell.
    #[must_use]
    pub const fn cell(&self) -> ProcedureCell {
        self.declaration.cell
    }

    /// The evidence ledger holding the respondents' rows and each retained source's.
    #[must_use]
    pub const fn evidence(&self) -> &ChoiceEvidence {
        &self.evidence
    }

    /// Number of options.
    #[must_use]
    pub fn options(&self) -> usize {
        self.declaration.option_labels.len()
    }

    /// Number of looks.
    #[must_use]
    pub fn looks(&self) -> usize {
        self.declaration.looks.as_ref().map_or(0, Vec::len)
    }

    /// Respondents enrolled at `look`, in enrolment order.
    ///
    /// # Errors
    /// `choice_inference.look_out_of_range` if `look >= self.looks()`.
    #[allow(
        clippy::result_large_err,
        reason = "a structured refusal preserves the requested index"
    )]
    pub fn respondents_at(&self, look: usize) -> Result<&[ChoiceRespondent], ExternalRefusal> {
        let looks = self.declaration.looks.as_ref().expect("validated contract");
        let enrolled = looks
            .get(look)
            .ok_or_else(|| {
                ExternalRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    STAGE_BIND,
                    "choice_inference.look_out_of_range",
                )
                .with_offending(format!("look {look}"))
                .with_remedy("select a look index below the number of looks")
            })?
            .respondents as usize;
        Ok(&self.declaration.respondents[..enrolled])
    }

    /// Authorize an inferential paradigm for this contract's data.
    ///
    /// # Errors
    /// A typed refusal from the matrix: a fixed-sample paradigm on data from an adaptive
    /// design, a closed or carried-forward cell, or a paradigm with no route for the cell.
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    pub fn admit_paradigm(
        &self,
        paradigm: InferenceParadigm,
    ) -> Result<RouteAuthorization, ExternalRefusal> {
        authorize_cell(&self.declaration.cell, paradigm)?;
        Ok(RouteAuthorization {
            cell: self.declaration.cell,
            paradigm,
            contract_identity: self.identity,
        })
    }
}

fn identity_of(d: &ChoiceDesignDeclaration) -> SemanticDigest {
    let mut w = CanonicalWriter::new("antecedent.choice_design.contract");
    w.word(CHOICE_DESIGN_VERSION).text(&d.label);
    w.text(d.cell.assignment.as_str())
        .text(d.cell.outcome.as_str())
        .text(d.cell.monitoring.as_str())
        .text(d.cell.stopping.as_str());
    w.texts(&d.option_labels).text(&d.meaning).text(&d.population);
    w.word(d.respondents.len() as u64);
    for r in &d.respondents {
        w.text(&r.id);
        let counts: Vec<u64> = r.counts.iter().map(|c| u64::from(*c)).collect();
        w.words(&counts);
    }
    w.reals(&d.initial_prior);
    w.word(d.retained_sources.len() as u64);
    for s in &d.retained_sources {
        w.text(&s.label).texts(&s.option_labels).text(&s.meaning).text(&s.population);
        w.reals(&s.counts).texts(&s.observation_ids).real(s.discount);
        match &s.transfer_rationale {
            Some(t) => w.word(1).text(t),
            None => w.word(0),
        };
    }
    match &d.looks {
        Some(looks) => {
            w.word(1).word(looks.len() as u64);
            for l in looks {
                w.word(u64::from(l.respondents));
            }
        }
        None => {
            w.word(0);
        }
    }
    match &d.stopping_rule {
        Some(rule) => {
            w.word(1).text(rule.kind.as_str()).word(u64::from(rule.ended_by_rule));
            match rule.spec {
                StoppingSpec::PosteriorProbabilityBest { threshold } => {
                    w.text("posterior_probability_best").real(threshold);
                }
                StoppingSpec::PosteriorExpectedLoss { loss_threshold } => {
                    w.text("posterior_expected_loss").real(loss_threshold);
                }
                _ => {
                    w.text("no_stopping");
                }
            }
        }
        None => {
            w.word(0);
        }
    }
    SemanticDigest::from_bytes(w.finish())
}
