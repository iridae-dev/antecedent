//! Durable study candidates: what a study could produce.
//!
//! A [`DurableStudyCandidate`] describes a study Antecedent need not conduct: its
//! kind, population, intervention and measurement regime, sample size,
//! recruitment, timing, unit and cluster rules, cost with units, feasibility,
//! the evidence it expects to deliver and an optional external provider. It
//! never claims evidence it cannot collect: a candidate that observes variables
//! in separate regimes or as separate marginals cannot claim the joint law, and
//! a candidate without unit, timing or cost semantics is refused.
//!
//! Its [`semantic_id`](DurableStudyCandidate::semantic_id) is a digest of the
//! canonical content, so reordering variables, levels, notes or expected
//! evidence never changes it and the human label is not part of it. The
//! [`StudyCandidate`](crate::StudyCandidate) of [`crate::study_planner`] is the
//! older, planner-specific declaration and is left untouched.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use antecedent_contract::{EvidenceOffer, MAX_OBLIGATION_COORDINATES};
use antecedent_core::{
    CanonicalEncode, CanonicalWriter, DistributionAvailability, EvidenceKind, EvidenceRegime,
    InterventionAssignment, RegimeId, RegimeKind, Value, VariableId, blake3_hex, is_blank,
    reason_code,
};

use crate::plan_common::id_list;
use crate::{CandidateDesign, DesignCost, ExperimentPlan, MeasurementPlan, SamplingPlan};

/// What the study does.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum StudyKind {
    /// Hard interventions on a declared set, with a measured margin.
    Experiment,
    /// Observation (or added measurement) with no hard intervention.
    Observation,
    /// More rows of an existing design; produces no new regime.
    SampleIncrease,
    /// Observation of an instrument, the treatment and the outcome on one joint
    /// law. The study supplies the data a Wald-type contrast is read from; it
    /// never establishes that the instrument is valid.
    Instrument,
    /// Observation that exploits variation the investigator does not set (a
    /// policy change, a lottery, an administrative cutoff). It sets no hard
    /// intervention, so it never supplies an interventional regime, and its
    /// as-if-random assignment is an assumption no study establishes.
    NaturalExperiment,
    /// Repeated measurements of the same units over a stated window, observed
    /// jointly. Within-unit comparisons remove confounding by causes that do not
    /// change over the window; that time invariance is an assumption.
    Panel,
}

impl StudyKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Experiment => "experiment",
            Self::Observation => "observation",
            Self::SampleIncrease => "sample_increase",
            Self::Instrument => "instrument",
            Self::NaturalExperiment => "natural_experiment",
            Self::Panel => "panel",
        }
    }

    /// Every kind, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::Experiment,
        Self::Observation,
        Self::SampleIncrease,
        Self::Instrument,
        Self::NaturalExperiment,
        Self::Panel,
    ];

    /// The kind named `name` (the [`Self::as_str`] spelling).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    /// Whether the kind sets no hard intervention and delivers observational laws
    /// only.
    #[must_use]
    pub const fn is_observational(self) -> bool {
        !matches!(self, Self::Experiment | Self::SampleIncrease)
    }
}

/// Declared cost with its units; ranking compares only equal unit labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StudyCostDeclaration {
    /// Cost in `unit_label` units, at least one.
    pub units: u64,
    /// What one unit is (a currency, person-days, ...). Never assumed to equal
    /// outcome utility.
    pub unit_label: Arc<str>,
    /// Sample budget consumed; a ranking tie-breaker, never a sufficiency input.
    pub sample_budget: u64,
}

/// Unit-of-analysis and cluster rules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitRules {
    /// The unit one row describes (`patient`, `site`, ...).
    pub unit: Arc<str>,
    /// Cluster identity when units are clustered.
    pub cluster: Option<Arc<str>>,
    /// Whether whole clusters are sampled or resampled together.
    pub whole_cluster_sampling: bool,
}

/// One law the study declares it would deliver.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedEvidence {
    /// Population of the law.
    pub population: Arc<str>,
    /// Hard-intervention set of the law (the study's own set).
    pub interventions: Arc<[VariableId]>,
    /// Concrete levels, empty for the unrestricted intervention domain.
    pub intervention_values: Arc<[InterventionAssignment]>,
    /// Conditioning coordinates already conditioned on.
    pub conditioned_on: Arc<[VariableId]>,
    /// Variables of the law.
    pub measured: Arc<[VariableId]>,
    /// Joint law or separate marginals.
    pub distribution: DistributionAvailability,
}

/// One durable, validated study description.
#[derive(Clone, Debug, PartialEq)]
pub struct DurableStudyCandidate {
    /// Human label; not part of the semantic identity.
    pub label: Arc<str>,
    /// What the study does.
    pub kind: StudyKind,
    /// Population the study is collected in.
    pub population: Arc<str>,
    /// Hard-intervention set of the study; empty unless an experiment.
    pub interventions: Arc<[VariableId]>,
    /// Variables the study measures.
    pub measured: Arc<[VariableId]>,
    /// Whether `measured` is observed jointly on each unit (one joint law).
    pub joint_measurement: bool,
    /// Planned sample size, at least one.
    pub sample_size: u64,
    /// Recruitment and sampling declaration; recorded, never a sufficiency input.
    pub recruitment: Arc<str>,
    /// Timing declaration (horizon, follow-up, delay to delivery).
    pub timing: Arc<str>,
    /// Unit and cluster rules.
    pub unit_rules: UnitRules,
    /// Declared cost and units.
    pub cost: StudyCostDeclaration,
    /// Whether the study is feasible as declared.
    pub feasible: bool,
    /// Feasibility notes.
    pub feasibility_notes: Arc<[Arc<str>]>,
    /// The laws the study declares it would deliver.
    pub expected_evidence: Arc<[ExpectedEvidence]>,
    /// External provider that would run or supply the study, if any.
    pub external_provider: Option<Arc<str>>,
}

/// A refused candidate: a registered reason code and a stable detail.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct StudyCandidateError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `study_candidate.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl StudyCandidateError {
    fn wrong_contract(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("design_signal_invalid"),
            detail: antecedent_core::reason_details::study_candidate::WRONG_CONTRACT,
            message: message.into(),
        }
    }
}

fn distinct(variables: &[VariableId]) -> bool {
    variables.iter().collect::<BTreeSet<_>>().len() == variables.len()
}

fn set(variables: &[VariableId]) -> BTreeSet<VariableId> {
    variables.iter().copied().collect()
}

/// Domain tag of the v2 semantic identity ([`DurableStudyCandidate::semantic_id`]).
const SEMANTIC_ID_V2_TAG: &str = "antecedent.design.study_candidate.semantic_id.v2";
/// Domain tag of one expected law inside the v2 semantic identity.
const EXPECTED_EVIDENCE_V2_TAG: &str = "antecedent.design.study_candidate.expected_evidence.v2";
/// Characters the v1 content string uses as field and list separators.
const V1_SEPARATORS: [char; 4] = [';', '|', '[', ']'];

/// Whether `text` cannot be confused with a v1 separator.
fn v1_plain(text: &str) -> bool {
    !text.contains(&V1_SEPARATORS[..])
}

/// A variable set: ascending, each variable once.
fn variable_set(variables: &[VariableId]) -> Vec<VariableId> {
    set(variables).into_iter().collect()
}

/// A variable set as the v1 comma-joined id list.
fn set_list(variables: &[VariableId]) -> String {
    variable_set(variables).iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(",")
}

fn write_variable_set(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    let variables = variable_set(variables);
    writer.word(variables.len() as u64);
    for variable in variables {
        variable.encode(writer);
    }
}

/// Whether a level's v1 rendering (`{:?}`) is unambiguous: no separator in a label and no NaN,
/// whose payload `{:?}` drops.
fn v1_plain_level(value: &Value) -> bool {
    match value {
        Value::Float64(level) => !level.is_nan(),
        Value::Label(label) => v1_plain(label),
        Value::Int64(_) | Value::Bool(_) | Value::Category(_) => true,
    }
}

impl ExpectedEvidence {
    /// Canonical, order-independent content string (the v1 form, kept for the identities of
    /// earlier releases; [`DurableStudyCandidate::semantic_id`] decides which form it uses).
    #[must_use]
    pub fn canonical(&self) -> String {
        self.v1_content(false)
    }

    /// The v1 content string. With `as_sets`, the variable lists and levels are written as
    /// sets (each element once), the meaning every consumer reads them with.
    fn v1_content(&self, as_sets: bool) -> String {
        let list = |variables: &[VariableId]| {
            if as_sets { set_list(variables) } else { id_list(variables) }
        };
        let mut levels: Vec<String> = self
            .intervention_values
            .iter()
            .map(|a| format!("{}={:?}", a.variable.raw(), a.value))
            .collect();
        levels.sort_unstable();
        if as_sets {
            levels.dedup();
        }
        let distribution = match &self.distribution {
            DistributionAvailability::Joint => "joint".to_owned(),
            DistributionAvailability::SeparateMarginals { variables } => {
                format!("marginals[{}]", list(variables))
            }
        };
        format!(
            "pop={};do=[{}];levels=[{}];cond=[{}];vars=[{}];dist={}",
            self.population,
            list(&self.interventions),
            levels.join("|"),
            list(&self.conditioned_on),
            list(&self.measured),
            distribution
        )
    }

    /// Whether the v1 content string of this law is unambiguous.
    fn v1_plain(&self) -> bool {
        v1_plain(&self.population)
            && self.intervention_values.iter().all(|a| v1_plain_level(&a.value))
    }

    /// The v2 digest of this law: every field framed on the canonical writer, variable lists
    /// and levels as sets, levels by their canonical (bit-exact) encoding.
    fn v2_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new(EXPECTED_EVIDENCE_V2_TAG);
        writer.text(&self.population);
        write_variable_set(&mut writer, &self.interventions);
        let mut levels: Vec<&InterventionAssignment> = self.intervention_values.iter().collect();
        levels.sort_by_key(|a| a.variable);
        levels.dedup();
        writer.word(levels.len() as u64);
        for level in levels {
            level.variable.encode(&mut writer);
            level.value.encode(&mut writer);
        }
        write_variable_set(&mut writer, &self.conditioned_on);
        write_variable_set(&mut writer, &self.measured);
        match &self.distribution {
            DistributionAvailability::Joint => {
                writer.word(0);
            }
            DistributionAvailability::SeparateMarginals { variables } => {
                writer.word(1);
                write_variable_set(&mut writer, variables);
            }
        }
        writer.finish()
    }

    fn check_against(&self, study: &DurableStudyCandidate) -> Result<(), StudyCandidateError> {
        let wrong = StudyCandidateError::wrong_contract;
        if self.population.as_ref() != study.population.as_ref()
            || set(&self.interventions) != set(&study.interventions)
        {
            return Err(wrong(
                "expected evidence must come from the study's own population and intervention set",
            ));
        }
        if self.measured.is_empty()
            || !distinct(&self.measured)
            || !distinct(&self.conditioned_on)
            || self.measured.iter().any(|v| !study.measured.contains(v))
            || self.conditioned_on.iter().any(|v| !self.measured.contains(v))
        {
            return Err(wrong(
                "expected evidence must name a non-empty margin the study measures, with its \
                 conditioning coordinates inside it",
            ));
        }
        let mut assigned = BTreeSet::new();
        if self
            .intervention_values
            .iter()
            .any(|a| !study.interventions.contains(&a.variable) || !assigned.insert(a.variable))
        {
            return Err(wrong("an intervention level names a variable the study does not set"));
        }
        match &self.distribution {
            DistributionAvailability::Joint => {
                if self.measured.len() > 1 && !study.joint_measurement {
                    return Err(wrong(
                        "a study that observes separate regimes or marginals cannot claim the \
                         joint law",
                    ));
                }
            }
            DistributionAvailability::SeparateMarginals { variables } => {
                if variables.iter().any(|v| !self.measured.contains(v)) {
                    return Err(wrong("marginals name a variable the evidence does not measure"));
                }
            }
        }
        Ok(())
    }
}

impl DurableStudyCandidate {
    /// Check the declaration: semantics (unit, timing, cost units, recruitment),
    /// kind consistency and that every expected law is one this study can
    /// produce.
    ///
    /// # Errors
    /// `study_candidate.wrong_contract` (reason `design_signal_invalid`) naming
    /// the first failed rule.
    pub fn validate(&self) -> Result<(), StudyCandidateError> {
        let wrong = StudyCandidateError::wrong_contract;
        if is_blank(&self.label)
            || is_blank(&self.population)
            || is_blank(&self.recruitment)
            || is_blank(&self.timing)
            || is_blank(&self.unit_rules.unit)
            || self.unit_rules.cluster.as_deref().is_some_and(is_blank)
            || self.external_provider.as_deref().is_some_and(is_blank)
        {
            return Err(wrong(
                "a candidate needs a label, population, recruitment, timing and unit rules",
            ));
        }
        if self.cost.units == 0 || is_blank(&self.cost.unit_label) || self.sample_size == 0 {
            return Err(wrong(
                "a candidate needs a positive cost with a unit label and a positive sample size",
            ));
        }
        if self.feasibility_notes.iter().any(|note| is_blank(note)) {
            return Err(wrong("a feasibility note is empty"));
        }
        if self.interventions.len() + self.measured.len() > MAX_OBLIGATION_COORDINATES
            || !distinct(&self.measured)
            || !distinct(&self.interventions)
            || self.measured.iter().any(|v| self.interventions.contains(v))
        {
            return Err(wrong(
                "interventions and measured variables must be distinct, disjoint and bounded",
            ));
        }
        match self.kind {
            StudyKind::Experiment if self.interventions.is_empty() || self.measured.is_empty() => {
                return Err(wrong("an experiment sets interventions and measures a margin"));
            }
            StudyKind::Observation
                if !self.interventions.is_empty() || self.measured.is_empty() =>
            {
                return Err(wrong("an observation sets no intervention and measures a margin"));
            }
            StudyKind::SampleIncrease if !self.expected_evidence.is_empty() => {
                return Err(wrong("a sample increase delivers no new regime"));
            }
            StudyKind::Instrument
                if !self.interventions.is_empty()
                    || self.measured.len() < 3
                    || !self.joint_measurement =>
            {
                return Err(wrong(
                    "an instrument study sets no intervention and jointly measures the \
                     instrument, the treatment and the outcome",
                ));
            }
            StudyKind::NaturalExperiment
                if !self.interventions.is_empty() || self.measured.len() < 2 =>
            {
                return Err(wrong(
                    "a natural experiment sets no intervention (the variation is not the \
                     investigator's) and measures at least the exposure and the outcome",
                ));
            }
            StudyKind::Panel
                if !self.interventions.is_empty()
                    || self.measured.len() < 2
                    || !self.joint_measurement =>
            {
                return Err(wrong(
                    "a panel sets no intervention and jointly measures at least the exposure and \
                     the outcome on the same units",
                ));
            }
            StudyKind::Experiment
            | StudyKind::Observation
            | StudyKind::SampleIncrease
            | StudyKind::Instrument
            | StudyKind::NaturalExperiment
            | StudyKind::Panel => {}
        }
        if self.kind != StudyKind::SampleIncrease && self.expected_evidence.is_empty() {
            return Err(wrong("a study declares the evidence it would deliver"));
        }
        self.expected_evidence.iter().try_for_each(|evidence| evidence.check_against(self))
    }

    /// Order-independent semantic identity; the label is not part of it.
    ///
    /// Variable lists, feasibility notes, levels and expected laws are sets: their order and
    /// repeats do not reach the identity. A candidate whose free text (population,
    /// recruitment, timing, unit rules, cost unit, notes, provider) and levels contain none of
    /// the v1 separators (`;`, `|`, `[`, `]`) keeps the v1 form of earlier releases
    /// (`sc1:<digest>`), whose content string is then unambiguous. Any other candidate uses
    /// the v2 form (`sc2:<digest>`), which frames every field, so free text can never move
    /// content between fields. Stored identities of either form verify through
    /// [`Self::accepts_semantic_id`].
    #[must_use]
    pub fn semantic_id(&self) -> Arc<str> {
        if self.v1_plain() { Self::sc1(&self.v1_content(true)) } else { self.semantic_id_v2() }
    }

    /// The v1 semantic identity exactly as earlier releases computed it (lists sorted, repeats
    /// kept, fields joined unframed). Computed only to verify identities stored under it.
    #[must_use]
    pub fn semantic_id_v1(&self) -> Arc<str> {
        Self::sc1(&self.v1_content(false))
    }

    /// Whether `stored` is this candidate's semantic identity, in the current form or as an
    /// earlier release wrote it.
    #[must_use]
    pub fn accepts_semantic_id(&self, stored: &str) -> bool {
        *self.semantic_id() == *stored || *self.semantic_id_v1() == *stored
    }

    fn sc1(content: &str) -> Arc<str> {
        let digest = blake3_hex(content.as_bytes());
        Arc::from(format!("sc1:{}", &digest[..32]))
    }

    fn v1_plain(&self) -> bool {
        [
            &*self.population,
            &*self.recruitment,
            &*self.timing,
            &*self.unit_rules.unit,
            &*self.cost.unit_label,
        ]
        .into_iter()
        .all(v1_plain)
            // `None` and `Some("")` both write nothing in the v1 form.
            && self.unit_rules.cluster.as_deref().is_none_or(|c| !c.is_empty() && v1_plain(c))
            && self.external_provider.as_deref().is_none_or(|p| !p.is_empty() && v1_plain(p))
            // An empty note joins to nothing: `[""]` and `[]` would share the v1 form.
            && self.feasibility_notes.iter().all(|note| !note.is_empty() && v1_plain(note))
            && self.expected_evidence.iter().all(ExpectedEvidence::v1_plain)
    }

    /// The v1 content string; with `as_sets`, lists, notes and laws are written as sets.
    fn v1_content(&self, as_sets: bool) -> String {
        let list = |variables: &[VariableId]| {
            if as_sets { set_list(variables) } else { id_list(variables) }
        };
        let mut notes: Vec<&str> = self.feasibility_notes.iter().map(AsRef::as_ref).collect();
        notes.sort_unstable();
        let mut evidence: Vec<String> =
            self.expected_evidence.iter().map(|e| e.v1_content(as_sets)).collect();
        evidence.sort_unstable();
        if as_sets {
            notes.dedup();
            evidence.dedup();
        }
        format!(
            "kind={};pop={};do=[{}];measured=[{}];joint={};n={};recruit={};timing={};unit={};\
             cluster={};whole={};cost={}/{}/{};feasible={};notes=[{}];provider={};evidence=[{}]",
            self.kind.as_str(),
            self.population,
            list(&self.interventions),
            list(&self.measured),
            self.joint_measurement,
            self.sample_size,
            self.recruitment,
            self.timing,
            self.unit_rules.unit,
            self.unit_rules.cluster.as_deref().unwrap_or(""),
            self.unit_rules.whole_cluster_sampling,
            self.cost.units,
            self.cost.unit_label,
            self.cost.sample_budget,
            self.feasible,
            notes.join("|"),
            self.external_provider.as_deref().unwrap_or(""),
            evidence.join(" ; "),
        )
    }

    /// The v2 semantic identity: every field framed on the canonical writer.
    fn semantic_id_v2(&self) -> Arc<str> {
        let mut writer = CanonicalWriter::new(SEMANTIC_ID_V2_TAG);
        writer.text(self.kind.as_str()).text(&self.population);
        write_variable_set(&mut writer, &self.interventions);
        write_variable_set(&mut writer, &self.measured);
        writer
            .word(u64::from(self.joint_measurement))
            .word(self.sample_size)
            .text(&self.recruitment)
            .text(&self.timing)
            .text(&self.unit_rules.unit)
            .opt_text(self.unit_rules.cluster.as_deref())
            .word(u64::from(self.unit_rules.whole_cluster_sampling))
            .word(self.cost.units)
            .text(&self.cost.unit_label)
            .word(self.cost.sample_budget)
            .word(u64::from(self.feasible));
        let mut notes: Vec<&str> = self.feasibility_notes.iter().map(AsRef::as_ref).collect();
        notes.sort_unstable();
        notes.dedup();
        writer.texts(&notes).opt_text(self.external_provider.as_deref());
        let mut evidence: Vec<[u8; 32]> =
            self.expected_evidence.iter().map(ExpectedEvidence::v2_digest).collect();
        evidence.sort_unstable();
        evidence.dedup();
        writer.word(evidence.len() as u64);
        for digest in &evidence {
            writer.bytes(digest);
        }
        Arc::from(format!("sc2:{}", &writer.finish_hex()[..32]))
    }

    /// What the study could supply, as screens an obligation can test with
    /// [`antecedent_contract::EvidenceObligation::addressed_by`]. A sample increase
    /// offers its rows over the declared design.
    #[must_use]
    pub fn offers(&self) -> Vec<EvidenceOffer> {
        if self.kind == StudyKind::SampleIncrease {
            return vec![EvidenceOffer {
                quantities: std::collections::BTreeMap::new(),
                population: Arc::clone(&self.population),
                interventions: Arc::clone(&self.interventions),
                conditioned_on: Arc::from([]),
                measured: Arc::clone(&self.measured),
                joint: self.joint_measurement,
                additional_samples: Some(self.sample_size),
            }];
        }
        self.expected_evidence
            .iter()
            .map(|evidence| EvidenceOffer {
                quantities: std::collections::BTreeMap::new(),
                population: Arc::clone(&evidence.population),
                interventions: Arc::clone(&evidence.interventions),
                conditioned_on: Arc::clone(&evidence.conditioned_on),
                measured: Arc::clone(&evidence.measured),
                joint: matches!(evidence.distribution, DistributionAvailability::Joint),
                additional_samples: Some(self.sample_size),
            })
            .collect()
    }

    /// The proposed regimes this study would deliver, with consecutive ids from
    /// `first_regime_id`, labelled `<semantic id>#<k>` and tied to the semantic
    /// id as their study. They are `Proposed`, never available evidence.
    ///
    /// # Errors
    /// An invalid candidate, or an id overflow.
    pub fn proposed_regimes(
        &self,
        first_regime_id: u32,
    ) -> Result<Vec<EvidenceRegime>, StudyCandidateError> {
        self.validate()?;
        let semantic = self.semantic_id();
        let mut next = first_regime_id;
        let mut regimes = Vec::with_capacity(self.expected_evidence.len());
        for (k, evidence) in self.expected_evidence.iter().enumerate() {
            let kind = if evidence.interventions.is_empty() {
                RegimeKind::Observational
            } else {
                RegimeKind::Experimental
            };
            let mut regime = EvidenceRegime::try_new(
                RegimeId::from_raw(next),
                kind,
                EvidenceKind::Proposed,
                Arc::clone(&evidence.interventions),
                Arc::clone(&evidence.intervention_values),
                Arc::clone(&evidence.measured),
                Arc::clone(&evidence.population),
                evidence.distribution.clone(),
            )
            .map_err(|e| StudyCandidateError::wrong_contract(e.to_string()))?;
            regime.conditioned_on = Arc::clone(&evidence.conditioned_on);
            regime.label = Some(Arc::from(format!("{semantic}#{k}")));
            regime.study = Some(Arc::clone(&semantic));
            regimes.push(regime);
            next = next.checked_add(1).ok_or_else(|| {
                StudyCandidateError::wrong_contract("regime ids overflow".to_owned())
            })?;
        }
        Ok(regimes)
    }

    /// The matching planner action of [`crate::candidate`], so a durable study
    /// can feed the existing specialized planners unchanged.
    #[must_use]
    pub fn design_action(&self) -> CandidateDesign {
        let cost =
            DesignCost { amount: self.cost.units as f64, sample_budget: self.cost.sample_budget };
        match self.kind {
            StudyKind::Experiment => CandidateDesign::Intervene(ExperimentPlan {
                targets: Arc::clone(&self.interventions),
                cost,
                tag: 0,
            }),
            StudyKind::Observation
            | StudyKind::Instrument
            | StudyKind::NaturalExperiment
            | StudyKind::Panel => CandidateDesign::Measure(MeasurementPlan {
                variables: Arc::clone(&self.measured),
                cost,
                tag: 0,
            }),
            StudyKind::SampleIncrease => CandidateDesign::IncreaseSamplingRate(SamplingPlan {
                additional_samples: self.sample_size,
                cost,
                tag: 0,
            }),
        }
    }
}

#[cfg(test)]
mod semantic_id_tests {
    use super::*;

    fn variable(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn candidate(notes: &[&str], recruitment: &str) -> DurableStudyCandidate {
        DurableStudyCandidate {
            label: Arc::from("label"),
            kind: StudyKind::Observation,
            population: Arc::from("target"),
            interventions: Arc::from([]),
            measured: Arc::from([variable(0), variable(1)]),
            joint_measurement: true,
            sample_size: 100,
            recruitment: Arc::from(recruitment),
            timing: Arc::from("now"),
            unit_rules: UnitRules {
                unit: Arc::from("patient"),
                cluster: None,
                whole_cluster_sampling: false,
            },
            cost: StudyCostDeclaration {
                units: 1,
                unit_label: Arc::from("usd"),
                sample_budget: 100,
            },
            feasible: true,
            feasibility_notes: notes.iter().map(|note| Arc::from(*note)).collect(),
            expected_evidence: Arc::from([ExpectedEvidence {
                population: Arc::from("target"),
                interventions: Arc::from([]),
                intervention_values: Arc::from([]),
                conditioned_on: Arc::from([]),
                measured: Arc::from([variable(1), variable(0)]),
                distribution: DistributionAvailability::Joint,
            }]),
            external_provider: None,
        }
    }

    /// A plain candidate keeps the v1 identity of earlier releases; a repeated note is the
    /// same set of notes.
    #[test]
    fn plain_candidates_keep_the_v1_identity_and_notes_are_a_set() {
        let plain = candidate(&["b", "a"], "mail");
        assert_eq!(plain.semantic_id(), plain.semantic_id_v1());
        assert!(plain.semantic_id().starts_with("sc1:"));
        let reordered = candidate(&["a", "b"], "mail");
        assert_eq!(plain.semantic_id(), reordered.semantic_id());
        let repeated = candidate(&["a", "b", "a"], "mail");
        assert_eq!(plain.semantic_id(), repeated.semantic_id());
        assert_ne!(repeated.semantic_id_v1(), repeated.semantic_id());
        assert!(repeated.accepts_semantic_id(&repeated.semantic_id_v1()));
        assert_ne!(plain.semantic_id(), candidate(&["a"], "mail").semantic_id());
    }

    /// The v1 notes join is unframed: notes `["a|b"]` and `["a", "b"]` shared a v1 identity.
    /// The v2 form keeps them apart.
    #[test]
    fn separator_in_free_text_no_longer_collides() {
        let joined = candidate(&["a|b"], "mail");
        let split = candidate(&["a", "b"], "mail");
        assert_eq!(joined.semantic_id_v1(), split.semantic_id_v1());
        assert_ne!(joined.semantic_id(), split.semantic_id());
        assert!(joined.semantic_id().starts_with("sc2:"));
        assert!(joined.accepts_semantic_id(&joined.semantic_id_v1()));
        assert!(!joined.accepts_semantic_id(&candidate(&["c"], "mail").semantic_id()));
    }
}
