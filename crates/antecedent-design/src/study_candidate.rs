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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use antecedent_core::{
    DistributionAvailability, EvidenceKind, EvidenceOffer, EvidenceRegime, InterventionAssignment,
    MAX_OBLIGATION_COORDINATES, RegimeId, RegimeKind, VariableId, reason_code,
};

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
}

impl StudyKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Experiment => "experiment",
            Self::Observation => "observation",
            Self::SampleIncrease => "sample_increase",
        }
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
            detail: "study_candidate.wrong_contract",
            message: message.into(),
        }
    }
}

fn blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn distinct(variables: &[VariableId]) -> bool {
    variables.iter().collect::<BTreeSet<_>>().len() == variables.len()
}

fn set(variables: &[VariableId]) -> BTreeSet<VariableId> {
    variables.iter().copied().collect()
}

fn id_list(variables: &[VariableId]) -> String {
    let mut ids: Vec<u32> = variables.iter().map(|v| v.raw()).collect();
    ids.sort_unstable();
    ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

impl ExpectedEvidence {
    /// Canonical, order-independent content string.
    #[must_use]
    pub fn canonical(&self) -> String {
        let mut levels: Vec<String> = self
            .intervention_values
            .iter()
            .map(|a| format!("{}={:?}", a.variable.raw(), a.value))
            .collect();
        levels.sort_unstable();
        let distribution = match &self.distribution {
            DistributionAvailability::Joint => "joint".to_owned(),
            DistributionAvailability::SeparateMarginals { variables } => {
                format!("marginals[{}]", id_list(variables))
            }
        };
        format!(
            "pop={};do=[{}];levels=[{}];cond=[{}];vars=[{}];dist={}",
            self.population,
            id_list(&self.interventions),
            levels.join("|"),
            id_list(&self.conditioned_on),
            id_list(&self.measured),
            distribution
        )
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
        if blank(&self.label)
            || blank(&self.population)
            || blank(&self.recruitment)
            || blank(&self.timing)
            || blank(&self.unit_rules.unit)
            || self.unit_rules.cluster.as_deref().is_some_and(blank)
            || self.external_provider.as_deref().is_some_and(blank)
        {
            return Err(wrong(
                "a candidate needs a label, population, recruitment, timing and unit rules",
            ));
        }
        if self.cost.units == 0 || blank(&self.cost.unit_label) || self.sample_size == 0 {
            return Err(wrong(
                "a candidate needs a positive cost with a unit label and a positive sample size",
            ));
        }
        if self.feasibility_notes.iter().any(|note| blank(note)) {
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
            StudyKind::Experiment | StudyKind::Observation | StudyKind::SampleIncrease => {}
        }
        if self.kind != StudyKind::SampleIncrease && self.expected_evidence.is_empty() {
            return Err(wrong("a study declares the evidence it would deliver"));
        }
        self.expected_evidence.iter().try_for_each(|evidence| evidence.check_against(self))
    }

    /// Order-independent semantic identity (`sc1:<digest>`); the label is not
    /// part of it.
    #[must_use]
    pub fn semantic_id(&self) -> Arc<str> {
        let mut notes: Vec<&str> = self.feasibility_notes.iter().map(AsRef::as_ref).collect();
        notes.sort_unstable();
        let mut evidence: Vec<String> =
            self.expected_evidence.iter().map(ExpectedEvidence::canonical).collect();
        evidence.sort_unstable();
        let content = format!(
            "kind={};pop={};do=[{}];measured=[{}];joint={};n={};recruit={};timing={};unit={};\
             cluster={};whole={};cost={}/{}/{};feasible={};notes=[{}];provider={};evidence=[{}]",
            self.kind.as_str(),
            self.population,
            id_list(&self.interventions),
            id_list(&self.measured),
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
        );
        let digest = blake3::hash(content.as_bytes()).to_hex();
        Arc::from(format!("sc1:{}", &digest.as_str()[..32]))
    }

    /// What the study could supply, as screens an obligation can test with
    /// [`antecedent_core::EvidenceObligation::addressed_by`]. A sample increase
    /// offers its rows over the declared design.
    #[must_use]
    pub fn offers(&self) -> Vec<EvidenceOffer> {
        if self.kind == StudyKind::SampleIncrease {
            return vec![EvidenceOffer {
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
            StudyKind::Observation => CandidateDesign::Measure(MeasurementPlan {
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
