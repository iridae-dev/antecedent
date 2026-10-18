//! The adaptive-procedure matrix: every combination of assignment policy,
//! outcome structure, monitoring strategy and stopping rule, and what the
//! engine does with it.
//!
//! Each cell is exactly one of:
//!
//! * [`CellStatus::Licensed`]: a design contract, an estimator and an
//!   uncertainty method exist for the cell, per inferential paradigm listed.
//!   A licensed cell still reports point values only: the interval and
//!   guarantee routes stay closed until whole-procedure coverage is measured.
//! * [`CellStatus::Closed`]: refused with a typed reason. Either the cell is
//!   not a coherent procedure, or it pairs components that are each delivered
//!   but whose pairing is not: a guarantee for two components does not license
//!   their pairing.
//! * [`CellStatus::CarriedForward`]: coherent, not delivered, and owned.
//!
//! The matrix is a total function of the four axes; [`procedure_cells`]
//! enumerates the whole product so a cell cannot be left unlisted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

macro_rules! axis_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $( $(#[$vmeta:meta])* $variant:ident => $text:literal ),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[non_exhaustive]
        pub enum $name {
            $( $(#[$vmeta])* $variant, )+
        }

        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )+ ];

            /// Stable `snake_case` name.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $text, )+
                }
            }

            /// Parse a wire name.
            #[must_use]
            pub fn from_name(name: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|value| value.as_str() == name)
            }
        }
    };
}

axis_enum! {
    /// How arms are assigned to arriving units.
    AssignmentFamily {
        /// Probabilities fixed in advance and constant over units.
        FixedRandomization => "fixed_randomization",
        /// Beta-Bernoulli Thompson sampling: arm probabilities are the
        /// posterior probabilities that each arm is best.
        ThompsonBetaBernoulli => "thompson_beta_bernoulli",
        /// Epsilon-greedy over running arm means.
        EpsilonGreedy => "epsilon_greedy",
        /// Upper-confidence-bound assignment.
        UpperConfidenceBound => "upper_confidence_bound",
        /// Assignment that depends on unit covariates.
        ContextualBandit => "contextual_bandit",
        /// Assignment probabilities updated between batches.
        BatchedAdaptive => "batched_adaptive",
    }
}

axis_enum! {
    /// Structure of the outcomes collected per unit.
    OutcomeStructure {
        /// One outcome per unit.
        SingleOutcome => "single_outcome",
        /// Repeated measurements of each assigned unit.
        RepeatedMeasures => "repeated_measures",
        /// Units nested in clusters, assigned by cluster.
        HierarchicalClustered => "hierarchical_clustered",
    }
}

axis_enum! {
    /// When the accumulating data are looked at.
    MonitoringStrategy {
        /// One analysis at the end.
        FinalAnalysisOnly => "final_analysis_only",
        /// A fixed number of interim looks.
        FixedInterimLooks => "fixed_interim_looks",
        /// Pre-specified looks with an alpha-spending function.
        GroupSequential => "group_sequential",
        /// A look after every observation.
        ContinuousMonitoring => "continuous_monitoring",
    }
}

axis_enum! {
    /// What a look can decide.
    StoppingRuleKind {
        /// Nothing: the planned horizon is run.
        NoStopping => "no_stopping",
        /// Stop for efficacy.
        Efficacy => "efficacy",
        /// Stop for futility.
        Futility => "futility",
        /// Stop for exhaustion of a resource.
        Resource => "resource",
        /// Stop when the expected loss of deciding falls below a threshold.
        DecisionTheoretic => "decision_theoretic",
    }
}

axis_enum! {
    /// Which inferential paradigm a result belongs to.
    InferenceParadigm {
        /// Posterior of a declared prior and likelihood.
        BayesianPosterior => "bayesian_posterior",
        /// A stopping decision from posterior expected loss.
        DecisionTheoreticStopping => "decision_theoretic_stopping",
        /// Inference as if the sample size and design were fixed in advance.
        FrequentistFixedSample => "frequentist_fixed_sample",
        /// Asymptotic frequentist inference from adaptively weighted scores.
        FrequentistAdaptiveWeighting => "frequentist_adaptive_weighting",
        /// Frequentist error control over pre-specified looks.
        FrequentistGroupSequential => "frequentist_group_sequential",
        /// Time-uniform frequentist inference from a confidence sequence.
        AnytimeValidFrequentist => "anytime_valid_frequentist",
    }
}

impl InferenceParadigm {
    /// What kind of statement the paradigm makes.
    #[must_use]
    pub const fn statement(self) -> &'static str {
        match self {
            Self::BayesianPosterior => {
                "a posterior under the declared prior, valid under any assignment rule that depends only on the observed history"
            }
            Self::DecisionTheoreticStopping => {
                "a stop-or-continue decision from posterior expected loss under a declared loss"
            }
            Self::FrequentistFixedSample => {
                "a sampling distribution that presumes a design fixed in advance"
            }
            Self::FrequentistAdaptiveWeighting => {
                "an adaptively weighted point estimate whose martingale variance estimate is uncalibrated"
            }
            Self::FrequentistGroupSequential => {
                "efficacy boundaries that spend a declared alpha over pre-specified looks"
            }
            Self::AnytimeValidFrequentist => {
                "a time-uniform confidence sequence for a bounded arm mean, uncalibrated as a whole procedure"
            }
        }
    }
}

/// One cell of the procedure matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ProcedureCell {
    /// How arms are assigned.
    pub assignment: AssignmentFamily,
    /// Structure of the outcomes.
    pub outcome: OutcomeStructure,
    /// When the data are looked at.
    pub monitoring: MonitoringStrategy,
    /// What a look can decide.
    pub stopping: StoppingRuleKind,
}

impl ProcedureCell {
    /// Stable identifier: the four axis names joined by `/`.
    #[must_use]
    pub fn identifier(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.assignment.as_str(),
            self.outcome.as_str(),
            self.monitoring.as_str(),
            self.stopping.as_str()
        )
    }

    /// Whether data from this cell were collected under a design that
    /// adapts to outcomes or stops on them.
    #[must_use]
    pub const fn is_adaptive(&self) -> bool {
        !matches!(self.assignment, AssignmentFamily::FixedRandomization)
            || !matches!(self.monitoring, MonitoringStrategy::FinalAnalysisOnly)
            || !matches!(self.stopping, StoppingRuleKind::NoStopping)
    }
}

/// What the engine does with a cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CellStatus {
    /// An estimator and uncertainty method exist for these paradigms.
    Licensed {
        /// Paradigms with a route for this cell.
        paradigms: &'static [InferenceParadigm],
    },
    /// Refused: the cell is not a coherent procedure.
    ClosedIncoherent {
        /// Why, and what would make it coherent.
        reason: &'static str,
    },
    /// Refused: every component is delivered but the pairing is not.
    ClosedPairing {
        /// Why, and what would license it.
        reason: &'static str,
    },
    /// Coherent but not delivered; the owner is the named work item.
    CarriedForward {
        /// Owning work item.
        owner: &'static str,
        /// What is missing.
        reason: &'static str,
    },
}

impl CellStatus {
    /// Stable `snake_case` class name.
    #[must_use]
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Licensed { .. } => "licensed",
            Self::ClosedIncoherent { .. } => "closed_incoherent",
            Self::ClosedPairing { .. } => "closed_pairing",
            Self::CarriedForward { .. } => "carried_forward",
        }
    }
}

const BAYES_AND_SEQUENCE: &[InferenceParadigm] =
    &[InferenceParadigm::BayesianPosterior, InferenceParadigm::AnytimeValidFrequentist];
const BAYES_AND_DECISION: &[InferenceParadigm] =
    &[InferenceParadigm::BayesianPosterior, InferenceParadigm::DecisionTheoreticStopping];
const BAYES_DECISION_AND_SEQUENCE: &[InferenceParadigm] = &[
    InferenceParadigm::BayesianPosterior,
    InferenceParadigm::DecisionTheoreticStopping,
    InferenceParadigm::AnytimeValidFrequentist,
];
const WEIGHTING_AND_SEQUENCE: &[InferenceParadigm] =
    &[InferenceParadigm::FrequentistAdaptiveWeighting, InferenceParadigm::AnytimeValidFrequentist];
const WEIGHTING_ONLY: &[InferenceParadigm] = &[InferenceParadigm::FrequentistAdaptiveWeighting];
const GROUP_SEQUENTIAL_ONLY: &[InferenceParadigm] =
    &[InferenceParadigm::FrequentistGroupSequential];
const FIXED_SAMPLE_ONLY: &[InferenceParadigm] = &[InferenceParadigm::FrequentistFixedSample];

/// What the engine does with `cell`. Total over the product of the axes.
#[must_use]
pub fn procedure_status(cell: &ProcedureCell) -> CellStatus {
    use AssignmentFamily as A;
    use MonitoringStrategy as M;
    use OutcomeStructure as O;
    use StoppingRuleKind as S;

    // Coherence first.
    if cell.monitoring == M::FinalAnalysisOnly && cell.stopping != S::NoStopping {
        return CellStatus::ClosedIncoherent {
            reason: "a stopping rule needs interim analyses to act on; declare interim looks or remove the rule",
        };
    }
    if cell.monitoring != M::FinalAnalysisOnly && cell.stopping == S::NoStopping {
        return CellStatus::ClosedIncoherent {
            reason: "interim analyses with no stopping rule have no decision to account for; declare the rule or analyze at the final look only",
        };
    }
    if cell.monitoring == M::GroupSequential && cell.stopping == S::DecisionTheoretic {
        return CellStatus::ClosedIncoherent {
            reason: "alpha spending governs error-rate boundaries; a decision-theoretic rule belongs to a different monitoring strategy",
        };
    }
    // Delivered cells, each with its own contract, estimator and method.
    let licensed = match (cell.assignment, cell.outcome, cell.monitoring, cell.stopping) {
        (A::ThompsonBetaBernoulli, O::SingleOutcome, M::FixedInterimLooks, S::Efficacy) => {
            Some(BAYES_AND_SEQUENCE)
        }
        (
            A::ThompsonBetaBernoulli,
            O::SingleOutcome,
            M::FixedInterimLooks,
            S::DecisionTheoretic,
        ) => Some(BAYES_AND_DECISION),
        // Participant-level binary responses (a beta-binomial count per
        // participant), Thompson allocation with a minimum exploration
        // probability, and decisions at completed-participant batch
        // boundaries: expected loss, futility and a recruitment cap, with a
        // posterior claim and an anytime-valid claim read at every look.
        (
            A::ThompsonBetaBernoulli,
            O::RepeatedMeasures,
            M::FixedInterimLooks,
            S::DecisionTheoretic,
        ) => Some(BAYES_DECISION_AND_SEQUENCE),
        (A::EpsilonGreedy, O::SingleOutcome, M::FixedInterimLooks, S::Efficacy) => {
            Some(WEIGHTING_AND_SEQUENCE)
        }
        (A::EpsilonGreedy, O::RepeatedMeasures, M::FinalAnalysisOnly, S::NoStopping) => {
            Some(WEIGHTING_ONLY)
        }
        (A::FixedRandomization, O::SingleOutcome, M::GroupSequential, S::Efficacy) => {
            Some(GROUP_SEQUENTIAL_ONLY)
        }
        (A::FixedRandomization, O::SingleOutcome, M::FinalAnalysisOnly, S::NoStopping) => {
            Some(FIXED_SAMPLE_ONLY)
        }
        _ => None,
    };
    if let Some(paradigms) = licensed {
        return CellStatus::Licensed { paradigms };
    }
    // Components that are not delivered at all, in a fixed priority order.
    let owner = match cell.assignment {
        A::UpperConfidenceBound => Some((
            "adaptive_inference/upper_confidence_bound",
            "no design contract, estimator or uncertainty method for upper-confidence-bound assignment",
        )),
        A::ContextualBandit => Some((
            "adaptive_inference/contextual_assignment",
            "assignment that depends on covariates needs a conditional propensity contract that is not delivered",
        )),
        A::BatchedAdaptive => Some((
            "adaptive_inference/batched_assignment",
            "batch-wise policy updates need a batch-level contract and variance method that are not delivered",
        )),
        _ => None,
    }
    .or(match cell.outcome {
        O::HierarchicalClustered => Some((
            "adaptive_inference/hierarchical_outcome",
            "cluster-level assignment with member outcomes needs a hierarchical variance method that is not delivered",
        )),
        _ => None,
    })
    .or(match cell.monitoring {
        M::ContinuousMonitoring => Some((
            "adaptive_inference/continuous_monitoring",
            "monitoring after every observation needs its own stopping-time analysis that is not delivered",
        )),
        _ => None,
    })
    .or(match cell.stopping {
        S::Futility => Some((
            "adaptive_inference/futility_boundaries",
            "futility boundaries and their effect on error rates are not delivered",
        )),
        S::Resource => Some((
            "adaptive_inference/resource_stopping",
            "stopping on resource exhaustion needs a cost contract that is not delivered",
        )),
        _ => None,
    });
    if let Some((owner, reason)) = owner {
        return CellStatus::CarriedForward { owner, reason };
    }
    CellStatus::ClosedPairing {
        reason: "each component is delivered separately but this pairing has no design contract, estimator or whole-procedure calibration design of its own; a guarantee for two components does not license their pairing",
    }
}

/// Every cell of the matrix, in a fixed order: assignment, then outcome, then
/// monitoring, then stopping.
#[must_use]
pub fn procedure_cells() -> Vec<ProcedureCell> {
    let mut cells = Vec::with_capacity(
        AssignmentFamily::ALL.len()
            * OutcomeStructure::ALL.len()
            * MonitoringStrategy::ALL.len()
            * StoppingRuleKind::ALL.len(),
    );
    for &assignment in AssignmentFamily::ALL {
        for &outcome in OutcomeStructure::ALL {
            for &monitoring in MonitoringStrategy::ALL {
                for &stopping in StoppingRuleKind::ALL {
                    cells.push(ProcedureCell { assignment, outcome, monitoring, stopping });
                }
            }
        }
    }
    cells
}

/// The cells with at least one licensed paradigm, with those paradigms.
#[must_use]
pub fn licensed_routes() -> Vec<(ProcedureCell, &'static [InferenceParadigm])> {
    procedure_cells()
        .into_iter()
        .filter_map(|cell| match procedure_status(&cell) {
            CellStatus::Licensed { paradigms } => Some((cell, paradigms)),
            _ => None,
        })
        .collect()
}
