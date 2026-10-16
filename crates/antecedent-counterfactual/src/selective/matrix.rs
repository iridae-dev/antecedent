//! The theorem-scoped capability matrix of cross-world execution.
//!
//! Each row names one quantity class, model class, conditioning, intervention
//! kind, population, history and evidence combination, and whether it is
//! executed here, keeps its own theorem-scoped route, or is closed. A request
//! outside every executed row is a **scientific** refusal that names the
//! unmet dimension and the declaration that would open it; it is never a cache
//! miss. The same rows are checked-in registry data in
//! `parity/cross_world_capability.toml`, and a test fails if the two differ.
//!
//! The matrix does not decide whether draws may be paired: a row only records
//! that a coupling is required. The pairing decision is the one joint-law rule
//! (`antecedent_contract::check_propagation`), applied by the executor.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::transported_gate::TransportedCounterfactualPrerequisites;

/// The class of cross-world quantity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QuantityClass {
    /// A counterfactual probability or mean contrast within the worlds' own
    /// marginals (no cross-world event).
    CounterfactualMarginal,
    /// A probability of an event over several worlds (needs the joint law).
    CounterfactualJoint,
    /// A natural or path-specific contrast (a world reads a mediator from
    /// another world).
    NaturalPathSpecific,
    /// A fixed-population temporal counterfactual over named action histories.
    Temporal,
    /// A cross-world quantity in a target population other than the one the
    /// mechanisms were declared for.
    TransportedCrossWorld,
}

impl QuantityClass {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CounterfactualMarginal => "counterfactual_marginal",
            Self::CounterfactualJoint => "counterfactual_joint",
            Self::NaturalPathSpecific => "natural_path_specific",
            Self::Temporal => "temporal",
            Self::TransportedCrossWorld => "transported_cross_world",
        }
    }
}

/// The structural-model class.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelClass {
    /// A static Markovian SCM over finite variables with finite independent
    /// exogenous terms.
    FiniteMarkovianScm,
    /// A fully declared static SCM over finite variables in which an exogenous
    /// term feeds several endogenous variables (a shared term). Every mechanism
    /// and every exogenous law is declared, so the counterfactual is enumerated.
    FiniteSharedExogenousScm,
    /// The fixed two-slice additive-noise linear-Gaussian temporal SCM.
    AdditiveLinearTwoSlice,
    /// The static additive linear SCM of the transported path-specific route.
    AdditiveLinearStaticTransported,
    /// Latent confounding whose mechanisms and exogenous law are not declared
    /// (they would have to be identified or fitted from observational data).
    SemiMarkovian,
    /// A nonparametric model over an ADMG with no declared mechanisms.
    NonparametricAdmg,
}

impl ModelClass {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::FiniteMarkovianScm => "finite_markovian_scm",
            Self::FiniteSharedExogenousScm => "finite_shared_exogenous_scm",
            Self::AdditiveLinearTwoSlice => "additive_linear_two_slice",
            Self::AdditiveLinearStaticTransported => "additive_linear_static_transported",
            Self::SemiMarkovian => "semi_markovian",
            Self::NonparametricAdmg => "nonparametric_admg",
        }
    }
}

/// Whether the quantity conditions on factual evidence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ConditioningKind {
    /// No factual evidence: the prior counterfactual.
    NoEvidence,
    /// Per-unit factual evidence (abduction).
    FactualEvidence,
}

/// The kind of intervention defining a world.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InterventionKind {
    /// Hard `do(V = v)`.
    Hard,
    /// A stochastic, soft or policy intervention.
    Soft,
}

/// Population scope.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PopulationKind {
    /// The population the mechanisms were declared for.
    Single,
    /// A different target population.
    Transported,
}

/// History scope.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HistoryKind {
    /// A static model: no history.
    NoHistory,
    /// One fixed-population two-slice history.
    FixedPopulationTwoSlice,
    /// Factual evidence gathered under a history or regime other than the one
    /// the mechanisms were declared for.
    Incompatible,
}

/// Where the mechanisms come from.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvidenceKind {
    /// Mechanisms declared by the caller (not fitted here).
    DeclaredMechanisms,
    /// A fitted mechanism identity supplied with its coefficients.
    SuppliedFittedIdentity,
    /// Data from which the mechanisms would be fitted by this route.
    DataToFit,
}

/// The coupling a row requires between worlds.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CouplingKind {
    /// No cross-world event: the marginals of each world suffice.
    NotRequired,
    /// One shared abduced exogenous term per unit, read by every world.
    SharedAbducedExogenous,
}

impl ConditioningKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoEvidence => "no_evidence",
            Self::FactualEvidence => "factual_evidence",
        }
    }
}

impl PopulationKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Transported => "transported",
        }
    }
}

impl HistoryKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoHistory => "no_history",
            Self::FixedPopulationTwoSlice => "fixed_population_two_slice",
            Self::Incompatible => "incompatible",
        }
    }
}

impl EvidenceKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DeclaredMechanisms => "declared_mechanisms",
            Self::SuppliedFittedIdentity => "supplied_fitted_identity",
            Self::DataToFit => "data_to_fit",
        }
    }
}

impl CouplingKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::SharedAbducedExogenous => "shared_abduced_exogenous",
        }
    }
}

/// What a row is.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CapabilityStatus {
    /// Executed by cross-world selective execution.
    Executed,
    /// Executed by its own existing theorem-scoped route; selective execution
    /// does not run it and does not widen its boundary.
    ExistingTheoremRoute,
    /// Closed with a typed refusal and the prerequisites that would open it.
    Closed,
}

impl CapabilityStatus {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Executed => "executed",
            Self::ExistingTheoremRoute => "existing_theorem_route",
            Self::Closed => "closed",
        }
    }
}

/// One row of the matrix.
#[derive(Clone, Copy, Debug)]
pub struct MatrixRow {
    /// Stable row identity.
    pub id: &'static str,
    /// Quantity class.
    pub quantity: QuantityClass,
    /// Model class.
    pub model: ModelClass,
    /// Allowed conditioning.
    pub conditioning: &'static [ConditioningKind],
    /// Allowed history.
    pub history: &'static [HistoryKind],
    /// Allowed evidence source.
    pub evidence: &'static [EvidenceKind],
    /// Population scope.
    pub population: PopulationKind,
    /// Coupling the row requires.
    pub coupling: CouplingKind,
    /// Status.
    pub status: CapabilityStatus,
    /// The route that executes the row, or the refusal's subject for a closed row.
    pub route: &'static str,
    /// The theorem or assumption set the row is scoped to.
    pub theorem: &'static str,
    /// Reason code of the refusal a closed row returns.
    pub refusal_code: Option<&'static str>,
    /// Detail of the refusal a closed row returns.
    pub refusal_detail: Option<&'static str>,
    /// What would open the row.
    pub remedy: &'static str,
}

impl MatrixRow {
    /// Whether the row needs one shared exogenous term across worlds.
    #[must_use]
    pub const fn needs_shared_exogenous(&self) -> bool {
        matches!(self.coupling, CouplingKind::SharedAbducedExogenous)
    }
}

const FINITE_THEOREM: &str = "Markovian SCM over finite variables with independent finite \
exogenous terms and declared mechanisms: the unit-level counterfactual joint law over worlds is \
the push-forward of the exogenous posterior given the factual evidence (abduction-action-\
prediction; Pearl 2009 ch. 7, Balke and Pearl 1994), enumerated exactly. Model adequacy is a \
premise, not a check.";

const SHARED_THEOREM: &str = "Fully declared SCM over finite variables in which an exogenous \
term may feed several mechanisms, with every mechanism and every finite exogenous law declared: \
the unit-level counterfactual joint law over worlds is the push-forward of the joint exogenous \
posterior given the factual evidence (abduction-action-prediction; Pearl 2009 ch. 7, Balke and \
Pearl 1994), enumerated exactly over the joint exogenous states. A shared term takes one value \
that every mechanism reading it sees in every world, so latent confounding raises no \
identification question when the model is declared. Model adequacy is a premise, not a check.";

/// Every row, in a fixed order.
pub static ROWS: [MatrixRow; 14] = [
    MatrixRow {
        id: "counterfactual.finite_markovian_scm.marginal",
        quantity: QuantityClass::CounterfactualMarginal,
        model: ModelClass::FiniteMarkovianScm,
        conditioning: &[ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::DeclaredMechanisms],
        population: PopulationKind::Single,
        coupling: CouplingKind::NotRequired,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::selective::FiniteScmExecutor",
        theorem: FINITE_THEOREM,
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "counterfactual.finite_markovian_scm.joint",
        quantity: QuantityClass::CounterfactualJoint,
        model: ModelClass::FiniteMarkovianScm,
        conditioning: &[ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::DeclaredMechanisms],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::selective::FiniteScmExecutor",
        theorem: FINITE_THEOREM,
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "natural_path_specific.finite_markovian_scm",
        quantity: QuantityClass::NaturalPathSpecific,
        model: ModelClass::FiniteMarkovianScm,
        conditioning: &[ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::DeclaredMechanisms],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::selective::FiniteScmExecutor",
        theorem: "As the finite Markovian row; a world reads a parent from another world \
(nested counterfactual, Pearl 2001 natural effects) and every world reads one exogenous \
posterior state, so the contrast is computed from coupled worlds only.",
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "temporal.fixed_population_two_slice",
        quantity: QuantityClass::Temporal,
        model: ModelClass::AdditiveLinearTwoSlice,
        conditioning: &[ConditioningKind::FactualEvidence],
        history: &[HistoryKind::FixedPopulationTwoSlice],
        evidence: &[EvidenceKind::SuppliedFittedIdentity],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::temporal_cross_world::staged::TemporalExecutor",
        theorem: "Fixed, fully observed, Markovian two-slice linear-Gaussian additive-noise SCM: \
each unit's exogenous history is recovered exactly from its factual trajectory and both named \
action histories are replayed against it (temporal_cross_world theorem).",
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "transported.path_specific_static_additive_linear",
        quantity: QuantityClass::TransportedCrossWorld,
        model: ModelClass::AdditiveLinearStaticTransported,
        conditioning: &[ConditioningKind::NoEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::SuppliedFittedIdentity],
        population: PopulationKind::Transported,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::ExistingTheoremRoute,
        route: "antecedent_counterfactual::transported_path_specific::\
evaluate_transported_path_specific",
        theorem: "Selection nodes only into covariates or the treatment, additive noise, \
invariant noise law, support inclusion (transported_path_specific theorem). It keeps its own \
boundary and does not abduce a target-population factual unit.",
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "transported.cross_world_execution",
        quantity: QuantityClass::TransportedCrossWorld,
        model: ModelClass::FiniteMarkovianScm,
        conditioning: &[],
        history: &[],
        evidence: &[],
        population: PopulationKind::Transported,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Closed,
        route: "none",
        theorem: "No joint theorem composes a transport license with factual abduction in a \
target population: a transport identification supplies no abduction, shared-noise or per-world \
receipts, and abduction in the source population is not abduction of a target unit.",
        refusal_code: Some("cell_not_licensed"),
        refusal_detail: Some("cross_world_execution.transported_row_closed"),
        remedy: "supply a transport license for the quantity, a fixed-population cross-world \
license, and the cross-population invariance of the exogenous law with target-population \
factual evidence; the static additive linear case has its own theorem-scoped route",
    },
    MatrixRow {
        id: "model.semi_markovian",
        quantity: QuantityClass::CounterfactualJoint,
        model: ModelClass::SemiMarkovian,
        conditioning: &[],
        history: &[],
        evidence: &[],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Closed,
        route: "none",
        theorem: "Latent confounding whose mechanisms and exogenous law are not declared, or are \
to be fitted from observational data, does not identify a cross-world joint law; the same \
shared exogenous term is executed when every mechanism and law is declared.",
        refusal_code: Some("cross_world_not_identified"),
        refusal_detail: Some("cross_world_execution.latent_confounding_not_identified"),
        remedy: "declare every mechanism and the finite law of every exogenous term (a fully \
declared finite SCM may share an exogenous term between mechanisms), or use an identification \
route that accepts the latent structure",
    },
    MatrixRow {
        id: "model.nonparametric_admg",
        quantity: QuantityClass::CounterfactualJoint,
        model: ModelClass::NonparametricAdmg,
        conditioning: &[],
        history: &[],
        evidence: &[],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Closed,
        route: "none",
        theorem: "A cross-world joint law is not identified by a graph alone; mechanisms and \
a noise law must be declared.",
        refusal_code: Some("cross_world_not_identified"),
        refusal_detail: Some("cross_world_execution.mechanisms_not_declared"),
        remedy: "declare the mechanisms and the exogenous law (a fully declared finite SCM or the \
two-slice additive linear model), or ask an interventional rather than a cross-world question",
    },
    MatrixRow {
        id: "history.incompatible",
        quantity: QuantityClass::Temporal,
        model: ModelClass::AdditiveLinearTwoSlice,
        conditioning: &[],
        history: &[],
        evidence: &[],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Closed,
        route: "none",
        theorem: "Mechanisms hold for the history or regime they were declared under; factual \
evidence from another history needs a new compatibility and identification check.",
        refusal_code: Some("route_not_supported"),
        refusal_detail: Some("cross_world_execution.incompatible_history"),
        remedy: "declare the mechanisms for the history the evidence was gathered under, or \
establish and supply the invariance of the mechanisms across the two histories",
    },
    MatrixRow {
        id: "intervention.soft",
        quantity: QuantityClass::CounterfactualMarginal,
        model: ModelClass::FiniteMarkovianScm,
        conditioning: &[],
        history: &[],
        evidence: &[],
        population: PopulationKind::Single,
        coupling: CouplingKind::NotRequired,
        status: CapabilityStatus::Closed,
        route: "none",
        theorem: "Worlds are defined by hard interventions; a stochastic or policy intervention \
adds a randomization law whose coupling across worlds is not declared.",
        refusal_code: Some("route_not_supported"),
        refusal_detail: Some("cross_world_execution.soft_intervention"),
        remedy: "express each world as hard interventions, or declare the intervention's own \
exogenous term as part of the SCM",
    },
    MatrixRow {
        id: "evidence.fit_from_data",
        quantity: QuantityClass::CounterfactualMarginal,
        model: ModelClass::FiniteMarkovianScm,
        conditioning: &[],
        history: &[],
        evidence: &[],
        population: PopulationKind::Single,
        coupling: CouplingKind::NotRequired,
        status: CapabilityStatus::Closed,
        route: "none",
        theorem: "A finite SCM's exogenous law is not identified from observational data alone \
(response-type mass is partially identified); fitting it needs declared constraints.",
        refusal_code: Some("route_not_supported"),
        refusal_detail: Some("cross_world_execution.finite_fitting_not_executed"),
        remedy: "declare the mechanisms and the exogenous law, or use the partial-identification \
route that returns bounds on the counterfactual",
    },
    MatrixRow {
        id: "counterfactual.finite_shared_exogenous_scm.marginal",
        quantity: QuantityClass::CounterfactualMarginal,
        model: ModelClass::FiniteSharedExogenousScm,
        conditioning: &[ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::DeclaredMechanisms],
        population: PopulationKind::Single,
        coupling: CouplingKind::NotRequired,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::selective::FiniteScmExecutor",
        theorem: SHARED_THEOREM,
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "counterfactual.finite_shared_exogenous_scm.joint",
        quantity: QuantityClass::CounterfactualJoint,
        model: ModelClass::FiniteSharedExogenousScm,
        conditioning: &[ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::DeclaredMechanisms],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::selective::FiniteScmExecutor",
        theorem: SHARED_THEOREM,
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
    MatrixRow {
        id: "natural_path_specific.finite_shared_exogenous_scm",
        quantity: QuantityClass::NaturalPathSpecific,
        model: ModelClass::FiniteSharedExogenousScm,
        conditioning: &[ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence],
        history: &[HistoryKind::NoHistory],
        evidence: &[EvidenceKind::DeclaredMechanisms],
        population: PopulationKind::Single,
        coupling: CouplingKind::SharedAbducedExogenous,
        status: CapabilityStatus::Executed,
        route: "antecedent_counterfactual::selective::FiniteScmExecutor",
        theorem: "As the shared-exogenous finite row; a world reads a parent from another \
world (nested counterfactual, Pearl 2001 natural effects) and every world reads one joint \
exogenous posterior state, so the contrast is computed from coupled worlds only.",
        refusal_code: None,
        refusal_detail: None,
        remedy: "",
    },
];

/// Every row of the matrix.
#[must_use]
pub fn rows() -> &'static [MatrixRow] {
    &ROWS
}

/// One request's coordinates in the matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixRequest {
    /// Quantity class.
    pub quantity: QuantityClass,
    /// Model class.
    pub model: ModelClass,
    /// Conditioning.
    pub conditioning: ConditioningKind,
    /// Intervention kind.
    pub interventions: InterventionKind,
    /// Population scope.
    pub population: PopulationKind,
    /// History scope.
    pub history: HistoryKind,
    /// Evidence source.
    pub evidence: EvidenceKind,
}

/// The refusal of a request outside every executed row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: &'static str,
    /// The matrix row (or dimension) the refusal is about.
    pub subject: &'static str,
    /// What would open the request.
    pub remedy: &'static str,
    /// Named unmet prerequisites.
    pub missing: Vec<String>,
}

/// A licensed row or a typed refusal.
pub type MatrixDecision = Result<&'static MatrixRow, MatrixRefusal>;

fn closed(id: &'static str) -> MatrixRefusal {
    let row = ROWS.iter().find(|r| r.id == id).unwrap_or(&ROWS[5]);
    MatrixRefusal {
        code: row.refusal_code.unwrap_or(antecedent_core::reason_code!("cell_not_licensed")),
        detail: row.refusal_detail.unwrap_or("cross_world_execution.cell_not_in_matrix"),
        subject: row.id,
        remedy: row.remedy,
        missing: Vec::new(),
    }
}

fn transported_refusal() -> MatrixRefusal {
    let prerequisites = TransportedCounterfactualPrerequisites {
        fixed_population_license: true,
        ..TransportedCounterfactualPrerequisites::default()
    };
    let mut missing: Vec<String> =
        prerequisites.missing_gates().iter().map(|g| g.name().to_owned()).collect();
    missing.push("target_population_abduction_receipt".to_owned());
    missing.push("shared_exogenous_invariance_across_populations".to_owned());
    MatrixRefusal { missing, ..closed("transported.cross_world_execution") }
}

/// Decide a request against the matrix.
///
/// # Errors
/// A [`MatrixRefusal`] naming the closed row or the first dimension no
/// executed row admits.
pub fn decide(request: &MatrixRequest) -> MatrixDecision {
    if request.population == PopulationKind::Transported {
        let own = ROWS.iter().find(|r| {
            r.status == CapabilityStatus::ExistingTheoremRoute
                && r.quantity == request.quantity
                && r.model == request.model
        });
        return own.ok_or_else(transported_refusal);
    }
    match request.model {
        ModelClass::SemiMarkovian => return Err(closed("model.semi_markovian")),
        ModelClass::NonparametricAdmg => return Err(closed("model.nonparametric_admg")),
        _ => {}
    }
    if request.history == HistoryKind::Incompatible {
        return Err(closed("history.incompatible"));
    }
    if request.interventions == InterventionKind::Soft {
        return Err(closed("intervention.soft"));
    }
    if request.evidence == EvidenceKind::DataToFit {
        return Err(closed("evidence.fit_from_data"));
    }
    let candidates = || {
        ROWS.iter().filter(|r| {
            r.status == CapabilityStatus::Executed
                && r.quantity == request.quantity
                && r.model == request.model
        })
    };
    let Some(row) = candidates().find(|r| {
        r.conditioning.contains(&request.conditioning)
            && r.history.contains(&request.history)
            && r.evidence.contains(&request.evidence)
    }) else {
        let subject = candidates().next().map_or("matrix", |r| r.id);
        return Err(MatrixRefusal {
            code: antecedent_core::reason_code!("route_not_supported"),
            detail: "cross_world_execution.cell_not_in_matrix",
            subject,
            remedy: "the quantity, model, conditioning, history and evidence source must match \
an executed row of the capability matrix (parity/cross_world_capability.toml)",
            missing: Vec::new(),
        });
    };
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite(quantity: QuantityClass) -> MatrixRequest {
        MatrixRequest {
            quantity,
            model: ModelClass::FiniteMarkovianScm,
            conditioning: ConditioningKind::FactualEvidence,
            interventions: InterventionKind::Hard,
            population: PopulationKind::Single,
            history: HistoryKind::NoHistory,
            evidence: EvidenceKind::DeclaredMechanisms,
        }
    }

    #[test]
    fn row_ids_are_unique_and_closed_rows_carry_a_refusal() {
        let mut ids: Vec<_> = ROWS.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ROWS.len());
        for row in &ROWS {
            let closed = row.status == CapabilityStatus::Closed;
            assert_eq!(row.refusal_code.is_some(), closed, "{}", row.id);
            assert_eq!(row.refusal_detail.is_some(), closed, "{}", row.id);
            assert_eq!(!row.remedy.is_empty(), closed, "{}", row.id);
        }
    }

    #[test]
    fn executed_cells_resolve_and_every_closed_cell_names_its_row() {
        let row = decide(&finite(QuantityClass::NaturalPathSpecific)).unwrap();
        assert_eq!(row.id, "natural_path_specific.finite_markovian_scm");
        assert!(row.needs_shared_exogenous());
        let marginal = decide(&finite(QuantityClass::CounterfactualMarginal)).unwrap();
        assert!(!marginal.needs_shared_exogenous());

        let shared = MatrixRequest {
            model: ModelClass::FiniteSharedExogenousScm,
            ..finite(QuantityClass::CounterfactualJoint)
        };
        assert_eq!(decide(&shared).unwrap().id, "counterfactual.finite_shared_exogenous_scm.joint");

        let semi = MatrixRequest {
            model: ModelClass::SemiMarkovian,
            ..finite(QuantityClass::CounterfactualJoint)
        };
        let refused = decide(&semi).unwrap_err();
        assert_eq!(refused.code, "cross_world_not_identified");
        assert_eq!(refused.detail, "cross_world_execution.latent_confounding_not_identified");

        let history =
            MatrixRequest { history: HistoryKind::Incompatible, ..finite(QuantityClass::Temporal) };
        assert_eq!(
            decide(&history).unwrap_err().detail,
            "cross_world_execution.incompatible_history"
        );

        let soft = MatrixRequest {
            interventions: InterventionKind::Soft,
            ..finite(QuantityClass::CounterfactualMarginal)
        };
        assert_eq!(decide(&soft).unwrap_err().detail, "cross_world_execution.soft_intervention");

        let fit = MatrixRequest {
            evidence: EvidenceKind::DataToFit,
            ..finite(QuantityClass::CounterfactualMarginal)
        };
        assert_eq!(
            decide(&fit).unwrap_err().detail,
            "cross_world_execution.finite_fitting_not_executed"
        );

        // A temporal quantity on a finite SCM is not a row.
        let mismatch = decide(&finite(QuantityClass::Temporal)).unwrap_err();
        assert_eq!(mismatch.detail, "cross_world_execution.cell_not_in_matrix");
    }

    #[test]
    fn a_transported_request_names_every_unmet_prerequisite() {
        let request = MatrixRequest {
            population: PopulationKind::Transported,
            ..finite(QuantityClass::TransportedCrossWorld)
        };
        let refused = decide(&request).unwrap_err();
        assert_eq!(refused.code, "cell_not_licensed");
        assert_eq!(refused.detail, "cross_world_execution.transported_row_closed");
        for gate in [
            "transport_license_missing",
            "cross_population_assumptions_missing",
            "target_population_abduction_receipt",
            "shared_exogenous_invariance_across_populations",
        ] {
            assert!(refused.missing.iter().any(|m| m == gate), "{gate} in {:?}", refused.missing);
        }
        // The static additive linear case keeps its own theorem-scoped route.
        let own = MatrixRequest {
            model: ModelClass::AdditiveLinearStaticTransported,
            conditioning: ConditioningKind::NoEvidence,
            evidence: EvidenceKind::SuppliedFittedIdentity,
            ..request
        };
        assert_eq!(decide(&own).unwrap().status, CapabilityStatus::ExistingTheoremRoute);
    }
}
