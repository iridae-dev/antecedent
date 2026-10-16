//! One declared transport scenario shared by the mechanism-composition tests
//! and the finite counterfactual-transport agreement test.
//!
//! Structure: treatment `x` and outcome `y` with `x -> y` and one shared
//! latent cause `u` of both (the finite derivation's bidirected edge `x <-> y`).
//! Mechanisms (the same structural numbers on both sides):
//!
//!   `x = 1 + 1 u + e_x`          (`e_x` sd 1)   fitted in the source
//!   `x = 3 + 1 u + e_x`          (`e_x` sd 1)   refitted in the target
//!   `y = 0.5 + 2 x + 1 u + e_y`  (`e_y` sd 1)   fitted in the source
//!
//! with `u ~ N(0, 1)`. Closed form in the target composed model with `x`
//! refitted and `y` carried:
//!
//!   E[x] = 3, E[y] = 0.5 + 2 * 3 = 6.5
//!   Var x = 2, Cov(x, u) = 1
//!   Var y = 4 * 2 + 1 + 1 + 2 * 2 * 1 = 14, Cov(x, y) = 2 * 2 + 1 = 5
//!   E[y | do(x = 1)] - E[y | do(x = 0)] = 2
//!
//! Two selection configurations are declared. In both, the finite
//! counterfactual transport derivation and the mechanism-composition check
//! must reach the same verdict about the outcome mechanism:
//!
//! * `OutcomeSelected`: a selection node points into `y`. The outcome
//!   mechanism is not invariant: composition refuses it and the finite
//!   derivation proves the effect non-transportable from the source
//!   experiment.
//! * `TreatmentSelected`: a selection node points into `x` only. The treatment
//!   mechanism is refitted in the target; the outcome mechanism is invariant
//!   and the effect transports.

#![allow(dead_code, reason = "each integration test uses part of the shared fixture")]

use antecedent_model::compose::{
    ContextCoordinate, Domain, InvarianceDeclaration, LatentDependence, MechanismContext,
    MechanismDescriptor, MechanismForm, MechanismOrigin, MechanismProvenance, ParentBinding,
    SelectionDiagramDeclaration, SelectionNodeDeclaration,
};

pub const SOURCE: &str = "source";
pub const TARGET: &str = "target";
pub const REGIME: &str = "observational";

/// Variable names with the identifiers the finite derivation uses.
pub const VARIABLES: [(&str, u32); 2] = [("x", 0), ("y", 1)];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Selection {
    OutcomeSelected,
    TreatmentSelected,
}

impl Selection {
    pub const ALL: [Self; 2] = [Self::OutcomeSelected, Self::TreatmentSelected];

    /// The variable a selection node points into.
    pub fn variable(self) -> &'static str {
        match self {
            Self::OutcomeSelected => "y",
            Self::TreatmentSelected => "x",
        }
    }
}

pub fn variable_id(name: &str) -> u32 {
    VARIABLES.iter().find(|(n, _)| *n == name).map(|(_, id)| *id).expect("declared variable")
}

pub fn diagram(selection: Selection) -> SelectionDiagramDeclaration {
    SelectionDiagramDeclaration {
        source_population: SOURCE.into(),
        target_population: TARGET.into(),
        selection_nodes: vec![SelectionNodeDeclaration {
            name: "S".into(),
            points_into: vec![selection.variable().into()],
        }],
    }
}

pub fn target_context() -> MechanismContext {
    MechanismContext { population: TARGET.into(), regime: REGIME.into() }
}

fn context(population: &str) -> MechanismContext {
    MechanismContext { population: population.into(), regime: REGIME.into() }
}

fn binding(variable: &str) -> ParentBinding {
    ParentBinding {
        variable: variable.into(),
        units: "u".into(),
        transform: "identity".into(),
        domain: Domain::Real,
    }
}

fn latent(shared_with: &str) -> LatentDependence {
    LatentDependence {
        latent: "u".into(),
        loading: Some(1.0),
        shared_with: vec![shared_with.into()],
    }
}

fn gaussian(
    variable: &str,
    population: &str,
    intercept: f64,
    coefficients: &[(&str, f64)],
    shared_with: &str,
) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: variable.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma: 1.0,
            coefficients: coefficients.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: coefficients.iter().map(|(p, _)| binding(p)).collect(),
        conditioning_set: coefficients.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: context(population),
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{variable}:{population}"),
            evidence_id: format!("data:{variable}:{population}"),
        },
        latents: vec![latent(shared_with)],
        uncertainty: None,
    }
}

pub fn source_x() -> MechanismDescriptor {
    gaussian("x", SOURCE, 1.0, &[], "y")
}

pub fn target_x() -> MechanismDescriptor {
    gaussian("x", TARGET, 3.0, &[], "y")
}

pub fn source_y() -> MechanismDescriptor {
    gaussian("y", SOURCE, 0.5, &[("x", 2.0)], "x")
}

/// A target refit of the outcome mechanism with a different slope (1.5).
pub fn target_y() -> MechanismDescriptor {
    gaussian("y", TARGET, 0.25, &[("x", 1.5)], "x")
}

pub fn edges() -> Vec<(String, String)> {
    vec![("x".into(), "y".into())]
}

/// The analyst's population invariance for the carried outcome mechanism,
/// required because it loads on the shared latent `u`.
pub fn outcome_invariance() -> InvarianceDeclaration {
    InvarianceDeclaration {
        variable: "y".into(),
        coordinate: ContextCoordinate::Population,
        source: SOURCE.into(),
        target: TARGET.into(),
        assumption_id: "assumption:outcome-mechanism-invariant".into(),
    }
}
