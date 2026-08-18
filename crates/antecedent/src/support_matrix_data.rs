//! Generated from `parity/support_*.toml`. Do not edit.

#![allow(missing_docs)]

#[derive(Clone, Copy, Debug)]
pub struct NaRule {
    pub queries: Option<&'static [&'static str]>,
    pub graph_classes: Option<&'static [&'static str]>,
    pub structures: Option<&'static [&'static str]>,
    pub inferences: Option<&'static [&'static str]>,
    pub validations: Option<&'static [&'static str]>,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct LicensedCell {
    pub query: &'static str,
    pub graph_class: &'static str,
    pub structure: &'static str,
    pub inference: &'static str,
    pub validation: &'static str,
}

pub static NA_RULES: &[NaRule] = &[
    NaRule {
        queries: Some(&["PulseEffect", "SustainedEffect", "TemporalMediationEffect"]),
        graph_classes: Some(&["Dag", "Admg", "Cpdag", "Pag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "Temporal contrast queries require a temporal graph class.",
    },
    NaRule {
        queries: Some(&[
            "AverageDerivative",
            "AverageEffect",
            "ConditionalEffect",
            "Counterfactual",
            "DirectionalDerivative",
            "Elasticity",
            "InterventionalDistribution",
            "InterventionResponse",
            "PathSpecificEffect",
            "PointDerivative",
            "ResponseCurve",
            "ResponseJacobian",
            "SemiElasticity",
        ]),
        graph_classes: Some(&["TemporalDag", "TemporalCpdag", "TemporalPag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "Static queries are not a temporal-graph cell; use PulseEffect or SustainedEffect.",
    },
    NaRule {
        queries: Some(&[
            "AverageDerivative",
            "DirectionalDerivative",
            "Elasticity",
            "InterventionResponse",
            "PointDerivative",
            "ResponseCurve",
            "ResponseJacobian",
            "SemiElasticity",
        ]),
        graph_classes: None,
        structures: Some(&["graph_posterior"]),
        inferences: None,
        validations: None,
        reason: "Structural uncertainty around curves is contrast-only; graph-posterior mixtures do not license a response cell.",
    },
];

pub static LICENSED: &[LicensedCell] = &[
    LicensedCell {
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "cheap",
    },
    LicensedCell {
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Frequentist",
        validation: "cheap",
    },
    LicensedCell {
        query: "ResponseCurve",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
];
