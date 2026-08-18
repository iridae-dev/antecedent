//! Generated from `parity/support_*.toml`. Do not edit.

#![allow(missing_docs)]
#![cfg_attr(rustfmt, rustfmt::skip)]

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
        queries: Some(&["AverageDerivative", "AverageEffect", "ConditionalEffect", "Counterfactual", "DirectionalDerivative", "Elasticity", "InterventionalDistribution", "InterventionResponse", "PathSpecificEffect", "PointDerivative", "ResponseCurve", "ResponseJacobian", "SemiElasticity"]),
        graph_classes: Some(&["TemporalDag", "TemporalCpdag", "TemporalPag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "Static queries are not a temporal-graph cell; use PulseEffect or SustainedEffect.",
    },
    NaRule {
        queries: Some(&["AverageDerivative", "DirectionalDerivative", "Elasticity", "InterventionResponse", "PointDerivative", "ResponseCurve", "ResponseJacobian", "SemiElasticity"]),
        graph_classes: None,
        structures: Some(&["graph_posterior"]),
        inferences: None,
        validations: None,
        reason: "Structural uncertainty around curves is contrast-only; graph-posterior mixtures do not license a response cell.",
    },
    NaRule {
        queries: Some(&["TransportQuery", "InterferenceQuery"]),
        graph_classes: Some(&["TemporalDag", "TemporalCpdag", "TemporalPag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "Stage transport and interference queries are not a temporal-graph cell.",
    }
];

pub static CLOSED_RULES: &[NaRule] = &[
    NaRule {
        queries: Some(&["Counterfactual"]),
        graph_classes: None,
        structures: None,
        inferences: None,
        validations: None,
        reason: "Counterfactual is not on the staged handle.",
    },
    NaRule {
        queries: Some(&["AverageDerivative", "DirectionalDerivative", "Elasticity", "PointDerivative", "ResponseJacobian", "SemiElasticity"]),
        graph_classes: None,
        structures: None,
        inferences: None,
        validations: None,
        reason: "Derivative cells are not licensed; only ResponseCurve on a Dag is staged.",
    },
    NaRule {
        queries: Some(&["ResponseCurve"]),
        graph_classes: Some(&["Pag", "Cpdag", "Admg"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "ResponseCurve is licensed only on a Dag.",
    },
    NaRule {
        queries: Some(&["PathSpecificEffect", "InterventionalDistribution"]),
        graph_classes: None,
        structures: Some(&["graph_posterior", "accepted"]),
        inferences: None,
        validations: None,
        reason: "Path and distribution queries are licensed only as explicit Dag cells; accepted and graph-posterior structures are not staged.",
    },
    NaRule {
        queries: Some(&["MediationEffect"]),
        graph_classes: None,
        structures: None,
        inferences: None,
        validations: None,
        reason: "MediationEffect is not on the staged handle.",
    },
    NaRule {
        queries: Some(&["SustainedEffect"]),
        graph_classes: None,
        structures: None,
        inferences: None,
        validations: None,
        reason: "SustainedEffect is not on the staged handle.",
    },
    NaRule {
        queries: Some(&["TransportQuery", "InterferenceQuery"]),
        graph_classes: None,
        structures: None,
        inferences: None,
        validations: None,
        reason: "Stage transport and interference APIs are not licensed analyze cells; sID outside the implemented subset is NotCertified.",
    },
    NaRule {
        queries: Some(&["ConditionalEffect"]),
        graph_classes: Some(&["Cpdag", "Admg", "Pag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "ConditionalEffect compiles only against a supplied static Dag; Cpdag/Admg/Pag have no ConditionalEffect compile arm.",
    },
    NaRule {
        queries: Some(&["PathSpecificEffect", "InterventionalDistribution"]),
        graph_classes: Some(&["Admg", "Pag"]),
        structures: Some(&["explicit"]),
        inferences: None,
        validations: None,
        reason: "Path and distribution queries execute only on a supplied static Dag; a directly supplied Admg/Pag hits the same static-Dag requirement (accepted/graph-posterior Admg and Pag are already closed above).",
    },
    NaRule {
        queries: Some(&["InterventionResponse"]),
        graph_classes: Some(&["Cpdag", "Admg", "Pag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "InterventionResponse executes only on a supplied static Dag, the same requirement ResponseCurve is closed on above; Cpdag/Admg/Pag have no Response compile arm.",
    },
    NaRule {
        queries: Some(&["TemporalMediationEffect"]),
        graph_classes: Some(&["TemporalCpdag", "TemporalPag"]),
        structures: None,
        inferences: None,
        validations: None,
        reason: "TemporalMediationEffect compiles only against a supplied TemporalDag; compile.rs has no Mediation arm for TemporalCpdag/TemporalPag.",
    },
    NaRule {
        queries: None,
        graph_classes: None,
        structures: Some(&["graph_posterior"]),
        inferences: Some(&["Frequentist"]),
        validations: None,
        reason: "Graph-posterior compilation requires Bayesian inference: a graph posterior is a mixture over structures, and only Bayesian inference can combine per-graph effect draws into an envelope.",
    }
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
        structure: "explicit",
        inference: "Frequentist",
        validation: "full",
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
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Frequentist",
        validation: "full",
    },
    LicensedCell {
        query: "ResponseCurve",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "ResponseCurve",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Bayesian",
        validation: "none",
    },
    LicensedCell {
        query: "ConditionalEffect",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "ConditionalEffect",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "PathSpecificEffect",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "InterventionResponse",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "InterventionResponse",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "InterventionalDistribution",
        graph_class: "Dag",
        structure: "explicit",
        inference: "Frequentist",
        validation: "none",
    },
    LicensedCell {
        query: "AverageEffect",
        graph_class: "Dag",
        structure: "accepted",
        inference: "Bayesian",
        validation: "none",
    }
];
