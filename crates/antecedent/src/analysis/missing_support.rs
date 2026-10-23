//! The declared missing-data support of every route cell, read from
//! `parity/missing_policy_support.toml`.
//!
//! A route cell is a query kind, a data modality and an inference mode. The registry is the
//! single statement of which `missing=` policies a cell executes, the policy it applies when
//! none is declared, the policy a refusal names first, and, for multiple imputation, the
//! estimators whose fits it pools. The builder refuses every declaration the registry does
//! not list for the study's cell; nothing is accepted and then ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use antecedent_core::CausalQueryKind;
use antecedent_data::MissingPolicy;

/// The registry text, compiled in.
const REGISTRY: &str = include_str!("../../../../parity/missing_policy_support.toml");

/// Every query kind, in declaration order. [`query_kind_index`] is an exhaustive match, so a
/// new [`CausalQueryKind`] fails to compile until it is listed here and given its rows (or
/// left without one, which refuses every declaration).
pub const ALL_QUERY_KINDS: [CausalQueryKind; 29] = [
    CausalQueryKind::AverageEffect,
    CausalQueryKind::TemporalEffect,
    CausalQueryKind::Counterfactual,
    CausalQueryKind::NestedCounterfactual,
    CausalQueryKind::AnomalyAttribution,
    CausalQueryKind::ChangeAttribution,
    CausalQueryKind::MechanismChange,
    CausalQueryKind::UnitChange,
    CausalQueryKind::Mediation,
    CausalQueryKind::ConditionalEffect,
    CausalQueryKind::Distribution,
    CausalQueryKind::PathSpecific,
    CausalQueryKind::Response,
    CausalQueryKind::Transport,
    CausalQueryKind::Interference,
    CausalQueryKind::RandomizedEffect,
    CausalQueryKind::PolicyValue,
    CausalQueryKind::ContinuousDoseResponse,
    CausalQueryKind::PanelDid,
    CausalQueryKind::SyntheticControl,
    CausalQueryKind::LocalPolynomialRatio,
    CausalQueryKind::Survival,
    CausalQueryKind::LongitudinalRegime,
    CausalQueryKind::SequentialContrast,
    CausalQueryKind::HistoryMatch,
    CausalQueryKind::Discover,
    CausalQueryKind::Attribute,
    CausalQueryKind::Simulate,
    CausalQueryKind::TemporalDesign,
];

/// Position of `kind` in [`ALL_QUERY_KINDS`].
#[must_use]
pub(crate) const fn query_kind_index(kind: CausalQueryKind) -> usize {
    match kind {
        CausalQueryKind::AverageEffect => 0,
        CausalQueryKind::TemporalEffect => 1,
        CausalQueryKind::Counterfactual => 2,
        CausalQueryKind::NestedCounterfactual => 3,
        CausalQueryKind::AnomalyAttribution => 4,
        CausalQueryKind::ChangeAttribution => 5,
        CausalQueryKind::MechanismChange => 6,
        CausalQueryKind::UnitChange => 7,
        CausalQueryKind::Mediation => 8,
        CausalQueryKind::ConditionalEffect => 9,
        CausalQueryKind::Distribution => 10,
        CausalQueryKind::PathSpecific => 11,
        CausalQueryKind::Response => 12,
        CausalQueryKind::Transport => 13,
        CausalQueryKind::Interference => 14,
        CausalQueryKind::RandomizedEffect => 15,
        CausalQueryKind::PolicyValue => 16,
        CausalQueryKind::ContinuousDoseResponse => 17,
        CausalQueryKind::PanelDid => 18,
        CausalQueryKind::SyntheticControl => 19,
        CausalQueryKind::LocalPolynomialRatio => 20,
        CausalQueryKind::Survival => 21,
        CausalQueryKind::LongitudinalRegime => 22,
        CausalQueryKind::SequentialContrast => 23,
        CausalQueryKind::HistoryMatch => 24,
        CausalQueryKind::Discover => 25,
        CausalQueryKind::Attribute => 26,
        CausalQueryKind::Simulate => 27,
        CausalQueryKind::TemporalDesign => 28,
    }
}

/// Registry name of a query kind (its variant name).
#[must_use]
pub(crate) fn query_kind_name(kind: CausalQueryKind) -> String {
    format!("{kind:?}")
}

/// The data modality of a route cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MissingModality {
    /// One row per observation.
    Tabular,
    /// One regular time series.
    Series,
    /// Event data aligned on a duration grid.
    Event,
    /// Several environments' series.
    MultiEnv,
    /// Per-unit series of a panel.
    Panel,
}

impl MissingModality {
    /// Every modality.
    pub const ALL: [Self; 5] =
        [Self::Tabular, Self::Series, Self::Event, Self::MultiEnv, Self::Panel];

    /// Registry name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tabular => "tabular",
            Self::Series => "series",
            Self::Event => "event",
            Self::MultiEnv => "multi_env",
            Self::Panel => "panel",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }
}

/// The inference mode of a route cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MissingInference {
    /// Frequentist point estimate and variance.
    Frequentist,
    /// Bayesian posterior.
    Bayesian,
}

impl MissingInference {
    /// Both modes.
    pub const ALL: [Self; 2] = [Self::Frequentist, Self::Bayesian];

    /// Registry name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Frequentist => "frequentist",
            Self::Bayesian => "bayesian",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }
}

/// One route cell: query kind, data modality, inference mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MissingRouteCell {
    /// Query kind.
    pub query: CausalQueryKind,
    /// Data modality.
    pub modality: MissingModality,
    /// Inference mode.
    pub inference: MissingInference,
}

impl MissingRouteCell {
    /// `query:modality:inference`.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{:?}:{}:{}", self.query, self.modality.name(), self.inference.name())
    }

    /// Every route cell, in query, modality, inference order.
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut cells = Vec::with_capacity(ALL_QUERY_KINDS.len() * 10);
        for query in ALL_QUERY_KINDS {
            for modality in MissingModality::ALL {
                for inference in MissingInference::ALL {
                    cells.push(Self { query, modality, inference });
                }
            }
        }
        cells
    }
}

/// One registry row: the policies a route cell executes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingRouteSupport {
    /// Query kind.
    pub query: CausalQueryKind,
    /// Data modality.
    pub modality: MissingModality,
    /// Inference modes the row covers.
    pub inference: Vec<MissingInference>,
    /// Executing route family: `static`, `randomized`, `panel` or `temporal`.
    pub family: String,
    /// Policy applied when none is declared.
    pub default: MissingPolicy,
    /// Policies the cell executes, in documentation order. Multiple imputation is listed
    /// with its default specification; membership is by policy name.
    pub supported: Vec<MissingPolicy>,
    /// The supported policy a refusal names first.
    pub next_attempt: MissingPolicy,
    /// Estimators multiple imputation pools on this cell.
    pub imputation_estimators: Vec<String>,
}

impl MissingRouteSupport {
    /// Whether the cell executes `policy` (by name: any imputation specification).
    #[must_use]
    pub fn supports(&self, policy: MissingPolicy) -> bool {
        self.supported.iter().any(|supported| supported.name() == policy.name())
    }

    /// Whether multiple imputation pools `estimator` on this cell.
    #[must_use]
    pub fn imputes_with(&self, estimator: &str) -> bool {
        self.imputation_estimators.iter().any(|name| name == estimator)
    }

    fn covers(&self, cell: MissingRouteCell) -> bool {
        self.query == cell.query
            && self.modality == cell.modality
            && self.inference.contains(&cell.inference)
    }
}

/// The parsed registry.
///
/// # Panics
///
/// When the compiled-in registry does not parse; `missing_policy_support` tests parse it.
#[must_use]
pub fn missing_support_table() -> &'static [MissingRouteSupport] {
    static TABLE: OnceLock<Vec<MissingRouteSupport>> = OnceLock::new();
    TABLE.get_or_init(|| {
        parse_registry(REGISTRY)
            .unwrap_or_else(|error| panic!("parity/missing_policy_support.toml: {error}"))
    })
}

/// The registry row of `cell`, or `None` when the cell executes no missing-data policy.
#[must_use]
pub fn route_missing_support(cell: MissingRouteCell) -> Option<&'static MissingRouteSupport> {
    missing_support_table().iter().find(|row| row.covers(cell))
}

/// Parse the registry text.
///
/// The format is the subset of TOML the registry uses: `schema = 1`, `[[route]]` tables,
/// string values and arrays of strings, comments and blank lines.
///
/// # Errors
///
/// A line outside that subset, an unknown key, field, query kind, modality, inference mode
/// or policy name, a missing field, a default or next attempt that is not supported, or two
/// rows that cover the same cell.
pub(crate) fn parse_registry(text: &str) -> Result<Vec<MissingRouteSupport>, String> {
    let mut rows: Vec<Vec<(String, Value)>> = Vec::new();
    let mut schema = None;
    for (number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[route]]" {
            rows.push(Vec::new());
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {}: expected `key = value`", number + 1));
        };
        let key = key.trim().to_owned();
        let value = parse_value(value.trim()).map_err(|e| format!("line {}: {e}", number + 1))?;
        match rows.last_mut() {
            Some(row) => row.push((key, value)),
            None if key == "schema" => schema = Some(value),
            None => return Err(format!("line {}: `{key}` outside a [[route]]", number + 1)),
        }
    }
    if !matches!(schema, Some(Value::Integer(1))) {
        return Err("schema must be 1".to_owned());
    }
    let mut table = Vec::with_capacity(rows.len());
    for (index, fields) in rows.into_iter().enumerate() {
        table.push(parse_row(index, fields)?);
    }
    for (i, a) in table.iter().enumerate() {
        for b in &table[i + 1..] {
            if a.query == b.query
                && a.modality == b.modality
                && a.inference.iter().any(|mode| b.inference.contains(mode))
            {
                return Err(format!("two rows cover {:?}:{}", a.query, a.modality.name()));
            }
        }
    }
    Ok(table)
}

/// Fields of a registry row.
const KEYS: [&str; 8] = [
    "query",
    "modality",
    "inference",
    "family",
    "default",
    "supported",
    "next_attempt",
    "imputation_estimators",
];

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Integer(i64),
    Text(String),
    List(Vec<String>),
}

fn parse_value(text: &str) -> Result<Value, String> {
    if let Some(inner) = text.strip_prefix('[') {
        let inner = inner.strip_suffix(']').ok_or("unterminated array")?.trim();
        if inner.is_empty() {
            return Ok(Value::List(Vec::new()));
        }
        return inner
            .split(',')
            .map(|item| parse_string(item.trim()))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List);
    }
    if text.starts_with('"') {
        return parse_string(text).map(Value::Text);
    }
    text.parse::<i64>().map(Value::Integer).map_err(|_| format!("unsupported value `{text}`"))
}

fn parse_string(text: &str) -> Result<String, String> {
    let inner = text
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .ok_or_else(|| format!("expected a quoted string, got `{text}`"))?;
    if inner.contains('"') || inner.contains('\\') {
        return Err(format!("unsupported string `{text}`"));
    }
    Ok(inner.to_owned())
}

fn parse_row(index: usize, fields: Vec<(String, Value)>) -> Result<MissingRouteSupport, String> {
    let at = |message: String| format!("route {}: {message}", index + 1);
    let text = |key: &str| -> Result<String, String> {
        match fields.iter().find(|(k, _)| k == key) {
            Some((_, Value::Text(value))) => Ok(value.clone()),
            Some(_) => Err(at(format!("`{key}` must be a string"))),
            None => Err(at(format!("`{key}` is missing"))),
        }
    };
    let query_name = text("query")?;
    let modality_name = text("modality")?;
    let family = text("family")?;
    let default_name = text("default")?;
    let next_name = text("next_attempt")?;
    let list = |key: &str| -> Result<Vec<String>, String> {
        match fields.iter().find(|(k, _)| k == key) {
            Some((_, Value::List(values))) => Ok(values.clone()),
            Some(_) => Err(at(format!("`{key}` must be an array of strings"))),
            None => Err(at(format!("`{key}` is missing"))),
        }
    };
    let inference_names = list("inference")?;
    let supported_names = list("supported")?;
    let imputation_estimators = list("imputation_estimators")?;
    if let Some((key, _)) = fields.iter().find(|(k, _)| !KEYS.contains(&k.as_str())) {
        return Err(at(format!("unknown field `{key}`")));
    }
    let query = ALL_QUERY_KINDS
        .into_iter()
        .find(|kind| query_kind_name(*kind) == query_name)
        .ok_or_else(|| at(format!("unknown query kind `{query_name}`")))?;
    let modality = MissingModality::from_name(&modality_name)
        .ok_or_else(|| at(format!("unknown modality `{modality_name}`")))?;
    let inference = inference_names
        .iter()
        .map(|name| {
            MissingInference::from_name(name)
                .ok_or_else(|| at(format!("unknown inference mode `{name}`")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if inference.is_empty() {
        return Err(at("`inference` is empty".to_owned()));
    }
    if !["static", "randomized", "panel", "temporal"].contains(&family.as_str()) {
        return Err(at(format!("unknown family `{family}`")));
    }
    let policy = |name: &str| {
        MissingPolicy::from_name(name).ok_or_else(|| at(format!("unknown policy `{name}`")))
    };
    let supported =
        supported_names.iter().map(|name| policy(name)).collect::<Result<Vec<_>, _>>()?;
    let default = policy(&default_name)?;
    let next_attempt = policy(&next_name)?;
    let row = MissingRouteSupport {
        query,
        modality,
        inference,
        family,
        default,
        supported,
        next_attempt,
        imputation_estimators,
    };
    if !row.supports(row.default) || !row.supports(row.next_attempt) {
        return Err(at("default and next_attempt must be supported policies".to_owned()));
    }
    let imputes = row.supported.iter().any(|p| p.imputation().is_some());
    if imputes == row.imputation_estimators.is_empty() {
        return Err(at(
            "imputation_estimators must be listed exactly when multiple_imputation is supported"
                .to_owned(),
        ));
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_parses_and_every_kind_has_its_index() {
        assert!(!missing_support_table().is_empty());
        for (index, kind) in ALL_QUERY_KINDS.into_iter().enumerate() {
            assert_eq!(query_kind_index(kind), index);
        }
    }

    #[test]
    fn malformed_registries_are_refused() {
        assert!(parse_registry("schema = 2\n").is_err());
        let row = "schema = 1\n[[route]]\nquery = \"AverageEffect\"\nmodality = \"tabular\"\n\
                   inference = [\"frequentist\"]\nfamily = \"static\"\ndefault = \"complete_case\"\n\
                   supported = [\"error\", \"complete_case\"]\nnext_attempt = \"complete_case\"\n\
                   imputation_estimators = []\n";
        assert_eq!(parse_registry(row).unwrap().len(), 1);
        assert!(parse_registry(&format!("{row}{}", &row[11..])).is_err(), "duplicate cell");
        assert!(parse_registry(&row.replace("AverageEffect", "Nope")).is_err());
        assert!(parse_registry(&row.replace("\"error\", ", "\"bogus\", ")).is_err());
        assert!(
            parse_registry(
                &row.replace(
                    "next_attempt = \"complete_case\"",
                    "next_attempt = \"balanced_panel\""
                )
            )
            .is_err()
        );
        assert!(
            parse_registry(
                &row.replace("imputation_estimators = []", "imputation_estimators = [\"aipw\"]")
            )
            .is_err()
        );
        assert!(parse_registry(&format!("{row}extra = \"x\"\n")).is_err());
    }
}
