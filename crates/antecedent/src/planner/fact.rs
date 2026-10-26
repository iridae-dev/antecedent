//! The typed fact tree a plan view states itself as.
//!
//! A view writes what it knows once, as a [`Fact`] tree: scalars, lists, maps by key, and
//! variables as variables. The two readers of that statement are derived from it and never
//! from each other. [`PlanView::to_json`](super::PlanView::to_json) renders the tree as JSON,
//! with a variable as its dense id; the report flattens the tree into labelled facts and
//! resolves a variable to its schema name from the node's own type, with no key-name
//! convention between the producer and the reader.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::VariableId;
use serde::Serialize;
use serde_json::Value as Json;

/// One node of a plan view's statement.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Fact {
    /// An absent layer; it states nothing.
    Null,
    /// A flag.
    Bool(bool),
    /// A signed integer.
    Int(i64),
    /// An unsigned integer.
    Uint(u64),
    /// A finite real. A non-finite value is [`Fact::Null`], as it is in JSON.
    Real(f64),
    /// Text.
    Text(String),
    /// A variable of the schema, stated by its dense id and rendered by its name.
    Variable(VariableId),
    /// An ordered list.
    List(Vec<Fact>),
    /// Facts by key, in key order.
    Map(BTreeMap<String, Fact>),
}

/// What [`Fact::get`] returns for an absent key or a node that is not a map.
static ABSENT: Fact = Fact::Null;

impl Fact {
    /// A map of the given entries; a repeated key keeps its last entry.
    pub(crate) fn map<const N: usize>(entries: [(&str, Self); N]) -> Self {
        Self::Map(entries.into_iter().map(|(key, value)| (key.to_owned(), value)).collect())
    }

    /// A list of variables.
    pub(crate) fn variables(variables: &[VariableId]) -> Self {
        Self::List(variables.iter().copied().map(Self::Variable).collect())
    }

    /// The child under `key`; absent for a missing key or a node that is not a map.
    pub(crate) fn get(&self, key: &str) -> &Self {
        match self {
            Self::Map(map) => map.get(key).unwrap_or(&ABSENT),
            _ => &ABSENT,
        }
    }

    /// Remove the child under `key` of a map.
    pub(crate) fn remove(&mut self, key: &str) {
        if let Self::Map(map) = self {
            map.remove(key);
        }
    }

    /// The text of a [`Fact::Text`].
    pub(crate) fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// A leaf lifted from a serialized value. A variable cannot arrive this way: a value that
    /// holds variables is built as a [`Fact`] by the view that knows them.
    fn from_json(json: Json) -> Self {
        match json {
            Json::Null => Self::Null,
            Json::Bool(flag) => Self::Bool(flag),
            Json::Number(number) => number
                .as_i64()
                .map(Self::Int)
                .or_else(|| number.as_u64().map(Self::Uint))
                .or_else(|| number.as_f64().map(Self::Real))
                .unwrap_or(Self::Null),
            Json::String(text) => Self::Text(text),
            Json::Array(items) => Self::List(items.into_iter().map(Self::from_json).collect()),
            Json::Object(map) => Self::Map(
                map.into_iter().map(|(key, value)| (key, Self::from_json(value))).collect(),
            ),
        }
    }
}

impl From<&Fact> for Json {
    /// The JSON of a tree: a variable is its dense id and an absent layer is `null`.
    fn from(fact: &Fact) -> Self {
        match fact {
            Fact::Null => Self::Null,
            Fact::Bool(flag) => Self::Bool(*flag),
            Fact::Int(value) => Self::from(*value),
            Fact::Uint(value) => Self::from(*value),
            Fact::Real(value) => Self::from(*value),
            Fact::Text(text) => Self::String(text.clone()),
            Fact::Variable(variable) => Self::from(variable.raw()),
            Fact::List(items) => Self::Array(items.iter().map(Self::from).collect()),
            Fact::Map(map) => Self::Object(
                map.iter().map(|(key, value)| (key.clone(), Self::from(value))).collect(),
            ),
        }
    }
}

/// A value that is already a fact tree (or a collection of them).
pub(crate) trait AsFact {
    fn as_fact(&self) -> Fact;
}

impl AsFact for Fact {
    fn as_fact(&self) -> Fact {
        self.clone()
    }
}

impl AsFact for Vec<Fact> {
    fn as_fact(&self) -> Fact {
        Fact::List(self.clone())
    }
}

impl AsFact for Option<Fact> {
    fn as_fact(&self) -> Fact {
        self.clone().unwrap_or(Fact::Null)
    }
}

/// Carrier of the two ways a value becomes a fact. A fact tree keeps its variables
/// ([`LiftTyped`]); any other serializable value is a leaf, as it is in `serde_json::json!`
/// ([`LiftSerialized`]). The by-value method is found first, so a value that is both is read as
/// a fact tree.
pub(crate) struct Lift<T>(pub(crate) T);

pub(crate) trait LiftTyped {
    fn lift(self) -> Fact;
}

pub(crate) trait LiftSerialized {
    fn lift(&self) -> Fact;
}

impl<T: AsFact + ?Sized> LiftTyped for Lift<&T> {
    fn lift(self) -> Fact {
        self.0.as_fact()
    }
}

impl<T: Serialize + ?Sized> LiftSerialized for Lift<&T> {
    fn lift(&self) -> Fact {
        serde_json::to_value(self.0).map_or(Fact::Null, Fact::from_json)
    }
}

/// A map fact written as `fact!({"key": value, ...})`: each value is a fact tree, or a
/// serializable leaf.
macro_rules! fact {
    ({ $($key:literal : $value:expr),* $(,)? }) => {{
        #[allow(unused_imports, reason = "each value picks one of the two lifting traits")]
        use $crate::planner::fact::{LiftSerialized as _, LiftTyped as _};
        $crate::planner::fact::Fact::map([
            $(($key, $crate::planner::fact::Lift(&$value).lift())),*
        ])
    }};
}

pub(crate) use fact;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_renders_variables_as_ids_and_leaves_as_json_would() {
        let tree = fact!({
            "name": "x",
            "count": 3_usize,
            "share": 0.5,
            "gone": Option::<f64>::None,
            "inputs": Fact::variables(&[VariableId::from_raw(2), VariableId::from_raw(0)]),
            "nested": fact!({"flag": true}),
            "rows": vec![fact!({"id": 1_u32}), fact!({"id": 2_u32})],
            "maybe": Some(fact!({"k": "v"})),
        });
        let json = Json::from(&tree);
        assert_eq!(
            json,
            serde_json::json!({
                "name": "x",
                "count": 3,
                "share": 0.5,
                "gone": null,
                "inputs": [2, 0],
                "nested": {"flag": true},
                "rows": [{"id": 1}, {"id": 2}],
                "maybe": {"k": "v"},
            })
        );
        assert_eq!(tree.get("nested").get("flag"), &Fact::Bool(true));
        assert_eq!(tree.get("absent").get("deeper"), &Fact::Null);
        assert_eq!(tree.get("name").as_text(), Some("x"));
    }

    #[test]
    fn a_non_finite_real_is_absent_as_it_is_in_json() {
        let tree = fact!({"nan": f64::NAN, "inf": f64::INFINITY});
        assert_eq!(tree.get("nan"), &Fact::Null);
        assert_eq!(tree.get("inf"), &Fact::Null);
    }
}
