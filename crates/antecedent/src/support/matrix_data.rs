//! Support-matrix tables loaded from `data/support_matrix.json`, with the
//! coordinate index the lookups use.
//!
//! `parity/support_*.toml` is the source of truth. `scripts/generate_support_matrix_docs.py`
//! resolves rules and licensed cells into that snapshot. The public statics stay
//! [`StaticTable`]s, not slices, so the rows are parsed on first use instead of
//! compiled as array literals. [`StaticTable::iter`] yields `'static` references
//! because the parsed rows are leaked.

#![allow(missing_docs)]

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::OnceLock;

use serde::Deserialize;

const SNAPSHOT: &str = include_str!("../../data/support_matrix.json");

/// Copy handle for a leaked slice. The static itself is const; the slice is built
/// on first use.
///
/// `iter` and `IntoIterator` return `'static` items. A [`std::sync::LazyLock`]
/// deref would borrow the lock and could not satisfy `&'static` row lookups.
#[derive(Clone, Copy, Debug)]
pub struct StaticTable<T: 'static> {
    load: fn() -> &'static [T],
}

impl<T: 'static> StaticTable<T> {
    /// `'static` iterator. Prefer this over the [`Deref`] to `[T]`, whose items
    /// are bound to the borrow of the handle.
    #[must_use = "iterate the table or use the returned rows"]
    pub fn iter(&self) -> std::slice::Iter<'static, T> {
        (self.load)().iter()
    }
}

impl<T: 'static> Deref for StaticTable<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        (self.load)()
    }
}

impl<T: 'static> IntoIterator for StaticTable<T> {
    type Item = &'static T;
    type IntoIter = std::slice::Iter<'static, T>;

    fn into_iter(self) -> Self::IntoIter {
        (self.load)().iter()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NaRule {
    pub queries: Option<&'static [&'static str]>,
    pub graph_classes: Option<&'static [&'static str]>,
    pub structures: Option<&'static [&'static str]>,
    pub inferences: Option<&'static [&'static str]>,
    pub validations: Option<&'static [&'static str]>,
    pub reason: &'static str,
    /// Closed rules only: the refused cell still executes, and its result stays
    /// unlicensed (the contract reports `refused`). Always `false` on n/a rules.
    pub executes_unlicensed: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct LicensedCell {
    pub query: &'static str,
    pub graph_class: &'static str,
    pub structure: &'static str,
    pub inference: &'static str,
    pub validation: &'static str,
    /// Compiler-plan estimator from `parity/licensed_routes.toml`, when present.
    pub route_estimator: Option<&'static str>,
    /// Union of route, explicitly licensed, and calibration-token estimators.
    pub estimators: &'static [&'static str],
}

const fn table<T: 'static>(load: fn() -> &'static [T]) -> StaticTable<T> {
    StaticTable { load }
}

pub static NA_RULES: StaticTable<NaRule> = table(na_rules);
pub static CLOSED_RULES: StaticTable<NaRule> = table(closed_rules);
pub static LICENSED: StaticTable<LicensedCell> = table(licensed);

fn na_rules() -> &'static [NaRule] {
    matrix().na_rules
}

fn closed_rules() -> &'static [NaRule] {
    matrix().closed_rules
}

fn licensed() -> &'static [LicensedCell] {
    matrix().licensed
}

/// One of the five axes of a matrix coordinate, in coordinate order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Axis {
    Query,
    GraphClass,
    Structure,
    Inference,
    Validation,
}

struct Matrix {
    na_rules: &'static [NaRule],
    closed_rules: &'static [NaRule],
    licensed: &'static [LicensedCell],
    /// Licensed rows by full five-axis coordinate (coordinate order), so a
    /// lookup is one hash instead of a scan of every row.
    by_coordinate: HashMap<[&'static str; 5], &'static LicensedCell>,
    /// Distinct licensed values of each axis, for interning a parsed coordinate.
    axis_values: [HashSet<&'static str>; 5],
}

/// The licensed row at exactly this `[query, graph_class, structure, inference,
/// validation]` coordinate.
pub(super) fn licensed_cell(coordinate: [&str; 5]) -> Option<&'static LicensedCell> {
    matrix().by_coordinate.get(&coordinate).copied()
}

/// The `'static` axis value equal to `value`, when some licensed row carries it
/// on `axis`.
pub(super) fn intern_axis(axis: Axis, value: &str) -> Option<&'static str> {
    matrix().axis_values[axis as usize].get(value).copied()
}

fn matrix() -> &'static Matrix {
    static CELL: OnceLock<Matrix> = OnceLock::new();
    CELL.get_or_init(|| {
        let raw: RawMatrix = serde_json::from_str(SNAPSHOT).expect("support matrix snapshot");
        let licensed: &'static [LicensedCell] =
            leak_slice(raw.licensed.into_iter().map(LicensedCell::from_raw).collect());
        let mut by_coordinate = HashMap::with_capacity(licensed.len());
        let mut axis_values: [HashSet<&'static str>; 5] = Default::default();
        for row in licensed {
            let coordinate =
                [row.query, row.graph_class, row.structure, row.inference, row.validation];
            // The first row at a coordinate wins, as the linear scan it replaces did.
            by_coordinate.entry(coordinate).or_insert(row);
            for (values, value) in axis_values.iter_mut().zip(coordinate) {
                values.insert(value);
            }
        }
        Matrix {
            na_rules: leak_slice(raw.na_rules.into_iter().map(NaRule::from_raw).collect()),
            closed_rules: leak_slice(raw.closed_rules.into_iter().map(NaRule::from_raw).collect()),
            licensed,
            by_coordinate,
            axis_values,
        }
    })
}

#[derive(Deserialize)]
struct RawMatrix {
    na_rules: Vec<RawRule>,
    closed_rules: Vec<RawRule>,
    licensed: Vec<RawLicensed>,
}

#[derive(Deserialize)]
struct RawRule {
    queries: Option<Vec<String>>,
    graph_classes: Option<Vec<String>>,
    structures: Option<Vec<String>>,
    inferences: Option<Vec<String>>,
    validations: Option<Vec<String>>,
    reason: String,
    #[serde(default)]
    executes_unlicensed: bool,
}

#[derive(Deserialize)]
struct RawLicensed {
    query: String,
    graph_class: String,
    structure: String,
    inference: String,
    validation: String,
    route_estimator: Option<String>,
    estimators: Vec<String>,
}

impl NaRule {
    fn from_raw(raw: RawRule) -> Self {
        Self {
            queries: opt_strs(raw.queries),
            graph_classes: opt_strs(raw.graph_classes),
            structures: opt_strs(raw.structures),
            inferences: opt_strs(raw.inferences),
            validations: opt_strs(raw.validations),
            reason: leak_str(raw.reason),
            executes_unlicensed: raw.executes_unlicensed,
        }
    }
}

impl LicensedCell {
    fn from_raw(raw: RawLicensed) -> Self {
        Self {
            query: leak_str(raw.query),
            graph_class: leak_str(raw.graph_class),
            structure: leak_str(raw.structure),
            inference: leak_str(raw.inference),
            validation: leak_str(raw.validation),
            route_estimator: raw.route_estimator.map(leak_str),
            estimators: leak_strs(raw.estimators),
        }
    }
}

fn leak_str(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn leak_slice<T>(items: Vec<T>) -> &'static [T] {
    Box::leak(items.into_boxed_slice())
}

fn leak_strs(items: Vec<String>) -> &'static [&'static str] {
    if items.is_empty() {
        return &[];
    }
    leak_slice(items.into_iter().map(leak_str).collect())
}

fn opt_strs(items: Option<Vec<String>>) -> Option<&'static [&'static str]> {
    items.map(leak_strs)
}
