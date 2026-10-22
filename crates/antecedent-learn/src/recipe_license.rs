// SPDX-License-Identifier: MIT OR Apache-2.0

//! Route and test obligations of a checked recipe.
//!
//! Accepting a provider proves that a recipe is type-compatible. Compilation,
//! preparation, and that acceptance are not identification and are not
//! calibration. They do not open a support-matrix route.
//!
//! A checked recipe owes two kinds of obligation: the route (the cell
//! coordinate and its estimators) and the test (the checked-execution citation
//! for each estimator). Identification and calibration stay on the explicit
//! license. A type-compatible recipe on a closed cell stays closed. A closed
//! cell is not relabeled `not_applicable`. A licensed cell's obligations are
//! the license's route and tests. An estimator the support matrix does not
//! list is not a new license.

use std::collections::BTreeSet;
use std::fmt;

use antecedent_core::{InferenceSelection, is_blank};

use crate::recipes::Recipe;

/// Name a route the support license lists.
pub const NAME_LICENSED_ROUTE: &str = "name a route the support license lists";

/// Keep the closed reason on this route.
pub const KEEP_CLOSED_REASON: &str = "keep the closed reason on this route";

/// Emit the route and test obligations recorded on the support license.
pub const MATCH_LICENSED_OBLIGATIONS: &str =
    "emit the route and test obligations recorded on the support license";

/// How a support-matrix cell stands. A closed cell is not [`Self::NotApplicable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CellStanding {
    /// Listed by an explicit support license.
    Licensed,
    /// A closed-reason record matches. The cell stays closed.
    Closed,
    /// Not licensed. Refusal is the default.
    Unlicensed,
    /// The cell is a typed impossibility, and no closed reason matches.
    NotApplicable,
}

impl CellStanding {
    /// Stable wire spelling.
    #[must_use = "the wire spelling is the support-matrix status"]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Licensed => "licensed",
            Self::Closed => "closed",
            Self::Unlicensed => "unlicensed",
            Self::NotApplicable => "not_applicable",
        }
    }
}

impl fmt::Display for CellStanding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One support-matrix route: query, graph class, structure, inference, validation.
///
/// The query is the estimand. The spellings are those of `support_matrix.json`
/// and `support_licensed.toml`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SupportRoute {
    query: String,
    graph_class: String,
    structure: String,
    inference: String,
    validation: String,
}

impl SupportRoute {
    /// Route with those axis spellings.
    ///
    /// # Errors
    ///
    /// A blank axis. Axes are read in coordinate order, and the first blank one
    /// is the refusal.
    #[must_use = "a blank route axis is refused"]
    pub fn try_new(
        query: &str,
        graph_class: &str,
        structure: &str,
        inference: &str,
        validation: &str,
    ) -> Result<Self, LicenseRefusal> {
        let axes = [
            ("query", query),
            ("graph_class", graph_class),
            ("structure", structure),
            ("inference", inference),
            ("validation", validation),
        ];
        for (name, value) in axes {
            if is_blank(value) {
                return Err(LicenseRefusal::blank_axis(name));
            }
        }
        Ok(Self {
            query: query.to_owned(),
            graph_class: graph_class.to_owned(),
            structure: structure.to_owned(),
            inference: inference.to_owned(),
            validation: validation.to_owned(),
        })
    }

    /// Estimand. This is the query axis.
    #[must_use = "the estimand is the query the refusal names"]
    pub fn estimand(&self) -> &str {
        &self.query
    }

    /// Graph class axis.
    #[must_use = "the graph class is part of the route"]
    pub fn graph_class(&self) -> &str {
        &self.graph_class
    }

    /// Structure axis.
    #[must_use = "the structure is part of the route"]
    pub fn structure(&self) -> &str {
        &self.structure
    }

    /// Inference axis, in support-matrix spelling.
    #[must_use = "the inference spelling is part of the route"]
    pub fn inference(&self) -> &str {
        &self.inference
    }

    /// Validation axis.
    #[must_use = "the validation is part of the route"]
    pub fn validation(&self) -> &str {
        &self.validation
    }

    /// `query:graph_class:structure:inference:validation`.
    #[must_use = "the coordinate is the support-matrix route"]
    pub fn coordinate(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.query, self.graph_class, self.structure, self.inference, self.validation
        )
    }
}

/// One checked-execution citation a licensed estimator owes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedCitation {
    estimator: String,
    test: String,
    assertion: String,
}

impl CheckedCitation {
    /// Citation of `estimator` in `test` at `assertion`.
    ///
    /// # Errors
    ///
    /// A blank estimator, test path, or assertion. The route is named when
    /// `route` is supplied by [`ExplicitLicense::try_new`]; this constructor
    /// names the estimator.
    #[must_use = "a blank checked-execution citation is refused"]
    pub fn try_new(estimator: &str, test: &str, assertion: &str) -> Result<Self, LicenseRefusal> {
        if is_blank(estimator) || is_blank(test) || is_blank(assertion) {
            return Err(LicenseRefusal::blank_citation());
        }
        Ok(Self {
            estimator: estimator.to_owned(),
            test: test.to_owned(),
            assertion: assertion.to_owned(),
        })
    }

    /// Licensed estimator this citation covers.
    #[must_use = "the estimator is the licensed route or citation"]
    pub fn estimator(&self) -> &str {
        &self.estimator
    }

    /// Test path, relative to the repository.
    #[must_use = "the test path is the obligation the license records"]
    pub fn test(&self) -> &str {
        &self.test
    }

    /// Assertion the test runs.
    #[must_use = "the assertion is the obligation the license records"]
    pub fn assertion(&self) -> &str {
        &self.assertion
    }
}

/// Explicit support license. Identification and calibration stay here.
///
/// They are not recipe obligations. [`Self::obligations`] copies the route and
/// the checked-execution citations only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplicitLicense {
    route: SupportRoute,
    route_estimator: String,
    estimators: Vec<String>,
    citations: Vec<CheckedCitation>,
    identification: Option<String>,
    calibration: Option<String>,
}

impl ExplicitLicense {
    /// License for `route`.
    ///
    /// `route_estimator` is the support-matrix route estimator and must be one
    /// of `estimators`. `citations` cover each estimator once. `identification`
    /// is the evidence kind, when the license records one. `calibration` is the
    /// calibration reason or record note, when the license records one.
    ///
    /// # Errors
    ///
    /// A blank route estimator, a duplicate or empty estimator, a citation that
    /// does not cover the estimator list, or a blank identification or
    /// calibration string. `None` means the license does not record that field.
    #[must_use = "an incomplete support license is refused"]
    pub fn try_new(
        route: SupportRoute,
        route_estimator: &str,
        estimators: &[&str],
        citations: &[CheckedCitation],
        identification: Option<&str>,
        calibration: Option<&str>,
    ) -> Result<Self, LicenseRefusal> {
        if is_blank(route_estimator) {
            return Err(LicenseRefusal::not_this_license(
                &route,
                format!("{} is missing its route estimator", route.coordinate()),
            ));
        }
        let owned = owned_estimators(&route, estimators)?;
        if !owned.iter().any(|estimator| estimator == route_estimator) {
            return Err(LicenseRefusal::not_this_license(
                &route,
                format!("{} is licensed for {}", route.coordinate(), join_estimators(&owned)),
            ));
        }
        if !citations_cover(&owned, citations) {
            return Err(LicenseRefusal::not_this_license(
                &route,
                format!("{} is licensed for {}", route.coordinate(), join_estimators(&owned)),
            ));
        }
        let identification = optional_note(&route, "identification", identification)?;
        let calibration = optional_note(&route, "calibration", calibration)?;
        Ok(Self {
            route,
            route_estimator: route_estimator.to_owned(),
            estimators: owned,
            citations: citations.to_vec(),
            identification,
            calibration,
        })
    }

    /// Licensed route.
    #[must_use = "the route is the licensed cell"]
    pub fn route(&self) -> &SupportRoute {
        &self.route
    }

    /// Route estimator recorded for this cell.
    #[must_use = "the route estimator is the license, not a new one"]
    pub fn route_estimator(&self) -> &str {
        &self.route_estimator
    }

    /// Estimators this license names, in declaration order.
    #[must_use = "the estimator list is the explicit license"]
    pub fn estimators(&self) -> &[String] {
        &self.estimators
    }

    /// Checked-execution citations, in declaration order.
    #[must_use = "the citations are the test obligations"]
    pub fn citations(&self) -> &[CheckedCitation] {
        &self.citations
    }

    /// Identification recorded on the license. A recipe does not emit this.
    #[must_use = "identification stays on the explicit license"]
    pub fn identification(&self) -> Option<&str> {
        self.identification.as_deref()
    }

    /// Calibration recorded on the license. A recipe does not emit this.
    #[must_use = "calibration stays on the explicit license"]
    pub fn calibration(&self) -> Option<&str> {
        self.calibration.as_deref()
    }
}

/// Estimators `support_matrix.json` lists for one licensed cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixEstimators<'a> {
    /// Estimator the matrix records as this cell's route.
    pub route_estimator: &'a str,
    /// Every estimator the matrix lists, including a calibration join.
    pub estimators: &'a [&'a str],
}

/// What a caller asks [`decide`] to do with one route.
///
/// A closed reason is read before the license and before a typed impossibility.
/// The license is then the only source of obligations.
#[derive(Clone, Copy, Debug)]
pub struct CellDisposition<'a> {
    /// Route under consideration.
    pub route: &'a SupportRoute,
    /// Closed-reason text when a `support_closed` rule matches.
    pub closed_reason: Option<&'a str>,
    /// Typed-impossibility text. Ignored when [`Self::closed_reason`] is set.
    pub not_applicable_reason: Option<&'a str>,
    /// Explicit license. Ignored when [`Self::closed_reason`] is set.
    pub license: Option<&'a ExplicitLicense>,
    /// Matrix estimators for this route, when the caller has the snapshot.
    pub matrix: Option<MatrixEstimators<'a>>,
    /// Estimators the caller wants emitted. `None` uses the license's list.
    pub claimed_estimators: Option<&'a [&'a str]>,
}

impl<'a> CellDisposition<'a> {
    /// A route with no closed reason, no typed impossibility, and no license.
    #[must_use = "a route without a license stays shut"]
    pub const fn bare(route: &'a SupportRoute) -> Self {
        Self {
            route,
            closed_reason: None,
            not_applicable_reason: None,
            license: None,
            matrix: None,
            claimed_estimators: None,
        }
    }
}

/// Route obligation: the cell coordinate and one licensed estimator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteObligation {
    route: SupportRoute,
    estimator: String,
}

impl RouteObligation {
    /// Cell this obligation names.
    #[must_use = "the route is the licensed cell"]
    pub fn route(&self) -> &SupportRoute {
        &self.route
    }

    /// Estimand. The query of [`Self::route`].
    #[must_use = "the estimand is the query the refusal names"]
    pub fn estimand(&self) -> &str {
        self.route.estimand()
    }

    /// Licensed estimator.
    #[must_use = "the estimator is the licensed route or citation"]
    pub fn estimator(&self) -> &str {
        &self.estimator
    }
}

/// Test obligation: the checked-execution citation for one estimator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TestObligation {
    estimator: String,
    test: String,
    assertion: String,
}

impl TestObligation {
    /// Estimator the citation covers.
    #[must_use = "the estimator is the licensed route or citation"]
    pub fn estimator(&self) -> &str {
        &self.estimator
    }

    /// Test path, relative to the repository.
    #[must_use = "the test path is the obligation the license records"]
    pub fn test(&self) -> &str {
        &self.test
    }

    /// Assertion the test runs.
    #[must_use = "the assertion is the obligation the license records"]
    pub fn assertion(&self) -> &str {
        &self.assertion
    }
}

/// The two obligations a checked recipe emits.
///
/// [`Self::identification`] and [`Self::calibration`] are always absent.
/// Those fields stay on [`ExplicitLicense`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedObligations {
    route_estimator: String,
    routes: Vec<RouteObligation>,
    tests: Vec<TestObligation>,
}

impl CheckedObligations {
    /// Route estimator copied from the license.
    #[must_use = "the route estimator is the license, not a new one"]
    pub fn route_estimator(&self) -> &str {
        &self.route_estimator
    }

    /// One route obligation per licensed estimator, in license order.
    #[must_use = "route obligations are what the recipe owes"]
    pub fn routes(&self) -> &[RouteObligation] {
        &self.routes
    }

    /// One test obligation per checked-execution citation, in license order.
    #[must_use = "test obligations are what the recipe owes"]
    pub fn tests(&self) -> &[TestObligation] {
        &self.tests
    }

    /// Identification is an explicit license, not an obligation.
    ///
    /// Compilation, preparation, and recipe acceptance do not identify.
    #[must_use = "identification stays on the explicit license"]
    pub const fn identification(&self) -> Option<&str> {
        None
    }

    /// Calibration is an explicit license, not an obligation.
    ///
    /// Compilation, preparation, and recipe acceptance do not calibrate.
    #[must_use = "calibration stays on the explicit license"]
    pub const fn calibration(&self) -> Option<&str> {
        None
    }
}

/// A route a recipe did not open.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LicenseRefusal {
    estimand: String,
    coordinate: String,
    standing: CellStanding,
    detail: &'static str,
    message: String,
    next: &'static str,
}

impl LicenseRefusal {
    /// Estimand the refusal names. The query of the route that has a standing.
    #[must_use = "the estimand is the query the refusal names"]
    pub fn estimand(&self) -> &str {
        &self.estimand
    }

    /// Coordinate of that route.
    #[must_use = "the coordinate is the support-matrix route"]
    pub fn coordinate(&self) -> &str {
        &self.coordinate
    }

    /// Standing of the route. Closed is never [`CellStanding::NotApplicable`].
    #[must_use = "the standing distinguishes a closed cell from not_applicable"]
    pub const fn standing(&self) -> CellStanding {
        self.standing
    }

    /// Whether this refusal is a typed impossibility.
    ///
    /// A closed cell returns false.
    #[must_use = "a closed cell is not a typed impossibility"]
    pub const fn is_not_applicable(&self) -> bool {
        matches!(self.standing, CellStanding::NotApplicable)
    }

    /// What was refused, without the next-attempt sentence.
    #[must_use = "the message names the route that has a standing"]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Namespaced detail.
    #[must_use = "the detail is the stable refusal id"]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// Next attempt.
    #[must_use = "the next attempt names how the route stays licensed or closed"]
    pub const fn next_attempt(&self) -> &'static str {
        self.next
    }

    fn blank_axis(axis: &str) -> Self {
        Self {
            estimand: axis.to_owned(),
            coordinate: axis.to_owned(),
            standing: CellStanding::Unlicensed,
            detail: "recipe_license.blank_route",
            message: format!("the {axis} of the route is blank"),
            next: NAME_LICENSED_ROUTE,
        }
    }

    fn blank_citation() -> Self {
        Self {
            estimand: "citation".to_owned(),
            coordinate: "citation".to_owned(),
            standing: CellStanding::Unlicensed,
            detail: "recipe_license.blank_citation",
            message: "a checked-execution citation needs an estimator, a test, and an assertion"
                .to_owned(),
            next: MATCH_LICENSED_OBLIGATIONS,
        }
    }

    fn closed(route: &SupportRoute, recipe: &Recipe, reason: &str) -> Self {
        Self {
            estimand: route.estimand().to_owned(),
            coordinate: route.coordinate(),
            standing: CellStanding::Closed,
            detail: "recipe_license.closed",
            message: format!(
                "recipe {} leaves {} closed: {reason}",
                recipe.kind(),
                route.estimand()
            ),
            next: KEEP_CLOSED_REASON,
        }
    }

    fn unlicensed(route: &SupportRoute, recipe: &Recipe) -> Self {
        Self {
            estimand: route.estimand().to_owned(),
            coordinate: route.coordinate(),
            standing: CellStanding::Unlicensed,
            detail: "recipe_license.unlicensed",
            message: format!("recipe {} leaves {} unlicensed", recipe.kind(), route.coordinate()),
            next: NAME_LICENSED_ROUTE,
        }
    }

    fn not_applicable(route: &SupportRoute, recipe: &Recipe, reason: &str) -> Self {
        Self {
            estimand: route.estimand().to_owned(),
            coordinate: route.coordinate(),
            standing: CellStanding::NotApplicable,
            detail: "recipe_license.not_applicable",
            message: format!(
                "recipe {} leaves {} not_applicable: {reason}",
                recipe.kind(),
                route.estimand()
            ),
            next: NAME_LICENSED_ROUTE,
        }
    }

    fn not_this_license(route: &SupportRoute, message: String) -> Self {
        Self {
            estimand: route.estimand().to_owned(),
            coordinate: route.coordinate(),
            standing: CellStanding::Licensed,
            detail: "recipe_license.not_this_license",
            message,
            next: MATCH_LICENSED_OBLIGATIONS,
        }
    }
}

impl fmt::Display for LicenseRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{detail}: {message}; next attempt: {next}",
            detail = self.detail,
            message = self.message,
            next = self.next
        )
    }
}

impl std::error::Error for LicenseRefusal {}

/// A type-compatible `recipe` does not open `route`.
///
/// The caller may already have accepted a provider for `recipe`. That
/// acceptance is not a support license.
///
/// # Errors
///
/// Always. The standing is [`CellStanding::Unlicensed`]. The estimand is the
/// query of `route`.
#[must_use = "a type-compatible recipe does not open a route"]
pub fn open_route(
    recipe: &Recipe,
    route: &SupportRoute,
) -> Result<CheckedObligations, LicenseRefusal> {
    decide(recipe, CellDisposition::bare(route))
}

/// Obligations `recipe` owes for `cell`, or the refusal that keeps the route shut.
///
/// A closed reason wins over a license and over a typed impossibility. The
/// standing of that refusal is [`CellStanding::Closed`]. Obligations are copied
/// from the explicit license: the route, its estimators, and the checked
/// executions. Identification and calibration are not copied. A claimed
/// estimator list that differs from the license is refused, and the refusal
/// names the licensed route. Matrix estimators that are only a calibration join
/// are not copied into the obligations. A license estimator the matrix does not
/// list is refused.
///
/// # Errors
///
/// The route stays closed, unlicensed, or a typed impossibility, or the claim
/// is not the license the cell already has.
#[must_use = "a closed or unlicensed route stays shut"]
pub fn decide(
    recipe: &Recipe,
    cell: CellDisposition<'_>,
) -> Result<CheckedObligations, LicenseRefusal> {
    if let Some(reason) = nonblank(cell.closed_reason) {
        return Err(LicenseRefusal::closed(cell.route, recipe, reason));
    }
    if let Some(license) = cell.license {
        return licensed_obligations(
            recipe,
            cell.route,
            license,
            cell.matrix,
            cell.claimed_estimators,
        );
    }
    if let Some(reason) = nonblank(cell.not_applicable_reason) {
        return Err(LicenseRefusal::not_applicable(cell.route, recipe, reason));
    }
    Err(LicenseRefusal::unlicensed(cell.route, recipe))
}

fn licensed_obligations(
    recipe: &Recipe,
    route: &SupportRoute,
    license: &ExplicitLicense,
    matrix: Option<MatrixEstimators<'_>>,
    claimed: Option<&[&str]>,
) -> Result<CheckedObligations, LicenseRefusal> {
    if license.route() != route {
        return Err(LicenseRefusal::not_this_license(
            license.route(),
            format!("{} is the licensed route", license.route().coordinate()),
        ));
    }
    match matrix_inference(recipe.inference()) {
        Some(spelling) if spelling == route.inference() => {}
        Some(spelling) => {
            return Err(LicenseRefusal::not_this_license(
                route,
                format!(
                    "{} is licensed at {}; recipe {} names {spelling}",
                    route.estimand(),
                    route.inference(),
                    recipe.kind()
                ),
            ));
        }
        None => {
            return Err(LicenseRefusal::not_this_license(
                route,
                format!(
                    "{} is licensed at {}; recipe {} names no support-matrix inference",
                    route.estimand(),
                    route.inference(),
                    recipe.kind()
                ),
            ));
        }
    }
    if let Some(claimed) = claimed {
        if !same_estimators(claimed, license.estimators()) {
            return Err(licensed_for(route, license));
        }
    }
    if let Some(matrix) = matrix {
        let route_estimator = matrix.route_estimator;
        let listed =
            license.estimators().iter().any(|estimator| estimator.as_str() == route_estimator);
        if is_blank(route_estimator) || !listed {
            return Err(LicenseRefusal::not_this_license(
                route,
                format!(
                    "{} is licensed for route estimator {}",
                    route.coordinate(),
                    matrix.route_estimator
                ),
            ));
        }
        for estimator in license.estimators() {
            let on_matrix = matrix.estimators.iter().any(|listed| *listed == estimator.as_str());
            if !on_matrix {
                return Err(licensed_for(route, license));
            }
        }
    }
    Ok(CheckedObligations {
        route_estimator: license.route_estimator().to_owned(),
        routes: license
            .estimators()
            .iter()
            .map(|estimator| RouteObligation {
                route: license.route().clone(),
                estimator: estimator.clone(),
            })
            .collect(),
        tests: license
            .citations()
            .iter()
            .map(|citation| TestObligation {
                estimator: citation.estimator().to_owned(),
                test: citation.test().to_owned(),
                assertion: citation.assertion().to_owned(),
            })
            .collect(),
    })
}

fn licensed_for(route: &SupportRoute, license: &ExplicitLicense) -> LicenseRefusal {
    LicenseRefusal::not_this_license(
        route,
        format!("{} is licensed for {}", route.coordinate(), join_estimators(license.estimators())),
    )
}

fn matrix_inference(inference: InferenceSelection) -> Option<&'static str> {
    match inference {
        InferenceSelection::Frequentist => Some("Frequentist"),
        InferenceSelection::Bayesian => Some("Bayesian"),
        _ => None,
    }
}

fn nonblank(text: Option<&str>) -> Option<&str> {
    match text {
        Some(text) if !is_blank(text) => Some(text),
        _ => None,
    }
}

fn owned_estimators(
    route: &SupportRoute,
    estimators: &[&str],
) -> Result<Vec<String>, LicenseRefusal> {
    if estimators.is_empty() {
        return Err(LicenseRefusal::not_this_license(
            route,
            format!("{} is licensed for no estimator", route.coordinate()),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut owned = Vec::with_capacity(estimators.len());
    for estimator in estimators {
        if is_blank(estimator) || !seen.insert(*estimator) {
            return Err(LicenseRefusal::not_this_license(
                route,
                format!("{} is licensed for {}", route.coordinate(), join_displayed(estimators)),
            ));
        }
        owned.push((*estimator).to_owned());
    }
    Ok(owned)
}

fn citations_cover(estimators: &[String], citations: &[CheckedCitation]) -> bool {
    if citations.len() != estimators.len() {
        return false;
    }
    let mut seen = BTreeSet::new();
    for citation in citations {
        if !seen.insert(citation.estimator()) {
            return false;
        }
    }
    estimators.iter().all(|estimator| seen.contains(estimator.as_str()))
}

fn optional_note(
    route: &SupportRoute,
    field: &str,
    note: Option<&str>,
) -> Result<Option<String>, LicenseRefusal> {
    match note {
        None => Ok(None),
        Some(note) if !is_blank(note) => Ok(Some(note.to_owned())),
        Some(_) => Err(LicenseRefusal::not_this_license(
            route,
            format!("{} records a blank {field}", route.coordinate()),
        )),
    }
}

fn same_estimators(claimed: &[&str], licensed: &[String]) -> bool {
    if claimed.len() != licensed.len() {
        return false;
    }
    let mut seen = BTreeSet::new();
    for estimator in claimed {
        if is_blank(estimator) || !seen.insert(*estimator) {
            return false;
        }
    }
    licensed.iter().all(|estimator| seen.contains(estimator.as_str()))
}

fn join_estimators(estimators: &[String]) -> String {
    estimators.join(", ")
}

fn join_displayed(estimators: &[&str]) -> String {
    estimators.join(", ")
}

#[cfg(test)]
mod tests {
    use super::{
        CellDisposition, CellStanding, CheckedCitation, ExplicitLicense, MatrixEstimators,
        SupportRoute, decide, open_route,
    };
    use crate::recipes::{Recipe, RecipeKind};
    use antecedent_core::{
        InferenceSelection, ProviderCapability, ProviderDeclaration, ProviderId,
    };

    fn route(
        query: &str,
        graph_class: &str,
        structure: &str,
        inference: &str,
        validation: &str,
    ) -> SupportRoute {
        SupportRoute::try_new(query, graph_class, structure, inference, validation).unwrap()
    }

    fn frequentist_linear() -> Recipe {
        Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::Frequentist).unwrap()
    }

    fn license() -> ExplicitLicense {
        let cell = route("NestedCounterfactualEffect", "Dag", "explicit", "Frequentist", "none");
        let citation = CheckedCitation::try_new(
            "mediation.linear",
            "crates/antecedent/tests/nested_counterfactual_route_evidence.rs",
            "natural_direct_effect_retains_worlds_and_replays_outcome_fit",
        )
        .unwrap();
        ExplicitLicense::try_new(
            cell,
            "mediation.linear",
            &["mediation.linear"],
            &[citation],
            Some("internal_known_truth"),
            Some("no_interval_reported"),
        )
        .unwrap()
    }

    #[test]
    fn type_compatible_recipe_does_not_open_a_route() {
        let recipe = frequentist_linear();
        let provider = ProviderDeclaration::new(
            ProviderId::from_raw(1),
            "linear",
            [ProviderCapability::EstimatingEquations],
        )
        .unwrap();
        let accepted = recipe.accept_provider(&provider).unwrap();
        assert_eq!(accepted.recipe().kind(), RecipeKind::LinearGaussian);
        let cell = route("AverageEffect", "Dag", "explicit", "Frequentist", "none");
        let err = open_route(&accepted.recipe(), &cell).unwrap_err();
        assert_eq!(err.estimand(), "AverageEffect");
        assert_eq!(err.coordinate(), "AverageEffect:Dag:explicit:Frequentist:none");
        assert_eq!(err.standing(), CellStanding::Unlicensed);
        assert!(!err.is_not_applicable());
        assert!(err.message().contains("AverageEffect"));
        assert!(err.message().contains("linear_gaussian"));
        assert!(!err.message().contains("not_applicable"));
        assert_eq!(err.detail(), "recipe_license.unlicensed");
        assert_eq!(err.next_attempt(), super::NAME_LICENSED_ROUTE);
        let generalized =
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::GeneralizedBayesian)
                .unwrap();
        let err = open_route(&generalized, &cell).unwrap_err();
        assert_eq!(err.standing(), CellStanding::Unlicensed);
        assert_eq!(err.estimand(), "AverageEffect");
    }

    #[test]
    fn compatible_recipe_on_a_closed_cell_stays_closed() {
        let recipe = frequentist_linear();
        let cell = route("Counterfactual", "Admg", "explicit", "Frequentist", "none");
        let licensed = license();
        let mut disposition = CellDisposition::bare(&cell);
        disposition.closed_reason = Some("Counterfactual requires a supplied static Dag.");
        disposition.not_applicable_reason =
            Some("Temporal contrast queries require a temporal graph class.");
        disposition.license = Some(&licensed);
        let err = decide(&recipe, disposition).unwrap_err();
        assert_eq!(err.standing(), CellStanding::Closed);
        assert!(!err.is_not_applicable());
        assert_eq!(err.estimand(), "Counterfactual");
        assert!(err.message().contains("Counterfactual"));
        assert!(err.message().contains("supplied static Dag"));
        assert!(!err.to_string().contains("not_applicable"));
        assert_eq!(err.next_attempt(), super::KEEP_CLOSED_REASON);
    }

    #[test]
    fn licensed_obligations_match_the_support_record() {
        let recipe = frequentist_linear();
        let licensed = license();
        let matrix = MatrixEstimators {
            route_estimator: "mediation.linear",
            estimators: &["mediation.linear"],
        };
        let mut disposition = CellDisposition::bare(licensed.route());
        disposition.license = Some(&licensed);
        disposition.matrix = Some(matrix);
        let obligations = decide(&recipe, disposition).unwrap();
        assert_eq!(obligations.route_estimator(), "mediation.linear");
        assert_eq!(obligations.routes().len(), 1);
        assert_eq!(obligations.routes()[0].estimand(), "NestedCounterfactualEffect");
        assert_eq!(obligations.routes()[0].estimator(), "mediation.linear");
        assert_eq!(
            obligations.routes()[0].route().coordinate(),
            "NestedCounterfactualEffect:Dag:explicit:Frequentist:none"
        );
        assert_eq!(obligations.tests().len(), 1);
        assert_eq!(obligations.tests()[0].estimator(), "mediation.linear");
        assert_eq!(
            obligations.tests()[0].assertion(),
            "natural_direct_effect_retains_worlds_and_replays_outcome_fit"
        );
        assert_eq!(obligations.identification(), None);
        assert_eq!(obligations.calibration(), None);
        assert_eq!(licensed.identification(), Some("internal_known_truth"));
        assert_eq!(licensed.calibration(), Some("no_interval_reported"));

        let mut invented = disposition;
        invented.claimed_estimators = Some(&["other.estimator"]);
        let err = decide(&recipe, invented).unwrap_err();
        assert_eq!(err.estimand(), "NestedCounterfactualEffect");
        assert_eq!(err.standing(), CellStanding::Licensed);
        assert!(!err.is_not_applicable());
        assert!(err.message().contains("mediation.linear"));
        assert!(!err.message().contains("other.estimator"));

        let mut extra = disposition;
        extra.matrix = Some(MatrixEstimators {
            route_estimator: "mediation.linear",
            estimators: &["mediation.linear", "cell.aipw"],
        });
        let obligations = decide(&recipe, extra).unwrap();
        assert_eq!(obligations.routes().len(), 1);
        assert_eq!(obligations.routes()[0].estimator(), "mediation.linear");
        assert!(obligations.routes().iter().all(|item| item.estimator() != "cell.aipw"));

        let generalized =
            Recipe::new(RecipeKind::LinearGaussian, InferenceSelection::GeneralizedBayesian)
                .unwrap();
        let err = decide(&generalized, disposition).unwrap_err();
        assert_eq!(err.estimand(), "NestedCounterfactualEffect");
        assert_eq!(err.standing(), CellStanding::Licensed);
        assert!(err.message().contains("Frequentist"));
    }

    #[test]
    fn typed_impossibility_is_not_the_label_of_a_closed_cell() {
        let recipe = frequentist_linear();
        let cell = route("PulseEffect", "Dag", "explicit", "Frequentist", "none");
        let mut disposition = CellDisposition::bare(&cell);
        disposition.not_applicable_reason =
            Some("Temporal contrast queries require a temporal graph class.");
        let err = decide(&recipe, disposition).unwrap_err();
        assert_eq!(err.standing(), CellStanding::NotApplicable);
        assert!(err.is_not_applicable());
        assert_eq!(err.estimand(), "PulseEffect");
        disposition.closed_reason = Some("a closed reason is on file");
        let err = decide(&recipe, disposition).unwrap_err();
        assert_eq!(err.standing(), CellStanding::Closed);
        assert!(!err.is_not_applicable());
        assert_eq!(err.estimand(), "PulseEffect");
    }
}
