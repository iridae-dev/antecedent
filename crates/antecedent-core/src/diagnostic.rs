//! Structured diagnostics.
//!
//! Route-native checks (positivity, censoring, dependent resampling, convergence,
//! posterior effective sample size, R-hat, constrained draws, and Monte Carlo
//! error) are records. A failed check stays failed. Unevaluable mass stays on
//! that record. Model restriction, computational failure, and unsupported
//! inference are different outcomes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Severity of a diagnostic condition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DiagnosticSeverity {
    /// Informational note.
    Info,
    /// Non-fatal scientific or operational warning.
    Warning,
    /// Severe condition that may invalidate interpretation.
    Error,
}

/// Stable diagnostic category.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiagnosticKind {
    /// Scientific / statistical condition.
    Scientific,
    /// Whether the data support the requested estimand: overlap, positivity, or an evaluation
    /// point outside the observed support. These populate the result's Support slot.
    Support,
    /// Execution / performance path choice.
    Execution,
}

/// Machine-readable diagnostic attached to analysis artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Stable diagnostic code (e.g. `exec.arrow_copied`).
    pub code: Arc<str>,
    /// Kind of diagnostic.
    pub kind: DiagnosticKind,
    /// Severity.
    pub severity: DiagnosticSeverity,
    /// Human-readable message.
    pub message: Arc<str>,
    /// Optional affected artifact identifier.
    pub artifact_id: Option<Arc<str>>,
    /// Optional structured fields as `key=value` pairs.
    pub fields: Arc<[(Arc<str>, Arc<str>)]>,
}

impl Diagnostic {
    /// Construct a diagnostic with no fields.
    #[must_use]
    pub fn new(
        code: impl Into<Arc<str>>,
        kind: DiagnosticKind,
        severity: DiagnosticSeverity,
        message: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            code: code.into(),
            kind,
            severity,
            message: message.into(),
            artifact_id: None,
            fields: Arc::from([]),
        }
    }

    /// Attach structured `key=value` fields.
    ///
    /// Readers take numbers from here; the message is prose for humans and is
    /// never parsed.
    #[must_use]
    pub fn with_fields<K, V>(mut self, fields: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<Arc<str>>,
        V: Into<Arc<str>>,
    {
        self.fields = fields.into_iter().map(|(key, value)| (key.into(), value.into())).collect();
        self
    }

    /// Attach `record`, replacing any earlier route fields.
    ///
    /// Other fields are kept. The message is not rewritten, and a failed check
    /// is not stored as a pass.
    #[must_use = "the route record is dropped if the diagnostic is discarded"]
    pub fn with_route(mut self, record: RouteNativeDiagnostic) -> Self {
        let owned = |key: &Arc<str>| {
            key.as_ref() == ROUTE_FIELD
                || key.as_ref() == OUTCOME_FIELD
                || key.as_ref() == MASS_BITS_FIELD
        };
        let mut fields: Vec<(Arc<str>, Arc<str>)> =
            self.fields.iter().filter(|(key, _)| !owned(key)).cloned().collect();
        fields.extend(record.fields());
        self.fields = Arc::from(fields);
        self
    }

    /// Unevaluable mass stored on this diagnostic, when the route record wrote one.
    ///
    /// Absence is not zero. A missing field is not rewritten into a measured mass.
    #[must_use = "a missing mass is not a measured zero"]
    pub fn unevaluable_mass(&self) -> Option<RouteMass> {
        mass_from_fields(&self.fields)
    }
}

/// Collection of diagnostics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiagnosticSet {
    /// Ordered diagnostics.
    pub entries: Vec<Diagnostic>,
}

impl DiagnosticSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a diagnostic.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.entries.push(diagnostic);
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append the diagnostic for `record`.
    pub fn push_route(&mut self, record: RouteNativeDiagnostic) {
        self.push(record.to_diagnostic());
    }

    /// Route records in entry order.
    ///
    /// A failed check is included. Unevaluable mass on that record is included.
    /// Nothing here rewrites a failure as a pass.
    #[must_use = "dropped records are not on the result"]
    pub fn route_diagnostics(&self) -> Vec<RouteNativeDiagnostic> {
        self.entries.iter().filter_map(RouteNativeDiagnostic::from_diagnostic).collect()
    }
}

const ROUTE_FIELD: &str = "route";
const OUTCOME_FIELD: &str = "outcome";
const MASS_BITS_FIELD: &str = "unevaluable_mass_bits";

const MASS_UNREPORTED: &str = "unreported";

const OUTCOME_FAILED: &str = "failed";
const OUTCOME_PASSED: &str = "passed";
const OUTCOME_MODEL_RESTRICTION: &str = "model_restriction";
const OUTCOME_COMPUTATIONAL_FAILURE: &str = "computational_failure";
const OUTCOME_UNSUPPORTED_INFERENCE: &str = "unsupported_inference";

/// Unevaluable mass was not a finite value in `[0, 1]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteMassError;

impl std::fmt::Display for RouteMassError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("unevaluable mass must be finite and in [0, 1]")
    }
}

impl std::error::Error for RouteMassError {}

/// Identified mass that was not evaluated.
///
/// Either a measured mass or an explicit [`Self::Unreported`]. A route that did
/// not report a mass has not measured zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteMass {
    /// A mass the route measured.
    Measured(MeasuredMass),
    /// The route did not report a mass.
    Unreported,
}

/// A finite mass in `[0, 1]`, compared by bits so it round-trips exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredMass {
    bits: u64,
}

impl RouteMass {
    /// Measured mass `mass`.
    ///
    /// # Errors
    ///
    /// [`RouteMassError`] when `mass` is non-finite or outside `[0, 1]`.
    #[must_use = "a refused mass was not stored"]
    pub fn new(mass: f64) -> Result<Self, RouteMassError> {
        if mass.is_finite() && (0.0..=1.0).contains(&mass) {
            Ok(Self::Measured(MeasuredMass { bits: mass.to_bits() }))
        } else {
            Err(RouteMassError)
        }
    }

    /// Measured zero. Not a stand-in for a mass the route did not report.
    #[must_use]
    pub const fn zero() -> Self {
        Self::Measured(MeasuredMass { bits: 0 })
    }

    /// A mass the route did not report.
    #[must_use]
    pub const fn unreported() -> Self {
        Self::Unreported
    }

    /// The measured mass, or `None` when unreported.
    #[must_use = "an unreported mass is not a measured zero"]
    pub const fn get(self) -> Option<f64> {
        match self {
            Self::Measured(measured) => Some(f64::from_bits(measured.bits)),
            Self::Unreported => None,
        }
    }

    /// Bit pattern of [`Self::get`], or `None` when unreported.
    #[must_use]
    pub const fn bits(self) -> Option<u64> {
        match self {
            Self::Measured(measured) => Some(measured.bits),
            Self::Unreported => None,
        }
    }

    /// Whether the route left the mass unreported.
    #[must_use]
    pub const fn is_unreported(self) -> bool {
        matches!(self, Self::Unreported)
    }
}

impl std::fmt::Display for RouteMass {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.get() {
            Some(mass) => write!(formatter, "{mass}"),
            None => formatter.write_str(MASS_UNREPORTED),
        }
    }
}

/// One route check and the unevaluable mass measured with it.
///
/// `failed` is not cleared because the mass is zero, and the mass is not
/// cleared because the check failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteCheck {
    failed: bool,
    unevaluable_mass: RouteMass,
}

impl RouteCheck {
    /// Check that failed when `failed` is set, carrying `unevaluable_mass`.
    #[must_use]
    pub const fn new(failed: bool, unevaluable_mass: RouteMass) -> Self {
        Self { failed, unevaluable_mass }
    }

    /// Whether the check failed.
    #[must_use]
    pub const fn failed(self) -> bool {
        self.failed
    }

    /// Unevaluable mass measured with this check.
    #[must_use]
    pub const fn unevaluable_mass(self) -> RouteMass {
        self.unevaluable_mass
    }
}

/// Which check a [`RouteNativeDiagnostic::Check`] records.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RouteKind {
    /// Treatment or action probability fell below the route's floor.
    Positivity,
    /// Censoring survival or censoring probability fell below the route's floor.
    Censoring,
    /// A resampling rule ignored dependence the design declared.
    DependentResampling,
    /// An optimizer or chain-movement check failed.
    Convergence,
    /// Posterior effective sample size failed its floor.
    PosteriorEss,
    /// Split R-hat failed its ceiling.
    RHat,
    /// A constrained draw or constrained fit failed its constraint check.
    ConstrainedDraws,
    /// Monte Carlo error of a reported number failed its tolerance.
    MonteCarloError,
}

impl RouteKind {
    const ALL: [Self; 8] = [
        Self::Positivity,
        Self::Censoring,
        Self::DependentResampling,
        Self::Convergence,
        Self::PosteriorEss,
        Self::RHat,
        Self::ConstrainedDraws,
        Self::MonteCarloError,
    ];

    /// Stable route name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Positivity => "positivity",
            Self::Censoring => "censoring",
            Self::DependentResampling => "dependent_resampling",
            Self::Convergence => "convergence",
            Self::PosteriorEss => "posterior_ess",
            Self::RHat => "r_hat",
            Self::ConstrainedDraws => "constrained_draws",
            Self::MonteCarloError => "monte_carlo_error",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    const fn diagnostic_kind(self) -> DiagnosticKind {
        match self {
            Self::Positivity | Self::Censoring => DiagnosticKind::Support,
            _ => DiagnosticKind::Scientific,
        }
    }
}

/// Diagnostic a route emits for a check it can fail, or for one of the three
/// outcomes that are not that check.
///
/// [`Self::ModelRestriction`], [`Self::ComputationalFailure`], and
/// [`Self::UnsupportedInference`] are different claims. A failed check is none
/// of them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RouteNativeDiagnostic {
    /// A check of `kind`, passed or failed.
    Check {
        /// Which check.
        kind: RouteKind,
        /// Outcome and unevaluable mass.
        check: RouteCheck,
    },
    /// The published functional is restricted by the model.
    ModelRestriction(RouteMass),
    /// A numerical step failed. Not an identification verdict.
    ComputationalFailure(RouteMass),
    /// This route does not perform the requested inference.
    UnsupportedInference(RouteMass),
}

impl RouteNativeDiagnostic {
    /// Check of `kind`.
    #[must_use]
    pub const fn check(kind: RouteKind, check: RouteCheck) -> Self {
        Self::Check { kind, check }
    }

    /// Model restriction carrying `unevaluable_mass`.
    #[must_use]
    pub const fn model_restriction(unevaluable_mass: RouteMass) -> Self {
        Self::ModelRestriction(unevaluable_mass)
    }

    /// Computational failure carrying `unevaluable_mass`.
    #[must_use]
    pub const fn computational_failure(unevaluable_mass: RouteMass) -> Self {
        Self::ComputationalFailure(unevaluable_mass)
    }

    /// Unsupported inference carrying `unevaluable_mass`.
    #[must_use]
    pub const fn unsupported_inference(unevaluable_mass: RouteMass) -> Self {
        Self::UnsupportedInference(unevaluable_mass)
    }

    /// Which check this records, or `None` for the three non-check outcomes.
    #[must_use]
    pub const fn kind(self) -> Option<RouteKind> {
        match self {
            Self::Check { kind, .. } => Some(kind),
            Self::ModelRestriction(_)
            | Self::ComputationalFailure(_)
            | Self::UnsupportedInference(_) => None,
        }
    }

    /// Stable route name.
    #[must_use]
    pub const fn route_name(self) -> &'static str {
        match self {
            Self::Check { kind, .. } => kind.name(),
            Self::ModelRestriction(_) => OUTCOME_MODEL_RESTRICTION,
            Self::ComputationalFailure(_) => OUTCOME_COMPUTATIONAL_FAILURE,
            Self::UnsupportedInference(_) => OUTCOME_UNSUPPORTED_INFERENCE,
        }
    }

    /// Whether this is a failed check.
    ///
    /// Model restriction, computational failure, and unsupported inference are
    /// not failed checks.
    #[must_use]
    pub const fn failed(self) -> bool {
        match self {
            Self::Check { check, .. } => check.failed(),
            Self::ModelRestriction(_)
            | Self::ComputationalFailure(_)
            | Self::UnsupportedInference(_) => false,
        }
    }

    /// Unevaluable mass on this record.
    #[must_use]
    pub const fn unevaluable_mass(self) -> RouteMass {
        match self {
            Self::Check { check, .. } => check.unevaluable_mass(),
            Self::ModelRestriction(mass)
            | Self::ComputationalFailure(mass)
            | Self::UnsupportedInference(mass) => mass,
        }
    }

    /// Whether this is a model restriction.
    #[must_use]
    pub const fn is_model_restriction(self) -> bool {
        matches!(self, Self::ModelRestriction(_))
    }

    /// Whether this is a computational failure.
    #[must_use]
    pub const fn is_computational_failure(self) -> bool {
        matches!(self, Self::ComputationalFailure(_))
    }

    /// Whether this is unsupported inference.
    #[must_use]
    pub const fn is_unsupported_inference(self) -> bool {
        matches!(self, Self::UnsupportedInference(_))
    }

    /// Diagnostic whose fields are this record.
    ///
    /// The message names the route. Readers take the failure and the mass from
    /// the fields, not from the message.
    #[must_use = "the diagnostic is the record on the result"]
    pub fn to_diagnostic(self) -> Diagnostic {
        let severity = match self {
            Self::Check { check, .. } if check.failed() => DiagnosticSeverity::Error,
            Self::Check { .. } => DiagnosticSeverity::Info,
            Self::ComputationalFailure(_) | Self::UnsupportedInference(_) => {
                DiagnosticSeverity::Error
            }
            Self::ModelRestriction(_) => DiagnosticSeverity::Warning,
        };
        let kind = match self {
            Self::Check { kind, .. } => kind.diagnostic_kind(),
            _ => DiagnosticKind::Scientific,
        };
        Diagnostic::new(
            format!("route.{}", self.route_name()),
            kind,
            severity,
            format!("route {} outcome {}", self.route_name(), self.outcome_name()),
        )
        .with_route(self)
    }

    /// Record previously written by [`Self::to_diagnostic`].
    ///
    /// A missing outcome is not a pass. A missing mass is not zero.
    #[must_use = "absence is not a passed check"]
    pub fn from_diagnostic(diagnostic: &Diagnostic) -> Option<Self> {
        let route = field(&diagnostic.fields, ROUTE_FIELD)?;
        let outcome = field(&diagnostic.fields, OUTCOME_FIELD)?;
        let mass = mass_from_fields(&diagnostic.fields)?;
        match (route, outcome) {
            (OUTCOME_MODEL_RESTRICTION, OUTCOME_MODEL_RESTRICTION) => {
                Some(Self::ModelRestriction(mass))
            }
            (OUTCOME_COMPUTATIONAL_FAILURE, OUTCOME_COMPUTATIONAL_FAILURE) => {
                Some(Self::ComputationalFailure(mass))
            }
            (OUTCOME_UNSUPPORTED_INFERENCE, OUTCOME_UNSUPPORTED_INFERENCE) => {
                Some(Self::UnsupportedInference(mass))
            }
            (name, OUTCOME_FAILED | OUTCOME_PASSED) => Some(Self::Check {
                kind: RouteKind::from_name(name)?,
                check: RouteCheck::new(outcome == OUTCOME_FAILED, mass),
            }),
            _ => None,
        }
    }

    /// Stable outcome name: `failed`, `passed`, or one of the three non-check outcomes.
    #[must_use]
    pub const fn outcome_name(self) -> &'static str {
        match self {
            Self::ModelRestriction(_) => OUTCOME_MODEL_RESTRICTION,
            Self::ComputationalFailure(_) => OUTCOME_COMPUTATIONAL_FAILURE,
            Self::UnsupportedInference(_) => OUTCOME_UNSUPPORTED_INFERENCE,
            Self::Check { check, .. } => {
                if check.failed() {
                    OUTCOME_FAILED
                } else {
                    OUTCOME_PASSED
                }
            }
        }
    }

    fn fields(self) -> [(Arc<str>, Arc<str>); 3] {
        let mass = match self.unevaluable_mass().bits() {
            Some(bits) => bits.to_string(),
            None => MASS_UNREPORTED.to_owned(),
        };
        [
            (Arc::from(ROUTE_FIELD), Arc::from(self.route_name())),
            (Arc::from(OUTCOME_FIELD), Arc::from(self.outcome_name())),
            (Arc::from(MASS_BITS_FIELD), Arc::from(mass)),
        ]
    }
}

impl std::fmt::Display for RouteNativeDiagnostic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "route {} outcome {} unevaluable_mass={}",
            self.route_name(),
            self.outcome_name(),
            self.unevaluable_mass()
        )
    }
}

fn field<'a>(fields: &'a [(Arc<str>, Arc<str>)], key: &str) -> Option<&'a str> {
    fields.iter().find(|(name, _)| name.as_ref() == key).map(|(_, value)| value.as_ref())
}

fn mass_from_fields(fields: &[(Arc<str>, Arc<str>)]) -> Option<RouteMass> {
    let stored = field(fields, MASS_BITS_FIELD)?;
    if stored == MASS_UNREPORTED {
        return Some(RouteMass::Unreported);
    }
    let bits: u64 = stored.parse().ok()?;
    RouteMass::new(f64::from_bits(bits)).ok()
}

#[cfg(test)]
mod tests {
    use super::{
        Diagnostic, DiagnosticKind, DiagnosticSet, DiagnosticSeverity, RouteCheck, RouteKind,
        RouteMass, RouteNativeDiagnostic,
    };

    fn quarter() -> RouteMass {
        RouteMass::new(0.25).expect("0.25 is a finite mass")
    }

    fn assert_round_trip(record: RouteNativeDiagnostic) {
        let restored = RouteNativeDiagnostic::from_diagnostic(&record.to_diagnostic());
        assert_eq!(restored, Some(record));
        let mut set = DiagnosticSet::new();
        set.push_route(record);
        assert_eq!(set.route_diagnostics(), vec![record]);
        assert_eq!(set.entries[0].unevaluable_mass(), Some(record.unevaluable_mass()));
    }

    #[test]
    fn positivity_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record =
            RouteNativeDiagnostic::check(RouteKind::Positivity, RouteCheck::new(true, quarter()));
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn censoring_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record =
            RouteNativeDiagnostic::check(RouteKind::Censoring, RouteCheck::new(true, quarter()));
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn dependent_resampling_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record = RouteNativeDiagnostic::check(
            RouteKind::DependentResampling,
            RouteCheck::new(true, quarter()),
        );
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn convergence_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record =
            RouteNativeDiagnostic::check(RouteKind::Convergence, RouteCheck::new(true, quarter()));
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn posterior_ess_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record =
            RouteNativeDiagnostic::check(RouteKind::PosteriorEss, RouteCheck::new(true, quarter()));
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn r_hat_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record =
            RouteNativeDiagnostic::check(RouteKind::RHat, RouteCheck::new(true, quarter()));
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn constrained_draws_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record = RouteNativeDiagnostic::check(
            RouteKind::ConstrainedDraws,
            RouteCheck::new(true, quarter()),
        );
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn monte_carlo_error_failure_keeps_the_check_and_the_unevaluable_mass() {
        let record = RouteNativeDiagnostic::check(
            RouteKind::MonteCarloError,
            RouteCheck::new(true, quarter()),
        );
        assert!(record.failed());
        assert_round_trip(record);
    }

    #[test]
    fn model_restriction_computational_failure_and_unsupported_inference_stay_distinct() {
        let mass = quarter();
        let restriction = RouteNativeDiagnostic::model_restriction(mass);
        let computational = RouteNativeDiagnostic::computational_failure(mass);
        let unsupported = RouteNativeDiagnostic::unsupported_inference(mass);
        let failed =
            RouteNativeDiagnostic::check(RouteKind::Positivity, RouteCheck::new(true, mass));
        assert_ne!(restriction, computational);
        assert_ne!(computational, unsupported);
        assert_ne!(restriction, unsupported);
        assert_ne!(failed, restriction);
        assert_ne!(failed, computational);
        assert_ne!(failed, unsupported);
        assert!(!restriction.failed());
        assert!(!computational.failed());
        assert!(!unsupported.failed());
        assert!(restriction.is_model_restriction());
        assert!(computational.is_computational_failure());
        assert!(unsupported.is_unsupported_inference());
        assert_round_trip(restriction);
        assert_round_trip(computational);
        assert_round_trip(unsupported);
    }

    #[test]
    fn a_missing_outcome_is_not_a_passed_check() {
        let diagnostic = Diagnostic::new(
            "route.positivity",
            DiagnosticKind::Support,
            DiagnosticSeverity::Info,
            "route positivity",
        )
        .with_fields([("route", "positivity"), ("unevaluable_mass_bits", "0")]);
        assert!(RouteNativeDiagnostic::from_diagnostic(&diagnostic).is_none());
    }

    #[test]
    fn an_unreported_mass_round_trips_and_is_not_zero() {
        let record = RouteNativeDiagnostic::unsupported_inference(RouteMass::unreported());
        assert!(record.unevaluable_mass().is_unreported());
        assert_ne!(RouteMass::unreported(), RouteMass::zero());
        assert_round_trip(record);
    }

    #[test]
    fn a_non_finite_unevaluable_mass_is_refused() {
        assert!(RouteMass::new(f64::NAN).is_err());
        assert!(RouteMass::new(1.5).is_err());
    }
}
