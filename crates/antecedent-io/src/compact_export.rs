//! Compact runtime export of a fitted linear-in-coefficients effect model.
//!
//! The artifact stores a coefficient vector, its full covariance, the finite
//! basis specification (intercept, linear, polynomial and one-hot terms of
//! named quantities), the declared support of every input and a refusal mask
//! of query regions the model must refuse. One BLAKE3 identity binds every
//! semantic field, so a change to any coefficient, covariance entry, term,
//! support bound, mask region or scope statement is refused even when the
//! container is resealed with recomputed section checksums.
//!
//! Scope. The export evaluates a point prediction `f(x) = b' phi(x)` inside the
//! declared support and outside the refusal mask, and nothing else. The only
//! uncertainty it can report is the model-based standard error
//! `se = sqrt(phi' V phi)` from the stored covariance. That number is marked as
//! model-based with calibration unmeasured; it is never an interval or a
//! coverage claim, and it carries no extrapolation or misspecification
//! uncertainty. The verifier evaluates from the stored fields only.

use std::collections::BTreeMap;

use antecedent_core::ScientificQuantity;
use serde::{Deserialize, Serialize};

use crate::container::{
    ArtifactManifest, CompressPolicy, EncodedArtifact, SectionBytes, section_descriptor_with_policy,
};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::quantity_wire::ScientificQuantityWire;
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// Body format version this module reads and writes.
pub const COMPACT_EXPORT_VERSION: u16 = 1;
/// Artifact kind tag in the container manifest.
pub const COMPACT_EXPORT_ARTIFACT_KIND: &str = "compact_runtime_export_v1";
/// Id of the single body section.
pub const COMPACT_EXPORT_SECTION: &str = "compact_export.body";
/// Stored statement of what the point prediction means.
pub const POINT_SCOPE: &str = "point prediction f(x) = b'phi(x) of the declared linear-in-coefficients basis, valid only inside the declared support and outside the refusal mask";
/// Stored statement of what the reported uncertainty means.
pub const UNCERTAINTY_SCOPE: &str = "model-based standard error se = sqrt(phi' V phi) from the stored covariance; not an interval, not a coverage claim, and without extrapolation or misspecification uncertainty";
/// Stored calibration status of the standard error.
pub const CALIBRATION_STATUS: &str = "calibration unmeasured";
/// Label carried by every reported standard error.
pub const SE_BASIS: &str = "model_based_sqrt_phi_v_phi";
/// Largest polynomial degree a term may declare.
pub const MAX_POLYNOMIAL_DEGREE: u32 = 8;

const SYMMETRY_TOLERANCE: f64 = 1e-9;
const PSD_TOLERANCE: f64 = 1e-9;
const NEGATIVE_VARIANCE_TOLERANCE: f64 = 1e-9;

/// Bounds a consumer applies before and after decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExportLimits {
    /// Largest artifact, and largest declared uncompressed payload, in bytes.
    pub max_bytes: usize,
    /// Largest number of basis terms.
    pub max_terms: usize,
    /// Largest number of declared input quantities.
    pub max_inputs: usize,
    /// Largest number of refusal-mask regions.
    pub max_mask_regions: usize,
    /// Largest number of levels in one level set.
    pub max_levels: usize,
}

impl Default for ExportLimits {
    fn default() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024,
            max_terms: 256,
            max_inputs: 64,
            max_mask_regions: 256,
            max_levels: 1024,
        }
    }
}

/// A typed refusal. `detail` is a stable `compact_export.<snake_case>` literal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExportRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Stable refusal detail under the `compact_export` namespace.
    pub detail: &'static str,
    /// The quantity, region, field or value the refusal is about.
    pub subject: String,
}

impl std::fmt::Display for ExportRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {} ({})",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.subject
        )
    }
}

impl std::error::Error for ExportRefusal {}

fn invalid(detail: &'static str, subject: impl Into<String>) -> ExportRefusal {
    ExportRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail,
        subject: subject.into(),
    }
}

fn unlicensed(detail: &'static str, subject: impl Into<String>) -> ExportRefusal {
    ExportRefusal {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        detail,
        subject: subject.into(),
    }
}

fn io_refusal(error: &IoError) -> ExportRefusal {
    let detail = match error {
        IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. } => {
            "compact_export.unsupported_version"
        }
        IoError::TooLarge => "compact_export.oversized",
        IoError::ChecksumMismatch { .. } => "compact_export.container_checksum_mismatch",
        _ => "compact_export.container_invalid",
    };
    invalid(detail, error.to_string())
}

/// Declared support or mask set of one named quantity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSupport {
    /// Closed finite interval `[lo, hi]` of a numeric quantity.
    Range {
        /// Lower bound.
        lo: f64,
        /// Upper bound.
        hi: f64,
    },
    /// Finite level set of a categorical quantity.
    Levels {
        /// Levels, strictly ascending after canonicalisation.
        levels: Vec<String>,
    },
}

/// One declared input: its scientific coordinate and its support.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportInput {
    /// Scientific coordinate (units included); `variable_id` is the query key.
    pub quantity: ScientificQuantityWire,
    /// Declared support.
    pub support: InputSupport,
}

/// One basis term. The derived order is the canonical term order.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TermSpec {
    /// The constant 1.
    Intercept,
    /// The numeric quantity itself.
    Linear {
        /// Input `variable_id`.
        quantity: String,
    },
    /// The numeric quantity raised to `degree` (2 to [`MAX_POLYNOMIAL_DEGREE`]).
    Power {
        /// Input `variable_id`.
        quantity: String,
        /// Exponent.
        degree: u32,
    },
    /// Indicator that a categorical quantity equals `level`.
    OneHot {
        /// Input `variable_id`.
        quantity: String,
        /// Level whose indicator this term is.
        level: String,
    },
}

/// One conjunctive condition of a refusal-mask region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskCondition {
    /// Input `variable_id`.
    pub quantity: String,
    /// Closed range or level subset the query value must lie in.
    pub within: InputSupport,
}

/// A query region the model must refuse: every condition holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskRegion {
    /// Stable region id, reported in the refusal.
    pub id: String,
    /// Conditions, one per quantity, ascending by quantity.
    pub conditions: Vec<MaskCondition>,
}

/// Stored scope statements, bound into the identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportScope {
    /// What the point prediction means.
    pub point: String,
    /// What the uncertainty means.
    pub uncertainty: String,
    /// Calibration status of the uncertainty.
    pub calibration: String,
}

impl ExportScope {
    /// The only scope this module writes and accepts.
    #[must_use]
    pub fn declared() -> Self {
        Self {
            point: POINT_SCOPE.into(),
            uncertainty: UNCERTAINTY_SCOPE.into(),
            calibration: CALIBRATION_STATUS.into(),
        }
    }
}

/// The stored body: every semantic field plus the digests that bind them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactExportBody {
    /// Body format version.
    pub version: u16,
    /// Scientific coordinate of the response.
    pub response: ScientificQuantityWire,
    /// Declared inputs, ascending by `variable_id`.
    pub inputs: Vec<ExportInput>,
    /// Basis terms in canonical order.
    pub terms: Vec<TermSpec>,
    /// Coefficients aligned with `terms`.
    pub coefficients: Vec<f64>,
    /// Row-major full covariance of the coefficients.
    pub covariance: Vec<f64>,
    /// Refusal mask, ascending by region id.
    pub mask: Vec<MaskRegion>,
    /// Scope statements.
    pub scope: ExportScope,
    /// BLAKE3 over everything except coefficients and covariance.
    pub premises_blake3: String,
    /// BLAKE3 over the coefficient and covariance payload.
    pub data_blake3: String,
    /// BLAKE3 over both digests; the identity a consumer retains.
    pub identity: String,
}

struct Canon(blake3::Hasher);

impl Canon {
    fn new(domain: &str) -> Self {
        let mut canon = Self(blake3::Hasher::new());
        canon.text(domain);
        canon
    }

    fn count(&mut self, value: usize) {
        self.0.update(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
    }

    fn text(&mut self, value: &str) {
        self.count(value.len());
        self.0.update(value.as_bytes());
    }

    fn real(&mut self, value: f64) {
        self.0.update(&value.to_bits().to_le_bytes());
    }

    fn finish(self) -> String {
        self.0.finalize().to_hex().to_string()
    }
}

fn hash_quantity(canon: &mut Canon, quantity: &ScientificQuantityWire) {
    canon.count(usize::from(quantity.version));
    for field in [
        &quantity.variable_id,
        &quantity.variable_name,
        &quantity.role,
        &quantity.units,
        &quantity.population_id,
        &quantity.regime_id,
    ] {
        canon.text(field);
    }
    canon.count(quantity.horizon as usize);
    canon.text(&quantity.functional_id);
    canon.count(quantity.conditioning.len());
    for condition in &quantity.conditioning {
        canon.text(&condition.variable_id);
        canon.text(&condition.value_id);
    }
    canon.text(&quantity.transform_id);
}

fn hash_support(canon: &mut Canon, support: &InputSupport) {
    match support {
        InputSupport::Range { lo, hi } => {
            canon.text("range");
            canon.real(*lo);
            canon.real(*hi);
        }
        InputSupport::Levels { levels } => {
            canon.text("levels");
            canon.count(levels.len());
            for level in levels {
                canon.text(level);
            }
        }
    }
}

fn hash_term(canon: &mut Canon, term: &TermSpec) {
    match term {
        TermSpec::Intercept => canon.text("intercept"),
        TermSpec::Linear { quantity } => {
            canon.text("linear");
            canon.text(quantity);
        }
        TermSpec::Power { quantity, degree } => {
            canon.text("power");
            canon.text(quantity);
            canon.count(*degree as usize);
        }
        TermSpec::OneHot { quantity, level } => {
            canon.text("one_hot");
            canon.text(quantity);
            canon.text(level);
        }
    }
}

impl CompactExportBody {
    fn premises_digest(&self) -> String {
        let mut canon = Canon::new("antecedent.compact_export.v1.premises");
        canon.count(usize::from(self.version));
        hash_quantity(&mut canon, &self.response);
        canon.count(self.inputs.len());
        for input in &self.inputs {
            hash_quantity(&mut canon, &input.quantity);
            hash_support(&mut canon, &input.support);
        }
        canon.count(self.terms.len());
        for term in &self.terms {
            hash_term(&mut canon, term);
        }
        canon.count(self.mask.len());
        for region in &self.mask {
            canon.text(&region.id);
            canon.count(region.conditions.len());
            for condition in &region.conditions {
                canon.text(&condition.quantity);
                hash_support(&mut canon, &condition.within);
            }
        }
        canon.text(&self.scope.point);
        canon.text(&self.scope.uncertainty);
        canon.text(&self.scope.calibration);
        canon.finish()
    }

    fn data_digest(&self) -> String {
        let mut canon = Canon::new("antecedent.compact_export.v1.data");
        canon.count(self.coefficients.len());
        for value in &self.coefficients {
            canon.real(*value);
        }
        canon.count(self.covariance.len());
        for value in &self.covariance {
            canon.real(*value);
        }
        canon.finish()
    }

    /// Recompute `(premises, data, identity)` from the semantic fields only.
    #[must_use]
    pub fn compute_digests(&self) -> (String, String, String) {
        let premises = self.premises_digest();
        let data = self.data_digest();
        let mut canon = Canon::new("antecedent.compact_export.v1.identity");
        canon.text(&premises);
        canon.text(&data);
        (premises, data, canon.finish())
    }

    /// Overwrite the stored digests and identity with the recomputed ones.
    pub fn seal(&mut self) {
        let (premises, data, identity) = self.compute_digests();
        self.premises_blake3 = premises;
        self.data_blake3 = data;
        self.identity = identity;
    }
}

/// Everything needed to build an export. Term, input, level, mask and
/// condition order are canonicalised; `coefficients` and the row-major
/// `covariance` are aligned with `terms` as given.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSpec {
    /// Scientific coordinate of the response.
    pub response: ScientificQuantityWire,
    /// Declared inputs.
    pub inputs: Vec<ExportInput>,
    /// Basis terms.
    pub terms: Vec<TermSpec>,
    /// Coefficients aligned with `terms`.
    pub coefficients: Vec<f64>,
    /// Row-major `terms.len()` squared covariance aligned with `terms`.
    pub covariance: Vec<f64>,
    /// Refusal mask.
    pub mask: Vec<MaskRegion>,
}

/// One query value.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryValue {
    /// A numeric quantity value.
    Number(f64),
    /// A categorical quantity level.
    Level(String),
}

/// A query: one value per declared input.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExportQuery {
    /// Values keyed by input `variable_id`.
    pub values: BTreeMap<String, QueryValue>,
}

impl ExportQuery {
    /// Empty query.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a numeric value.
    #[must_use]
    pub fn with_number(mut self, quantity: &str, value: f64) -> Self {
        self.values.insert(quantity.into(), QueryValue::Number(value));
        self
    }

    /// Set a categorical level.
    #[must_use]
    pub fn with_level(mut self, quantity: &str, level: &str) -> Self {
        self.values.insert(quantity.into(), QueryValue::Level(level.into()));
        self
    }
}

/// A point prediction with its model-based standard error.
#[derive(Clone, Debug, PartialEq)]
pub struct PointWithSe {
    /// `b' phi(x)`.
    pub point: f64,
    /// `sqrt(phi' V phi)` from the stored covariance. Not an interval.
    pub model_based_se: f64,
    /// Always [`SE_BASIS`].
    pub se_basis: &'static str,
    /// Always [`CALIBRATION_STATUS`].
    pub calibration: &'static str,
    /// Identity of the export that produced the numbers.
    pub export_identity: String,
}

/// A validated compact export.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactExport {
    body: CompactExportBody,
}

fn check_support(
    support: &InputSupport,
    subject: &str,
    limits: &ExportLimits,
) -> Result<(), ExportRefusal> {
    match support {
        InputSupport::Range { lo, hi } => {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(invalid("compact_export.invalid_support", subject));
            }
        }
        InputSupport::Levels { levels } => {
            if levels.len() > limits.max_levels {
                return Err(invalid("compact_export.too_many_levels", subject));
            }
            if levels.is_empty()
                || levels.iter().any(|level| level.trim().is_empty())
                || levels.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid("compact_export.invalid_support", subject));
            }
        }
    }
    Ok(())
}

fn check_quantity(quantity: &ScientificQuantityWire) -> Result<(), ExportRefusal> {
    ScientificQuantity::try_from(quantity.clone())
        .map(|_| ())
        .map_err(|_| invalid("compact_export.invalid_quantity", quantity.variable_id.clone()))
}

/// Dimension and count checks that run before any index is trusted.
fn check_shape(body: &CompactExportBody, limits: &ExportLimits) -> Result<(), ExportRefusal> {
    let n = body.terms.len();
    if n == 0 {
        return Err(invalid("compact_export.no_terms", "terms"));
    }
    if n > limits.max_terms {
        return Err(invalid("compact_export.too_many_terms", n.to_string()));
    }
    if body.inputs.len() > limits.max_inputs {
        return Err(invalid("compact_export.too_many_inputs", body.inputs.len().to_string()));
    }
    if body.mask.len() > limits.max_mask_regions {
        return Err(invalid("compact_export.too_many_mask_regions", body.mask.len().to_string()));
    }
    if body.coefficients.len() != n {
        return Err(invalid("compact_export.dimension_mismatch", "coefficients"));
    }
    if n.checked_mul(n) != Some(body.covariance.len()) {
        return Err(invalid("compact_export.dimension_mismatch", "covariance"));
    }
    Ok(())
}

fn check_inputs(
    body: &CompactExportBody,
    limits: &ExportLimits,
) -> Result<BTreeMap<String, InputSupport>, ExportRefusal> {
    check_quantity(&body.response)?;
    if body
        .inputs
        .windows(2)
        .any(|pair| pair[0].quantity.variable_id >= pair[1].quantity.variable_id)
    {
        return Err(invalid("compact_export.inputs_not_canonical", "inputs"));
    }
    let mut supports = BTreeMap::new();
    for input in &body.inputs {
        check_quantity(&input.quantity)?;
        check_support(&input.support, &input.quantity.variable_id, limits)?;
        supports.insert(input.quantity.variable_id.clone(), input.support.clone());
    }
    Ok(supports)
}

fn check_terms(
    body: &CompactExportBody,
    supports: &BTreeMap<String, InputSupport>,
) -> Result<(), ExportRefusal> {
    if body.terms.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid("compact_export.terms_not_canonical", "terms"));
    }
    let support_of = |quantity: &String| {
        supports
            .get(quantity)
            .ok_or_else(|| invalid("compact_export.unknown_quantity", quantity.clone()))
    };
    for term in &body.terms {
        match term {
            TermSpec::Intercept => {}
            TermSpec::Linear { quantity } | TermSpec::Power { quantity, .. } => {
                if !matches!(support_of(quantity)?, InputSupport::Range { .. }) {
                    return Err(invalid("compact_export.term_support_mismatch", quantity.clone()));
                }
                if let TermSpec::Power { degree, .. } = term {
                    if !(2..=MAX_POLYNOMIAL_DEGREE).contains(degree) {
                        return Err(invalid("compact_export.invalid_degree", quantity.clone()));
                    }
                }
            }
            TermSpec::OneHot { quantity, level } => match support_of(quantity)? {
                InputSupport::Levels { levels } => {
                    if !levels.contains(level) {
                        return Err(invalid("compact_export.unknown_level", level.clone()));
                    }
                }
                InputSupport::Range { .. } => {
                    return Err(invalid("compact_export.term_support_mismatch", quantity.clone()));
                }
            },
        }
    }
    Ok(())
}

fn check_covariance(body: &CompactExportBody) -> Result<(), ExportRefusal> {
    let n = body.terms.len();
    if body.coefficients.iter().chain(&body.covariance).any(|value| !value.is_finite()) {
        return Err(invalid("compact_export.non_finite_value", "coefficients_or_covariance"));
    }
    let at = |i: usize, j: usize| body.covariance[i * n + j];
    for i in 0..n {
        if at(i, i) < 0.0 {
            return Err(invalid("compact_export.covariance_not_psd", format!("diagonal {i}")));
        }
        for j in (i + 1)..n {
            let (a, b) = (at(i, j), at(j, i));
            if (a - b).abs() > SYMMETRY_TOLERANCE * (1.0 + a.abs().max(b.abs())) {
                return Err(invalid(
                    "compact_export.covariance_asymmetric",
                    format!("entry {i},{j}"),
                ));
            }
            let bound = (at(i, i) * at(j, j)).sqrt() * (1.0 + PSD_TOLERANCE);
            if a.abs().max(b.abs()) > bound + f64::MIN_POSITIVE {
                return Err(invalid("compact_export.covariance_not_psd", format!("entry {i},{j}")));
            }
        }
    }
    Ok(())
}

fn check_region(
    region: &MaskRegion,
    supports: &BTreeMap<String, InputSupport>,
    limits: &ExportLimits,
) -> Result<(), ExportRefusal> {
    if region.id.trim().is_empty() {
        return Err(invalid("compact_export.invalid_mask_region", region.id.clone()));
    }
    if region.conditions.is_empty() {
        return Err(invalid("compact_export.empty_mask_region", region.id.clone()));
    }
    if region.conditions.windows(2).any(|pair| pair[0].quantity >= pair[1].quantity) {
        return Err(invalid("compact_export.mask_not_canonical", region.id.clone()));
    }
    for condition in &region.conditions {
        let declared = supports.get(&condition.quantity).ok_or_else(|| {
            invalid("compact_export.unknown_quantity", condition.quantity.clone())
        })?;
        check_support(&condition.within, &region.id, limits)?;
        match (&condition.within, declared) {
            (InputSupport::Range { .. }, InputSupport::Range { .. }) => {}
            (InputSupport::Levels { levels: chosen }, InputSupport::Levels { levels: all }) => {
                if chosen.iter().any(|level| !all.contains(level)) {
                    return Err(invalid("compact_export.unknown_level", region.id.clone()));
                }
            }
            _ => {
                return Err(invalid(
                    "compact_export.mask_support_mismatch",
                    condition.quantity.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn check_mask(
    body: &CompactExportBody,
    supports: &BTreeMap<String, InputSupport>,
    limits: &ExportLimits,
) -> Result<(), ExportRefusal> {
    if body.mask.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return Err(invalid("compact_export.mask_not_canonical", "mask"));
    }
    body.mask.iter().try_for_each(|region| check_region(region, supports, limits))
}

fn check_semantics(body: &CompactExportBody, limits: &ExportLimits) -> Result<(), ExportRefusal> {
    if body.scope != ExportScope::declared() {
        return Err(invalid("compact_export.scope_mismatch", "scope"));
    }
    let supports = check_inputs(body, limits)?;
    check_terms(body, &supports)?;
    check_covariance(body)?;
    check_mask(body, &supports, limits)
}

fn check_input_value(input: &ExportInput, query: &ExportQuery) -> Result<(), ExportRefusal> {
    let name = &input.quantity.variable_id;
    let value = query
        .values
        .get(name)
        .ok_or_else(|| unlicensed("compact_export.missing_quantity", name.clone()))?;
    match (&input.support, value) {
        (InputSupport::Range { lo, hi }, QueryValue::Number(x)) => {
            if !x.is_finite() {
                Err(unlicensed("compact_export.non_finite_input", name.clone()))
            } else if x < lo || x > hi {
                Err(unlicensed("compact_export.out_of_support", name.clone()))
            } else {
                Ok(())
            }
        }
        (InputSupport::Levels { levels }, QueryValue::Level(level)) => {
            if levels.contains(level) {
                Ok(())
            } else {
                Err(unlicensed("compact_export.out_of_support", name.clone()))
            }
        }
        _ => Err(unlicensed("compact_export.query_type_mismatch", name.clone())),
    }
}

fn region_matches(region: &MaskRegion, query: &ExportQuery) -> bool {
    region.conditions.iter().all(|condition| {
        match (&condition.within, query.values.get(&condition.quantity)) {
            (InputSupport::Range { lo, hi }, Some(QueryValue::Number(x))) => x >= lo && x <= hi,
            (InputSupport::Levels { levels }, Some(QueryValue::Level(level))) => {
                levels.contains(level)
            }
            _ => false,
        }
    })
}

fn number_of(query: &ExportQuery, quantity: &str) -> Result<f64, ExportRefusal> {
    match query.values.get(quantity) {
        Some(QueryValue::Number(x)) => Ok(*x),
        _ => Err(unlicensed("compact_export.query_type_mismatch", quantity)),
    }
}

fn basis_row(terms: &[TermSpec], query: &ExportQuery) -> Result<Vec<f64>, ExportRefusal> {
    terms
        .iter()
        .map(|term| match term {
            TermSpec::Intercept => Ok(1.0),
            TermSpec::Linear { quantity } => number_of(query, quantity),
            TermSpec::Power { quantity, degree } => number_of(query, quantity)
                .map(|x| x.powi(i32::try_from(*degree).unwrap_or(i32::MAX))),
            TermSpec::OneHot { quantity, level } => match query.values.get(quantity) {
                Some(QueryValue::Level(chosen)) => Ok(if chosen == level { 1.0 } else { 0.0 }),
                _ => Err(unlicensed("compact_export.query_type_mismatch", quantity.clone())),
            },
        })
        .collect()
}

/// Evaluate from the stored fields only: support, then mask, then the basis.
fn evaluate_body(
    body: &CompactExportBody,
    query: &ExportQuery,
) -> Result<PointWithSe, ExportRefusal> {
    let known = |name: &str| body.inputs.iter().any(|input| input.quantity.variable_id == name);
    if let Some(extra) = query.values.keys().find(|key| !known(key.as_str())) {
        return Err(unlicensed("compact_export.unknown_quantity", extra.clone()));
    }
    for input in &body.inputs {
        check_input_value(input, query)?;
    }
    if let Some(region) = body.mask.iter().find(|region| region_matches(region, query)) {
        return Err(unlicensed("compact_export.masked_region", region.id.clone()));
    }
    let phi = basis_row(&body.terms, query)?;
    let n = phi.len();
    let point: f64 = body.coefficients.iter().zip(&phi).map(|(b, p)| b * p).sum();
    let variance: f64 = body
        .covariance
        .chunks_exact(n.max(1))
        .zip(&phi)
        .map(|(row, pi)| pi * row.iter().zip(&phi).map(|(v, pj)| v * pj).sum::<f64>())
        .sum();
    let scale: f64 = body.covariance.iter().step_by(n + 1).zip(&phi).map(|(v, p)| v * p * p).sum();
    if variance < -NEGATIVE_VARIANCE_TOLERANCE * (1.0 + scale.abs()) {
        return Err(unlicensed("compact_export.negative_variance", "phi_v_phi"));
    }
    let se = variance.max(0.0).sqrt();
    if !point.is_finite() || !se.is_finite() {
        return Err(unlicensed("compact_export.non_finite_result", "evaluation"));
    }
    Ok(PointWithSe {
        point,
        model_based_se: se,
        se_basis: SE_BASIS,
        calibration: CALIBRATION_STATUS,
        export_identity: body.identity.clone(),
    })
}

fn canonical_support(support: &mut InputSupport) {
    if let InputSupport::Levels { levels } = support {
        levels.sort();
    }
}

fn verify_body(
    body: &CompactExportBody,
    limits: &ExportLimits,
    expected_identity: &str,
) -> Result<(), ExportRefusal> {
    if body.version != COMPACT_EXPORT_VERSION {
        return Err(invalid("compact_export.unsupported_version", body.version.to_string()));
    }
    check_shape(body, limits)?;
    let (premises, data, identity) = body.compute_digests();
    if body.premises_blake3 != premises {
        return Err(invalid("compact_export.premises_digest_mismatch", "premises_blake3"));
    }
    if body.data_blake3 != data {
        return Err(invalid("compact_export.data_digest_mismatch", "data_blake3"));
    }
    if body.identity != identity {
        return Err(invalid("compact_export.identity_mismatch", "identity"));
    }
    if body.identity != expected_identity {
        return Err(invalid("compact_export.identity_unexpected", expected_identity));
    }
    check_semantics(body, limits)
}

impl CompactExport {
    /// Validate a specification, canonicalise its order and seal its identity.
    ///
    /// The covariance must be finite, symmetric, with a nonnegative diagonal and
    /// every off-diagonal within the Cauchy-Schwarz bound (necessary conditions
    /// for positive semidefiniteness; a full factorisation is not attempted).
    ///
    /// # Errors
    /// A malformed term, support, mask, quantity or covariance refuses.
    pub fn build(spec: ExportSpec) -> Result<Self, ExportRefusal> {
        let limits = ExportLimits::default();
        let n = spec.terms.len();
        if spec.coefficients.len() != n {
            return Err(invalid("compact_export.dimension_mismatch", "coefficients"));
        }
        if n.checked_mul(n) != Some(spec.covariance.len()) {
            return Err(invalid("compact_export.dimension_mismatch", "covariance"));
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| spec.terms[a].cmp(&spec.terms[b]));
        if order.windows(2).any(|pair| spec.terms[pair[0]] == spec.terms[pair[1]]) {
            return Err(invalid("compact_export.duplicate_term", "terms"));
        }
        let source = &spec.covariance;
        let covariance: Vec<f64> =
            order.iter().flat_map(|&i| order.iter().map(move |&j| source[i * n + j])).collect();
        let mut inputs = spec.inputs;
        inputs.sort_by(|a, b| a.quantity.variable_id.cmp(&b.quantity.variable_id));
        inputs.iter_mut().for_each(|input| canonical_support(&mut input.support));
        let mut mask = spec.mask;
        for region in &mut mask {
            region.conditions.iter_mut().for_each(|c| canonical_support(&mut c.within));
            region.conditions.sort_by(|a, b| a.quantity.cmp(&b.quantity));
        }
        mask.sort_by(|a, b| a.id.cmp(&b.id));
        let mut body = CompactExportBody {
            version: COMPACT_EXPORT_VERSION,
            response: spec.response,
            inputs,
            terms: order.iter().map(|&i| spec.terms[i].clone()).collect(),
            coefficients: order.iter().map(|&i| spec.coefficients[i]).collect(),
            covariance,
            mask,
            scope: ExportScope::declared(),
            premises_blake3: String::new(),
            data_blake3: String::new(),
            identity: String::new(),
        };
        check_shape(&body, &limits)?;
        check_semantics(&body, &limits)?;
        body.seal();
        Ok(Self { body })
    }

    /// The stored, validated body.
    #[must_use]
    pub fn body(&self) -> &CompactExportBody {
        &self.body
    }

    /// The identity a consumer retains to verify the artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.body.identity
    }

    /// Point prediction and model-based standard error at a query.
    ///
    /// # Errors
    /// Out-of-support, masked, missing, unknown, mistyped or non-finite queries
    /// refuse with a `compact_export.*` detail; nothing is extrapolated.
    pub fn evaluate(&self, query: &ExportQuery) -> Result<PointWithSe, ExportRefusal> {
        evaluate_body(&self.body, query)
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// A blank artifact id, an oversized payload or an encoding failure refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, ExportRefusal> {
        if artifact_id.trim().is_empty() {
            return Err(invalid("compact_export.missing_artifact_id", "artifact_id"));
        }
        let payload = to_cbor(&self.body).map_err(|error| io_refusal(&error))?;
        let descriptor = section_descriptor_with_policy(
            COMPACT_EXPORT_SECTION,
            "application/cbor",
            &payload,
            CompressPolicy::Never,
        );
        let library_version = SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
            .map_err(|error| io_refusal(&error))?;
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: crate::migrate::STABLE_FORMAT,
                minimum_reader_version: crate::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(COMPACT_EXPORT_ARTIFACT_KIND.into()),
                library_version,
                artifact_id: artifact_id.into(),
                sections: vec![descriptor],
                provenance: ProvenanceWire { note: "compact_runtime_export".into() },
            },
            sections: vec![SectionBytes::new(COMPACT_EXPORT_SECTION, payload)],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).map_err(|error| io_refusal(&error))?;
        if bytes.len() > ExportLimits::default().max_bytes {
            return Err(invalid("compact_export.oversized", bytes.len().to_string()));
        }
        Ok(bytes)
    }

    /// Independent verifier: decode with bounded sizes, recompute the digests
    /// and identity from the stored fields, require the consumer's retained
    /// identity, then validate support, mask, basis and covariance.
    ///
    /// # Errors
    /// Oversized claims, unknown versions, a tampered coefficient, covariance,
    /// term, support, mask or scope (even when the container is resealed), an
    /// identity other than `expected_identity`, or a malformed body refuse.
    pub fn consume(
        bytes: &[u8],
        limits: &ExportLimits,
        expected_identity: &str,
    ) -> Result<Self, ExportRefusal> {
        if bytes.len() > limits.max_bytes {
            return Err(invalid("compact_export.oversized", bytes.len().to_string()));
        }
        let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))
            .map_err(|error| io_refusal(&error))?;
        let (layout_ok, declared) = {
            let manifest = reader.manifest();
            let layout_ok = manifest.artifact_kind
                == ArtifactKind::Other(COMPACT_EXPORT_ARTIFACT_KIND.into())
                && manifest.sections.len() == 1
                && manifest.sections[0].id == COMPACT_EXPORT_SECTION;
            let declared = manifest
                .sections
                .iter()
                .try_fold(0u64, |total, section| total.checked_add(section.uncompressed_size));
            (layout_ok, declared)
        };
        if !layout_ok {
            return Err(invalid("compact_export.layout_invalid", "manifest"));
        }
        let limit = u64::try_from(limits.max_bytes).unwrap_or(u64::MAX);
        if declared.is_none_or(|total| total > limit) {
            return Err(invalid("compact_export.oversized", "declared_uncompressed_size"));
        }
        let section =
            reader.load_section(COMPACT_EXPORT_SECTION).map_err(|error| io_refusal(&error))?;
        if section.as_bytes().len() > limits.max_bytes {
            return Err(invalid("compact_export.oversized", "section"));
        }
        let body: CompactExportBody =
            from_cbor(section.as_bytes()).map_err(|error| io_refusal(&error))?;
        verify_body(&body, limits, expected_identity)?;
        Ok(Self { body })
    }
}
