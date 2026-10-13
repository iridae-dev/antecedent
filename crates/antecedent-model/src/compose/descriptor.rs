//! Plain, model-level descriptors of independently fitted conditional mechanisms.
//!
//! A descriptor is the *declaration* a composition is built from: which variable
//! a mechanism produces, which parents it consumes and in what units and
//! transform, the domain it was fitted on, the population and regime it was
//! fitted in, where it came from, and the parameter uncertainty its owning fit
//! reported. The numbers of the mechanism itself travel with it. Nothing here
//! executes; [`super::coherence::check_composition`] decides whether a set of
//! descriptors establishes one joint law, and the family modules execute it.
//!
//! Descriptors are keyed by *names*, never by position: parents, coefficients,
//! latent loadings, table rows and covariance entries are all addressed by the
//! variable or term they belong to, so reordering any list cannot change a
//! composition ([`MechanismDescriptor::canonical`] and
//! [`ComposedModelSpec::canonical`] sort every list into one order).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Version of the descriptor semantics. Bumped whenever the meaning of a field
/// changes, so a stored descriptor is never reread under newer semantics.
pub const MECHANISM_DESCRIPTOR_VERSION: u32 = 1;
/// Most mechanisms one composition may hold.
pub const MAX_COMPOSED_MECHANISMS: usize = 64;
/// Most parents one mechanism may declare.
pub const MAX_PARENTS_PER_MECHANISM: usize = 16;
/// Most distinct shared latent causes one composition may declare.
pub const MAX_COMPOSED_LATENTS: usize = 32;
/// Most levels one finite variable may declare.
pub const MAX_LEVELS_PER_VARIABLE: usize = 64;
/// Most joint states a finite composition may enumerate.
pub const MAX_FINITE_STATES: usize = 1 << 20;

/// The domain a quantity ranges over, or that a mechanism was fitted on.
#[derive(Clone, Debug, PartialEq)]
pub enum Domain {
    /// Every real value: the mechanism is declared valid on the whole line.
    Real,
    /// A closed numeric interval; either end may be infinite.
    Interval {
        /// Lower end.
        lo: f64,
        /// Upper end.
        hi: f64,
    },
    /// A finite set of named levels.
    Levels(Vec<String>),
}

/// How one domain sits inside another.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Containment {
    /// Every value of the inner domain is a value of the outer one.
    Contained,
    /// The domains share values but the inner one is not contained.
    Partial,
    /// The domains share no value.
    Disjoint,
    /// One domain is numeric and the other is a set of levels.
    KindMismatch,
}

impl Domain {
    fn bounds(&self) -> Option<(f64, f64)> {
        match self {
            Self::Real => Some((f64::NEG_INFINITY, f64::INFINITY)),
            Self::Interval { lo, hi } => Some((*lo, *hi)),
            Self::Levels(_) => None,
        }
    }

    /// Whether this domain is well formed: ordered non-NaN interval ends, or a
    /// non-empty set of distinct non-blank levels.
    ///
    /// # Errors
    /// A description of the defect.
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Real => Ok(()),
            Self::Interval { lo, hi } => {
                if lo.is_nan() || hi.is_nan() {
                    Err("interval end is NaN")
                } else if lo > hi {
                    Err("interval is empty (lo > hi)")
                } else {
                    Ok(())
                }
            }
            Self::Levels(levels) => {
                if levels.is_empty() {
                    return Err("level set is empty");
                }
                if levels.len() > MAX_LEVELS_PER_VARIABLE {
                    return Err("level set exceeds the level bound");
                }
                if levels.iter().any(|l| l.trim().is_empty()) {
                    return Err("blank level name");
                }
                let mut sorted: Vec<&String> = levels.iter().collect();
                sorted.sort();
                if sorted.windows(2).any(|w| w[0] == w[1]) {
                    return Err("duplicate level");
                }
                Ok(())
            }
        }
    }

    /// Whether `value` lies in a numeric domain. A level domain contains no
    /// number.
    #[must_use]
    pub fn contains_value(&self, value: f64) -> bool {
        self.bounds().is_some_and(|(lo, hi)| value >= lo && value <= hi)
    }

    /// How this domain sits inside `outer`.
    #[must_use]
    pub fn contained_in(&self, outer: &Self) -> Containment {
        match (self, outer) {
            (Self::Levels(inner), Self::Levels(wide)) => {
                let shared = inner.iter().filter(|l| wide.contains(l)).count();
                if shared == inner.len() {
                    Containment::Contained
                } else if shared > 0 {
                    Containment::Partial
                } else {
                    Containment::Disjoint
                }
            }
            (Self::Levels(_), _) | (_, Self::Levels(_)) => Containment::KindMismatch,
            _ => {
                let (alo, ahi) = self.bounds().unwrap_or((f64::NAN, f64::NAN));
                let (blo, bhi) = outer.bounds().unwrap_or((f64::NAN, f64::NAN));
                if alo >= blo && ahi <= bhi {
                    Containment::Contained
                } else if alo.max(blo) <= ahi.min(bhi) {
                    Containment::Partial
                } else {
                    Containment::Disjoint
                }
            }
        }
    }

    /// Text form for refusals.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Real => "real line".to_owned(),
            Self::Interval { lo, hi } => format!("[{lo}, {hi}]"),
            Self::Levels(levels) => format!("levels {{{}}}", levels.join(",")),
        }
    }
}

/// One parent of a mechanism, as that mechanism was fitted to see it.
#[derive(Clone, Debug, PartialEq)]
pub struct ParentBinding {
    /// Stable identity of the parent variable (never a display label).
    pub variable: String,
    /// Units the mechanism expects the parent in.
    pub units: String,
    /// Scale or transform the mechanism expects (`identity`, `log`, ...).
    pub transform: String,
    /// Domain of the parent the mechanism was fitted on.
    pub domain: Domain,
}

/// The kind of mechanism family a descriptor carries.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MechanismFamily {
    /// Additive-noise linear-Gaussian mechanisms.
    LinearGaussian,
    /// Finite-discrete conditional probability tables.
    Tabular,
}

/// One conditional probability row of a finite mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct TabularRow {
    /// Parent configuration: `(parent variable, level)` for every parent.
    pub parent_levels: Vec<(String, String)>,
    /// `(level, probability)` for every level of the produced variable.
    pub probabilities: Vec<(String, f64)>,
}

/// The numbers of a mechanism.
#[derive(Clone, Debug, PartialEq)]
pub enum MechanismForm {
    /// `Y = intercept + sum_p coefficient_p * Pa_p + sum_l loading_l * U_l + sigma * eps`.
    ///
    /// `sigma` is the *idiosyncratic* residual standard deviation: the part of
    /// the fitted residual that is not explained by a declared shared latent.
    LinearGaussian {
        /// Intercept.
        intercept: f64,
        /// Idiosyncratic residual standard deviation.
        sigma: f64,
        /// `(parent variable, coefficient)`.
        coefficients: Vec<(String, f64)>,
    },
    /// Finite conditional table `P(Y | parents)`.
    Tabular {
        /// Levels of the produced variable.
        levels: Vec<String>,
        /// One row per parent configuration.
        rows: Vec<TabularRow>,
    },
}

impl MechanismForm {
    /// The family of this form.
    #[must_use]
    pub fn family(&self) -> MechanismFamily {
        match self {
            Self::LinearGaussian { .. } => MechanismFamily::LinearGaussian,
            Self::Tabular { .. } => MechanismFamily::Tabular,
        }
    }
}

/// Population and regime a mechanism was fitted in, or a composition targets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismContext {
    /// Stable population identity.
    pub population: String,
    /// Exact regime identity; `observational` is explicit.
    pub regime: String,
}

/// Where a mechanism came from.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MechanismOrigin {
    /// Fitted by this library from retained evidence.
    Fitted,
    /// Produced by an external provider under a verified or attested binding.
    ExternalBinding,
    /// Written down by the analyst, not estimated from evidence.
    Declared,
}

/// Provenance of a mechanism.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismProvenance {
    /// Origin of the mechanism.
    pub origin: MechanismOrigin,
    /// Identity (digest) of the owning fit or external binding.
    pub artifact_id: String,
    /// Identity of the evidence the mechanism was estimated from. Two
    /// mechanisms that name the same evidence were not estimated independently.
    pub evidence_id: String,
}

/// One declared dependence of a mechanism on an unobserved shared cause.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentDependence {
    /// Stable identity of the latent cause.
    pub latent: String,
    /// Loading on a unit-variance latent, when it is known. `None` states that
    /// the dependence exists but its strength is not declared.
    pub loading: Option<f64>,
    /// The other mechanisms' variables that share this latent.
    pub shared_with: Vec<String>,
}

/// What the randomness of a parameter law represents.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum UncertaintyMeaning {
    /// Bayesian posterior covariance of the parameters.
    Posterior,
    /// Repeated-sampling covariance of the estimator.
    SamplingDistribution,
}

/// A parameter of one linear-Gaussian mechanism.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ParameterTerm {
    /// The intercept.
    Intercept,
    /// The coefficient on the named parent.
    Coefficient(String),
}

/// A parameter of the whole composition.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ParameterId {
    /// Variable of the owning mechanism.
    pub mechanism: String,
    /// Which parameter of that mechanism.
    pub term: ParameterTerm,
}

/// Marginal parameter covariance one fit reported for its own mechanism.
///
/// `covariance` is row-major `k x k` over `parameters`, in that order. The
/// residual scale is not part of the parameter vector: the delta-method route
/// covers functionals of the conditional means only.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismUncertainty {
    /// What the law means.
    pub meaning: UncertaintyMeaning,
    /// Parameters the covariance is over.
    pub parameters: Vec<ParameterTerm>,
    /// Row-major covariance.
    pub covariance: Vec<f64>,
}

/// One independently fitted conditional mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismDescriptor {
    /// Stable identity of the variable the mechanism produces.
    pub variable: String,
    /// The numbers.
    pub form: MechanismForm,
    /// Parents as the fit saw them.
    pub parents: Vec<ParentBinding>,
    /// Variables the fit conditioned on (must equal the declared parents).
    pub conditioning_set: Vec<String>,
    /// Units the mechanism produces.
    pub output_units: String,
    /// Scale or transform the mechanism produces.
    pub output_transform: String,
    /// Support of the produced variable.
    pub output_support: Domain,
    /// Population and regime of the fit.
    pub context: MechanismContext,
    /// Provenance.
    pub provenance: MechanismProvenance,
    /// Declared dependence on unobserved shared causes.
    pub latents: Vec<LatentDependence>,
    /// Marginal parameter uncertainty reported by the fit.
    pub uncertainty: Option<MechanismUncertainty>,
}

fn permute_square(values: &[f64], k: usize, perm: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(k * k);
    for &i in perm {
        for &j in perm {
            out.push(values[i * k + j]);
        }
    }
    out
}

impl MechanismDescriptor {
    /// This descriptor with every list in canonical order.
    ///
    /// Sorting never removes an entry, so a duplicate stays visible to the
    /// checker.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let mut out = self.clone();
        out.parents.sort_by(|a, b| a.variable.cmp(&b.variable));
        out.conditioning_set.sort();
        out.latents.sort_by(|a, b| a.latent.cmp(&b.latent));
        for latent in &mut out.latents {
            latent.shared_with.sort();
        }
        match &mut out.form {
            MechanismForm::LinearGaussian { coefficients, .. } => {
                coefficients.sort_by(|a, b| a.0.cmp(&b.0));
            }
            MechanismForm::Tabular { levels, rows } => {
                levels.sort();
                for row in rows.iter_mut() {
                    row.parent_levels.sort();
                    row.probabilities.sort_by(|a, b| a.0.cmp(&b.0));
                }
                rows.sort_by(|a, b| a.parent_levels.cmp(&b.parent_levels));
            }
        }
        if let Domain::Levels(levels) = &mut out.output_support {
            levels.sort();
        }
        for parent in &mut out.parents {
            if let Domain::Levels(levels) = &mut parent.domain {
                levels.sort();
            }
        }
        if let Some(u) = &mut out.uncertainty {
            let k = u.parameters.len();
            if u.covariance.len() == k * k {
                let mut perm: Vec<usize> = (0..k).collect();
                perm.sort_by(|&i, &j| u.parameters[i].cmp(&u.parameters[j]));
                u.covariance = permute_square(&u.covariance, k, &perm);
                u.parameters = perm.iter().map(|&i| u.parameters[i].clone()).collect();
            }
        }
        out
    }
}

/// A declared selection node and the variables it points into.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionNodeDeclaration {
    /// Identity of the selection node.
    pub name: String,
    /// Variables with a selection arrow into them.
    pub points_into: Vec<String>,
}

/// A declared selection diagram between two populations.
///
/// The transport theory is owned elsewhere; this only records which variables
/// carry a selection arrow, so a mechanism of a variable no selection node
/// points into can be admitted from the source population.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionDiagramDeclaration {
    /// Population the mechanisms were fitted in.
    pub source_population: String,
    /// Population the composition targets.
    pub target_population: String,
    /// Selection nodes.
    pub selection_nodes: Vec<SelectionNodeDeclaration>,
}

/// Which context coordinate an invariance declaration is about.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ContextCoordinate {
    /// The population.
    Population,
    /// The regime.
    Regime,
}

/// An explicit assumption that one mechanism is invariant across contexts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvarianceDeclaration {
    /// Variable whose mechanism is declared invariant.
    pub variable: String,
    /// Which coordinate differs.
    pub coordinate: ContextCoordinate,
    /// Source value of the coordinate (where the mechanism was fitted).
    pub source: String,
    /// Target value of the coordinate.
    pub target: String,
    /// Identity of the assumption's statement and its owner.
    pub assumption_id: String,
}

/// One declared cross-mechanism covariance block.
///
/// `values` is row-major `first_parameters.len() x second_parameters.len()`.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossCovarianceBlock {
    /// Variable of the first mechanism.
    pub first: String,
    /// Variable of the second mechanism.
    pub second: String,
    /// Rows: parameters of the first mechanism.
    pub first_parameters: Vec<ParameterTerm>,
    /// Columns: parameters of the second mechanism.
    pub second_parameters: Vec<ParameterTerm>,
    /// Row-major covariance block.
    pub values: Vec<f64>,
}

/// What is declared about the dependence between the mechanisms' parameter
/// estimates.
#[derive(Clone, Debug, PartialEq)]
pub enum PosteriorDependence {
    /// Nothing is declared. Point execution is unaffected; every uncertainty
    /// claim over more than one mechanism is refused.
    Unknown,
    /// The mechanisms were estimated from disjoint, independently generated
    /// evidence, as declared by the analyst. Never a default.
    IndependentByDesign {
        /// Why the estimates are independent.
        justification: String,
    },
    /// One joint parameter covariance, e.g. from a joint fit.
    Joint {
        /// What the law means.
        meaning: UncertaintyMeaning,
        /// Parameters, in the order of `covariance`.
        parameters: Vec<ParameterId>,
        /// Row-major covariance over `parameters`.
        covariance: Vec<f64>,
    },
    /// Explicit covariance between every pair of mechanisms; each mechanism's
    /// own marginal comes from its descriptor.
    CrossCovariance {
        /// One block per unordered pair of mechanisms.
        blocks: Vec<CrossCovarianceBlock>,
    },
}

/// Everything a composition is built from.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedModelSpec {
    /// Population and regime the composition targets.
    pub target: MechanismContext,
    /// The mechanisms.
    pub mechanisms: Vec<MechanismDescriptor>,
    /// Declared causal graph as `(parent, child)` variable pairs.
    pub edges: Vec<(String, String)>,
    /// Declared invariances admitting mechanisms from another context.
    pub invariances: Vec<InvarianceDeclaration>,
    /// Declared selection diagrams admitting mechanisms from another population.
    pub selection_diagrams: Vec<SelectionDiagramDeclaration>,
    /// What is declared about posterior dependence.
    pub dependence: PosteriorDependence,
    /// Acknowledge that [`MechanismOrigin::Declared`] mechanisms are admitted;
    /// every claim then carries that they are assumed, not estimated.
    pub admit_declared_mechanisms: bool,
}

impl ComposedModelSpec {
    /// This specification with every list in canonical order, so two
    /// declarations that differ only in listing order are equal.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let mut out = self.clone();
        out.mechanisms = self.mechanisms.iter().map(MechanismDescriptor::canonical).collect();
        out.mechanisms.sort_by(|a, b| a.variable.cmp(&b.variable));
        out.edges.sort();
        out.invariances.sort_by(|a, b| {
            (&a.variable, a.coordinate, &a.source, &a.target, &a.assumption_id).cmp(&(
                &b.variable,
                b.coordinate,
                &b.source,
                &b.target,
                &b.assumption_id,
            ))
        });
        for diagram in &mut out.selection_diagrams {
            for node in &mut diagram.selection_nodes {
                node.points_into.sort();
            }
            diagram.selection_nodes.sort_by(|a, b| a.name.cmp(&b.name));
        }
        out.selection_diagrams.sort_by(|a, b| {
            (&a.source_population, &a.target_population)
                .cmp(&(&b.source_population, &b.target_population))
        });
        out.dependence = match &self.dependence {
            PosteriorDependence::Joint { meaning, parameters, covariance } => {
                let k = parameters.len();
                if covariance.len() == k * k {
                    let mut perm: Vec<usize> = (0..k).collect();
                    perm.sort_by(|&i, &j| parameters[i].cmp(&parameters[j]));
                    PosteriorDependence::Joint {
                        meaning: *meaning,
                        parameters: perm.iter().map(|&i| parameters[i].clone()).collect(),
                        covariance: permute_square(covariance, k, &perm),
                    }
                } else {
                    self.dependence.clone()
                }
            }
            PosteriorDependence::CrossCovariance { blocks } => {
                let mut canonical: Vec<CrossCovarianceBlock> =
                    blocks.iter().map(canonical_block).collect();
                canonical.sort_by(|a, b| (&a.first, &a.second).cmp(&(&b.first, &b.second)));
                PosteriorDependence::CrossCovariance { blocks: canonical }
            }
            other => other.clone(),
        };
        out
    }
}

fn canonical_block(block: &CrossCovarianceBlock) -> CrossCovarianceBlock {
    let rows = block.first_parameters.len();
    let cols = block.second_parameters.len();
    if block.values.len() != rows * cols {
        return block.clone();
    }
    // Orient so that `first <= second`, transposing the block when swapped.
    let (first, second, fp, sp, values, r, c) = if block.first <= block.second {
        (
            block.first.clone(),
            block.second.clone(),
            block.first_parameters.clone(),
            block.second_parameters.clone(),
            block.values.clone(),
            rows,
            cols,
        )
    } else {
        let mut transposed = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                transposed.push(block.values[i * cols + j]);
            }
        }
        (
            block.second.clone(),
            block.first.clone(),
            block.second_parameters.clone(),
            block.first_parameters.clone(),
            transposed,
            cols,
            rows,
        )
    };
    let mut row_perm: Vec<usize> = (0..r).collect();
    row_perm.sort_by(|&i, &j| fp[i].cmp(&fp[j]));
    let mut col_perm: Vec<usize> = (0..c).collect();
    col_perm.sort_by(|&i, &j| sp[i].cmp(&sp[j]));
    let mut sorted = Vec::with_capacity(r * c);
    for &i in &row_perm {
        for &j in &col_perm {
            sorted.push(values[i * c + j]);
        }
    }
    CrossCovarianceBlock {
        first,
        second,
        first_parameters: row_perm.iter().map(|&i| fp[i].clone()).collect(),
        second_parameters: col_perm.iter().map(|&j| sp[j].clone()).collect(),
        values: sorted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containment_distinguishes_contained_partial_disjoint_and_kind() {
        let wide = Domain::Interval { lo: 0.0, hi: 10.0 };
        assert_eq!(
            Domain::Interval { lo: 1.0, hi: 2.0 }.contained_in(&wide),
            Containment::Contained
        );
        assert_eq!(
            Domain::Interval { lo: 5.0, hi: 20.0 }.contained_in(&wide),
            Containment::Partial
        );
        assert_eq!(
            Domain::Interval { lo: 11.0, hi: 20.0 }.contained_in(&wide),
            Containment::Disjoint
        );
        assert_eq!(Domain::Real.contained_in(&wide), Containment::Partial);
        assert_eq!(wide.contained_in(&Domain::Real), Containment::Contained);
        assert_eq!(
            Domain::Levels(vec!["a".into()]).contained_in(&wide),
            Containment::KindMismatch
        );
        assert_eq!(
            Domain::Levels(vec!["a".into(), "b".into()])
                .contained_in(&Domain::Levels(vec!["a".into(), "b".into(), "c".into()])),
            Containment::Contained
        );
    }

    #[test]
    fn cross_block_orientation_is_canonical() {
        let block = CrossCovarianceBlock {
            first: "z".into(),
            second: "a".into(),
            first_parameters: vec![ParameterTerm::Intercept],
            second_parameters: vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("q".into())],
            values: vec![1.0, 2.0],
        };
        let canon = canonical_block(&block);
        assert_eq!(canon.first, "a");
        assert_eq!(canon.second, "z");
        assert_eq!(canon.values, vec![1.0, 2.0]);
        assert_eq!(canon.first_parameters.len(), 2);
        assert_eq!(canon.second_parameters.len(), 1);
    }
}
