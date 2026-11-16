//! A prior on the outcome-regression nuisance of the cross-fitted AIPW score.
//!
//! The prior is a Gaussian coefficient prior hydrated from a posterior artifact. It
//! regularizes the per-arm outcome regressions inside every cross-fit fold: each fold fits the
//! posterior mode of the arm's linear outcome model centred on the prior mean (see
//! [`antecedent_learn::GaussianPriorRegression`]). The doubly robust correction, the
//! cross-fitting and the influence-function interval are those of the unregularized score, so
//! the prior shapes the efficiency of the estimate and never its identification: a consistent
//! propensity keeps the estimate consistent whatever the prior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    PriorAssumption,
};
use antecedent_prob::{PosteriorQuantityKind, PriorSet};

use crate::bayesian::{HydrateMapping, HydrateSource, HydrateTarget, hydrate_prior};
use crate::error::EstimationError;

/// Learner name recorded on a score table and in an estimator identity.
pub(crate) const OUTCOME_PRIOR_LEARNER: &str = "gaussian_prior_linear";

/// Restriction id of the prior in the assumptions of a result.
pub const OUTCOME_PRIOR_ASSUMPTION_ID: &str = "aipw_outcome_nuisance_prior";

/// A posterior artifact's coefficient summaries, as the outcome-nuisance prior of a
/// cross-fitted AIPW estimate.
///
/// The source is decoded once by the facade; the target design is known only when the problem
/// is prepared, so hydration into the design's coefficient coordinates happens at fit time.
#[derive(Clone, Debug, PartialEq)]
pub struct OutcomePriorSource {
    /// Source posterior quantity kinds (artifact column schema).
    pub quantities: Vec<PosteriorQuantityKind>,
    /// Source posterior means, aligned to [`Self::quantities`].
    pub mean: Vec<f64>,
    /// Source posterior SDs, aligned to [`Self::quantities`].
    pub sd: Vec<f64>,
    /// Source's absolute coefficient covariance (row-major `p²`), when its draws were
    /// retained. `None` hydrates the diagonal prior.
    pub coefficient_covariance: Option<Vec<f64>>,
    /// How the source coefficients map onto the outcome design: identical coefficient
    /// subspace or explicit named pairs. An effect-functional mapping is refused.
    pub mapping: HydrateMapping,
    /// Content identity of the source artifact (a digest of its bytes).
    pub identity: Arc<str>,
}

impl OutcomePriorSource {
    /// A source from decoded posterior summaries.
    ///
    /// # Errors
    ///
    /// Summaries whose lengths disagree, or an effect-functional mapping, which has no
    /// coordinate in the outcome regression to land on.
    pub fn new(
        quantities: Vec<PosteriorQuantityKind>,
        mean: Vec<f64>,
        sd: Vec<f64>,
        coefficient_covariance: Option<Vec<f64>>,
        mapping: HydrateMapping,
        identity: impl Into<Arc<str>>,
    ) -> Result<Self, EstimationError> {
        if mean.len() != quantities.len() || sd.len() != quantities.len() {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                "the outcome-nuisance prior source has mean/sd summaries that do not match its \
                 quantity schema; supply the artifact of a fitted posterior",
            ));
        }
        if matches!(mapping, HydrateMapping::EffectFunctional { .. }) {
            return Err(effect_functional_refusal());
        }
        Ok(Self {
            quantities,
            mean,
            sd,
            coefficient_covariance,
            mapping,
            identity: identity.into(),
        })
    }

    /// Stable identity of the prior: the source digest and the declared mapping.
    #[must_use]
    pub fn identity_key(&self) -> String {
        let mapping = match &self.mapping {
            HydrateMapping::IdenticalCoefficientSubspace => "identical".to_owned(),
            HydrateMapping::EffectFunctional { source_quantity } => {
                format!("effect_functional:{source_quantity}")
            }
            HydrateMapping::NamedParameters { pairs } => {
                let pairs: Vec<String> = pairs.iter().map(|(a, b)| format!("{a}->{b}")).collect();
                format!("named:{}", pairs.join(","))
            }
        };
        format!("{OUTCOME_PRIOR_LEARNER};prior={};mapping={mapping}", self.identity)
    }

    /// Source coefficient names in coefficient-index order (`None` for an unnamed one).
    fn source_coefficient_names(&self) -> Vec<Option<&str>> {
        let mut coefficients: Vec<(usize, Option<&str>)> = self
            .quantities
            .iter()
            .filter_map(|q| match q {
                PosteriorQuantityKind::Coefficient { index, name } => {
                    Some((*index, name.as_deref()))
                }
                _ => None,
            })
            .collect();
        coefficients.sort_by_key(|(index, _)| *index);
        coefficients.into_iter().map(|(_, name)| name).collect()
    }

    /// Hydrate the source into a Gaussian coefficient prior over `design_names`, the design
    /// columns of the outcome regression (`intercept`, then `coef_<covariate>` per adjustment
    /// variable).
    ///
    /// # Errors
    ///
    /// A prior whose coordinates do not map onto the design: a coefficient count that differs
    /// from the design width (identical mapping), source names that disagree with the design
    /// names, or a named pair naming a coefficient the source or the design does not have. The
    /// refusal names the declaration that would map them.
    pub(crate) fn hydrate(&self, design_names: &[Arc<str>]) -> Result<PriorSet, EstimationError> {
        let design_list = design_names.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(", ");
        let source_names = self.source_coefficient_names();
        let source_list = source_names
            .iter()
            .enumerate()
            .map(|(i, name)| name.map_or_else(|| format!("#{i}"), str::to_owned))
            .collect::<Vec<_>>()
            .join(", ");
        let remedy = format!(
            "declare PriorMapping.named_parameters([...]) pairing source coefficients [{source_list}] \
             with outcome-regression columns [{design_list}] (the outcome regression carries no \
             treatment coefficient), or supply a prior fitted on exactly those columns"
        );
        let unmapped = |detail: String| {
            crate::util::refuse(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                antecedent_core::reason_details::mapped_posterior_transfer::COEFFICIENT_MAP,
                &format!(
                    "the outcome-nuisance prior does not map onto the outcome regression: \
                     {detail}; {remedy}"
                ),
            )
        };
        match &self.mapping {
            HydrateMapping::EffectFunctional { .. } => return Err(effect_functional_refusal()),
            HydrateMapping::IdenticalCoefficientSubspace => {
                if source_names.len() != design_names.len() {
                    return Err(unmapped(format!(
                        "the source has {} coefficients and the design {}",
                        source_names.len(),
                        design_names.len()
                    )));
                }
                if source_names.iter().all(Option::is_some)
                    && source_names
                        .iter()
                        .zip(design_names)
                        .any(|(source, design)| source.is_some_and(|s| s != &**design))
                {
                    return Err(unmapped(
                        "the source coefficient names differ from the design column names"
                            .to_owned(),
                    ));
                }
            }
            HydrateMapping::NamedParameters { pairs } => {
                for (source, target) in pairs {
                    if !design_names.iter().any(|name| &**name == target.as_str()) {
                        return Err(unmapped(format!(
                            "`{target}` is not a column of the outcome regression"
                        )));
                    }
                    if !source_names.contains(&Some(source.as_str())) {
                        return Err(unmapped(format!(
                            "`{source}` is not a named coefficient of the source"
                        )));
                    }
                }
            }
        }
        let baseline = PriorSet::weakly_informative(design_names.len());
        hydrate_prior(
            &self.mapping,
            &HydrateSource::summaries(&self.quantities, &self.mean, &self.sd)
                .with_coefficient_covariance(self.coefficient_covariance.as_deref()),
            &HydrateTarget { baseline: &baseline, coef_names: design_names, treatment_col: None },
        )
        .map_err(|error| unmapped(error.to_string()))
    }

    /// The assumption records a result that used `prior` carries: the prior and its source, and
    /// every restriction the hydration recorded.
    pub(crate) fn assumptions(
        &self,
        prior: &PriorSet,
        design_names: &[Arc<str>],
    ) -> Vec<AssumptionRecord> {
        let design_list = design_names.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(", ");
        let mut records = vec![AssumptionRecord {
            assumption: Assumption::PriorRestriction(PriorAssumption {
                id: Arc::from(OUTCOME_PRIOR_ASSUMPTION_ID),
                description: Arc::from(format!(
                    "a Gaussian coefficient prior (source {}; mapping onto [{design_list}]) \
                     regularizes the per-arm outcome-regression nuisance of the cross-fitted \
                     AIPW score as a posterior-mode fit centred on the prior mean in every \
                     fold; the doubly robust correction, the propensity nuisance and the \
                     influence-function interval are unchanged, so the prior shapes the \
                     efficiency of the estimate and never its identification",
                    self.identity_key()
                )),
            }),
            source: AssumptionSource::Artifact,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Untestable,
        }];
        records.extend(prior.restrictions.iter().map(|restriction| AssumptionRecord {
            assumption: Assumption::PriorRestriction(restriction.clone()),
            source: AssumptionSource::Artifact,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Untestable,
        }));
        records
    }
}

fn effect_functional_refusal() -> EstimationError {
    crate::util::refuse(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        antecedent_core::reason_details::mapped_posterior_transfer::COEFFICIENT_MAP,
        "an effect-functional mapping has no coordinate in the outcome regression to land on \
         (the per-arm outcome regression carries no treatment coefficient); declare \
         PriorMapping.named_parameters([...]) onto the outcome-regression columns or use \
         the identical coefficient subspace",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(items: &[&str]) -> Vec<Arc<str>> {
        items.iter().map(|n| Arc::from(*n)).collect()
    }

    fn source(mapping: HydrateMapping) -> OutcomePriorSource {
        let quantities = vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
            PosteriorQuantityKind::Coefficient { index: 2, name: Some(Arc::from("coef_z")) },
            PosteriorQuantityKind::ResidualVariance,
        ];
        OutcomePriorSource::new(
            quantities,
            vec![0.5, 2.0, 1.5, 4.0],
            vec![0.1, 0.2, 0.3, 0.0],
            None,
            mapping,
            "blake3:test",
        )
        .unwrap()
    }

    #[test]
    fn named_pairs_hydrate_the_design_coordinates_on_the_conjugate_scale() {
        let mapping = HydrateMapping::NamedParameters {
            pairs: vec![
                ("intercept".into(), "intercept".into()),
                ("coef_z".into(), "coef_z".into()),
            ],
        };
        let prior = source(mapping).hydrate(&names(&["intercept", "coef_z"])).unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        assert_eq!(coef.mean.as_ref(), &[0.5, 1.5]);
        // Absolute variance sd^2 over the source residual variance 4.
        assert!((coef.variance[0] - 0.01 / 4.0).abs() <= 1e-15);
        assert!((coef.variance[1] - 0.09 / 4.0).abs() <= 1e-15);
    }

    #[test]
    fn coordinates_that_do_not_map_are_refused_with_the_next_attempt() {
        let design = names(&["intercept", "coef_z"]);
        let identical = source(HydrateMapping::IdenticalCoefficientSubspace);
        let refused = identical.hydrate(&design).unwrap_err().to_string();
        assert!(refused.contains("3 coefficients and the design 2"), "{refused}");
        assert!(refused.contains("PriorMapping.named_parameters"), "{refused}");
        let unknown = source(HydrateMapping::NamedParameters {
            pairs: vec![("coef_z".into(), "coef_w".into())],
        });
        let refused = unknown.hydrate(&design).unwrap_err().to_string();
        assert!(
            refused.contains("`coef_w` is not a column of the outcome regression"),
            "{refused}"
        );
        let missing = source(HydrateMapping::NamedParameters {
            pairs: vec![("coef_q".into(), "coef_z".into())],
        });
        let refused = missing.hydrate(&design).unwrap_err().to_string();
        assert!(refused.contains("`coef_q` is not a named coefficient"), "{refused}");
    }

    #[test]
    fn an_effect_functional_mapping_is_refused_at_construction() {
        let refused = OutcomePriorSource::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
            "blake3:test",
        );
        assert!(refused.is_err());
    }

    #[test]
    fn the_identity_key_names_the_source_and_the_mapping() {
        let key = source(HydrateMapping::IdenticalCoefficientSubspace).identity_key();
        assert_eq!(key, "gaussian_prior_linear;prior=blake3:test;mapping=identical");
    }
}
