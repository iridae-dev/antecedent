//! Keys for the nuisances the estimators reuse, derived from the inputs each fit reads.
//!
//! Every function hashes the actual vectors: the design matrix, the treatment, the outcome,
//! the weights, the realized folds, the row identities. None accepts a digest from a caller, so
//! a fit that reads an input the key omits cannot be written without also writing the omission.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::NumericalPolicy;
use antecedent_core::SemanticDigest;
use antecedent_core::identity::ReusableProduct;
use antecedent_learn::LearnerSpec;
use antecedent_stats::GlmOptions;

use crate::nuisance_cache::{
    FoldIdentity, NuisanceCacheKey, NuisanceKeyBuilder, NuisanceOptions, NumericalDependencies,
    NumericalDependency, ProviderDependency, TargetDependency,
};
use crate::propensity::PreparedPropensityProblem;

/// Version of the producing crate: a different build is a different provider.
pub(crate) const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn glm_numerical(options: &GlmOptions) -> Option<NumericalDependencies> {
    let mut dependencies = vec![
        NumericalDependency::try_new("glm_tolerance", options.tol).ok()?,
        NumericalDependency::try_new("glm_max_iterations", f64::from(options.max_iter)).ok()?,
    ];
    if let Some(ridge) = options.ridge_on_separation {
        dependencies.push(NumericalDependency::try_new("glm_ridge_on_separation", ridge).ok()?);
    }
    NumericalDependencies::try_new(NumericalPolicy::strict(), &dependencies).ok()
}

/// The inputs a cross-fitted GLM propensity and the arm-specific outcome regressions share.
///
/// `fold_ids` are the realized folds of the complete-case rows, `sampling` the weights both
/// stages are fit under. The builder is keyed for the propensity; an outcome key adds its
/// target and its outcome column.
pub(crate) fn glm_crossfit_inputs(
    problem: &PreparedPropensityProblem,
    fold_ids: &[u32],
    folds: usize,
    glm_options: &GlmOptions,
    sampling: Option<&[f64]>,
) -> Option<NuisanceKeyBuilder> {
    let options = NuisanceOptions::new("glm_crossfit")
        .text("glm", &format!("{glm_options:?}"))
        .word("folds", folds as u64)
        .digest(
            "fold_units",
            problem
                .fold_units
                .as_deref()
                .map(|units| SemanticDigest::of_u32s("antecedent.nuisance.fold_units", units)),
        );
    Some(
        NuisanceKeyBuilder::new(ReusableProduct::PropensityFit)
            .design_matrix(problem.nrows, problem.design_ncols, &problem.design_matrix)
            .rows_by_id(problem.row_index.iter().map(|row| row.get()))
            .folds(FoldIdentity::of_fold_ids(u32::try_from(folds).ok()?, fold_ids).ok()?)
            .treatment_values("treatment", &[&*problem.treatment])
            .weights(sampling)
            .provider(ProviderDependency::named("glm_crossfit", PROVIDER_VERSION))
            .options(&options)
            .numerical(glm_numerical(glm_options)?),
    )
}

/// The outcome functional an arm-specific regression is fit to: the mean, or the indicator of
/// exceeding `threshold`.
pub(crate) fn outcome_target(threshold: Option<f64>) -> TargetDependency {
    match threshold {
        None => TargetDependency::of_functional("outcome_mean", &[]),
        Some(c) => TargetDependency::of_functional("outcome_exceeds", &[c]),
    }
}

/// Key of the learner-backed cross-fitted bundle (outcome regression, treatment propensity).
///
/// `fold_ids` are the realized folds; the declared fold-plan inputs (seed, units, kind,
/// supplied assignment) are keyed as well so a changed plan input is a miss even when it
/// happens to realize the same folds.
pub(crate) fn learner_bundle_key(
    problem: &PreparedPropensityProblem,
    outcome: LearnerSpec,
    treatment: LearnerSpec,
    folds: usize,
    seed: u64,
    fold_ids: &[u16],
    admit_boundary_saturation: bool,
) -> Option<NuisanceCacheKey> {
    let realized: Vec<u32> = fold_ids.iter().map(|&fold| u32::from(fold)).collect();
    let options = NuisanceOptions::new("learner_bundle")
        .word("admit_boundary_saturation", u64::from(admit_boundary_saturation))
        .text("outcome_learner", &outcome.identity())
        .text("treatment_learner", &treatment.identity())
        .word("folds", folds as u64)
        .word("fold_seed", problem.fold_seed)
        .text("fold_unit_kind", &format!("{:?}", problem.fold_unit_kind))
        .digest(
            "fold_units",
            problem
                .fold_units
                .as_deref()
                .map(|units| SemanticDigest::of_u32s("antecedent.nuisance.fold_units", units)),
        )
        .digest(
            "fold_assignment",
            problem
                .fold_assignment
                .as_deref()
                .map(|ids| SemanticDigest::of_u32s("antecedent.nuisance.fold_assignment", ids)),
        );
    NuisanceKeyBuilder::new(ReusableProduct::LearnerBundle)
        .design_matrix(problem.nrows, problem.design_ncols, &problem.design_matrix)
        .rows_by_id(problem.row_index.iter().map(|row| row.get()))
        .folds(FoldIdentity::of_fold_ids(u32::try_from(folds).ok()?, &realized).ok()?)
        .treatment_values("treatment", &[&*problem.treatment])
        .outcome(&problem.outcome)
        .weights(problem.sampling_weights.as_ref().map(|weights| &*weights.weights))
        .provider(ProviderDependency::named("learner_crossfit", PROVIDER_VERSION))
        .options(&options)
        .seed(seed)
        .numerical(NumericalDependencies::try_new(NumericalPolicy::strict(), &[]).ok()?)
        .build()
        .ok()
}
