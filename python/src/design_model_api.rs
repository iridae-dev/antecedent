//! Declared models of the randomized-design Bayesian estimators, from their Python wire form.
//!
//! A posterior needs a stated likelihood and prior, so the declaration crosses the boundary as a
//! small dict (`kind` plus the numbers) and becomes a `DesignPosteriorModel` here; the numerical
//! checks, refusals and the posterior are Rust's.

use antecedent::DesignPosteriorModel;
use antecedent_estimate::randomized_posterior::{ArmBetaPrior, ArmNormalPrior, ArmOutcomeModel};
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn invalid(message: String) -> PyErr {
    crate::refusal(antecedent_core::reason_code!("invalid_argument"), message)
}

fn real(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .filter(|value| !value.is_none())
        .ok_or_else(|| invalid(format!("the declared design model needs `{key}`")))?
        .extract::<f64>()
}

fn text(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    dict.get_item(key)?
        .filter(|value| !value.is_none())
        .ok_or_else(|| invalid(format!("the declared design model needs `{key}`")))?
        .extract::<String>()
}

fn arm_outcome(dict: &Bound<'_, PyDict>) -> PyResult<DesignPosteriorModel> {
    let model = match text(dict, "outcome")?.as_str() {
        "bernoulli" => ArmOutcomeModel::Bernoulli(ArmBetaPrior {
            alpha: real(dict, "alpha")?,
            beta: real(dict, "beta")?,
        }),
        "gaussian" => ArmOutcomeModel::Gaussian(ArmNormalPrior {
            mean: real(dict, "mean")?,
            pseudo_units: real(dict, "pseudo_units")?,
            variance_shape: real(dict, "variance_shape")?,
            variance_scale: real(dict, "variance_scale")?,
        }),
        other => {
            return Err(invalid(format!(
                "unknown arm outcome {other:?}; declare `bernoulli` or `gaussian`"
            )));
        }
    };
    Ok(DesignPosteriorModel::ArmOutcome { model })
}

fn switchback(dict: &Bound<'_, PyDict>) -> PyResult<DesignPosteriorModel> {
    let carryover = match text(dict, "carryover")?.as_str() {
        "none" => CarryoverDeclaration::None,
        "first_order" => CarryoverDeclaration::FirstOrder,
        "unrestricted" => CarryoverDeclaration::Unrestricted,
        other => {
            return Err(invalid(format!(
                "unknown carryover {other:?}; declare `none`, `first_order` or `unrestricted`"
            )));
        }
    };
    let dependence = match text(dict, "dependence")?.as_str() {
        "independent" => SequenceDependence::Independent,
        "compound_symmetric" => SequenceDependence::CompoundSymmetric { rho: real(dict, "rho")? },
        "autoregressive" => SequenceDependence::Autoregressive { rho: real(dict, "rho")? },
        "unspecified" => SequenceDependence::Unspecified,
        other => {
            return Err(invalid(format!(
                "unknown dependence {other:?}; declare `independent`, `compound_symmetric` or \
                 `autoregressive`"
            )));
        }
    };
    let periods = match text(dict, "periods")?.as_str() {
        "none" => PeriodAdjustment::None,
        "fixed_effects" => PeriodAdjustment::FixedEffects,
        other => {
            return Err(invalid(format!(
                "unknown period adjustment {other:?}; use `none` or `fixed_effects`"
            )));
        }
    };
    Ok(DesignPosteriorModel::Switchback {
        model: SwitchbackModel {
            carryover,
            dependence,
            periods,
            prior: SwitchbackPrior {
                effect_precision: real(dict, "effect_precision")?,
                nuisance_precision: real(dict, "nuisance_precision")?,
                variance_shape: real(dict, "variance_shape")?,
                variance_scale: real(dict, "variance_scale")?,
            },
        },
    })
}

fn reals(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    dict.get_item(key)?
        .filter(|value| !value.is_none())
        .ok_or_else(|| invalid(format!("the declared design model needs `{key}`")))?
        .extract::<Vec<f64>>()
}

fn optional_real(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    dict.get_item(key)?.filter(|value| !value.is_none()).map(|value| value.extract()).transpose()
}

fn piecewise_hazard(dict: &Bound<'_, PyDict>) -> PyResult<DesignPosteriorModel> {
    Ok(DesignPosteriorModel::PiecewiseHazard {
        cuts: reals(dict, "cuts")?.into(),
        prior_shape: real(dict, "prior_shape")?,
        prior_rate: real(dict, "prior_rate")?,
    })
}

fn quantile_grid(dict: &Bound<'_, PyDict>) -> PyResult<DesignPosteriorModel> {
    Ok(DesignPosteriorModel::QuantileGrid {
        thresholds: reals(dict, "thresholds")?.into(),
        bin_concentration: real(dict, "bin_concentration")?,
    })
}

fn dose_polynomial(dict: &Bound<'_, PyDict>) -> PyResult<DesignPosteriorModel> {
    use antecedent_estimate::dose_basis_regression::ResidualPrior;
    let support = reals(dict, "support")?;
    let &[low, high] = support.as_slice() else {
        return Err(invalid("the dose support must be a (low, high) pair".into()));
    };
    let residual = match (
        optional_real(dict, "residual_variance")?,
        optional_real(dict, "residual_shape")?,
        optional_real(dict, "residual_scale")?,
    ) {
        (Some(variance), None, None) => ResidualPrior::KnownVariance(variance),
        (None, Some(shape), Some(scale)) => ResidualPrior::InverseGamma { shape, scale },
        _ => {
            return Err(invalid(
                "declare either residual_variance, or residual_shape and residual_scale".into(),
            ));
        }
    };
    Ok(DesignPosteriorModel::DosePolynomial {
        support: (low, high),
        coefficient_scales: reals(dict, "coefficient_scales")?.into(),
        residual,
    })
}

/// The declared model of a Python `Bayesian(design_model=...)` wire dict.
pub(crate) fn parse_design_model(dict: &Bound<'_, PyDict>) -> PyResult<DesignPosteriorModel> {
    match text(dict, "kind")?.as_str() {
        "arm_outcome" => arm_outcome(dict),
        "switchback" => switchback(dict),
        "piecewise_hazard" => piecewise_hazard(dict),
        "dose_polynomial" => dose_polynomial(dict),
        "quantile_grid" => quantile_grid(dict),
        other => Err(invalid(format!("unknown design model kind {other:?}"))),
    }
}
