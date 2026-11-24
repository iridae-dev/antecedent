//! A caller learner's fit returned as a portable prediction map.
//!
//! The compiled smoothed dose-response transport with caller learners
//! ([`super::compiled_smoothed_dose_learners`]) keeps every fold's nuisance model as a
//! provider-independent numerical map and evaluates the grid from those maps, so a caller
//! learner it accepts returns each fit as such a map: a linear predictor, optionally through a
//! logistic link, or an ordered tree ensemble. [`portable_fit`] decodes one map, binds it to the
//! design width of the fit and to the caller's provenance, and validates it; the result is the
//! fitted predictor whose portable export is that map, so the route stores and evaluates it as
//! it does a library learner's fit.
//!
//! The map is the caller's declaration of its fitted model. It is checked for shape and
//! finiteness, never for how it was fitted.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_learn::{
    FittedPredictor, LearnError, LearnerProvenance, PortablePredictor, PredictionMap,
};

/// The codec version of a portable prediction map.
const PORTABLE_MAP_VERSION: u32 = 1;

/// Decode a portable prediction map and bind it to a fit over `columns` design columns.
///
/// `map_json` is one map in its wire form: `{"kind": "linear", "coefficients": [...],
/// "logistic": false}` with one coefficient per design column (including any intercept), or
/// `{"kind": "trees", ...}`.
///
/// # Errors
///
/// [`LearnError::Unsupported`] when the text is not a portable prediction map, and
/// [`LearnError::Shape`] when the map does not fit `columns` or holds a non-finite number.
pub fn portable_fit(
    map_json: &str,
    columns: usize,
    provenance: LearnerProvenance,
) -> Result<Box<dyn FittedPredictor>, LearnError> {
    let model: PredictionMap =
        serde_json::from_str(map_json).map_err(|_| LearnError::Unsupported {
            message: "the caller learner's fit is not a portable prediction map",
        })?;
    let predictor = PortablePredictor { version: PORTABLE_MAP_VERSION, columns, provenance, model };
    predictor.validate()?;
    Ok(Box::new(predictor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::ExecutionContext;
    use antecedent_learn::DesignView;

    fn provenance() -> LearnerProvenance {
        LearnerProvenance {
            spec: "external_provider".into(),
            implementation: "test".into(),
            version: "1".into(),
        }
    }

    #[test]
    fn a_linear_map_predicts_its_linear_function_and_exports_itself() {
        let fitted = portable_fit(
            r#"{"kind": "linear", "coefficients": [1.0, 2.0], "logistic": false}"#,
            2,
            provenance(),
        )
        .unwrap();
        // Column-major design: an intercept column and x = (0, 1, 3).
        let design = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0];
        let view = DesignView::from_column_major(&design, 3, 2).unwrap();
        let mut out = [0.0; 3];
        fitted.predict(view, &mut out, &ExecutionContext::production_default(0)).unwrap();
        assert_eq!(out.map(f64::to_bits), [1.0_f64, 3.0, 7.0].map(f64::to_bits));
        let exported = fitted.portable().unwrap();
        assert_eq!(exported.columns, 2);
        assert!(matches!(exported.model, PredictionMap::Linear { logistic: false, .. }));
    }

    #[test]
    fn a_map_of_the_wrong_width_or_text_that_is_no_map_is_refused() {
        let wrong = portable_fit(
            r#"{"kind": "linear", "coefficients": [1.0], "logistic": false}"#,
            2,
            provenance(),
        );
        assert!(matches!(wrong, Err(LearnError::Shape { .. })));
        let text = portable_fit("0.5", 2, provenance());
        assert!(matches!(text, Err(LearnError::Unsupported { .. })));
    }
}
