//! Python bridge for the bounded compiled categorical posterior candidate.

use crate::{value_err, with_reason_code};
use antecedent::analysis::categorical_treatment::{
    CategoricalPosteriorDeclaration, CategoricalSpecWire, CategoricalTreatmentRequest,
};
use pyo3::prelude::*;

// The handle owns the compiled candidate; its source and receipt confer no authority.
#[pyclass(name = "CompiledCategoricalPosteriorCandidateHandle", unsendable)]
struct PyCompiledCategoricalPosteriorCandidate {
    prepared: antecedent::analysis::categorical_posterior_candidate::PreparedCategoricalPosteriorCandidate,
}

fn candidate_error(error: antecedent::CausalError) -> PyErr {
    let code = error.reason_code().unwrap_or("invalid_argument").to_owned();
    with_reason_code(value_err(error.to_string()), &code)
}

#[pymethods]
impl PyCompiledCategoricalPosteriorCandidate {
    #[new]
    fn new(
        outcome: Vec<f64>,
        levels: Vec<String>,
        declared_levels: Vec<String>,
        subject_ids: Vec<String>,
        row_snapshot: String,
        sampling_design: &str,
        seed: u64,
    ) -> PyResult<Self> {
        use antecedent::analysis::categorical_posterior_candidate::{
            CategoricalPosteriorCandidateRequest, CategoricalSamplingPremise, prepare,
        };
        use antecedent::analysis::categorical_treatment::{CovarianceKindWire, LevelScaleWire};
        use antecedent_core::sampling_design::SamplingDesign;
        if sampling_design != "independent_subject_trajectories" {
            return Err(value_err(
                "categorical candidate requires independent_subject_trajectories",
            ));
        }
        let request = CategoricalPosteriorCandidateRequest {
            data: CategoricalTreatmentRequest {
                outcome,
                levels,
                row_snapshot,
                adjustment: vec![],
                spec: CategoricalSpecWire {
                    // The first declared level is the reference of every reported contrast.
                    reference: declared_levels.first().cloned().unwrap_or_default(),
                    declared_levels,
                    scale: LevelScaleWire::Unordered,
                    min_level_rows: 2,
                    pairwise: vec![],
                    monotonicity: None,
                    covariance: CovarianceKindWire::ModelBased,
                },
            },
            posterior: CategoricalPosteriorDeclaration {
                level_precision: 1.0,
                nuisance_precision: 0.5,
                variance_shape: 2.0,
                variance_scale: 1.0,
                credible_level: 0.95,
            },
            sampling: CategoricalSamplingPremise {
                design: SamplingDesign::IndependentSubjectTrajectories,
                subject_ids,
            },
        };
        let ctx = crate::execution_api::context(seed);
        Ok(Self { prepared: prepare(request, &ctx).map_err(candidate_error)? })
    }

    /// Both values are data: independently retain the expectation to check the source later.
    fn export(&self) -> (Vec<u8>, Vec<u8>) {
        (self.prepared.export_source(), self.prepared.receipt().to_bytes().to_vec())
    }

    fn inspect(&self) -> (String, bool, Vec<u8>, Vec<u8>) {
        let receipt = self.prepared.receipt();
        (
            self.prepared.program().route().to_owned(),
            self.prepared.program().scientific().licensed(),
            receipt.source().to_vec(),
            receipt.program().to_vec(),
        )
    }

    fn interval(
        &self,
        seed: u64,
    ) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
        self.measured_interval(seed)
    }

    fn measured_interval(
        &self,
        seed: u64,
    ) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
        let ctx = crate::execution_api::context(seed);
        let measured = self
            .prepared
            .measured_interval(&ctx)
            .map_err(crate::intervention_program_api::io_err)?;
        crate::measured_inference_api::NativeMeasuredInference::from_io(measured)
    }

    #[staticmethod]
    fn consume(source: Vec<u8>, expectation: Vec<u8>, seed: u64) -> PyResult<Self> {
        use antecedent::analysis::categorical_posterior_candidate::{
            CategoricalPosteriorCompileReceipt, consume_source,
        };
        let expected = CategoricalPosteriorCompileReceipt::from_bytes(&expectation)
            .map_err(candidate_error)?;
        let ctx = crate::execution_api::context(seed);
        Ok(Self { prepared: consume_source(&source, expected, &ctx).map_err(candidate_error)? })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCompiledCategoricalPosteriorCandidate>()
}
