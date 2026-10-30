use super::*;
use antecedent_core::execution::{AdmissionError, MapExecutionError};

#[test]
fn worker_refusals_remain_operational_and_work_errors_keep_their_type() {
    assert!(matches!(
        CausalError::from(EstimationError::Admission(AdmissionError::Cancelled)),
        CausalError::Cancelled { .. }
    ));
    assert!(matches!(
        CausalError::from(EstimationError::Admission(AdmissionError::WorkerSpawn)),
        CausalError::Resource { .. }
    ));
    let work: MapExecutionError<EstimationError> =
        MapExecutionError::Work(EstimationError::unsupported("declared test refusal"));
    assert!(matches!(
        CausalError::from(work),
        CausalError::Estimate(EstimationError::Unsupported { .. })
    ));
    assert!(matches!(
        CausalError::from(DesignError::Resource("worker unavailable".into())),
        CausalError::Resource { .. }
    ));
}
