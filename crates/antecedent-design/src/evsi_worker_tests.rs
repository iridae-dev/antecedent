use super::*;
use antecedent_core::execution::{AdmissionError, MapExecutionError};

#[test]
fn scheduler_refusals_do_not_become_signal_refusals() {
    let cancelled = EvsiError::from(MapExecutionError::Admission(AdmissionError::Cancelled));
    assert!(matches!(cancelled, EvsiError::Cancelled));
    let resource = EvsiError::from(MapExecutionError::Admission(AdmissionError::WorkerSpawn));
    assert!(matches!(&resource, EvsiError::Design(DesignError::Resource(_))));
    assert_eq!(resource.to_refusal().code, "execution_resource_limit");
}
