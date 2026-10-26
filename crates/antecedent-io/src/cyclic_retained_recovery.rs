// SPDX-License-Identifier: MIT OR Apache-2.0
//! Borrowed continuations with the unchanged independent-consumer wire contract.
use super::{
    CyclicArtifactError, CyclicFamilyAnswer as A, CyclicFamilyReport, CyclicFamilyRequest as R,
    CyclicWorldWire, RecoveredLawProduct, invalid,
};
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use std::sync::Arc;

/// A live or independently verified source retained without copying observations or draws.
#[derive(Clone, Debug)]
struct Source {
    request: Arc<R>,
    report: Arc<CyclicFamilyReport>,
    // Pins the originating program/consumer and its input/output admission leases.
    _owner: Arc<dyn std::fmt::Debug + Send + Sync>,
}
impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        self.request == other.request && self.report == other.report
    }
}

/// Borrowed first fields of the original experimental-recovery encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedRecoveryInputs(Arc<Source>);
/// Borrowed last fields of that encoding. Both adapters retain the same source Arc.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedRecoveryProvenance(Arc<Source>);

impl RetainedRecoveryInputs {
    /// Original scientific recipe, never a reconstructed live preparation.
    pub fn recipe(&self) -> &R {
        self.0.request.recovery_recipe()
    }
    /// Exact native source report; its admission reservations remain owned.
    pub fn report(&self) -> &CyclicFamilyReport {
        &self.0.report
    }
    /// Exact recovered law allocation, including original origin/status/failed slots.
    pub fn law(&self) -> &Arc<RecoveredLawProduct> {
        let A::ExperimentalRecovery(report) = &self.0.report.answer else {
            unreachable!("retained recovery source is validated at construction")
        };
        &report.law
    }
    /// Original declared scientific source, independent of the selected worlds.
    pub fn source_id(&self) -> &str {
        let R::ExperimentalRecovery { source_id, .. } = self.recipe() else {
            unreachable!("retained recovery recipe is validated at construction")
        };
        source_id
    }
    /// Original ordered evidence declarations for the retained recovery recipe.
    pub fn certifying_evidence(&self) -> &[String] {
        let R::ExperimentalRecovery { certifying_evidence, .. } = self.recipe() else {
            unreachable!("retained recovery recipe is validated at construction")
        };
        certifying_evidence
    }
}

impl Serialize for RetainedRecoveryInputs {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let R::ExperimentalRecovery { regimes, dropped, samples, options, inference, .. } =
            self.recipe()
        else {
            return Err(serde::ser::Error::custom("retained recovery recipe"));
        };
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry("regimes", regimes)?;
        map.serialize_entry("dropped", dropped)?;
        map.serialize_entry("samples", samples)?;
        map.serialize_entry("options", options)?;
        map.serialize_entry("inference", inference)?;
        map.end()
    }
}
impl Serialize for RetainedRecoveryProvenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let source = RetainedRecoveryInputs(Arc::clone(&self.0));
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("source_id", source.source_id())?;
        map.serialize_entry("certifying_evidence", source.certifying_evidence())?;
        map.end()
    }
}

impl R {
    /// Borrow the original experiment declarations while retaining its own worlds separately.
    pub fn recovery_recipe(&self) -> &Self {
        match self {
            Self::RetainedExperimentalRecovery { inputs, .. } => inputs.recipe(),
            _ => self,
        }
    }

    /// Branch a checked native execution using its exact source data and law allocations.
    ///
    /// The caller must obtain source owners from execution or independent consumption.
    /// This never fits/samples and never serializes an artifact. The ordinary compiler
    /// admits and validates each new world and retains the source Arcs unchanged.
    pub fn retained_recovery_worlds(
        request: Arc<Self>,
        report: Arc<CyclicFamilyReport>,
        owner: Arc<dyn std::fmt::Debug + Send + Sync>,
        worlds: Vec<CyclicWorldWire>,
        outcome: u32,
    ) -> Result<Self, CyclicArtifactError> {
        if !matches!(request.recovery_recipe(), Self::ExperimentalRecovery { .. })
            || !matches!(report.answer, A::ExperimentalRecovery(_))
        {
            return Err(invalid("retained recovery worlds require an experimental recovered law"));
        }
        let source = Arc::new(Source { request, report, _owner: owner });
        Ok(Self::RetainedExperimentalRecovery {
            inputs: RetainedRecoveryInputs(Arc::clone(&source)),
            worlds,
            outcome,
            provenance: RetainedRecoveryProvenance(source),
        })
    }
}

/// Hash JSON without retaining an encoded copy of a source dataset or recovered law.
pub(super) fn json_digest<T: Serialize + ?Sized>(
    tag: &str,
    value: &T,
) -> Result<[u8; 32], CyclicArtifactError> {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .filter(|n| *n <= super::MAX_CYCLIC_ARTIFACT_BYTES)
                .ok_or_else(|| std::io::Error::other("cyclic identity byte limit"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    struct Hash(blake3::Hasher);
    impl std::io::Write for Hash {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut size = Count(0);
    serde_json::to_writer(&mut size, value).map_err(invalid)?;
    let mut hash = Hash(blake3::Hasher::new());
    hash.0.update(&(tag.len() as u64).to_le_bytes());
    hash.0.update(tag.as_bytes());
    hash.0.update(&(size.0 as u64).to_le_bytes());
    serde_json::to_writer(&mut hash, value).map_err(invalid)?;
    Ok(*hash.0.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::super::{
        CompiledCyclicFamilyProgram, CyclicInterventionWire, consume_cyclic_family,
    };
    use super::*;
    use antecedent_core::ExecutionContext;

    fn original() -> R {
        serde_json::from_str(include_str!(
            "../../../parity/cyclic/recovered-asymptotic-normal/request.json"
        ))
        .unwrap()
    }
    fn worlds() -> Vec<CyclicWorldWire> {
        vec![CyclicWorldWire {
            world: 47,
            interventions: vec![CyclicInterventionWire::Set { variable: 0, value: 3.0 }],
            initial_states: vec![],
        }]
    }
    #[test]
    fn continuation_shares_source_and_law_without_another_fit_and_keeps_old_codec() {
        let ctx = ExecutionContext::for_tests(19);
        let original = CompiledCyclicFamilyProgram::compile(&original(), &ctx).unwrap();
        let report = Arc::new(original.execute(&ctx).unwrap());
        let A::ExperimentalRecovery(source) = &report.answer else { panic!("recovery fixture") };
        let source_law = Arc::clone(&source.law);
        let count = super::super::worlds::recovery_fit_count();
        let retained = R::retained_recovery_worlds(
            original.shared_request(),
            Arc::clone(&report),
            Arc::new(original.clone()),
            worlds(),
            1,
        )
        .unwrap();
        let R::RetainedExperimentalRecovery { inputs, provenance, .. } = &retained else {
            panic!("native continuation")
        };
        assert!(Arc::ptr_eq(&inputs.0, &provenance.0));
        assert!(Arc::ptr_eq(&inputs.0.request, &original.shared_request()));
        assert!(Arc::ptr_eq(&inputs.0.report, &report));
        let mut ordinary = original.request().clone();
        let R::ExperimentalRecovery { worlds: selected, outcome, .. } = &mut ordinary else {
            panic!("source recipe")
        };
        *selected = worlds();
        *outcome = 1;
        assert_eq!(serde_json::to_vec(&retained).unwrap(), serde_json::to_vec(&ordinary).unwrap());
        let independent: R =
            serde_json::from_slice(&serde_json::to_vec(&retained).unwrap()).unwrap();
        assert!(matches!(independent, R::ExperimentalRecovery { .. }));
        let prepared = CompiledCyclicFamilyProgram::compile(&retained, &ctx).unwrap();
        let continued = prepared.execute(&ctx).unwrap();
        let A::ExperimentalRecovery(answer) = &continued.answer else { panic!("continued law") };
        assert!(Arc::ptr_eq(&source_law, &answer.law));
        assert_eq!(super::super::worlds::recovery_fit_count(), count);
        let artifact = prepared.encode_execution(&continued, &ctx).unwrap();
        assert_eq!(super::super::worlds::recovery_fit_count(), count);
        let consumed = consume_cyclic_family(&artifact, &continued.identity, &ctx).unwrap();
        assert_eq!(consumed, continued);
        assert_eq!(super::super::worlds::recovery_fit_count(), count + 1);
        assert!(consume_cyclic_family(&artifact, &"0".repeat(64), &ctx).is_err());
        let ordinary_program = CompiledCyclicFamilyProgram::compile(&ordinary, &ctx).unwrap();
        assert_eq!(prepared.identity(), ordinary_program.identity());
        assert_eq!(prepared.contract().unwrap(), ordinary_program.contract().unwrap());
        let ordinary_report = ordinary_program.execute(&ctx).unwrap();
        assert_eq!(artifact, ordinary_program.encode_execution(&ordinary_report, &ctx).unwrap());
        drop(original);
        drop(report);
        drop(retained);
        assert_eq!(answer.law.origin, source_law.origin);
    }
    #[test]
    fn retained_recovery_does_not_reinterpret_unsupported_worlds() {
        let ctx = ExecutionContext::for_tests(19);
        let source = CompiledCyclicFamilyProgram::compile(&original(), &ctx).unwrap();
        let report = Arc::new(source.execute(&ctx).unwrap());
        let requested = vec![CyclicWorldWire {
            world: 7,
            interventions: vec![CyclicInterventionWire::Shift { variable: 0, delta: 1.0 }],
            initial_states: vec![],
        }];
        let retained = R::retained_recovery_worlds(
            source.shared_request(),
            report,
            Arc::new(source.clone()),
            requested,
            1,
        )
        .unwrap();
        assert!(CompiledCyclicFamilyProgram::compile(&retained, &ctx).is_err());
    }
}
