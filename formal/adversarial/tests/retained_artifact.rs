//! Every `*_expecting` loader admits bytes only as the artifact its consumer retained.
//!
//! The real BLAKE3 artifact digest is used. Positive cases retain the identity of the very bytes
//! handed over: the loader then runs and its outcome is the unbound loader's outcome. Negative
//! cases retain the identity of other bytes (one byte flipped, one appended, one removed, or
//! unrelated bytes): the bytes are refused with `external_binding_mismatch` before the loader
//! reads them.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::counterfactual_id_artifact::CounterfactualIdArtifactWire;
use antecedent_io_core::retained_artifact::{
    RETAINED_ARTIFACT_REFUSAL, RetainedArtifactIdentity, bind_retained_artifact, load_expecting,
};
use antecedent_io_core::staleness_artifact::StalenessBaselineWire;
use antecedent_io_core::temporal_recalc_artifact::TemporalRecalcArtifactWire;
use antecedent_io_core::{
    IoError, decode_causal_posterior_bytes, decode_causal_posterior_bytes_expecting,
    decode_prior_source_meta, decode_prior_source_meta_expecting, load_analysis_result,
    load_analysis_result_expecting,
};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

/// Bytes that differ from `bytes`: one byte flipped, one appended, one removed, or unrelated.
fn other_bytes(bytes: &[u8], how: u8, index: usize, mask: u8) -> Vec<u8> {
    let mut other = bytes.to_vec();
    match how % 4 {
        0 if !other.is_empty() => {
            let at = index % other.len();
            other[at] ^= mask;
        }
        1 => other.push(mask),
        2 if !other.is_empty() => {
            other.remove(index % other.len());
        }
        _ => {
            other = b"another artifact".to_vec();
            if other == bytes {
                other.push(0);
            }
        }
    }
    other
}

/// The outcome of a loader, as text: `None` when it admitted the bytes, the error otherwise.
fn outcome<T, E: core::fmt::Debug>(result: Result<T, E>) -> Option<String> {
    result.err().map(|error| format!("{error:?}"))
}

fn is_binding_refusal(text: &Option<String>) -> bool {
    text.as_deref().is_some_and(|text| text.contains(RETAINED_ARTIFACT_REFUSAL))
}

/// The unbound and bound forms of one production loader, applied to the same bytes.
type LoaderPair = (
    &'static str,
    fn(&[u8]) -> Option<String>,
    fn(&[u8], &RetainedArtifactIdentity) -> Option<String>,
);

fn loaders() -> Vec<LoaderPair> {
    let mut loaders: Vec<LoaderPair> = Vec::new();
    loaders.push((
        "load_analysis_result",
        |bytes| outcome(load_analysis_result(bytes)),
        |bytes, retained| outcome(load_analysis_result_expecting(bytes, retained)),
    ));
    loaders.push((
        "decode_causal_posterior_bytes",
        |bytes| outcome(decode_causal_posterior_bytes(bytes)),
        |bytes, retained| outcome(decode_causal_posterior_bytes_expecting(bytes, retained)),
    ));
    loaders.push((
        "decode_prior_source_meta",
        |bytes| outcome(decode_prior_source_meta(bytes)),
        |bytes, retained| outcome(decode_prior_source_meta_expecting(bytes, retained)),
    ));
    loaders.push((
        "StalenessBaselineWire::consume",
        |bytes| outcome(StalenessBaselineWire::consume(bytes)),
        |bytes, retained| outcome(StalenessBaselineWire::consume_expecting(bytes, retained)),
    ));
    loaders.push((
        "TemporalRecalcArtifactWire::decode",
        |bytes| outcome(TemporalRecalcArtifactWire::decode(bytes)),
        |bytes, retained| outcome(TemporalRecalcArtifactWire::decode_expecting(bytes, retained)),
    ));
    loaders.push((
        "CounterfactualIdArtifactWire::decode",
        |bytes| outcome(CounterfactualIdArtifactWire::decode(bytes)),
        |bytes, retained| outcome(CounterfactualIdArtifactWire::decode_expecting(bytes, retained)),
    ));
    loaders
}

proptest! {
    #![proptest_config(config())]

    /// Retaining the identity of the bytes handed over admits them: the loader runs on exactly
    /// those bytes.
    #[test]
    fn retained_bytes_are_admitted(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        let retained = RetainedArtifactIdentity::of_bytes(&bytes);
        prop_assert_eq!(bind_retained_artifact(&bytes, &retained), Ok(()));
        let read: Result<Vec<u8>, IoError> =
            load_expecting(&bytes, &retained, |admitted| Ok(admitted.to_vec()));
        prop_assert_eq!(read, Ok(bytes));
    }

    /// Bytes other than the retained artifact are refused with `external_binding_mismatch`, and
    /// the loader never reads them.
    #[test]
    fn other_bytes_are_refused_unread(
        bytes in prop::collection::vec(any::<u8>(), 0..256),
        how in any::<u8>(),
        index in any::<usize>(),
        mask in 1..=255u8,
    ) {
        let retained = RetainedArtifactIdentity::of_bytes(&bytes);
        let other = other_bytes(&bytes, how, index, mask);
        prop_assume!(other != bytes);
        let mut read = false;
        let outcome: Result<(), IoError> = load_expecting(&other, &retained, |_| {
            read = true;
            Ok(())
        });
        prop_assert!(!read, "the loader read bytes that are not the retained artifact");
        prop_assert!(matches!(
            outcome,
            Err(IoError::Refused { code: "external_binding_mismatch", .. })
        ));
    }

    /// Each production loader's bound form, given the identity of the bytes, returns what the
    /// unbound loader returns; given the identity of other bytes, it is refused as not the
    /// retained artifact.
    #[test]
    fn production_loaders_bind_the_retained_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..256),
        how in any::<u8>(),
        index in any::<usize>(),
        mask in 1..=255u8,
    ) {
        let matching = RetainedArtifactIdentity::of_bytes(&bytes);
        let other = other_bytes(&bytes, how, index, mask);
        prop_assume!(other != bytes);
        let mismatched = RetainedArtifactIdentity::of_bytes(&other);
        for (name, unbound, bound) in loaders() {
            let plain = unbound(&bytes);
            prop_assert!(!is_binding_refusal(&plain), "{name}: unbound form refused as unretained");
            prop_assert_eq!(bound(&bytes, &matching), plain, "{}: matching identity", name);
            prop_assert!(
                is_binding_refusal(&bound(&bytes, &mismatched)),
                "{name}: mismatched identity was not refused as unretained"
            );
        }
    }
}
