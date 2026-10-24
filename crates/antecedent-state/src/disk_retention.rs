//! Budgeted disk-backed retention of prepared refresh state.
//!
//! A prepared refresh result (the rows, the summary, the source identity) is serialised into a
//! digest-checked frame and written under one byte budget by a [`SpillRegistry`]. The registry
//! is the realisation of [`RetentionMode::DiskBacked`]:
//!
//! * a spill reserves its frame bytes against the budget before anything is written, and
//!   evicts the least recently used spilled results to make room; a frame larger than the
//!   whole budget is refused and nothing is evicted for it;
//! * [`SpillHandle`] serves a read of the declared artifact ([`SpillHandle::reload`]) and
//!   refuses every resident operation with the next attempt, exactly as
//!   [`RetentionMode::DiskBacked`] declares;
//! * a reload verifies the frame digest, the kind and the length, so a file that was changed or
//!   truncated is refused, and an evicted entry is refused with the rebuild it needs;
//! * a frame is written to a temporary name and renamed into place, so a failed write leaves no
//!   target and returns its reservation;
//! * remaining files are removed when the registry drops.
//!
//! The directory handed to [`SpillRegistry::new`] belongs to the registry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use antecedent_core::CanonicalWriter;

use crate::error::StateError;
use crate::retention::{RetainedOperation, RetentionMode};

const FRAME_MAGIC: &[u8; 8] = b"ANTSPILL";
const FRAME_TAG: &str = "antecedent.state.disk_retention.frame";
const KEY_TAG: &str = "antecedent.state.disk_retention.key";
const DIGEST_HEX_LEN: usize = 64;

/// A prepared state that can be written to and read back from a frame.
pub trait Spillable: Sized {
    /// Stable name of the encoded state, checked on reload.
    const KIND: &'static str;

    /// Encode the complete state.
    ///
    /// # Errors
    /// The state does not hold what a reload needs (for example, rows dropped by summary-only
    /// retention).
    fn encode(&self) -> Result<Vec<u8>, StateError>;

    /// Decode a state encoded by [`Self::encode`].
    ///
    /// # Errors
    /// The bytes are not a state of this kind.
    fn decode(bytes: &[u8]) -> Result<Self, StateError>;
}

/// Little-endian writer for [`Spillable::encode`].
#[derive(Debug, Default)]
pub struct SpillEncoder {
    bytes: Vec<u8>,
}

impl SpillEncoder {
    /// An empty encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// One `u64` word.
    pub fn word(&mut self, word: u64) -> &mut Self {
        self.bytes.extend_from_slice(&word.to_le_bytes());
        self
    }

    /// One real as its bit pattern.
    pub fn real(&mut self, value: f64) -> &mut Self {
        self.word(value.to_bits())
    }

    /// A flag as one word.
    pub fn flag(&mut self, value: bool) -> &mut Self {
        self.word(u64::from(value))
    }

    /// Length-prefixed text.
    pub fn text(&mut self, text: &str) -> &mut Self {
        self.word(text.len() as u64);
        self.bytes.extend_from_slice(text.as_bytes());
        self
    }

    /// A count-prefixed sequence of reals.
    pub fn reals(&mut self, values: &[f64]) -> &mut Self {
        self.word(values.len() as u64);
        for value in values {
            self.real(*value);
        }
        self
    }

    /// The encoded bytes.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Little-endian reader for [`Spillable::decode`]. Every read is bounds-checked.
#[derive(Debug)]
pub struct SpillDecoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

fn truncated() -> StateError {
    StateError::StaleCache("spilled state is truncated or malformed".into())
}

impl<'a> SpillDecoder<'a> {
    /// A reader over `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], StateError> {
        let end = self.position.checked_add(count).ok_or_else(truncated)?;
        let slice = self.bytes.get(self.position..end).ok_or_else(truncated)?;
        self.position = end;
        Ok(slice)
    }

    /// One `u64` word.
    ///
    /// # Errors
    /// The bytes end first.
    pub fn word(&mut self) -> Result<u64, StateError> {
        let raw: [u8; 8] = self.take(8)?.try_into().map_err(|_| truncated())?;
        Ok(u64::from_le_bytes(raw))
    }

    /// One count read as `usize`, refused when it cannot index the remaining bytes.
    ///
    /// # Errors
    /// The bytes end first, or the count exceeds what the remaining bytes can hold at
    /// `unit` bytes per item.
    pub fn count(&mut self, unit: usize) -> Result<usize, StateError> {
        let count = usize::try_from(self.word()?).map_err(|_| truncated())?;
        let needed = count.checked_mul(unit.max(1)).ok_or_else(truncated)?;
        if needed > self.bytes.len().saturating_sub(self.position) {
            return Err(truncated());
        }
        Ok(count)
    }

    /// One real.
    ///
    /// # Errors
    /// The bytes end first.
    pub fn real(&mut self) -> Result<f64, StateError> {
        Ok(f64::from_bits(self.word()?))
    }

    /// One flag.
    ///
    /// # Errors
    /// The bytes end first, or the word is neither 0 nor 1.
    pub fn flag(&mut self) -> Result<bool, StateError> {
        match self.word()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(truncated()),
        }
    }

    /// Length-prefixed text.
    ///
    /// # Errors
    /// The bytes end first, or the text is not UTF-8.
    pub fn text(&mut self) -> Result<String, StateError> {
        let length = self.count(1)?;
        let raw = self.take(length)?;
        String::from_utf8(raw.to_vec()).map_err(|_| truncated())
    }

    /// A count-prefixed sequence of reals.
    ///
    /// # Errors
    /// The bytes end first.
    pub fn reals(&mut self) -> Result<Vec<f64>, StateError> {
        let length = self.count(8)?;
        (0..length).map(|_| self.real()).collect()
    }

    /// Require that every byte was read.
    ///
    /// # Errors
    /// Bytes remain.
    pub fn finish(&self) -> Result<(), StateError> {
        if self.position == self.bytes.len() { Ok(()) } else { Err(truncated()) }
    }
}

fn io_error(context: &str, error: &std::io::Error) -> StateError {
    StateError::InvalidEvent(format!("{context}: {error}"))
}

fn frame_digest(kind: &str, payload: &[u8]) -> String {
    let mut writer = CanonicalWriter::new(FRAME_TAG);
    writer.text(kind).bytes(payload);
    writer.finish_hex()
}

fn build_frame(kind: &str, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(payload.len() + kind.len() + 32 + DIGEST_HEX_LEN);
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&(kind.len() as u64).to_le_bytes());
    frame.extend_from_slice(kind.as_bytes());
    frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(frame_digest(kind, payload).as_bytes());
    frame
}

/// Verify a frame and return its payload and digest.
fn open_frame<'a>(frame: &'a [u8], expected_kind: &str) -> Result<(&'a [u8], String), StateError> {
    let mut reader = SpillDecoder::new(frame);
    if reader.take(FRAME_MAGIC.len())? != FRAME_MAGIC.as_slice() {
        return Err(StateError::StaleCache("spilled file is not a retention frame".into()));
    }
    let kind_length = reader.count(1)?;
    let kind = reader.take(kind_length)?;
    if kind != expected_kind.as_bytes() {
        return Err(StateError::StaleCache(format!(
            "spilled file holds a different kind of state than {expected_kind}"
        )));
    }
    let payload_length = reader.count(1)?;
    let payload = reader.take(payload_length)?;
    let stored = reader.take(DIGEST_HEX_LEN)?;
    reader.finish()?;
    let digest = frame_digest(expected_kind, payload);
    if stored != digest.as_bytes() {
        return Err(StateError::StaleCache(
            "spilled file was changed after it was written; rebuild it from the source rows".into(),
        ));
    }
    Ok((payload, digest))
}

#[derive(Debug)]
struct Entry {
    path: PathBuf,
    bytes: u64,
    digest: String,
    kind: &'static str,
    last_used: u64,
}

#[derive(Debug, Default)]
struct Inner {
    used: u64,
    tick: u64,
    sequence: u64,
    evictions: u64,
    reloads: u64,
    spills: u64,
    entries: BTreeMap<String, Entry>,
}

/// Counters of one registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpillStats {
    /// Spilled results currently on disk.
    pub entries: usize,
    /// Bytes they hold.
    pub used_bytes: u64,
    /// Byte budget.
    pub budget_bytes: u64,
    /// Results written.
    pub spills: u64,
    /// Results removed to make room.
    pub evictions: u64,
    /// Successful reloads.
    pub reloads: u64,
}

/// Spilled prepared results under one byte budget.
#[derive(Debug)]
pub struct SpillRegistry {
    directory: PathBuf,
    budget_bytes: u64,
    inner: Mutex<Inner>,
}

impl SpillRegistry {
    /// A registry writing under `directory`, which it creates, holding at most `budget_bytes`.
    ///
    /// # Errors
    /// The directory cannot be created.
    pub fn new(directory: impl Into<PathBuf>, budget_bytes: u64) -> Result<Arc<Self>, StateError> {
        let directory = directory.into();
        fs::create_dir_all(&directory)
            .map_err(|error| io_error("cannot create the spill directory", &error))?;
        Ok(Arc::new(Self { directory, budget_bytes, inner: Mutex::new(Inner::default()) }))
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Byte budget.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Directory the registry writes under.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> SpillStats {
        let inner = self.lock();
        SpillStats {
            entries: inner.entries.len(),
            used_bytes: inner.used,
            budget_bytes: self.budget_bytes,
            spills: inner.spills,
            evictions: inner.evictions,
            reloads: inner.reloads,
        }
    }

    /// Whether `key` is currently on disk.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.lock().entries.contains_key(key)
    }

    /// Keys on disk, least recently used first.
    #[must_use]
    pub fn keys_by_recency(&self) -> Vec<String> {
        let inner = self.lock();
        let mut keys: Vec<(&u64, &String)> =
            inner.entries.iter().map(|(key, entry)| (&entry.last_used, key)).collect();
        keys.sort();
        keys.into_iter().map(|(_, key)| key.clone()).collect()
    }

    fn path_for(&self, key: &str, sequence: u64) -> PathBuf {
        let mut writer = CanonicalWriter::new(KEY_TAG);
        writer.text(key).word(sequence);
        self.directory.join(format!("{}.spill", writer.finish_hex()))
    }

    /// Write `value` under `key`, evicting the least recently used results to fit.
    ///
    /// A key already on disk is replaced. The returned handle serves a read of the artifact
    /// and nothing resident.
    ///
    /// # Errors
    /// The state cannot be encoded, the frame is larger than the whole budget (nothing is
    /// evicted for it), or the file cannot be written (the reservation is returned and no
    /// target is left).
    pub fn spill<T: Spillable>(
        self: &Arc<Self>,
        key: &str,
        value: &T,
    ) -> Result<SpillHandle, StateError> {
        let payload = value.encode()?;
        let frame = build_frame(T::KIND, &payload);
        let need = frame.len() as u64;
        if need > self.budget_bytes {
            return Err(StateError::CacheBudget { need, remaining: self.budget_bytes });
        }
        let mut inner = self.lock();
        let held = inner.entries.get(key).map_or(0, |entry| entry.bytes);
        let mut evicted = Vec::new();
        while inner.used.saturating_sub(held).saturating_add(need) > self.budget_bytes {
            let oldest = inner
                .entries
                .iter()
                .filter(|(other, _)| other.as_str() != key)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(other, _)| other.clone());
            let Some(oldest) = oldest else { break };
            if let Some(entry) = inner.entries.remove(&oldest) {
                let _ = fs::remove_file(&entry.path);
                inner.used = inner.used.saturating_sub(entry.bytes);
                inner.evictions = inner.evictions.saturating_add(1);
            }
            evicted.push(oldest);
        }
        inner.sequence = inner.sequence.saturating_add(1);
        let path = self.path_for(key, inner.sequence);
        let temporary = path.with_extension("tmp");
        let written = fs::File::create(&temporary)
            .and_then(|mut file| file.write_all(&frame).and_then(|()| file.sync_all()))
            .and_then(|()| fs::rename(&temporary, &path));
        if let Err(error) = written {
            let _ = fs::remove_file(&temporary);
            return Err(io_error("cannot write the spilled state", &error));
        }
        if let Some(previous) = inner.entries.remove(key) {
            let _ = fs::remove_file(&previous.path);
            inner.used = inner.used.saturating_sub(previous.bytes);
        }
        inner.tick = inner.tick.saturating_add(1);
        let tick = inner.tick;
        let digest = frame_digest(T::KIND, &payload);
        inner.used = inner.used.saturating_add(need);
        inner.spills = inner.spills.saturating_add(1);
        inner.entries.insert(
            key.to_owned(),
            Entry { path, bytes: need, digest: digest.clone(), kind: T::KIND, last_used: tick },
        );
        drop(inner);
        Ok(SpillHandle {
            registry: Arc::clone(self),
            key: key.to_owned(),
            bytes: need,
            digest,
            evicted,
        })
    }

    /// Read `key` back and decode it.
    ///
    /// # Errors
    /// The key is not on disk (never spilled, evicted, or removed: rebuild it from the
    /// source rows), or the file was changed, truncated or is of another kind.
    pub fn reload<T: Spillable>(&self, key: &str) -> Result<T, StateError> {
        let (path, digest) = {
            let inner = self.lock();
            let entry = inner.entries.get(key).ok_or_else(|| {
                StateError::UnknownId(format!(
                    "spilled state {key} is not on disk (never spilled, evicted under the byte \
                     budget, or removed); rebuild it from the source rows"
                ))
            })?;
            if entry.kind != T::KIND {
                return Err(StateError::StaleCache(format!(
                    "spilled state {key} is a {}, not a {}",
                    entry.kind,
                    T::KIND
                )));
            }
            (entry.path.clone(), entry.digest.clone())
        };
        let frame = fs::read(&path).map_err(|error| io_error("cannot read the spill", &error))?;
        let (payload, actual) = open_frame(&frame, T::KIND)?;
        if actual != digest {
            return Err(StateError::StaleCache(
                "spilled file differs from the one this registry wrote; rebuild it from the \
                 source rows"
                    .into(),
            ));
        }
        let value = T::decode(payload)?;
        let mut inner = self.lock();
        inner.tick = inner.tick.saturating_add(1);
        let tick = inner.tick;
        if let Some(entry) = inner.entries.get_mut(key) {
            entry.last_used = tick;
        }
        inner.reloads = inner.reloads.saturating_add(1);
        drop(inner);
        Ok(value)
    }

    /// Remove `key` from disk. Returns whether it was there.
    pub fn remove(&self, key: &str) -> bool {
        let mut inner = self.lock();
        match inner.entries.remove(key) {
            Some(entry) => {
                let _ = fs::remove_file(&entry.path);
                inner.used = inner.used.saturating_sub(entry.bytes);
                true
            }
            None => false,
        }
    }
}

impl Drop for SpillRegistry {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        for entry in inner.entries.values() {
            let _ = fs::remove_file(&entry.path);
        }
        inner.entries.clear();
    }
}

/// A spilled prepared result. Its mode is [`RetentionMode::DiskBacked`].
#[derive(Clone, Debug)]
pub struct SpillHandle {
    registry: Arc<SpillRegistry>,
    key: String,
    bytes: u64,
    digest: String,
    evicted: Vec<String>,
}

impl SpillHandle {
    /// Key the result was spilled under.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Frame bytes reserved against the budget.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Digest of the frame payload.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Keys evicted to make room for this spill.
    #[must_use]
    pub fn evicted(&self) -> &[String] {
        &self.evicted
    }

    /// The mode of a spilled result.
    #[must_use]
    pub const fn mode(&self) -> RetentionMode {
        RetentionMode::DiskBacked
    }

    /// Whether the file is still on disk (it is not once evicted or removed).
    #[must_use]
    pub fn on_disk(&self) -> bool {
        self.registry.contains(&self.key)
    }

    /// Accept `operation` on the spilled result, or refuse it with the next attempt.
    ///
    /// # Errors
    /// The disk-backed mode cannot serve `operation`; the next attempt is to reload the
    /// artifact and run the operation on the resident state.
    pub fn require(&self, operation: RetainedOperation) -> Result<(), StateError> {
        self.mode().require(operation).map_err(|refusal| {
            StateError::InvalidEvent(format!(
                "{refusal}; reload the artifact with SpillHandle::reload and run the operation \
                 on the resident state"
            ))
        })
    }

    /// Read the artifact back as the resident state.
    ///
    /// # Errors
    /// The errors of [`SpillRegistry::reload`].
    pub fn reload<T: Spillable>(&self) -> Result<T, StateError> {
        self.require(RetainedOperation::ReadDiskArtifact)?;
        let value = self.registry.reload::<T>(&self.key)?;
        Ok(value)
    }

    /// Remove the artifact from disk and return its reservation.
    pub fn remove(self) -> bool {
        self.registry.remove(&self.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Blob(Vec<f64>);

    impl Spillable for Blob {
        const KIND: &'static str = "test.blob";
        fn encode(&self) -> Result<Vec<u8>, StateError> {
            let mut encoder = SpillEncoder::new();
            encoder.reals(&self.0);
            Ok(encoder.finish())
        }
        fn decode(bytes: &[u8]) -> Result<Self, StateError> {
            let mut decoder = SpillDecoder::new(bytes);
            let values = decoder.reals()?;
            decoder.finish()?;
            Ok(Self(values))
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("antecedent-spill-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn lru_eviction_keeps_the_budget_and_refuses_the_evicted_key() {
        let directory = scratch("lru");
        let blob = |seed: f64| Blob((0..32).map(|i| seed + f64::from(i)).collect());
        let frame = build_frame(Blob::KIND, &blob(0.0).encode().unwrap()).len() as u64;
        let registry = SpillRegistry::new(&directory, frame * 2).unwrap();
        let first = registry.spill("a", &blob(1.0)).unwrap();
        registry.spill("b", &blob(2.0)).unwrap();
        assert_eq!(first.reload::<Blob>().unwrap(), blob(1.0));
        let third = registry.spill("c", &blob(3.0)).unwrap();
        assert_eq!(third.evicted(), ["b"], "b was least recently used");
        assert!(registry.stats().used_bytes <= frame * 2);
        assert!(registry.reload::<Blob>("b").is_err());
        assert_eq!(registry.reload::<Blob>("a").unwrap(), blob(1.0));
        assert_eq!(registry.stats().evictions, 1);
        drop((first, third));
        drop(registry);
        assert!(fs::read_dir(&directory).map(|mut d| d.next().is_none()).unwrap_or(true));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_changed_file_is_refused() {
        let directory = scratch("tamper");
        let registry = SpillRegistry::new(&directory, 1 << 20).unwrap();
        let handle = registry.spill("a", &Blob(vec![1.0, 2.0, 3.0])).unwrap();
        let path = fs::read_dir(&directory).unwrap().next().unwrap().unwrap().path();
        let mut bytes = fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(handle.reload::<Blob>(), Err(StateError::StaleCache(_))));
        bytes.truncate(bytes.len() - 5);
        fs::write(&path, &bytes).unwrap();
        assert!(handle.reload::<Blob>().is_err());
        drop(registry);
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_frame_over_the_whole_budget_is_refused_without_evicting() {
        let directory = scratch("over");
        let registry = SpillRegistry::new(&directory, 400).unwrap();
        registry.spill("small", &Blob(vec![1.0])).unwrap();
        let error = registry.spill("big", &Blob(vec![0.0; 200])).unwrap_err();
        assert!(matches!(error, StateError::CacheBudget { .. }));
        assert!(registry.contains("small"));
        assert_eq!(registry.stats().evictions, 0);
        drop(registry);
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_spilled_result_refuses_resident_operations() {
        let directory = scratch("modes");
        let registry = SpillRegistry::new(&directory, 1 << 20).unwrap();
        let handle = registry.spill("a", &Blob(vec![1.0])).unwrap();
        assert!(handle.require(RetainedOperation::ReadDiskArtifact).is_ok());
        for operation in RetentionMode::DiskBacked.no_longer_serves() {
            assert!(handle.require(*operation).is_err(), "{operation}");
        }
        drop(registry);
        let _ = fs::remove_dir_all(&directory);
    }
}
