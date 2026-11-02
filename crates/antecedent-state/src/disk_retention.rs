//! Budgeted disk-backed retention of prepared refresh state.
//!
//! A prepared refresh result (the rows, the summary, the source identity) is serialised into a
//! digest-checked frame and written under one byte budget by a [`SpillRegistry`]. The registry
//! is the realisation of [`RetentionMode::DiskBacked`]:
//!
//! * a frame larger than the logical retained-byte budget is refused without eviction;
//!   a new frame is written before evicting old results, so commit may temporarily use one
//!   incoming frame beyond that logical disk budget;
//! * [`SpillHandle`] serves a read of the declared artifact ([`SpillHandle::reload`]) and
//!   refuses every resident operation with the next attempt, exactly as
//!   [`RetentionMode::DiskBacked`] declares;
//! * a reload verifies the frame digest, the kind and the length, so a file that was changed or
//!   truncated is refused, and an evicted entry is refused with the rebuild it needs;
//! * a frame is exclusively created under a unique name and registered only after writing and
//!   syncing it; a failed write removes only that new file and preserves the retained entries;
//! * remaining files are removed when the registry drops.
//!
//! Registries can share a directory; each registry removes only its own files.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use antecedent_core::CanonicalWriter;

use crate::error::StateError;
use crate::retention::{RetainedOperation, RetentionMode};

const FRAME_MAGIC: &[u8; 8] = b"ANTSPILL";
const FRAME_TAG: &str = "antecedent.state.disk_retention.frame";
const KEY_TAG: &str = "antecedent.state.disk_retention.key";
const DIGEST_HEX_LEN: usize = 64;
static SPILL_NONCE: AtomicU64 = AtomicU64::new(0);

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
    generation: u64,
}

/// A file whose removal failed. Its bytes stay charged to the budget until it is gone, so the
/// logical account never claims space the disk still holds.
#[derive(Debug)]
struct Orphan {
    path: PathBuf,
    bytes: u64,
}

#[derive(Debug, Default)]
struct Inner {
    used: u64,
    tick: u64,
    sequence: u64,
    evictions: u64,
    reloads: u64,
    spills: u64,
    recovered: u64,
    entries: BTreeMap<String, Entry>,
    orphans: Vec<Orphan>,
}

impl Inner {
    fn orphaned_bytes(&self) -> u64 {
        self.orphans.iter().fold(0, |sum, orphan| sum.saturating_add(orphan.bytes))
    }
}

/// Counters of one registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpillStats {
    /// Spilled results currently on disk.
    pub entries: usize,
    /// Bytes they hold, including files whose removal is still pending.
    pub used_bytes: u64,
    /// Byte budget.
    pub budget_bytes: u64,
    /// Results written.
    pub spills: u64,
    /// Results removed to make room.
    pub evictions: u64,
    /// Successful reloads.
    pub reloads: u64,
    /// Files that could not be removed and are retried; their bytes stay in `used_bytes`.
    pub orphaned_files: usize,
    /// Files left by a dead process that this registry removed when it opened the directory.
    pub recovered_files: u64,
}

fn unlink(path: &Path) -> std::io::Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Removes a file that was created but never registered if the writer unwinds. Ordinary
/// failures are handled explicitly so a removal failure is accounted, not dropped.
struct Unregistered(Option<PathBuf>);

impl Unregistered {
    fn release(&mut self) -> Option<PathBuf> {
        self.0.take()
    }
}

impl Drop for Unregistered {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            // Best effort during unwinding; the next registry on this directory recovers it.
            let _ = fs::remove_file(path);
        }
    }
}

/// The owning process of a spill file name `<pid>-<64 hex digits>.spill`, if it has that shape.
fn owner_of(name: &str) -> Option<u32> {
    let stem = name.strip_suffix(".spill")?;
    let (pid, digest) = stem.split_once('-')?;
    (digest.len() == DIGEST_HEX_LEN && digest.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| pid.parse().ok())
        .flatten()
}

#[cfg(unix)]
fn process_may_be_alive(pid: u32) -> bool {
    use std::process::{Command, Stdio};
    // Without a way to ask, assume the owner lives: a file is only ever removed once its
    // owner is known to be gone. A reused pid keeps a dead process's file, never the reverse.
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_or(true, |status| status.success())
}

/// Remove the spill files that processes no longer running left in `directory`, and only
/// those this user owns. Returns how many were removed.
#[cfg(unix)]
fn recover_directory(directory: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let probe = directory.join(format!(".probe-{}-{}", std::process::id(), unique_nonce()));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let Ok(file) = options.open(&probe) else { return 0 };
    let uid = file.metadata().map(|meta| meta.uid());
    drop(file);
    let _ = unlink(&probe);
    let Ok(uid) = uid else { return 0 };
    let Ok(listing) = fs::read_dir(directory) else { return 0 };
    let own = std::process::id();
    let mut alive: BTreeMap<u32, bool> = BTreeMap::new();
    let mut removed = 0;
    for item in listing.flatten() {
        let Some(owner) = item.file_name().to_str().and_then(owner_of) else { continue };
        if owner == own {
            continue;
        }
        let Ok(meta) = item.metadata() else { continue };
        if !meta.file_type().is_file() || meta.uid() != uid {
            continue;
        }
        if *alive.entry(owner).or_insert_with(|| process_may_be_alive(owner)) {
            continue;
        }
        if unlink(&item.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(not(unix))]
fn recover_directory(_directory: &Path) -> u64 {
    0
}

fn unique_nonce() -> u64 {
    SPILL_NONCE.fetch_add(1, Ordering::Relaxed)
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
    /// Spill files that a process which is no longer running left in `directory` (a crash
    /// skips the cleanup on drop) are removed here; files of live registries, including other
    /// registries in this process, are never touched. [`SpillStats::recovered_files`] counts
    /// them.
    ///
    /// # Errors
    /// The directory cannot be created.
    pub fn new(directory: impl Into<PathBuf>, budget_bytes: u64) -> Result<Arc<Self>, StateError> {
        let directory = directory.into();
        fs::create_dir_all(&directory)
            .map_err(|error| io_error("cannot create the spill directory", &error))?;
        let inner = Inner { recovered: recover_directory(&directory), ..Inner::default() };
        Ok(Arc::new(Self { directory, budget_bytes, inner: Mutex::new(inner) }))
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
            orphaned_files: inner.orphans.len(),
            recovered_files: inner.recovered,
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

    fn path_for(&self, key: &str, nonce: u64) -> PathBuf {
        let mut writer = CanonicalWriter::new(KEY_TAG);
        writer.text(key).word(u64::from(std::process::id())).word(nonce);
        self.directory.join(format!("{}-{}.spill", std::process::id(), writer.finish_hex()))
    }

    fn create_frame(&self, key: &str) -> Result<(PathBuf, fs::File), StateError> {
        for _ in 0..1024 {
            let nonce = SPILL_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = self.path_for(key, nonce);
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            match options.open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(io_error("cannot create the spilled state", &error)),
            }
        }
        Err(StateError::InvalidEvent("cannot allocate a unique spill file".into()))
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
        self.reap_orphans();
        // Do not change retained entries until the incoming frame is durable. The exclusive
        // file is visible only to this registry until it is registered below, and the sync
        // runs outside the registry lock so other keys keep reloading and spilling.
        let (path, mut file) = self.create_frame(key)?;
        let mut unregistered = Unregistered(Some(path.clone()));
        if let Err(error) = file.write_all(&frame).and_then(|()| file.sync_all()) {
            drop(file);
            self.discard(unregistered.release(), need);
            return Err(io_error("cannot write the spilled state", &error));
        }
        drop(file);
        let digest = frame_digest(T::KIND, &payload);
        let mut inner = self.lock();
        let orphaned = inner.orphaned_bytes();
        if orphaned.saturating_add(need) > self.budget_bytes {
            drop(inner);
            self.discard(unregistered.release(), need);
            return Err(StateError::CacheBudget {
                need,
                remaining: self.budget_bytes.saturating_sub(orphaned),
            });
        }
        let Some(generation) = inner.sequence.checked_add(1) else {
            drop(inner);
            self.discard(unregistered.release(), need);
            return Err(StateError::InvalidEvent("spill generation exhausted".into()));
        };
        inner.sequence = generation;
        let held = inner.entries.get(key).map_or(0, |entry| entry.bytes);
        let mut projected = inner.used.saturating_sub(held);
        let mut evicted = Vec::new();
        let mut doomed = Vec::new();
        while projected.saturating_add(need) > self.budget_bytes {
            let oldest = inner
                .entries
                .iter()
                .filter(|(other, _)| other.as_str() != key)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(other, _)| other.clone());
            let Some(oldest) = oldest else { break };
            if let Some(entry) = inner.entries.remove(&oldest) {
                projected = projected.saturating_sub(entry.bytes);
                inner.evictions = inner.evictions.saturating_add(1);
                doomed.push(Orphan { path: entry.path, bytes: entry.bytes });
            }
            evicted.push(oldest);
        }
        if let Some(previous) = inner.entries.remove(key) {
            doomed.push(Orphan { path: previous.path, bytes: previous.bytes });
        }
        inner.tick = inner.tick.saturating_add(1);
        let tick = inner.tick;
        inner.used = inner.used.saturating_add(need);
        inner.spills = inner.spills.saturating_add(1);
        inner.entries.insert(
            key.to_owned(),
            Entry {
                path,
                bytes: need,
                digest: digest.clone(),
                kind: T::KIND,
                last_used: tick,
                generation,
            },
        );
        drop(inner);
        let _registered = unregistered.release();
        self.settle(doomed);
        Ok(SpillHandle {
            registry: Arc::clone(self),
            key: key.to_owned(),
            bytes: need,
            digest,
            generation,
            evicted,
        })
    }

    /// Unlink files that left the index. A file is released from the byte account only once
    /// it is gone; one that cannot be removed stays charged and is retried.
    fn settle(&self, doomed: Vec<Orphan>) {
        if doomed.is_empty() {
            return;
        }
        let mut released = 0_u64;
        let mut failed = Vec::new();
        for file in doomed {
            match unlink(&file.path) {
                Ok(()) => released = released.saturating_add(file.bytes),
                Err(_) => failed.push(file),
            }
        }
        let mut inner = self.lock();
        inner.used = inner.used.saturating_sub(released);
        inner.orphans.extend(failed);
    }

    /// Remove a file that was written but never registered. A failed removal is charged as
    /// an orphan, because the disk holds bytes the index does not.
    fn discard(&self, path: Option<PathBuf>, bytes: u64) {
        let Some(path) = path else { return };
        if unlink(&path).is_err() {
            let mut inner = self.lock();
            inner.used = inner.used.saturating_add(bytes);
            inner.orphans.push(Orphan { path, bytes });
        }
    }

    /// Retry the removal of files that earlier failed to unlink.
    fn reap_orphans(&self) {
        let pending = {
            let mut inner = self.lock();
            if inner.orphans.is_empty() {
                return;
            }
            std::mem::take(&mut inner.orphans)
        };
        self.settle(pending);
    }

    /// Read `key` back and decode it.
    ///
    /// # Errors
    /// The key is not on disk (never spilled, evicted, or removed: rebuild it from the
    /// source rows), or the file was changed, truncated or is of another kind.
    pub fn reload<T: Spillable>(&self, key: &str) -> Result<T, StateError> {
        self.reload_generation(key, None)
    }

    fn reload_generation<T: Spillable>(
        &self,
        key: &str,
        generation: Option<u64>,
    ) -> Result<T, StateError> {
        let (frame, digest, loaded_generation) = {
            let inner = self.lock();
            let entry = inner.entries.get(key).ok_or_else(|| {
                StateError::UnknownId(format!(
                    "spilled state {key} is not on disk (never spilled, evicted under the byte \
                     budget, or removed); rebuild it from the source rows"
                ))
            })?;
            if generation.is_some_and(|expected| expected != entry.generation) {
                return Err(StateError::StaleCache(
                    "spilled state was replaced; rebuild it from the source rows".into(),
                ));
            }
            if entry.kind != T::KIND {
                return Err(StateError::StaleCache(format!(
                    "spilled state {key} is a {}, not a {}",
                    entry.kind,
                    T::KIND
                )));
            }
            // Read while the entry is pinned by the registry lock: replacement or eviction
            // must not unlink the frame between lookup and reading its snapshot.
            let file = fs::File::open(&entry.path)
                .map_err(|error| io_error("cannot read the spill", &error))?;
            let mut frame = Vec::new();
            file.take(entry.bytes.saturating_add(1))
                .read_to_end(&mut frame)
                .map_err(|error| io_error("cannot read the spill", &error))?;
            if frame.len() as u64 != entry.bytes {
                return Err(StateError::StaleCache(
                    "spilled file size differs from its reserved frame".into(),
                ));
            }
            (frame, entry.digest.clone(), entry.generation)
        };
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
            if entry.generation == loaded_generation {
                entry.last_used = tick;
            }
        }
        inner.reloads = inner.reloads.saturating_add(1);
        drop(inner);
        Ok(value)
    }

    /// Remove `key` from disk. Returns whether it was there.
    pub fn remove(&self, key: &str) -> bool {
        self.remove_generation(key, None)
    }

    fn remove_generation(&self, key: &str, generation: Option<u64>) -> bool {
        let mut inner = self.lock();
        if generation.is_some_and(|expected| {
            inner.entries.get(key).is_none_or(|entry| entry.generation != expected)
        }) {
            return false;
        }
        let Some(entry) = inner.entries.remove(key) else { return false };
        drop(inner);
        self.settle(vec![Orphan { path: entry.path, bytes: entry.bytes }]);
        true
    }
}

impl Drop for SpillRegistry {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        let paths = std::mem::take(&mut inner.entries)
            .into_values()
            .map(|entry| entry.path)
            .chain(std::mem::take(&mut inner.orphans).into_iter().map(|orphan| orphan.path));
        for path in paths {
            // Nothing can report from drop; a file that stays is removed by the next registry
            // on this directory once this process has exited.
            let _ = fs::remove_file(path);
        }
    }
}

/// A spilled prepared result. Its mode is [`RetentionMode::DiskBacked`].
#[derive(Clone, Debug)]
pub struct SpillHandle {
    registry: Arc<SpillRegistry>,
    key: String,
    bytes: u64,
    digest: String,
    generation: u64,
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

    /// Whether this generation is still on disk (false after replacement, eviction or removal).
    #[must_use]
    pub fn on_disk(&self) -> bool {
        self.registry
            .lock()
            .entries
            .get(&self.key)
            .is_some_and(|entry| entry.generation == self.generation)
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
        self.registry.reload_generation::<T>(&self.key, Some(self.generation))
    }

    /// Remove this generation's artifact from disk and return its reservation.
    /// Returns false if the generation was already removed or replaced.
    #[must_use = "false means this handle's generation was not removed"]
    pub fn remove(self) -> bool {
        self.registry.remove_generation(&self.key, Some(self.generation))
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

    #[test]
    fn replaced_handles_cannot_reload_or_remove_the_new_generation() {
        let directory = scratch("replacement");
        let registry = SpillRegistry::new(&directory, 4096).unwrap();
        let old = registry.spill("same", &Blob(vec![1.0])).unwrap();
        let current = registry.spill("same", &Blob(vec![2.0])).unwrap();
        assert!(!old.on_disk());
        assert!(matches!(old.reload::<Blob>(), Err(StateError::StaleCache(_))));
        assert!(!old.remove());
        assert!(current.on_disk());
        assert_eq!(current.reload::<Blob>().unwrap(), Blob(vec![2.0]));
        // Equal content is still a separate retained generation.
        let replacement = registry.spill("same", &Blob(vec![2.0])).unwrap();
        assert!(!current.on_disk());
        assert!(!current.remove());
        assert_eq!(replacement.reload::<Blob>().unwrap(), Blob(vec![2.0]));
        assert!(replacement.remove());
        assert_eq!(registry.stats().used_bytes, 0);
        drop(registry);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn enlarged_spill_is_refused_against_the_reserved_frame_size() {
        let directory = scratch("enlarged");
        let registry = SpillRegistry::new(&directory, 4096).unwrap();
        let handle = registry.spill("a", &Blob(vec![1.0])).unwrap();
        let path = registry.lock().entries["a"].path.clone();
        let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(&[0; 4096]).unwrap();
        assert!(matches!(handle.reload::<Blob>(), Err(StateError::StaleCache(_))));
        assert_eq!(registry.stats().reloads, 0);
        assert!(handle.remove());
        drop(registry);
        let _ = fs::remove_dir_all(directory);
    }

    fn scratch(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("antecedent-spill-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn failed_write_preserves_existing_spills_and_accounting() {
        let directory = scratch("write-failure");
        let blob = Blob(vec![1.0]);
        let frame = build_frame(Blob::KIND, &blob.encode().unwrap()).len() as u64;
        let registry = SpillRegistry::new(&directory, frame).unwrap();
        let old = registry.spill("old", &blob).unwrap();
        let before = registry.stats();
        // Temporarily move the directory so exclusive creation fails without requiring
        // platform-specific permission behavior. Restore it before reading the old handle.
        let moved = directory.with_extension("moved");
        let _ = fs::remove_dir_all(&moved);
        fs::rename(&directory, &moved).unwrap();
        assert!(registry.spill("new", &blob).is_err());
        fs::rename(&moved, &directory).unwrap();
        assert_eq!(registry.stats(), before);
        assert_eq!(old.reload::<Blob>().unwrap(), blob);
        drop(old);
        drop(registry);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn registries_in_same_directory_cannot_overwrite_each_other() {
        let directory = scratch("shared-directory");
        let first = SpillRegistry::new(&directory, 4096).unwrap();
        let second = SpillRegistry::new(&directory, 4096).unwrap();
        let a = first.spill("same", &Blob(vec![1.0])).unwrap();
        let b = second.spill("same", &Blob(vec![2.0])).unwrap();
        let path_a = first.lock().entries["same"].path.clone();
        let path_b = second.lock().entries["same"].path.clone();
        assert_ne!(path_a, path_b);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&path_a).unwrap().permissions().mode() & 0o177, 0);
            assert_eq!(fs::metadata(&path_b).unwrap().permissions().mode() & 0o177, 0);
        }
        assert_eq!(a.reload::<Blob>().unwrap(), Blob(vec![1.0]));
        assert_eq!(b.reload::<Blob>().unwrap(), Blob(vec![2.0]));
        assert!(a.remove());
        assert_eq!(b.reload::<Blob>().unwrap(), Blob(vec![2.0]));
        drop(b);
        drop(first);
        drop(second);
        let _ = fs::remove_dir_all(directory);
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

    #[cfg(unix)]
    #[test]
    fn a_file_that_cannot_be_removed_stays_charged_until_it_is_gone() {
        let directory = scratch("orphan");
        let blob = Blob(vec![1.0]);
        let frame = build_frame(Blob::KIND, &blob.encode().unwrap()).len() as u64;
        let registry = SpillRegistry::new(&directory, frame * 2).unwrap();
        let handle = registry.spill("a", &blob).unwrap();
        let path = registry.lock().entries["a"].path.clone();
        // A non-empty directory at the file's path makes the unlink fail with something
        // other than "not found".
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        fs::write(path.join("occupant"), b"x").unwrap();
        assert!(handle.remove());
        let stats = registry.stats();
        assert_eq!((stats.entries, stats.orphaned_files), (0, 1));
        assert_eq!(stats.used_bytes, frame, "the unremoved file still holds its bytes");
        // With the orphan charged, a frame that needs the whole budget is refused.
        let wide = Blob(vec![0.0; 3]);
        let wide_frame = build_frame(Blob::KIND, &wide.encode().unwrap()).len() as u64;
        assert!(wide_frame + frame > frame * 2);
        assert!(matches!(registry.spill("wide", &wide), Err(StateError::CacheBudget { .. })));
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1, "only the orphan remains");
        fs::remove_dir_all(&path).unwrap();
        let next = registry.spill("b", &blob).unwrap();
        let stats = registry.stats();
        assert_eq!((stats.orphaned_files, stats.used_bytes), (0, frame));
        drop(next);
        drop(registry);
        let _ = fs::remove_dir_all(&directory);
    }

    #[cfg(unix)]
    #[test]
    fn a_new_registry_removes_files_left_by_a_dead_process_and_nothing_else() {
        let directory = scratch("recovery");
        fs::create_dir_all(&directory).unwrap();
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let dead = child.id();
        child.wait().unwrap();
        let digest = "0".repeat(DIGEST_HEX_LEN);
        let stale = directory.join(format!("{dead}-{digest}.spill"));
        let live = directory.join(format!("{}-{digest}.spill", std::process::id()));
        let foreign = directory.join(format!("{dead}-not-a-digest.spill"));
        let caller = directory.join("notes.txt");
        for file in [&stale, &live, &foreign, &caller] {
            fs::write(file, b"x").unwrap();
        }
        let registry = SpillRegistry::new(&directory, 4096).unwrap();
        assert_eq!(registry.stats().recovered_files, 1);
        assert!(!stale.exists());
        assert!(live.exists() && foreign.exists() && caller.exists());
        drop(registry);
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn concurrent_spills_keep_the_account_equal_to_the_files_on_disk() {
        let directory = scratch("concurrent");
        let blob = |seed: f64| Blob((0..16).map(|i| seed + f64::from(i)).collect());
        let frame = build_frame(Blob::KIND, &blob(0.0).encode().unwrap()).len() as u64;
        let registry = SpillRegistry::new(&directory, frame * 3).unwrap();
        std::thread::scope(|scope| {
            for worker in 0..4_u32 {
                let registry = &registry;
                scope.spawn(move || {
                    for round in 0..12_u32 {
                        let key = format!("k{}", (worker + round) % 5);
                        let _ = registry.spill(&key, &blob(f64::from(worker * 100 + round)));
                    }
                });
            }
        });
        let stats = registry.stats();
        assert!(stats.used_bytes <= frame * 3);
        assert_eq!(stats.orphaned_files, 0);
        assert_eq!(stats.used_bytes, stats.entries as u64 * frame);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), stats.entries);
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
