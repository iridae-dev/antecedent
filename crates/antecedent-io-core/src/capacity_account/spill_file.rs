//! Spill frame and its atomic file write, shared by every spill in the account.
//!
//! A frame is the magic `ASPL`, a version, the payload length, the blake3
//! digest of the payload, and then the payload, all little-endian. A reader
//! checks the magic, the version, the length against the bytes it holds, and
//! the digest, and refuses a changed or truncated file with a typed
//! [`SpillFault`].
//!
//! A frame is written to a temporary name in the target directory, flushed to
//! disk, renamed over the target, and read back. The temporary file is removed
//! on every error, and so is the target when the read-back disagrees, so no
//! partial file stays behind.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::TryReserveError;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 4] = b"ASPL";
const VERSION: u32 = 1;
const DIGEST_DOMAIN: &str = "antecedent.capacity.spill-frame.v1";
const LENGTH_AT: usize = 8;
const DIGEST_AT: usize = 16;

/// Bytes before the payload: magic, version, payload length, digest.
pub(super) const FRAME_HEADER_BYTES: u64 = 48;
const HEADER_LEN: usize = 48;

/// How a spill frame failed to match its layout or its handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SpillFault {
    /// The first four bytes are not the spill magic.
    #[error("bad spill magic")]
    BadMagic,
    /// The version field is not one this reader decodes.
    #[error("unsupported spill version {found}")]
    UnsupportedVersion {
        /// Version in the header.
        found: u32,
    },
    /// The file is shorter than a frame header.
    #[error("spill file is shorter than its header")]
    ShortFrame,
    /// The file length is not what the header or the handle declares.
    ///
    /// Trailing bytes and truncation both land here.
    #[error("spill file is {actual_bytes} bytes but {expected_bytes} were declared")]
    LengthMismatch {
        /// Bytes the header or the handle declares.
        expected_bytes: u64,
        /// Bytes the file holds.
        actual_bytes: u64,
    },
    /// The payload does not hash to the digest the frame carries or the handle holds.
    #[error("spill digest does not match the payload")]
    DigestMismatch,
    /// The payload does not have the layout of the summary it should hold.
    #[error("spill payload does not have the expected layout")]
    PayloadLayout,
    /// The fit backend has no code in the spill layout.
    #[error("fit backend has no spill code")]
    UnknownBackend,
}

/// An operating-system failure with the step it happened in.
#[derive(Debug)]
pub(super) struct IoFail {
    pub(super) context: &'static str,
    pub(super) source: io::Error,
}

fn io_fail(context: &'static str) -> impl FnOnce(io::Error) -> IoFail {
    move |source| IoFail { context, source }
}

/// An allocation the allocator refused.
#[derive(Debug)]
pub(super) struct AllocFail {
    pub(super) bytes: u64,
    pub(super) source: TryReserveError,
}

/// Length and digest of a frame that was written and read back.
#[derive(Clone, Copy, Debug)]
pub(super) struct FrameReceipt {
    pub(super) frame_bytes: u64,
    pub(super) digest: [u8; 32],
}

/// Total bytes of a frame around `payload_len` payload bytes.
pub(super) fn frame_len(payload_len: u64) -> Option<u64> {
    payload_len.checked_add(FRAME_HEADER_BYTES)
}

/// A buffer holding a blank header, with room for `payload_len` payload bytes.
pub(super) fn begin_frame(payload_len: u64) -> Result<Vec<u8>, AllocFail> {
    let bytes = frame_len(payload_len).unwrap_or(u64::MAX);
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(usize::try_from(bytes).unwrap_or(usize::MAX))
        .map_err(|source| AllocFail { bytes, source })?;
    frame.resize(HEADER_LEN, 0);
    Ok(frame)
}

/// Fill the header of a frame whose payload has been appended after [`begin_frame`].
pub(super) fn seal_frame(frame: &mut [u8]) {
    let payload_len = (frame.len() - HEADER_LEN) as u64;
    let digest = payload_digest(&frame[HEADER_LEN..]);
    frame[0..4].copy_from_slice(MAGIC);
    frame[4..8].copy_from_slice(&VERSION.to_le_bytes());
    frame[LENGTH_AT..DIGEST_AT].copy_from_slice(&payload_len.to_le_bytes());
    frame[DIGEST_AT..HEADER_LEN].copy_from_slice(&digest);
}

fn payload_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(DIGEST_DOMAIN);
    hasher.update(payload);
    hasher.finalize().into()
}

/// The digest a sealed frame carries.
pub(super) fn frame_digest(frame: &[u8]) -> Result<[u8; 32], SpillFault> {
    let digest = frame.get(DIGEST_AT..HEADER_LEN).ok_or(SpillFault::ShortFrame)?;
    digest.try_into().map_err(|_| SpillFault::ShortFrame)
}

/// Check the header against the bytes held and return the payload.
pub(super) fn verify_frame(frame: &[u8]) -> Result<&[u8], SpillFault> {
    if frame.len() < HEADER_LEN {
        return Err(SpillFault::ShortFrame);
    }
    if &frame[0..4] != MAGIC {
        return Err(SpillFault::BadMagic);
    }
    let version = u32::from_le_bytes(word(&frame[4..8])?);
    if version != VERSION {
        return Err(SpillFault::UnsupportedVersion { found: version });
    }
    let declared = u64::from_le_bytes(word(&frame[LENGTH_AT..DIGEST_AT])?);
    let payload = &frame[HEADER_LEN..];
    if declared != payload.len() as u64 {
        return Err(SpillFault::LengthMismatch {
            expected_bytes: frame_len(declared).unwrap_or(u64::MAX),
            actual_bytes: frame.len() as u64,
        });
    }
    if payload_digest(payload) != frame_digest(frame)? {
        return Err(SpillFault::DigestMismatch);
    }
    Ok(payload)
}

fn word<const N: usize>(bytes: &[u8]) -> Result<[u8; N], SpillFault> {
    bytes.try_into().map_err(|_| SpillFault::ShortFrame)
}

/// What a handle declares about the file it names.
#[derive(Clone, Copy, Debug)]
pub(super) struct Declared {
    pub(super) frame_bytes: u64,
    pub(super) digest: [u8; 32],
}

/// Read a frame, check it, and return its bytes. The caller slices past
/// [`FRAME_HEADER_BYTES`] for the payload.
///
/// The file length is checked against `declared` before any buffer is
/// allocated.
pub(super) fn read_verified<E>(path: &Path, declared: Declared) -> Result<Vec<u8>, E>
where
    E: From<IoFail> + From<SpillFault> + From<AllocFail>,
{
    let mut file = File::open(path).map_err(io_fail("open the spill file"))?;
    let actual_bytes = file.metadata().map_err(io_fail("measure the spill file"))?.len();
    if actual_bytes != declared.frame_bytes {
        return Err(E::from(SpillFault::LengthMismatch {
            expected_bytes: declared.frame_bytes,
            actual_bytes,
        }));
    }
    let size = usize::try_from(actual_bytes).unwrap_or(usize::MAX);
    let mut frame = Vec::new();
    frame.try_reserve_exact(size).map_err(|source| AllocFail { bytes: actual_bytes, source })?;
    frame.resize(size, 0);
    file.read_exact(&mut frame).map_err(io_fail("read the spill file"))?;
    verify_frame(&frame)?;
    if frame_digest(&frame)? != declared.digest {
        return Err(E::from(SpillFault::DigestMismatch));
    }
    Ok(frame)
}

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Removes the file it names unless it was disarmed.
struct Cleanup(Option<PathBuf>);

impl Cleanup {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn temp_name(target: &Path) -> Result<PathBuf, IoFail> {
    let name = target.file_name().ok_or_else(|| IoFail {
        context: "name the spill file",
        source: io::Error::new(io::ErrorKind::InvalidInput, "spill path has no file name"),
    })?;
    let mut temp = OsString::from(".");
    temp.push(name);
    let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    temp.push(format!(".tmp-{}-{serial}", std::process::id()));
    Ok(target.with_file_name(temp))
}

/// Write `frame` to `target` all or nothing, then read it back.
///
/// `write` stores the frame in the temporary file. `before_rename` runs once
/// the temporary file is complete and can stop the commit. The frame buffer is
/// dropped before the read-back so the two are never resident together.
pub(super) fn commit_frame<E>(
    target: &Path,
    frame: Vec<u8>,
    write: impl FnOnce(&mut File, &[u8]) -> io::Result<()>,
    before_rename: impl FnOnce() -> Result<(), E>,
) -> Result<FrameReceipt, E>
where
    E: From<IoFail> + From<SpillFault> + From<AllocFail>,
{
    let digest = frame_digest(&frame)?;
    let frame_bytes = frame.len() as u64;
    let temp = temp_name(target)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(io_fail("create the temporary spill file"))?;
    let mut cleanup = Cleanup(Some(temp.clone()));
    write(&mut file, &frame).map_err(io_fail("write the spill file"))?;
    file.flush().map_err(io_fail("flush the spill file"))?;
    file.sync_all().map_err(io_fail("sync the spill file"))?;
    drop(file);
    drop(frame);
    before_rename()?;
    fs::rename(&temp, target).map_err(io_fail("move the spill file into place"))?;
    cleanup.disarm();
    let mut published = Cleanup(Some(target.to_path_buf()));
    read_verified::<E>(target, Declared { frame_bytes, digest })?;
    published.disarm();
    Ok(FrameReceipt { frame_bytes, digest })
}

/// Write the whole frame. The default `write` step of [`commit_frame`].
pub(super) fn write_whole(file: &mut File, frame: &[u8]) -> io::Result<()> {
    file.write_all(frame)
}

/// Remove a spill file. A file that is already gone is not an error.
pub(super) fn remove_file(path: &Path) -> Result<(), IoFail> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(IoFail { context: "remove the spill file", source: error }),
    }
}
