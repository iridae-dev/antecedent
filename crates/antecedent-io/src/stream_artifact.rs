//! Checksummed stream artifact: bounds, cheap metadata, and shared payloads.
//!
//! The reader validates a byte stream the caller already holds. It does not open files, map
//! them, sleep, or spawn threads. [`StreamBounds`] records the section and retained-output
//! limits and both the writer and the reader enforce them. A truncated stream, a checksum
//! mismatch, and a format newer than [`STREAM_ARTIFACT_VERSION`] are refusals.
//!
//! [`StreamArtifact::open`] cancels between sections only. [`StreamArtifact::encode_to`] and
//! [`StreamArtifact::read_to`] are the incremental forms: they lease a chunk-sized working set
//! from the operation's admission pool before moving a byte, never hold the payload (logical
//! or on-wire) whole, check cancellation between chunks, and bound decompression by the
//! declared logical size as the bytes arrive. The writer sizes and checksums the payload in
//! passes over a re-openable source and then streams it; the reader hashes the on-wire payload
//! before it decodes it, so a sink only receives bytes whose checksum verified. An accelerator-produced artifact is consumable on CPU when its device payload is
//! absent or already materialized as bytes; a live device pointer is refused.
//! [`StreamArtifact::read_metadata`] does not retain the payload. A payload the caller still
//! holds is borrowed (`Arc` clone), not copied.
//!
//! ```text
//! magic (8) | version (u16 LE) | section_count (u32 LE)
//! section: name_len (u16 LE) | name | codec (u8) | logical_len (u64 LE)
//!          | on_wire_len (u64 LE) | checksum (32) | on_wire
//! ```
//!
//! The checksum is the domain-separated BLAKE3 of the on-wire bytes
//! (`antecedent.stream_artifact.section.v1`). The codec byte is `0` (the on-wire bytes are the
//! logical bytes) or `1` (a Zstandard frame of the logical bytes, written and read by the
//! incremental forms; [`StreamArtifact::open`] borrows stored bytes and refuses it). Any other
//! codec is refused. The first section is `meta`; an optional second section is `payload`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use antecedent_core::{CancellationToken, CanonicalWriter};
use antecedent_io_core::byte_cursor::ByteCursor;
use antecedent_io_core::container::{
    DomainDigest, stream_encoder_workspace_bytes, zstd_compress_chunked, zstd_decompress_chunked,
};
use antecedent_io_core::error::IoError;

use crate::width::lossless_u64;

pub use antecedent_io_core::container::{LogicalSource, StreamReceipt, StreamWork};

/// Magic bytes: `ASTREAM\0`.
pub const STREAM_ARTIFACT_MAGIC: [u8; 8] = *b"ASTREAM\0";

/// Stream format this reader writes and accepts.
pub const STREAM_ARTIFACT_VERSION: u16 = 1;

const CHECKSUM_TAG: &str = "antecedent.stream_artifact.section.v1";
const META_SECTION: &str = "meta";
const PAYLOAD_SECTION: &str = "payload";
const MAX_SECTION_NAME_LEN: usize = 64;
const MAX_ARTIFACT_ID_LEN: usize = 256;
const STORED: u8 = 0;
const CODEC_ZSTD: u8 = 1;
/// Largest metadata section the incremental reader accepts; real ones are under 300 bytes.
const MAX_META_SECTION_BYTES: u64 = 1024;
const PRODUCER_CPU: u8 = 1;
const PRODUCER_ACCELERATOR: u8 = 2;
const DEVICE_ABSENT: u8 = 0;
const DEVICE_MATERIALIZED: u8 = 1;
const DEVICE_LIVE: u8 = 2;

/// Limits recorded for a stream and enforced before a section is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamBounds {
    /// Largest on-wire or declared logical size of one section.
    pub max_section_bytes: u64,
    /// Largest payload a reader may keep. Metadata does not count.
    pub max_retained_bytes: u64,
    /// Most sections, including metadata.
    pub max_sections: u32,
}

impl StreamBounds {
    /// Default section cap: 16 MiB.
    pub const DEFAULT_MAX_SECTION_BYTES: u64 = 16 * 1024 * 1024;
    /// Default retained-payload cap: 16 MiB.
    pub const DEFAULT_MAX_RETAINED_BYTES: u64 = 16 * 1024 * 1024;
    /// Default section count cap.
    pub const DEFAULT_MAX_SECTIONS: u32 = 8;
}

impl Default for StreamBounds {
    fn default() -> Self {
        Self {
            max_section_bytes: Self::DEFAULT_MAX_SECTION_BYTES,
            max_retained_bytes: Self::DEFAULT_MAX_RETAINED_BYTES,
            max_sections: Self::DEFAULT_MAX_SECTIONS,
        }
    }
}

/// Who produced the artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Producer {
    /// Host code.
    Cpu,
    /// An accelerator. CPU consumption depends on [`DevicePayload`].
    Accelerator,
}

/// Where the device payload lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DevicePayload {
    /// No device buffer was declared.
    Absent,
    /// The device buffer is already host bytes in the payload section.
    Materialized,
    /// A live device address. Not consumable on CPU.
    LivePointer {
        /// Device that owns the allocation.
        device_id: u32,
        /// Device address. Not a host pointer.
        address: u64,
        /// Declared device allocation size.
        bytes: u64,
    },
}

/// Metadata kept without the payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamMetadata {
    /// Caller-chosen artifact id. Non-empty, at most 256 bytes.
    pub artifact_id: String,
    /// Producer of the record.
    pub producer: Producer,
    /// Device payload declaration.
    pub device: DevicePayload,
}

/// Bytes the encoder writes for the optional payload section.
#[derive(Clone, Copy, Debug)]
pub enum PayloadSource<'a> {
    /// No payload section.
    Absent,
    /// Uncompressed logical bytes.
    Stored(&'a [u8]),
    /// Uncompressed logical bytes the caller still holds.
    ///
    /// Encoding writes those bytes into the stream. Opening with the same `Arc` borrows it.
    Shared(&'a Arc<[u8]>),
}

/// Host view of an accelerator (or CPU) payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuPayload<'a> {
    /// Nothing was declared for the device.
    Absent,
    /// Bytes already on the host.
    Bytes(&'a [u8]),
}

/// Why a stream artifact was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StreamArtifactError {
    /// The buffer ended before a declared field or section.
    #[error("stream artifact is truncated")]
    Truncated,
    /// The on-wire checksum does not match the section bytes.
    #[error("stream artifact checksum mismatch in section {section}")]
    ChecksumMismatch {
        /// `meta` or `payload`.
        section: &'static str,
    },
    /// `version` is newer than [`STREAM_ARTIFACT_VERSION`].
    #[error("stream artifact format {version} is newer than this reader")]
    NewerFormat {
        /// Version word from the stream.
        version: u16,
    },
    /// The version word is not the one this reader accepts.
    #[error("stream artifact format {version} is not supported")]
    UnsupportedFormat {
        /// Version word from the stream.
        version: u16,
    },
    /// The caller token was set between sections.
    #[error("stream artifact was cancelled between sections")]
    Cancelled,
    /// A declared section size is over [`StreamBounds::max_section_bytes`].
    #[error("stream artifact section exceeds {limit} bytes")]
    SectionBound {
        /// The limit that was recorded.
        limit: u64,
    },
    /// Keeping the payload would pass [`StreamBounds::max_retained_bytes`].
    #[error("stream artifact retained output exceeds {limit} bytes")]
    RetainedOutput {
        /// The limit that was recorded.
        limit: u64,
    },
    /// More sections than [`StreamBounds::max_sections`].
    #[error("stream artifact has more than {limit} sections")]
    TooManySections {
        /// The limit that was recorded.
        limit: u32,
    },
    /// [`DevicePayload::LivePointer`] cannot be read on CPU.
    #[error("accelerator artifact holds a live device pointer")]
    LiveDevicePointer,
    /// The bytes are not this format.
    #[error("malformed stream artifact: {0}")]
    Malformed(&'static str),
    /// The admission pool refused the working set of an incremental operation.
    #[error("stream artifact working set was refused by the operation budget")]
    BudgetRefused,
    /// The payload frame is corrupt or expands past its declared size.
    #[error("stream artifact payload failed to decompress")]
    Decompress,
    /// A source, sink or reader failed.
    #[error("stream artifact I/O failed: {kind:?}")]
    Io {
        /// Kind of the underlying failure.
        kind: std::io::ErrorKind,
    },
}

/// How an incremental encode stores the payload section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadCodec {
    /// On-wire bytes are the logical bytes.
    Stored,
    /// On-wire bytes are a Zstandard frame of the logical bytes.
    Zstd,
}

impl PayloadCodec {
    const fn byte(self) -> u8 {
        match self {
            Self::Stored => STORED,
            Self::Zstd => CODEC_ZSTD,
        }
    }

    const fn workspace_bytes(self) -> u64 {
        match self {
            Self::Stored => 0,
            Self::Zstd => stream_encoder_workspace_bytes(),
        }
    }
}

#[derive(Clone, Debug)]
enum PayloadBytes<'a> {
    Borrowed(&'a [u8]),
    Shared(Arc<[u8]>),
}

impl PayloadBytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Shared(bytes) => bytes.as_ref(),
        }
    }
}

/// A validated stream artifact.
///
/// Payload bytes are borrowed from the caller buffer (the lifetime a mapped file would have).
/// A shared payload aliases the caller's `Arc`.
#[derive(Clone, Debug)]
pub struct StreamArtifact<'a> {
    metadata: StreamMetadata,
    payload: Option<PayloadBytes<'a>>,
}

impl PartialEq for StreamArtifact<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata && self.payload() == other.payload()
    }
}

impl Eq for StreamArtifact<'_> {}

impl<'a> StreamArtifact<'a> {
    /// Encode `metadata` and `payload` under `bounds`.
    pub fn encode(
        metadata: &StreamMetadata,
        payload: PayloadSource<'_>,
        bounds: &StreamBounds,
    ) -> Result<Vec<u8>, StreamArtifactError> {
        let logical = payload.logical();
        validate_declaration(metadata, logical.is_some())?;
        let sections = if logical.is_some() { 2u32 } else { 1 };
        if sections > bounds.max_sections {
            return Err(StreamArtifactError::TooManySections { limit: bounds.max_sections });
        }
        let meta = encode_metadata(metadata)?;
        check_section_size(u64_len(meta.len())?, bounds)?;
        let encoded = encode_payload(payload, bounds)?;
        if let Some(bytes) = logical {
            charge(0, u64_len(bytes.len())?, bounds.max_retained_bytes)?;
        }

        let mut out = Vec::new();
        out.extend_from_slice(&STREAM_ARTIFACT_MAGIC);
        out.extend_from_slice(&STREAM_ARTIFACT_VERSION.to_le_bytes());
        out.extend_from_slice(&sections.to_le_bytes());
        write_section(&mut out, META_SECTION, STORED, &meta, &meta)?;
        if let Some(section) = encoded {
            write_section(
                &mut out,
                PAYLOAD_SECTION,
                section.codec,
                section.logical,
                section.on_wire,
            )?;
        }
        Ok(out)
    }

    /// Validate `stream` and retain the payload when the bounds allow it.
    ///
    /// `shared_payload`, when it still holds the logical payload bytes, is borrowed. The token
    /// is observed once per section, before that section is read.
    pub fn open(
        stream: &'a [u8],
        bounds: &StreamBounds,
        cancel: &CancellationToken,
        shared_payload: Option<&Arc<[u8]>>,
    ) -> Result<Self, StreamArtifactError> {
        let (metadata, payload) = read_sections(stream, bounds, cancel, true, shared_payload)?;
        Ok(Self { metadata, payload })
    }

    /// Validate the stream and return metadata. The payload is not retained.
    ///
    /// Declared section sizes are still refused. Retained-output limits are not charged, because
    /// the payload is not kept.
    pub fn read_metadata(
        stream: &[u8],
        bounds: &StreamBounds,
        cancel: &CancellationToken,
    ) -> Result<StreamMetadata, StreamArtifactError> {
        let (metadata, _payload) = read_sections(stream, bounds, cancel, false, None)?;
        Ok(metadata)
    }

    /// Metadata without the payload bytes.
    #[must_use = "metadata is the cheap view of the artifact"]
    pub fn metadata(&self) -> &StreamMetadata {
        &self.metadata
    }

    /// Logical payload bytes, borrowed from this artifact.
    #[must_use = "the payload bytes are the retained section"]
    pub fn payload(&self) -> Option<&[u8]> {
        self.payload.as_ref().map(PayloadBytes::as_slice)
    }

    /// Payload borrowed from the caller buffer, when the section was stored uncompressed and
    /// the caller did not pass a shared buffer.
    #[must_use = "the borrowed view is only valid with the caller buffer"]
    pub fn borrowed_payload(&self) -> Option<&'a [u8]> {
        match &self.payload {
            Some(PayloadBytes::Borrowed(bytes)) => Some(*bytes),
            _ => None,
        }
    }

    /// The caller's `Arc`, when open reused it instead of copying the payload.
    #[must_use = "the shared payload aliases the caller's allocation"]
    pub fn shared_payload(&self) -> Option<&Arc<[u8]>> {
        match &self.payload {
            Some(PayloadBytes::Shared(bytes)) => Some(bytes),
            _ => None,
        }
    }

    /// CPU view of an accelerator-produced artifact.
    ///
    /// Absent and materialized payloads are readable. A live device pointer is refused.
    pub fn consume_on_cpu(&self) -> Result<CpuPayload<'_>, StreamArtifactError> {
        match self.metadata.device {
            DevicePayload::Absent => Ok(CpuPayload::Absent),
            DevicePayload::Materialized => {
                let bytes =
                    self.payload().ok_or(StreamArtifactError::Malformed("missing payload"))?;
                Ok(CpuPayload::Bytes(bytes))
            }
            DevicePayload::LivePointer { .. } => Err(StreamArtifactError::LiveDevicePointer),
        }
    }
}

struct EncodedPayload<'a> {
    codec: u8,
    logical: &'a [u8],
    on_wire: &'a [u8],
}

impl<'a> PayloadSource<'a> {
    fn logical(self) -> Option<&'a [u8]> {
        match self {
            Self::Absent => None,
            Self::Stored(bytes) => Some(bytes),
            Self::Shared(bytes) => Some(bytes.as_ref()),
        }
    }
}

fn encode_payload<'a>(
    payload: PayloadSource<'a>,
    bounds: &StreamBounds,
) -> Result<Option<EncodedPayload<'a>>, StreamArtifactError> {
    let Some(logical) = payload.logical() else {
        return Ok(None);
    };
    check_section_size(u64_len(logical.len())?, bounds)?;
    Ok(Some(EncodedPayload { codec: STORED, logical, on_wire: logical }))
}

fn encode_metadata(metadata: &StreamMetadata) -> Result<Vec<u8>, StreamArtifactError> {
    let id = metadata.artifact_id.as_bytes();
    if id.is_empty() || id.len() > MAX_ARTIFACT_ID_LEN {
        return Err(StreamArtifactError::Malformed("artifact id"));
    }
    let mut body = Vec::new();
    body.push(match metadata.producer {
        Producer::Cpu => PRODUCER_CPU,
        Producer::Accelerator => PRODUCER_ACCELERATOR,
    });
    match metadata.device {
        DevicePayload::Absent => body.push(DEVICE_ABSENT),
        DevicePayload::Materialized => body.push(DEVICE_MATERIALIZED),
        DevicePayload::LivePointer { device_id, address, bytes } => {
            body.push(DEVICE_LIVE);
            body.extend_from_slice(&device_id.to_le_bytes());
            body.extend_from_slice(&address.to_le_bytes());
            body.extend_from_slice(&bytes.to_le_bytes());
        }
    }
    let id_len =
        u16::try_from(id.len()).map_err(|_| StreamArtifactError::Malformed("artifact id"))?;
    body.extend_from_slice(&id_len.to_le_bytes());
    body.extend_from_slice(id);
    Ok(body)
}

fn validate_identity(metadata: &StreamMetadata) -> Result<(), StreamArtifactError> {
    if metadata.artifact_id.is_empty() || metadata.artifact_id.len() > MAX_ARTIFACT_ID_LEN {
        return Err(StreamArtifactError::Malformed("artifact id"));
    }
    if matches!(metadata.producer, Producer::Cpu)
        && matches!(metadata.device, DevicePayload::LivePointer { .. })
    {
        return Err(StreamArtifactError::Malformed("cpu artifact has no device pointer"));
    }
    Ok(())
}

fn validate_declaration(
    metadata: &StreamMetadata,
    has_payload: bool,
) -> Result<(), StreamArtifactError> {
    validate_identity(metadata)?;
    if matches!(metadata.device, DevicePayload::LivePointer { .. }) && has_payload {
        return Err(StreamArtifactError::Malformed("live device pointer has no host payload"));
    }
    let wants_payload = matches!(metadata.device, DevicePayload::Materialized);
    if wants_payload != has_payload {
        return Err(StreamArtifactError::Malformed(
            "payload does not match the device declaration",
        ));
    }
    Ok(())
}

fn write_section(
    out: &mut Vec<u8>,
    name: &str,
    codec: u8,
    logical: &[u8],
    on_wire: &[u8],
) -> Result<(), StreamArtifactError> {
    let name_len =
        u16::try_from(name.len()).map_err(|_| StreamArtifactError::Malformed("section name"))?;
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.push(codec);
    out.extend_from_slice(&u64_len(logical.len())?.to_le_bytes());
    out.extend_from_slice(&u64_len(on_wire.len())?.to_le_bytes());
    out.extend_from_slice(&checksum(on_wire));
    out.extend_from_slice(on_wire);
    Ok(())
}

fn checksum(bytes: &[u8]) -> [u8; 32] {
    CanonicalWriter::digest_bytes(CHECKSUM_TAG, bytes)
}

fn check_section_size(len: u64, bounds: &StreamBounds) -> Result<(), StreamArtifactError> {
    if len > bounds.max_section_bytes {
        return Err(StreamArtifactError::SectionBound { limit: bounds.max_section_bytes });
    }
    Ok(())
}

fn charge(retained: u64, add: u64, limit: u64) -> Result<u64, StreamArtifactError> {
    let next = retained.checked_add(add).ok_or(StreamArtifactError::RetainedOutput { limit })?;
    if next > limit {
        return Err(StreamArtifactError::RetainedOutput { limit });
    }
    Ok(next)
}

fn u64_len(len: usize) -> Result<u64, StreamArtifactError> {
    u64::try_from(len).map_err(|_| StreamArtifactError::SectionBound { limit: u64::MAX })
}

fn read_sections<'a>(
    stream: &'a [u8],
    bounds: &StreamBounds,
    cancel: &CancellationToken,
    retain_payload: bool,
    shared: Option<&Arc<[u8]>>,
) -> Result<(StreamMetadata, Option<PayloadBytes<'a>>), StreamArtifactError> {
    let mut reader = open_reader(stream)?;
    let section_count = reader.u32()?;
    if section_count == 0 {
        return Err(StreamArtifactError::Malformed("missing metadata"));
    }
    if section_count > bounds.max_sections {
        return Err(StreamArtifactError::TooManySections { limit: bounds.max_sections });
    }
    observe(cancel)?;
    let metadata = {
        let meta = read_section(&mut reader, bounds)?;
        if meta.name != META_SECTION || meta.codec != STORED {
            return Err(StreamArtifactError::Malformed("metadata section"));
        }
        if meta.on_wire.len() != meta.logical_len {
            return Err(StreamArtifactError::Malformed("stored length"));
        }
        decode_metadata(meta.on_wire)?
    };
    let (saw_payload, payload) = if section_count == 1 {
        (false, None)
    } else if section_count == 2 {
        observe(cancel)?;
        (true, read_payload_section(&mut reader, bounds, retain_payload, shared)?)
    } else {
        return Err(StreamArtifactError::Malformed("section count"));
    };
    let declared_absent =
        matches!(metadata.device, DevicePayload::Absent | DevicePayload::LivePointer { .. });
    if saw_payload == declared_absent {
        return Err(StreamArtifactError::Malformed(
            "payload does not match the device declaration",
        ));
    }
    reader.finish(trailing_bytes())?;
    let kept = if retain_payload { payload } else { None };
    Ok((metadata, kept))
}

struct SectionView<'a> {
    name: &'a str,
    codec: u8,
    logical_len: usize,
    on_wire: &'a [u8],
}

fn read_payload_section<'a>(
    reader: &mut Reader<'a>,
    bounds: &StreamBounds,
    retain_payload: bool,
    shared: Option<&Arc<[u8]>>,
) -> Result<Option<PayloadBytes<'a>>, StreamArtifactError> {
    let section = read_section(reader, bounds)?;
    if section.name != PAYLOAD_SECTION {
        return Err(StreamArtifactError::Malformed("payload section"));
    }
    if !retain_payload {
        return Ok(None);
    }
    if section.codec != STORED {
        return Err(StreamArtifactError::Malformed("codec"));
    }
    if section.on_wire.len() != section.logical_len {
        return Err(StreamArtifactError::Malformed("stored length"));
    }
    let logical_len = u64_len(section.logical_len)?;
    charge(0, logical_len, bounds.max_retained_bytes)?;
    Ok(Some(adopt_stored(section.on_wire, shared)))
}

fn adopt_stored<'a>(on_wire: &'a [u8], shared: Option<&Arc<[u8]>>) -> PayloadBytes<'a> {
    if let Some(shared) = shared.filter(|held| held.as_ref() == on_wire) {
        return PayloadBytes::Shared(Arc::clone(shared));
    }
    PayloadBytes::Borrowed(on_wire)
}

fn read_section<'a>(
    reader: &mut Reader<'a>,
    bounds: &StreamBounds,
) -> Result<SectionView<'a>, StreamArtifactError> {
    let name_len = usize::from(reader.u16()?);
    if name_len == 0 || name_len > MAX_SECTION_NAME_LEN {
        return Err(StreamArtifactError::Malformed("section name"));
    }
    let name_bytes = reader.take(name_len)?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| StreamArtifactError::Malformed("section name"))?;
    let codec = reader.u8()?;
    let logical_len = reader.u64()?;
    let on_wire_len = reader.u64()?;
    check_section_size(logical_len, bounds)?;
    check_section_size(on_wire_len, bounds)?;
    if codec != STORED && codec != CODEC_ZSTD {
        return Err(StreamArtifactError::Malformed("codec"));
    }
    let checksum_bytes = reader.take(32)?;
    let on_wire_len = usize::try_from(on_wire_len)
        .map_err(|_| StreamArtifactError::SectionBound { limit: bounds.max_section_bytes })?;
    let on_wire = reader.take(on_wire_len)?;
    let mut expected = [0u8; 32];
    expected.copy_from_slice(checksum_bytes);
    if expected != checksum(on_wire) {
        let section = if name == META_SECTION { META_SECTION } else { PAYLOAD_SECTION };
        return Err(StreamArtifactError::ChecksumMismatch { section });
    }
    let logical_len = usize::try_from(logical_len)
        .map_err(|_| StreamArtifactError::SectionBound { limit: bounds.max_section_bytes })?;
    Ok(SectionView { name, codec, logical_len, on_wire })
}

fn decode_metadata(bytes: &[u8]) -> Result<StreamMetadata, StreamArtifactError> {
    let mut reader = Reader::new(bytes, truncated);
    let producer = match reader.u8()? {
        PRODUCER_CPU => Producer::Cpu,
        PRODUCER_ACCELERATOR => Producer::Accelerator,
        _ => return Err(StreamArtifactError::Malformed("producer")),
    };
    let device = match reader.u8()? {
        DEVICE_ABSENT => DevicePayload::Absent,
        DEVICE_MATERIALIZED => DevicePayload::Materialized,
        DEVICE_LIVE => DevicePayload::LivePointer {
            device_id: reader.u32()?,
            address: reader.u64()?,
            bytes: reader.u64()?,
        },
        _ => return Err(StreamArtifactError::Malformed("device payload")),
    };
    let id_len = usize::from(reader.u16()?);
    if id_len == 0 || id_len > MAX_ARTIFACT_ID_LEN {
        return Err(StreamArtifactError::Malformed("artifact id"));
    }
    let id = reader.take(id_len)?;
    let artifact_id = std::str::from_utf8(id)
        .map_err(|_| StreamArtifactError::Malformed("artifact id"))?
        .to_owned();
    reader.finish(trailing_bytes())?;
    let metadata = StreamMetadata { artifact_id, producer, device };
    validate_identity(&metadata)?;
    Ok(metadata)
}

fn from_io(error: IoError) -> StreamArtifactError {
    match error {
        IoError::StreamCancelled => StreamArtifactError::Cancelled,
        IoError::StreamBudgetRefused { .. } => StreamArtifactError::BudgetRefused,
        IoError::ChecksumMismatch { .. } => {
            StreamArtifactError::ChecksumMismatch { section: PAYLOAD_SECTION }
        }
        IoError::Decompress { .. } => StreamArtifactError::Decompress,
        IoError::TooLarge => StreamArtifactError::Malformed("size"),
        IoError::Io(failure) => StreamArtifactError::Io { kind: failure.kind() },
        _ => StreamArtifactError::Io { kind: std::io::ErrorKind::Other },
    }
}

fn from_std(error: &std::io::Error) -> StreamArtifactError {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        StreamArtifactError::Truncated
    } else {
        StreamArtifactError::Io { kind: error.kind() }
    }
}

fn tick(work: &StreamWork<'_>) -> Result<(), StreamArtifactError> {
    work.tick().map_err(from_io)
}

/// Counts the bytes written through it and forwards them.
struct CountWriter<W: Write> {
    inner: W,
    written: u64,
}

impl<W: Write> Write for CountWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let taken = self.inner.write(buf)?;
        self.written += lossless_u64(taken);
        Ok(taken)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Feeds a [`DomainDigest`] with everything written through it, and forwards it.
struct DigestWriter<W: Write> {
    inner: W,
    digest: DomainDigest,
    written: u64,
    largest: usize,
}

impl<W: Write> Write for DigestWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let taken = self.inner.write(buf)?;
        self.digest.update(&buf[..taken]);
        self.written += lossless_u64(taken);
        self.largest = self.largest.max(taken);
        Ok(taken)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Run `source` through `codec` into `sink`, a chunk at a time. Returns the logical bytes read.
fn encode_payload_pass<W: Write>(
    source: &dyn LogicalSource,
    codec: PayloadCodec,
    sink: &mut W,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
) -> Result<u64, StreamArtifactError> {
    let mut reader = source.open().map_err(|e| from_std(&e))?;
    let moved = match codec {
        PayloadCodec::Zstd => {
            zstd_compress_chunked(&mut reader, sink, buffer, work).map_err(from_io)?
        }
        PayloadCodec::Stored => {
            let mut total = 0u64;
            loop {
                tick(work)?;
                let taken = reader.read(buffer).map_err(|e| from_std(&e))?;
                if taken == 0 {
                    break total;
                }
                sink.write_all(&buffer[..taken]).map_err(|e| from_std(&e))?;
                total += lossless_u64(taken);
            }
        }
    };
    if moved != source.logical_len() {
        return Err(StreamArtifactError::Malformed("payload length"));
    }
    Ok(moved)
}

impl StreamArtifact<'_> {
    /// Encode `metadata` and an optional payload into `w` without holding the payload.
    ///
    /// The payload is read from a re-openable source in chunks. The first passes only count
    /// and checksum the on-wire bytes (the layout puts both before the payload); the last
    /// pass writes them, and fails with [`StreamArtifactError::ChecksumMismatch`] if the
    /// source changed between passes. The bytes are the same a buffered [`Self::encode`]
    /// writes for a stored payload. The working set is leased from `work.ctx` before the
    /// first byte is written, and the declared sizes are checked against `bounds` first.
    ///
    /// # Errors
    ///
    /// The declaration and bound errors of [`Self::encode`], [`StreamArtifactError::BudgetRefused`],
    /// [`StreamArtifactError::Cancelled`] between chunks, and I/O failures.
    pub fn encode_to<W: Write>(
        metadata: &StreamMetadata,
        payload: Option<(&dyn LogicalSource, PayloadCodec)>,
        bounds: &StreamBounds,
        work: &StreamWork<'_>,
        mut w: W,
    ) -> Result<StreamReceipt, StreamArtifactError> {
        validate_declaration(metadata, payload.is_some())?;
        let sections = if payload.is_some() { 2u32 } else { 1 };
        if sections > bounds.max_sections {
            return Err(StreamArtifactError::TooManySections { limit: bounds.max_sections });
        }
        let meta = encode_metadata(metadata)?;
        check_section_size(u64_len(meta.len())?, bounds)?;
        if let Some((source, _)) = payload {
            check_section_size(source.logical_len(), bounds)?;
            charge(0, source.logical_len(), bounds.max_retained_bytes)?;
        }
        let workspace = payload.map_or(0, |(_, codec)| codec.workspace_bytes());
        let mut lease = work.admit(workspace).map_err(from_io)?;
        let mut receipt = StreamReceipt {
            leased_bytes: lease.bytes(),
            chunk_bytes: work.chunk_bytes,
            ..StreamReceipt::default()
        };

        let sized = match payload {
            None => None,
            Some((source, codec)) => {
                let on_wire_len = match codec {
                    PayloadCodec::Stored => source.logical_len(),
                    PayloadCodec::Zstd => {
                        let mut count = CountWriter { inner: std::io::sink(), written: 0 };
                        encode_payload_pass(source, codec, &mut count, lease.buffer(), work)?;
                        count.written
                    }
                };
                check_section_size(on_wire_len, bounds)?;
                let mut digest = DigestWriter {
                    inner: std::io::sink(),
                    digest: DomainDigest::new(CHECKSUM_TAG, on_wire_len),
                    written: 0,
                    largest: 0,
                };
                encode_payload_pass(source, codec, &mut digest, lease.buffer(), work)?;
                let checksum = digest
                    .digest
                    .finish()
                    .ok_or(StreamArtifactError::Malformed("payload length"))?;
                Some((source, codec, on_wire_len, checksum))
            }
        };

        let mut head = Vec::new();
        head.extend_from_slice(&STREAM_ARTIFACT_MAGIC);
        head.extend_from_slice(&STREAM_ARTIFACT_VERSION.to_le_bytes());
        head.extend_from_slice(&sections.to_le_bytes());
        write_section(&mut head, META_SECTION, STORED, &meta, &meta)?;
        if let Some((source, codec, on_wire_len, checksum)) = &sized {
            let name_len = u16::try_from(PAYLOAD_SECTION.len())
                .map_err(|_| StreamArtifactError::Malformed("section name"))?;
            head.extend_from_slice(&name_len.to_le_bytes());
            head.extend_from_slice(PAYLOAD_SECTION.as_bytes());
            head.push(codec.byte());
            head.extend_from_slice(&source.logical_len().to_le_bytes());
            head.extend_from_slice(&on_wire_len.to_le_bytes());
            head.extend_from_slice(checksum);
        }
        w.write_all(&head).map_err(|e| from_std(&e))?;
        if let Some((source, codec, on_wire_len, checksum)) = sized {
            let mut out = DigestWriter {
                inner: &mut w,
                digest: DomainDigest::new(CHECKSUM_TAG, on_wire_len),
                written: 0,
                largest: 0,
            };
            let logical = encode_payload_pass(source, codec, &mut out, lease.buffer(), work)?;
            if out.written != on_wire_len || out.digest.finish() != Some(checksum) {
                return Err(StreamArtifactError::ChecksumMismatch { section: PAYLOAD_SECTION });
            }
            receipt.largest_write = receipt.largest_write.max(out.largest);
            receipt.logical_bytes = logical;
            receipt.on_wire_bytes = on_wire_len;
        }
        w.flush().map_err(|e| from_std(&e))?;
        Ok(receipt)
    }

    /// Validate a seekable stream and hand its payload to `sink` in chunks.
    ///
    /// Header, metadata and payload sizes are checked against `bounds` before the payload
    /// moves. The on-wire payload is checksummed in a first pass; only then is it decoded,
    /// so `sink` never sees bytes of a corrupt payload. A Zstandard payload is cut at its
    /// declared logical size as it decodes. Nothing but one chunk is held.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::open`], [`StreamArtifactError::BudgetRefused`],
    /// [`StreamArtifactError::Cancelled`] between chunks, [`StreamArtifactError::Decompress`],
    /// and the first error `sink` returns.
    #[allow(
        clippy::too_many_lines,
        reason = "one function reads the header, the metadata and the payload in order"
    )]
    pub fn read_to<R: Read + Seek>(
        mut r: R,
        bounds: &StreamBounds,
        work: &StreamWork<'_>,
        mut sink: impl FnMut(&[u8]) -> Result<(), StreamArtifactError>,
    ) -> Result<(StreamMetadata, StreamReceipt), StreamArtifactError> {
        let mut lease = work.admit(0).map_err(from_io)?;
        let mut receipt = StreamReceipt {
            leased_bytes: lease.bytes(),
            chunk_bytes: work.chunk_bytes,
            ..StreamReceipt::default()
        };
        let mut fixed = [0u8; 8];
        read_fixed(&mut r, &mut fixed)?;
        if fixed != STREAM_ARTIFACT_MAGIC {
            return Err(StreamArtifactError::Malformed("magic"));
        }
        let mut word = [0u8; 2];
        read_fixed(&mut r, &mut word)?;
        let version = u16::from_le_bytes(word);
        if version > STREAM_ARTIFACT_VERSION {
            return Err(StreamArtifactError::NewerFormat { version });
        }
        if version != STREAM_ARTIFACT_VERSION {
            return Err(StreamArtifactError::UnsupportedFormat { version });
        }
        let mut count = [0u8; 4];
        read_fixed(&mut r, &mut count)?;
        let section_count = u32::from_le_bytes(count);
        if section_count == 0 {
            return Err(StreamArtifactError::Malformed("missing metadata"));
        }
        if section_count > bounds.max_sections {
            return Err(StreamArtifactError::TooManySections { limit: bounds.max_sections });
        }
        if section_count > 2 {
            return Err(StreamArtifactError::Malformed("section count"));
        }
        tick(work)?;
        let head = read_head(&mut r, bounds)?;
        if head.name != META_SECTION
            || head.codec != STORED
            || head.on_wire_len != head.logical_len
            || head.on_wire_len > MAX_META_SECTION_BYTES
        {
            return Err(StreamArtifactError::Malformed("metadata section"));
        }
        let mut meta_bytes = vec![0u8; usize::try_from(head.on_wire_len).unwrap_or(0)];
        read_fixed(&mut r, &mut meta_bytes)?;
        if checksum(&meta_bytes) != head.checksum {
            return Err(StreamArtifactError::ChecksumMismatch { section: META_SECTION });
        }
        let metadata = decode_metadata(&meta_bytes)?;

        let declared_absent =
            matches!(metadata.device, DevicePayload::Absent | DevicePayload::LivePointer { .. });
        if (section_count == 2) == declared_absent {
            return Err(StreamArtifactError::Malformed(
                "payload does not match the device declaration",
            ));
        }
        if section_count == 2 {
            tick(work)?;
            let head = read_head(&mut r, bounds)?;
            if head.name != PAYLOAD_SECTION {
                return Err(StreamArtifactError::Malformed("payload section"));
            }
            if head.codec == STORED && head.on_wire_len != head.logical_len {
                return Err(StreamArtifactError::Malformed("stored length"));
            }
            charge(0, head.logical_len, bounds.max_retained_bytes)?;
            // Only a compressed payload needs a decoder; a stored one is leased the chunk alone.
            let _decoder_lease = if head.codec == CODEC_ZSTD {
                let bytes = work.decoder_workspace_bytes();
                Some(
                    work.ctx
                        .acquire_admission(0, bytes)
                        .map_err(|_| StreamArtifactError::BudgetRefused)?,
                )
            } else {
                None
            };
            let start = r.stream_position().map_err(|e| from_std(&e))?;

            // Pass one: the on-wire checksum, before any byte reaches the sink.
            let mut digest = DomainDigest::new(CHECKSUM_TAG, head.on_wire_len);
            let mut remaining = head.on_wire_len;
            while remaining > 0 {
                tick(work)?;
                let want =
                    usize::try_from(remaining).unwrap_or(usize::MAX).min(lease.buffer().len());
                let buffer = lease.buffer();
                read_fixed(&mut r, &mut buffer[..want])?;
                digest.update(&buffer[..want]);
                remaining -= lossless_u64(want);
                receipt.chunks += 1;
                receipt.largest_read = receipt.largest_read.max(want);
            }
            if digest.finish() != Some(head.checksum) {
                return Err(StreamArtifactError::ChecksumMismatch { section: PAYLOAD_SECTION });
            }
            receipt.on_wire_bytes = head.on_wire_len;

            // Pass two: decode the verified bytes into the sink.
            r.seek(SeekFrom::Start(start)).map_err(|e| from_std(&e))?;
            let mut failure: Option<StreamArtifactError> = None;
            let mut delivered = 0u64;
            let mut deliver = |chunk: &[u8]| -> Result<(), IoError> {
                match sink(chunk) {
                    Ok(()) => {
                        delivered += lossless_u64(chunk.len());
                        Ok(())
                    }
                    Err(error) => {
                        failure = Some(error);
                        Err(IoError::io("stream artifact sink refused a chunk"))
                    }
                }
            };
            let on_wire = (&mut r).take(head.on_wire_len);
            let decoded = if head.codec == CODEC_ZSTD {
                zstd_decompress_chunked(
                    on_wire,
                    head.logical_len,
                    lease.buffer(),
                    work,
                    &mut deliver,
                )
            } else {
                stored_chunked(on_wire, head.logical_len, lease.buffer(), work, &mut deliver)
            };
            if let Some(error) = failure {
                return Err(error);
            }
            let total = decoded.map_err(from_io)?;
            if total != head.logical_len || delivered != head.logical_len {
                return Err(StreamArtifactError::Decompress);
            }
            r.seek(SeekFrom::Start(start + head.on_wire_len)).map_err(|e| from_std(&e))?;
            receipt.logical_bytes = delivered;
            receipt.chunks += delivered.div_ceil(lossless_u64(work.chunk_bytes.max(1)));
        }
        let at = r.stream_position().map_err(|e| from_std(&e))?;
        let end = r.seek(SeekFrom::End(0)).map_err(|e| from_std(&e))?;
        if at != end {
            return Err(StreamArtifactError::Malformed("trailing bytes"));
        }
        Ok((metadata, receipt))
    }
}

/// Deliver a stored section in chunks, refusing more than `max_logical` bytes.
fn stored_chunked<R: Read>(
    mut on_wire: R,
    max_logical: u64,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    on_chunk: &mut impl FnMut(&[u8]) -> Result<(), IoError>,
) -> Result<u64, IoError> {
    let mut total = 0u64;
    loop {
        work.tick()?;
        let taken = on_wire.read(buffer).map_err(IoError::from)?;
        if taken == 0 {
            return Ok(total);
        }
        total += lossless_u64(taken);
        if total > max_logical {
            return Err(IoError::Decompress {
                section: PAYLOAD_SECTION.into(),
                message: "stored payload exceeds its declared size".into(),
            });
        }
        on_chunk(&buffer[..taken])?;
    }
}

fn read_fixed<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), StreamArtifactError> {
    r.read_exact(buf).map_err(|e| from_std(&e))
}

struct SectionHead {
    name: String,
    codec: u8,
    logical_len: u64,
    on_wire_len: u64,
    checksum: [u8; 32],
}

fn read_head<R: Read>(
    r: &mut R,
    bounds: &StreamBounds,
) -> Result<SectionHead, StreamArtifactError> {
    let mut word = [0u8; 2];
    read_fixed(r, &mut word)?;
    let name_len = usize::from(u16::from_le_bytes(word));
    if name_len == 0 || name_len > MAX_SECTION_NAME_LEN {
        return Err(StreamArtifactError::Malformed("section name"));
    }
    let mut name = vec![0u8; name_len];
    read_fixed(r, &mut name)?;
    let name =
        String::from_utf8(name).map_err(|_| StreamArtifactError::Malformed("section name"))?;
    let mut codec = [0u8; 1];
    read_fixed(r, &mut codec)?;
    let mut long = [0u8; 8];
    read_fixed(r, &mut long)?;
    let logical_len = u64::from_le_bytes(long);
    read_fixed(r, &mut long)?;
    let on_wire_len = u64::from_le_bytes(long);
    check_section_size(logical_len, bounds)?;
    check_section_size(on_wire_len, bounds)?;
    if codec[0] != STORED && codec[0] != CODEC_ZSTD {
        return Err(StreamArtifactError::Malformed("codec"));
    }
    let mut checksum = [0u8; 32];
    read_fixed(r, &mut checksum)?;
    Ok(SectionHead { name, codec: codec[0], logical_len, on_wire_len, checksum })
}

fn observe(cancel: &CancellationToken) -> Result<(), StreamArtifactError> {
    if cancel.is_cancelled() {
        return Err(StreamArtifactError::Cancelled);
    }
    Ok(())
}

type Reader<'a> = ByteCursor<'a, StreamArtifactError>;

fn truncated() -> StreamArtifactError {
    StreamArtifactError::Truncated
}

fn trailing_bytes() -> StreamArtifactError {
    StreamArtifactError::Malformed("trailing bytes")
}

/// A reader positioned after the magic and a supported version.
fn open_reader(bytes: &[u8]) -> Result<Reader<'_>, StreamArtifactError> {
    let mut reader = Reader::new(bytes, truncated);
    let magic = reader.take(STREAM_ARTIFACT_MAGIC.len())?;
    if magic != STREAM_ARTIFACT_MAGIC {
        return Err(StreamArtifactError::Malformed("magic"));
    }
    let version = reader.u16()?;
    if version > STREAM_ARTIFACT_VERSION {
        return Err(StreamArtifactError::NewerFormat { version });
    }
    if version != STREAM_ARTIFACT_VERSION {
        return Err(StreamArtifactError::UnsupportedFormat { version });
    }
    Ok(reader)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::CancellationToken;

    use super::{
        CpuPayload, DevicePayload, PayloadSource, Producer, STREAM_ARTIFACT_MAGIC, StreamArtifact,
        StreamArtifactError, StreamBounds, StreamMetadata,
    };

    fn bounds() -> StreamBounds {
        StreamBounds::default()
    }

    fn token() -> CancellationToken {
        CancellationToken::new()
    }

    fn metadata(producer: Producer, device: DevicePayload) -> StreamMetadata {
        StreamMetadata { artifact_id: "artifact-1".into(), producer, device }
    }

    fn host(payload: &[u8]) -> Vec<u8> {
        StreamArtifact::encode(
            &metadata(Producer::Cpu, DevicePayload::Materialized),
            PayloadSource::Stored(payload),
            &bounds(),
        )
        .expect("encode")
    }

    #[test]
    fn truncated_stream_is_refused() {
        let bytes = host(b"payload-bytes");
        let err = StreamArtifact::open(&bytes[..bytes.len() - 1], &bounds(), &token(), None)
            .expect_err("short stream");
        assert_eq!(err, StreamArtifactError::Truncated);
        let err = StreamArtifact::open(&[], &bounds(), &token(), None).expect_err("empty");
        assert_eq!(err, StreamArtifactError::Truncated);
    }

    #[test]
    fn corrupt_checksum_is_refused() {
        let mut bytes = host(b"payload-bytes");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let err = StreamArtifact::open(&bytes, &bounds(), &token(), None).expect_err("corrupt");
        assert_eq!(err, StreamArtifactError::ChecksumMismatch { section: "payload" });
    }

    #[test]
    fn newer_format_is_refused_before_the_body() {
        let mut bytes = Vec::from(STREAM_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&[0xff; 8]);
        let err = StreamArtifact::open(&bytes, &bounds(), &token(), None).expect_err("newer");
        assert_eq!(err, StreamArtifactError::NewerFormat { version: 2 });
    }

    #[test]
    fn cancellation_is_checked_between_sections() {
        let bytes = host(b"payload-bytes");
        let mut corrupt = bytes.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        let early = CancellationToken::cancel_after_checks(0);
        let err = StreamArtifact::open(&corrupt, &bounds(), &early, None).expect_err("cancelled");
        assert_eq!(err, StreamArtifactError::Cancelled);

        let absent = StreamArtifact::encode(
            &metadata(Producer::Cpu, DevicePayload::Absent),
            PayloadSource::Absent,
            &bounds(),
        )
        .expect("meta only");
        let one_check = CancellationToken::cancel_after_checks(1);
        StreamArtifact::open(&absent, &bounds(), &one_check, None).expect("section completes");
        assert!(one_check.is_cancelled());

        let mut truncated_next = bytes;
        let count_at = STREAM_ARTIFACT_MAGIC.len() + 2;
        truncated_next[count_at..count_at + 4].copy_from_slice(&2u32.to_le_bytes());
        truncated_next.pop();
        let between = CancellationToken::cancel_after_checks(1);
        let err = StreamArtifact::open(&truncated_next, &bounds(), &between, None)
            .expect_err("between sections");
        assert_eq!(err, StreamArtifactError::Cancelled);
    }

    #[test]
    fn mmap_view_lifetime_is_the_caller_buffer() {
        let payload = b"mapped-bytes";
        let bytes = host(payload);
        let opened = StreamArtifact::open(&bytes, &bounds(), &token(), None).expect("open");
        let view = opened.borrowed_payload().expect("borrowed from the caller buffer");
        assert!(bytes.as_ptr_range().contains(&view.as_ptr()));
        assert_eq!(view, payload);
        assert!(opened.shared_payload().is_none());
    }

    #[test]
    fn accelerator_produced_artifact_is_consumable_on_cpu_without_a_live_pointer() {
        let absent = StreamArtifact::encode(
            &metadata(Producer::Accelerator, DevicePayload::Absent),
            PayloadSource::Absent,
            &bounds(),
        )
        .expect("absent");
        let absent = StreamArtifact::open(&absent, &bounds(), &token(), None).expect("open");
        assert_eq!(absent.consume_on_cpu().expect("cpu"), CpuPayload::Absent);

        let payload = b"device-bytes-on-host";
        let materialized = StreamArtifact::encode(
            &metadata(Producer::Accelerator, DevicePayload::Materialized),
            PayloadSource::Stored(payload),
            &bounds(),
        )
        .expect("materialized");
        let materialized =
            StreamArtifact::open(&materialized, &bounds(), &token(), None).expect("open");
        assert_eq!(materialized.consume_on_cpu().expect("cpu"), CpuPayload::Bytes(payload));

        let live = DevicePayload::LivePointer { device_id: 3, address: 0x1000, bytes: 4096 };
        let bytes = StreamArtifact::encode(
            &metadata(Producer::Accelerator, live),
            PayloadSource::Absent,
            &bounds(),
        )
        .expect("live pointer");
        let opened = StreamArtifact::open(&bytes, &bounds(), &token(), None).expect("open");
        assert_eq!(opened.metadata().device, live);
        assert_eq!(opened.consume_on_cpu(), Err(StreamArtifactError::LiveDevicePointer));
    }

    #[test]
    fn metadata_read_keeps_no_payload() {
        let payload = vec![7u8; 64];
        let bytes = host(&payload);
        let tight = StreamBounds { max_retained_bytes: 0, ..bounds() };
        let meta = StreamArtifact::read_metadata(&bytes, &tight, &token()).expect("metadata");
        assert_eq!(meta.artifact_id, "artifact-1");
        assert_eq!(meta.device, DevicePayload::Materialized);
        let err = StreamArtifact::open(&bytes, &tight, &token(), None).expect_err("retained");
        assert_eq!(err, StreamArtifactError::RetainedOutput { limit: 0 });
    }

    #[test]
    fn shared_payload_is_borrowed_while_the_caller_holds_it() {
        let payload: Arc<[u8]> = Arc::from(&b"shared-payload"[..]);
        let before = Arc::strong_count(&payload);
        let bytes = StreamArtifact::encode(
            &metadata(Producer::Cpu, DevicePayload::Materialized),
            PayloadSource::Shared(&payload),
            &bounds(),
        )
        .expect("encode");
        let opened =
            StreamArtifact::open(&bytes, &bounds(), &token(), Some(&payload)).expect("open");
        let shared = opened.shared_payload().expect("shared");
        assert!(Arc::ptr_eq(shared, &payload));
        assert_eq!(Arc::strong_count(&payload), before + 1);
        assert!(opened.borrowed_payload().is_none());
        assert_eq!(opened.payload(), Some(payload.as_ref()));
    }

    #[test]
    fn unknown_codec_is_refused() {
        let mut bytes = host(b"payload-bytes");
        let codec_at = payload_logical_offset(&bytes) - 1;
        bytes[codec_at] = 1;
        let err = StreamArtifact::open(&bytes, &bounds(), &token(), None).expect_err("codec");
        assert_eq!(err, StreamArtifactError::Malformed("codec"));
    }

    #[test]
    fn section_bound_refuses_a_declared_size_before_the_body() {
        let bytes = host(b"abcd");
        let mut declared = bytes;
        let on_wire_at = payload_logical_offset(&declared) + 8;
        declared[on_wire_at..on_wire_at + 8].copy_from_slice(&17u64.to_le_bytes());
        let tight = StreamBounds { max_section_bytes: 16, ..bounds() };
        let err = StreamArtifact::open(&declared, &tight, &token(), None).expect_err("section");
        assert_eq!(err, StreamArtifactError::SectionBound { limit: 16 });
    }

    #[test]
    fn stored_payload_round_trips() {
        let stored = host(b"stored");
        let opened = StreamArtifact::open(&stored, &bounds(), &token(), None).expect("stored");
        assert_eq!(opened.payload(), Some(&b"stored"[..]));
        assert_eq!(opened.borrowed_payload(), Some(&b"stored"[..]));
    }

    fn payload_logical_offset(bytes: &[u8]) -> usize {
        let mut at = STREAM_ARTIFACT_MAGIC.len() + 2 + 4;
        let name_len = u16::from_le_bytes(bytes[at..at + 2].try_into().expect("meta name length"));
        at += 2 + usize::from(name_len) + 1 + 8 + 8 + 32;
        let logical = at - 32 - 8 - 8;
        let meta_len =
            u64::from_le_bytes(bytes[logical..logical + 8].try_into().expect("meta length"));
        at += usize::try_from(meta_len).expect("meta length");
        let payload_name =
            u16::from_le_bytes(bytes[at..at + 2].try_into().expect("payload name length"));
        at + 2 + usize::from(payload_name) + 1
    }

    mod incremental {
        use std::io::{Cursor, Read, Write};

        use antecedent_core::ExecutionContext;
        use antecedent_core::execution::AdmissionPool;

        use super::{
            CpuPayload, DevicePayload, PayloadSource, Producer, STREAM_ARTIFACT_MAGIC,
            StreamArtifact, StreamArtifactError, StreamBounds, StreamMetadata, metadata,
            payload_logical_offset,
        };
        use crate::stream_artifact::{LogicalSource, PayloadCodec, StreamWork};

        /// A compressible payload produced on demand.
        struct Pattern {
            len: u64,
        }

        struct PatternReader {
            left: u64,
            at: u64,
        }

        impl Read for PatternReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = usize::try_from(self.left).unwrap_or(usize::MAX).min(buf.len());
                for byte in &mut buf[..n] {
                    self.at += 1;
                    *byte = b'a' + ((self.at / 100) % 11) as u8;
                }
                self.left -= n as u64;
                Ok(n)
            }
        }

        impl LogicalSource for Pattern {
            fn logical_len(&self) -> u64 {
                self.len
            }

            fn open(&self) -> std::io::Result<Box<dyn Read + '_>> {
                Ok(Box::new(PatternReader { left: self.len, at: 0 }))
            }
        }

        fn materialized() -> StreamMetadata {
            metadata(Producer::Accelerator, DevicePayload::Materialized)
        }

        fn wide_bounds() -> StreamBounds {
            StreamBounds {
                max_section_bytes: 128 << 20,
                max_retained_bytes: 128 << 20,
                ..StreamBounds::default()
            }
        }

        fn study(pool: &AdmissionPool) -> ExecutionContext {
            ExecutionContext::for_tests(1).with_admission(pool.clone())
        }

        #[test]
        fn a_stored_incremental_encode_is_byte_identical_and_reads_back_in_chunks() {
            let payload: Vec<u8> = (0..50_000u32).map(|i| (i % 253) as u8).collect();
            let buffered = StreamArtifact::encode(
                &materialized(),
                PayloadSource::Stored(&payload),
                &StreamBounds::default(),
            )
            .unwrap();
            let pool = AdmissionPool::fixed(1, 1 << 20);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(1000);
            let mut streamed = Vec::new();
            let receipt = StreamArtifact::encode_to(
                &materialized(),
                Some((&payload, PayloadCodec::Stored)),
                &StreamBounds::default(),
                &work,
                &mut streamed,
            )
            .unwrap();
            assert_eq!(streamed, buffered);
            assert_eq!(receipt.logical_bytes, payload.len() as u64);
            assert_eq!(pool.available().bytes(), 1 << 20);

            let mut got = Vec::new();
            let mut largest = 0;
            let (meta, read) = StreamArtifact::read_to(
                Cursor::new(&streamed),
                &StreamBounds::default(),
                &work,
                |chunk| {
                    largest = largest.max(chunk.len());
                    got.extend_from_slice(chunk);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(meta, materialized());
            assert_eq!(got, payload);
            assert!(largest <= 1000, "{largest}");
            assert_eq!(read.logical_bytes, payload.len() as u64);

            let absent = StreamArtifact::encode_to(
                &metadata(Producer::Cpu, DevicePayload::Absent),
                None,
                &StreamBounds::default(),
                &work,
                Vec::new(),
            );
            assert!(absent.is_ok());
        }

        #[test]
        fn a_compressed_payload_larger_than_the_budget_streams_both_ways() {
            const BUDGET: u64 = 24 << 20;
            let pool = AdmissionPool::fixed(1, BUDGET);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(32 * 1024);
            let source = Pattern { len: 64 << 20 };
            let path = std::env::temp_dir()
                .join(format!("antecedent-stream-artifact-{}", std::process::id()));
            let file = std::fs::File::create(&path).unwrap();
            let written = StreamArtifact::encode_to(
                &materialized(),
                Some((&source, PayloadCodec::Zstd)),
                &wide_bounds(),
                &work,
                file,
            )
            .unwrap();
            assert!(written.leased_bytes <= BUDGET);
            assert!(
                written.leased_bytes < source.len,
                "the working set is smaller than the payload"
            );
            assert!(written.on_wire_bytes < source.len / 10);
            assert!(written.largest_write <= 1 << 20, "{}", written.largest_write);
            assert_eq!(pool.available().bytes(), BUDGET);

            let mut hasher = blake3::Hasher::new();
            let mut total = 0u64;
            let mut largest = 0usize;
            let (meta, read) = StreamArtifact::read_to(
                std::fs::File::open(&path).unwrap(),
                &wide_bounds(),
                &work,
                |chunk| {
                    largest = largest.max(chunk.len());
                    total += chunk.len() as u64;
                    hasher.update(chunk);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(meta, materialized());
            assert_eq!(total, source.len);
            assert!(largest <= 32 * 1024, "{largest}");
            assert!(read.leased_bytes <= BUDGET);
            let mut expected = blake3::Hasher::new();
            let mut again = source.open().unwrap();
            let mut buf = vec![0u8; 8192];
            loop {
                let n = again.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                expected.update(&buf[..n]);
            }
            assert_eq!(hasher.finalize(), expected.finalize());
            assert_eq!(pool.available().bytes(), BUDGET);

            // The slice reader keeps its cheap metadata read and refuses to borrow a frame.
            let bytes = std::fs::read(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            let cancel = antecedent_core::CancellationToken::new();
            let meta_only = StreamArtifact::read_metadata(&bytes, &wide_bounds(), &cancel).unwrap();
            assert_eq!(meta_only, materialized());
            let refused = StreamArtifact::open(&bytes, &wide_bounds(), &cancel, None).unwrap_err();
            assert_eq!(refused, StreamArtifactError::Malformed("codec"));
        }

        #[test]
        fn the_budget_is_leased_before_anything_is_written_and_cancellation_returns_it() {
            let source = Pattern { len: 1 << 20 };
            let chunk = 4096usize;
            let need =
                2 * chunk as u64 + antecedent_io_core::container::stream_encoder_workspace_bytes();
            let pool = AdmissionPool::fixed(1, need - 1);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(chunk);
            let mut out = Vec::new();
            let refused = StreamArtifact::encode_to(
                &materialized(),
                Some((&source, PayloadCodec::Zstd)),
                &wide_bounds(),
                &work,
                &mut out,
            )
            .unwrap_err();
            assert_eq!(refused, StreamArtifactError::BudgetRefused);
            assert!(out.is_empty());
            assert_eq!(pool.available().bytes(), need - 1);

            struct CountSeen {
                seen: usize,
            }
            impl Write for CountSeen {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    self.seen += buf.len();
                    Ok(buf.len())
                }

                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }
            let big = AdmissionPool::fixed(1, 64 << 20);
            let ctx = study(&big);
            let work = StreamWork::new(&ctx).with_chunk_bytes(chunk);
            let noise = Pattern { len: 8 << 20 };
            let token = ctx.cancellation.clone();
            let mut sink = CountSeen { seen: 0 };
            // Cancel from a reader that has delivered a megabyte.
            struct Cancelling<'a> {
                inner: Pattern,
                token: &'a antecedent_core::CancellationToken,
                after: u64,
            }
            impl LogicalSource for Cancelling<'_> {
                fn logical_len(&self) -> u64 {
                    self.inner.len
                }

                fn open(&self) -> std::io::Result<Box<dyn Read + '_>> {
                    struct R<'b> {
                        inner: PatternReader,
                        token: &'b antecedent_core::CancellationToken,
                        after: u64,
                        total: u64,
                    }
                    impl Read for R<'_> {
                        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                            let n = self.inner.read(buf)?;
                            self.total += n as u64;
                            if self.total >= self.after {
                                self.token.cancel();
                            }
                            Ok(n)
                        }
                    }
                    Ok(Box::new(R {
                        inner: PatternReader { left: self.inner.len, at: 0 },
                        token: self.token,
                        after: self.after,
                        total: 0,
                    }))
                }
            }
            let cancelling = Cancelling { inner: noise, token: &token, after: 1 << 20 };
            let refused = StreamArtifact::encode_to(
                &materialized(),
                Some((&cancelling, PayloadCodec::Stored)),
                &wide_bounds(),
                &work,
                &mut sink,
            )
            .unwrap_err();
            assert_eq!(refused, StreamArtifactError::Cancelled);
            assert!(sink.seen < 4 << 20, "{}", sink.seen);
            assert_eq!(big.available().bytes(), 64 << 20);
        }

        #[test]
        fn corrupt_truncated_newer_and_expanding_streams_never_reach_the_sink_wrongly() {
            let pool = AdmissionPool::fixed(1, 64 << 20);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(2048);
            let source = Pattern { len: 1 << 20 };
            let mut bytes = Vec::new();
            StreamArtifact::encode_to(
                &materialized(),
                Some((&source, PayloadCodec::Zstd)),
                &wide_bounds(),
                &work,
                &mut bytes,
            )
            .unwrap();

            let mut calls = 0;
            let mut corrupt = bytes.clone();
            let last = corrupt.len() / 2 + 40;
            corrupt[last] ^= 0x55;
            let refused =
                StreamArtifact::read_to(Cursor::new(&corrupt), &wide_bounds(), &work, |_| {
                    calls += 1;
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(refused, StreamArtifactError::ChecksumMismatch { section: "payload" });
            assert_eq!(calls, 0, "the checksum is verified before the first chunk");

            let refused = StreamArtifact::read_to(
                Cursor::new(&bytes[..bytes.len() - 3]),
                &wide_bounds(),
                &work,
                |_| Ok(()),
            )
            .unwrap_err();
            assert_eq!(refused, StreamArtifactError::Truncated);

            let mut newer = Vec::from(STREAM_ARTIFACT_MAGIC);
            newer.extend_from_slice(&2u16.to_le_bytes());
            newer.extend_from_slice(&u32::MAX.to_le_bytes());
            let refused =
                StreamArtifact::read_to(Cursor::new(&newer), &wide_bounds(), &work, |_| Ok(()))
                    .unwrap_err();
            assert_eq!(refused, StreamArtifactError::NewerFormat { version: 2 });

            // The checksum covers the on-wire frame, so declaring a smaller logical size
            // leaves it valid; the frame then expands past what the header allows.
            let mut declared = bytes.clone();
            let at = payload_logical_offset(&declared);
            declared[at..at + 8].copy_from_slice(&1000u64.to_le_bytes());
            let mut delivered = 0usize;
            let refused =
                StreamArtifact::read_to(Cursor::new(&declared), &wide_bounds(), &work, |chunk| {
                    delivered += chunk.len();
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(refused, StreamArtifactError::Decompress);
            assert!(delivered <= 1000, "{delivered}");

            let tight = StreamBounds { max_retained_bytes: 10, ..wide_bounds() };
            let refused = StreamArtifact::read_to(Cursor::new(&bytes), &tight, &work, |_| Ok(()))
                .unwrap_err();
            assert_eq!(refused, StreamArtifactError::RetainedOutput { limit: 10 });
            let _ = CpuPayload::Absent;
        }
    }
}
