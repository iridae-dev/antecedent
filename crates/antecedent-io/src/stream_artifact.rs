//! Checksummed stream artifact: bounds, cheap metadata, and shared payloads.
//!
//! The reader validates a byte stream the caller already holds. It does not open files, map
//! them, sleep, or spawn threads. [`StreamBounds`] records the decompression, section, and
//! retained-output limits and both the writer and the reader enforce them. A truncated stream,
//! a checksum mismatch, and a format newer than [`STREAM_ARTIFACT_VERSION`] are refusals.
//!
//! Cancellation is a caller [`CancellationToken`] checked between sections, not inside a
//! section. An accelerator-produced artifact is consumable on CPU when its device payload is
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
//! (`antecedent.stream_artifact.section.v1`). Codec `0` stores the logical bytes. Codec `1` is
//! a run-length frame whose expansion is capped at the declared logical length. The first
//! section is `meta`; an optional second section is `payload`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CancellationToken, CanonicalWriter};

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
const RLE: u8 = 1;
const PRODUCER_CPU: u8 = 1;
const PRODUCER_ACCELERATOR: u8 = 2;
const DEVICE_ABSENT: u8 = 0;
const DEVICE_MATERIALIZED: u8 = 1;
const DEVICE_LIVE: u8 = 2;

/// Limits recorded for a stream and enforced before a section is expanded or retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamBounds {
    /// Largest on-wire or declared logical size of one section.
    pub max_section_bytes: u64,
    /// Largest logical size a compressed section may expand to.
    pub max_decompressed_bytes: u64,
    /// Largest payload a reader may keep. Metadata does not count.
    pub max_retained_bytes: u64,
    /// Most sections, including metadata.
    pub max_sections: u32,
}

impl StreamBounds {
    /// Default section cap: 16 MiB.
    pub const DEFAULT_MAX_SECTION_BYTES: u64 = 16 * 1024 * 1024;
    /// Default decompression cap: 16 MiB.
    pub const DEFAULT_MAX_DECOMPRESSED_BYTES: u64 = 16 * 1024 * 1024;
    /// Default retained-payload cap: 16 MiB.
    pub const DEFAULT_MAX_RETAINED_BYTES: u64 = 16 * 1024 * 1024;
    /// Default section count cap.
    pub const DEFAULT_MAX_SECTIONS: u32 = 8;
}

impl Default for StreamBounds {
    fn default() -> Self {
        Self {
            max_section_bytes: Self::DEFAULT_MAX_SECTION_BYTES,
            max_decompressed_bytes: Self::DEFAULT_MAX_DECOMPRESSED_BYTES,
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
    /// A declared logical size is over [`StreamBounds::max_decompressed_bytes`].
    #[error("stream artifact decompression exceeds {limit} bytes")]
    DecompressionBound {
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
    /// A compressed frame does not expand to its declared logical size.
    #[error("stream artifact decompression failed")]
    Decompress,
    /// [`DevicePayload::LivePointer`] cannot be read on CPU.
    #[error("accelerator artifact holds a live device pointer")]
    LiveDevicePointer,
    /// The bytes are not this format.
    #[error("malformed stream artifact: {0}")]
    Malformed(&'static str),
}

#[derive(Clone, Debug)]
enum PayloadBytes<'a> {
    Borrowed(&'a [u8]),
    Shared(Arc<[u8]>),
    Owned(Arc<[u8]>),
}

impl PayloadBytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Shared(bytes) | Self::Owned(bytes) => bytes.as_ref(),
        }
    }
}

/// A validated stream artifact.
///
/// Uncompressed payload bytes are borrowed from the caller buffer (the lifetime a mapped
/// file would have). Compressed bytes are owned. A shared payload aliases the caller's `Arc`.
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
    /// Declared section and decompression sizes are still refused. Retained-output limits are
    /// not charged, because the payload is not kept.
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
    let mut reader = Reader::new(stream)?;
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
    reader.finish()?;
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
    let logical_len = u64_len(section.logical_len)?;
    charge(0, logical_len, bounds.max_retained_bytes)?;
    let body = match section.codec {
        STORED => {
            if section.on_wire.len() != section.logical_len {
                return Err(StreamArtifactError::Malformed("stored length"));
            }
            adopt_stored(section.on_wire, shared)
        }
        RLE => {
            let owned = decompress(section.on_wire, section.logical_len)?;
            adopt_owned(owned, shared)
        }
        _ => return Err(StreamArtifactError::Malformed("codec")),
    };
    Ok(Some(body))
}

fn adopt_stored<'a>(on_wire: &'a [u8], shared: Option<&Arc<[u8]>>) -> PayloadBytes<'a> {
    if let Some(shared) = shared.filter(|held| held.as_ref() == on_wire) {
        return PayloadBytes::Shared(Arc::clone(shared));
    }
    PayloadBytes::Borrowed(on_wire)
}

fn adopt_owned<'a>(owned: Vec<u8>, shared: Option<&Arc<[u8]>>) -> PayloadBytes<'a> {
    if let Some(shared) = shared.filter(|held| held.as_ref() == owned.as_slice()) {
        return PayloadBytes::Shared(Arc::clone(shared));
    }
    PayloadBytes::Owned(Arc::from(owned))
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
    if codec == RLE && logical_len > bounds.max_decompressed_bytes {
        return Err(StreamArtifactError::DecompressionBound {
            limit: bounds.max_decompressed_bytes,
        });
    }
    if codec != STORED && codec != RLE {
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
    let mut reader = Reader { bytes, at: 0 };
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
    reader.finish()?;
    let metadata = StreamMetadata { artifact_id, producer, device };
    validate_identity(&metadata)?;
    Ok(metadata)
}

fn decompress(on_wire: &[u8], logical_len: usize) -> Result<Vec<u8>, StreamArtifactError> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while out.len() < logical_len {
        let op = *on_wire.get(at).ok_or(StreamArtifactError::Decompress)?;
        at = at.checked_add(1).ok_or(StreamArtifactError::Decompress)?;
        match op {
            0 => copy_literal(on_wire, &mut at, &mut out, logical_len)?,
            1 => copy_repeat(on_wire, &mut at, &mut out, logical_len)?,
            _ => return Err(StreamArtifactError::Decompress),
        }
    }
    if at != on_wire.len() || out.len() != logical_len {
        return Err(StreamArtifactError::Decompress);
    }
    Ok(out)
}

fn copy_literal(
    on_wire: &[u8],
    at: &mut usize,
    out: &mut Vec<u8>,
    logical_len: usize,
) -> Result<(), StreamArtifactError> {
    let len =
        usize::try_from(read_u32(on_wire, at)?).map_err(|_| StreamArtifactError::Decompress)?;
    let end = at.checked_add(len).ok_or(StreamArtifactError::Decompress)?;
    let literal = on_wire.get(*at..end).ok_or(StreamArtifactError::Decompress)?;
    let new_len = out.len().checked_add(literal.len()).ok_or(StreamArtifactError::Decompress)?;
    if new_len > logical_len {
        return Err(StreamArtifactError::Decompress);
    }
    out.extend_from_slice(literal);
    *at = end;
    Ok(())
}

fn copy_repeat(
    on_wire: &[u8],
    at: &mut usize,
    out: &mut Vec<u8>,
    logical_len: usize,
) -> Result<(), StreamArtifactError> {
    let byte = *on_wire.get(*at).ok_or(StreamArtifactError::Decompress)?;
    *at = at.checked_add(1).ok_or(StreamArtifactError::Decompress)?;
    let count =
        usize::try_from(read_u32(on_wire, at)?).map_err(|_| StreamArtifactError::Decompress)?;
    let new_len = out.len().checked_add(count).ok_or(StreamArtifactError::Decompress)?;
    if new_len > logical_len {
        return Err(StreamArtifactError::Decompress);
    }
    out.resize(new_len, byte);
    Ok(())
}

fn read_u32(bytes: &[u8], at: &mut usize) -> Result<u32, StreamArtifactError> {
    let end = at.checked_add(4).ok_or(StreamArtifactError::Decompress)?;
    let word = bytes.get(*at..end).ok_or(StreamArtifactError::Decompress)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(word);
    *at = end;
    Ok(u32::from_le_bytes(buf))
}

fn observe(cancel: &CancellationToken) -> Result<(), StreamArtifactError> {
    if cancel.is_cancelled() {
        return Err(StreamArtifactError::Cancelled);
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, StreamArtifactError> {
        let mut reader = Self { bytes, at: 0 };
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], StreamArtifactError> {
        let end = self.at.checked_add(len).ok_or(StreamArtifactError::Truncated)?;
        let slice = self.bytes.get(self.at..end).ok_or(StreamArtifactError::Truncated)?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, StreamArtifactError> {
        let mut buf = [0u8; 1];
        buf.copy_from_slice(self.take(1)?);
        Ok(buf[0])
    }

    fn u16(&mut self) -> Result<u16, StreamArtifactError> {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(buf))
    }

    fn u32(&mut self) -> Result<u32, StreamArtifactError> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn u64(&mut self) -> Result<u64, StreamArtifactError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn finish(self) -> Result<(), StreamArtifactError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(StreamArtifactError::Malformed("trailing bytes"))
        }
    }
}

/// Encode a compressed payload.
///
/// The run-length frame is owned, so it cannot borrow `logical` for the life of the returned
/// stream. [`StreamArtifact::encode`] keeps stored sections borrowed; this path is the one
/// that records a decompression.
pub fn encode_compressed(
    metadata: &StreamMetadata,
    logical: &[u8],
    bounds: &StreamBounds,
) -> Result<Vec<u8>, StreamArtifactError> {
    validate_declaration(metadata, true)?;
    if bounds.max_sections < 2 {
        return Err(StreamArtifactError::TooManySections { limit: bounds.max_sections });
    }
    let logical_len = u64_len(logical.len())?;
    check_section_size(logical_len, bounds)?;
    if logical_len > bounds.max_decompressed_bytes {
        return Err(StreamArtifactError::DecompressionBound {
            limit: bounds.max_decompressed_bytes,
        });
    }
    charge(0, logical_len, bounds.max_retained_bytes)?;
    let frame = encode_rle(logical)?;
    check_section_size(u64_len(frame.len())?, bounds)?;
    let meta = encode_metadata(metadata)?;
    check_section_size(u64_len(meta.len())?, bounds)?;
    let mut out = Vec::new();
    out.extend_from_slice(&STREAM_ARTIFACT_MAGIC);
    out.extend_from_slice(&STREAM_ARTIFACT_VERSION.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    write_section(&mut out, META_SECTION, STORED, &meta, &meta)?;
    write_section(&mut out, PAYLOAD_SECTION, RLE, logical, &frame)?;
    Ok(out)
}

fn encode_rle(logical: &[u8]) -> Result<Vec<u8>, StreamArtifactError> {
    let len = u32::try_from(logical.len())
        .map_err(|_| StreamArtifactError::SectionBound { limit: u64::from(u32::MAX) })?;
    let mut frame = Vec::with_capacity(logical.len().saturating_add(5));
    frame.push(0);
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(logical);
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::CancellationToken;

    use super::{
        CpuPayload, DevicePayload, PayloadSource, Producer, STREAM_ARTIFACT_MAGIC,
        StreamArtifact, StreamArtifactError, StreamBounds, StreamMetadata,
        encode_compressed,
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
    fn decompression_bound_is_enforced_before_expansion() {
        let wide = StreamBounds {
            max_section_bytes: u64::MAX,
            max_decompressed_bytes: 32,
            max_retained_bytes: u64::MAX,
            max_sections: 8,
        };
        let err = encode_compressed(
            &metadata(Producer::Cpu, DevicePayload::Materialized),
            &[1u8; 33],
            &wide,
        )
        .expect_err("declared expansion");
        assert_eq!(err, StreamArtifactError::DecompressionBound { limit: 32 });

        let bytes = encode_compressed(
            &metadata(Producer::Cpu, DevicePayload::Materialized),
            b"abc",
            &bounds(),
        )
        .expect("compressed");
        let mut hostile = bytes;
        // Rewrite the payload logical length to 33 while leaving a short frame, after raising
        // the section cap so the refusal is the decompression bound.
        let logical_at = payload_logical_offset(&hostile);
        hostile[logical_at..logical_at + 8].copy_from_slice(&33u64.to_le_bytes());
        let err = StreamArtifact::open(&hostile, &wide, &token(), None).expect_err("bound");
        assert_eq!(err, StreamArtifactError::DecompressionBound { limit: 32 });
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
    fn stored_and_compressed_payloads_round_trip() {
        let stored = host(b"stored");
        let opened = StreamArtifact::open(&stored, &bounds(), &token(), None).expect("stored");
        assert_eq!(opened.payload(), Some(&b"stored"[..]));

        let compressed = encode_compressed(
            &metadata(Producer::Accelerator, DevicePayload::Materialized),
            b"compressed",
            &bounds(),
        )
        .expect("rle");
        let opened = StreamArtifact::open(&compressed, &bounds(), &token(), None).expect("rle");
        assert_eq!(opened.payload(), Some(&b"compressed"[..]));
        assert!(opened.borrowed_payload().is_none());
        assert_eq!(opened.consume_on_cpu().expect("cpu"), CpuPayload::Bytes(b"compressed"));
    }

    #[test]
    fn expanding_frame_stops_at_the_declared_logical_length() {
        let logical = [0xab, 0xab, 0xab, 0xab];
        let bytes = encode_compressed(
            &metadata(Producer::Cpu, DevicePayload::Materialized),
            &logical,
            &bounds(),
        )
        .expect("frame");
        let mut hostile = bytes;
        let on_wire_at = payload_logical_offset(&hostile) + 8;
        let on_wire_len = u64::from_le_bytes(
            hostile[on_wire_at..on_wire_at + 8].try_into().expect("on-wire length"),
        );
        let checksum_at = on_wire_at + 8;
        let body_at = checksum_at + 32;
        assert_eq!(on_wire_len, 1 + 4 + 4);
        hostile.truncate(body_at);
        hostile.extend_from_slice(&[1, 0xab, 100, 0, 0, 0]);
        let digest = super::checksum(&hostile[body_at..]);
        hostile[checksum_at..body_at].copy_from_slice(&digest);
        let on_wire_len_bytes =
            u64::try_from(hostile.len() - body_at).expect("frame length").to_le_bytes();
        hostile[on_wire_at..on_wire_at + 8].copy_from_slice(&on_wire_len_bytes);
        let err = StreamArtifact::open(&hostile, &bounds(), &token(), None).expect_err("expand");
        assert_eq!(err, StreamArtifactError::Decompress);
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
}
