//! Bounded streaming container and compression operations.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AUTO_COMPRESS_MIN_BYTES, AdmissionLease, Arc, ArtifactManifest, COMPRESSION_ZSTD,
    CONTAINER_VERSION, CompressPolicy, EncodedArtifact, ExecutionContext, IoError, MAGIC,
    MAX_SECTION_BYTES, Read, SectionDescriptor, Seek, SeekFrom, Write, ZSTD_LEVEL,
    read_header_and_manifest,
};

/// Chunk size of streamed artifact operations unless the caller chooses.
pub const DEFAULT_STREAM_CHUNK_BYTES: usize = 64 * 1024;

/// Largest Zstandard window (log2 bytes) a streamed decode accepts: 8 MiB.
///
/// A frame that declares a larger window is refused instead of sizing the decoder
/// from untrusted input. Frames written at the default level use a 2 MiB window.
pub const STREAM_DECODER_WINDOW_LOG_MAX: u32 = 23;

/// Largest window log the Zstandard decoder accepts at all.
const ZSTD_WINDOW_LOG_CEILING: u32 = 31;

/// Bytes leased for the Zstandard compression context and its output buffer.
const STREAM_ENCODER_WORKSPACE_BYTES: u64 = 8 * 1024 * 1024;
/// Bytes leased for the Zstandard decoding window and its input buffers.
const STREAM_DECODER_WORKSPACE_BYTES: u64 = (1 << STREAM_DECODER_WINDOW_LOG_MAX) + (1 << 20);

/// The operation a streamed artifact call runs under: its admission pool and
/// cancellation token, and the chunk size every buffer is sized from.
#[derive(Clone, Copy, Debug)]
pub struct StreamWork<'a> {
    /// Context whose admission pool funds the working set and whose token stops the call.
    pub ctx: &'a ExecutionContext,
    /// Bytes moved per read. At least one.
    pub chunk_bytes: usize,
    /// Largest single on-wire section a streamed read accepts.
    pub max_section_bytes: usize,
    /// Largest Zstandard window (log2 bytes) a streamed decode accepts. A frame that
    /// declares a larger one is refused, and the decode lease is sized from this value.
    pub decoder_window_log_max: u32,
}

impl<'a> StreamWork<'a> {
    /// Work under `ctx` with [`DEFAULT_STREAM_CHUNK_BYTES`].
    #[must_use]
    pub const fn new(ctx: &'a ExecutionContext) -> Self {
        Self {
            ctx,
            chunk_bytes: DEFAULT_STREAM_CHUNK_BYTES,
            max_section_bytes: MAX_SECTION_BYTES,
            decoder_window_log_max: STREAM_DECODER_WINDOW_LOG_MAX,
        }
    }

    /// The same work with `chunk_bytes` per read.
    #[must_use]
    pub const fn with_chunk_bytes(self, chunk_bytes: usize) -> Self {
        Self { chunk_bytes, ..self }
    }

    /// The same work accepting on-wire sections of at most `max_section_bytes`.
    #[must_use]
    pub const fn with_max_section_bytes(self, max_section_bytes: usize) -> Self {
        Self { max_section_bytes, ..self }
    }

    /// The same work accepting Zstandard windows up to `2^decoder_window_log_max` bytes.
    #[must_use]
    pub const fn with_decoder_window_log_max(self, decoder_window_log_max: u32) -> Self {
        Self { decoder_window_log_max, ..self }
    }

    /// Bytes a streamed decode under this work leases for its codec on top of two chunk
    /// buffers: the decoding window and its input buffers.
    #[must_use]
    pub fn decoder_workspace_bytes(&self) -> u64 {
        (1_u64 << self.decoder_window_log_max.min(ZSTD_WINDOW_LOG_CEILING)) + (1 << 20)
    }

    /// Check cancellation. Called between chunks.
    ///
    /// # Errors
    ///
    /// [`IoError::StreamCancelled`].
    pub fn tick(&self) -> Result<(), IoError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(IoError::StreamCancelled);
        }
        Ok(())
    }

    /// Lease the working set for a call that needs `codec_bytes` for its codec, and
    /// allocate the chunk buffer only after the lease is held.
    ///
    /// # Errors
    ///
    /// Cancellation, a zero chunk, or [`IoError::StreamBudgetRefused`].
    pub fn admit(&self, codec_bytes: u64) -> Result<StreamLease, IoError> {
        self.tick()?;
        if self.chunk_bytes == 0 {
            return Err(IoError::io("artifact stream chunk must be at least one byte"));
        }
        let chunk = u64::try_from(self.chunk_bytes).map_err(|_| IoError::TooLarge)?;
        let bytes = chunk
            .checked_mul(2)
            .and_then(|buffers| buffers.checked_add(codec_bytes))
            .ok_or(IoError::TooLarge)?;
        let lease = self.ctx.acquire_admission(0, bytes).map_err(|error| {
            IoError::StreamBudgetRefused { message: format!("{bytes} bytes were needed: {error}") }
        })?;
        Ok(StreamLease { _lease: lease, buffer: vec![0; self.chunk_bytes], bytes })
    }
}

/// The leased working set of one streamed call.
#[derive(Debug)]
pub struct StreamLease {
    _lease: AdmissionLease,
    buffer: Vec<u8>,
    bytes: u64,
}

impl StreamLease {
    /// Bytes the pool granted.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The reusable chunk buffer.
    pub fn buffer(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
}

/// What a streamed call moved, and the largest single transfer it made.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamReceipt {
    /// Bytes leased from the admission pool for the call.
    pub leased_bytes: u64,
    /// Chunk size the call was sized from.
    pub chunk_bytes: usize,
    /// Logical bytes read from sources or delivered to sinks.
    pub logical_bytes: u64,
    /// On-wire bytes written or read.
    pub on_wire_bytes: u64,
    /// Chunks moved across all passes.
    pub chunks: u64,
    /// Largest single read buffer used.
    pub largest_read: usize,
    /// Largest single write handed to the output.
    pub largest_write: usize,
}

/// A section's logical bytes that can be read more than once.
///
/// Streaming a section is two passes (size and checksum, then write), so the source
/// must reopen. A byte slice reopens for free; a file reopens by path.
pub trait LogicalSource {
    /// Logical length in bytes.
    fn logical_len(&self) -> u64;

    /// A reader positioned at the first byte.
    ///
    /// # Errors
    ///
    /// The source cannot be opened.
    fn open(&self) -> std::io::Result<Box<dyn Read + '_>>;
}

impl LogicalSource for [u8] {
    fn logical_len(&self) -> u64 {
        crate::error::lossless_u64(self.len())
    }

    fn open(&self) -> std::io::Result<Box<dyn Read + '_>> {
        Ok(Box::new(self))
    }
}

impl LogicalSource for Vec<u8> {
    fn logical_len(&self) -> u64 {
        crate::error::lossless_u64(self.len())
    }

    fn open(&self) -> std::io::Result<Box<dyn Read + '_>> {
        Ok(Box::new(self.as_slice()))
    }
}

impl LogicalSource for Arc<[u8]> {
    fn logical_len(&self) -> u64 {
        crate::error::lossless_u64(self.len())
    }

    fn open(&self) -> std::io::Result<Box<dyn Read + '_>> {
        Ok(Box::new(self.as_ref()))
    }
}

/// Counts and hashes everything written through it, and forwards it.
struct Tally<W: Write> {
    inner: W,
    hasher: blake3::Hasher,
    written: u64,
    largest_write: usize,
}

impl<W: Write> Tally<W> {
    fn new(inner: W) -> Self {
        Self { inner, hasher: blake3::Hasher::new(), written: 0, largest_write: 0 }
    }
}

impl<W: Write> Write for Tally<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let taken = self.inner.write(buf)?;
        self.hasher.update(&buf[..taken]);
        self.written += crate::error::lossless_u64(taken);
        self.largest_write = self.largest_write.max(taken);
        Ok(taken)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Read `source` to its end in chunks into `sink`, ticking between chunks.
///
/// Returns the bytes moved; more than `cap` is refused as it arrives.
fn pump<R: Read + ?Sized, W: Write>(
    source: &mut R,
    sink: &mut W,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    cap: u64,
    receipt: &mut StreamReceipt,
) -> Result<u64, IoError> {
    let mut total = 0u64;
    loop {
        work.tick()?;
        let taken = source.read(buffer)?;
        if taken == 0 {
            return Ok(total);
        }
        total += crate::error::lossless_u64(taken);
        if total > cap {
            return Err(IoError::ManifestMismatch { message: "section logical size mismatch" });
        }
        sink.write_all(&buffer[..taken])?;
        receipt.chunks += 1;
        receipt.largest_read = receipt.largest_read.max(taken);
    }
}

fn zstd_io(error: &std::io::Error) -> IoError {
    IoError::io(format!("zstd encode: {error}"))
}

/// Run one section's logical bytes through `compression` into `sink`.
fn encode_pass<W: Write>(
    source: &dyn LogicalSource,
    compression: Option<&str>,
    sink: &mut W,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    receipt: &mut StreamReceipt,
) -> Result<u64, IoError> {
    let mut reader = source.open()?;
    let cap = source.logical_len();
    let moved = match compression {
        None => pump(&mut reader, sink, buffer, work, cap, receipt)?,
        Some(COMPRESSION_ZSTD) => {
            let mut encoder = zstd::stream::write::Encoder::new(&mut *sink, ZSTD_LEVEL)
                .map_err(|e| zstd_io(&e))?;
            let moved = pump(&mut reader, &mut encoder, buffer, work, cap, receipt)?;
            encoder.finish().map_err(|e| zstd_io(&e))?;
            moved
        }
        Some(other) => return Err(IoError::UnsupportedCompression { algo: other.into() }),
    };
    if moved != cap {
        return Err(IoError::ManifestMismatch { message: "section logical size mismatch" });
    }
    receipt.logical_bytes += moved;
    Ok(moved)
}

/// On-wire length and BLAKE3 of one section under `compression`, without keeping any of it.
fn measure_pass(
    source: &dyn LogicalSource,
    compression: Option<&str>,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    receipt: &mut StreamReceipt,
) -> Result<(u64, [u8; 32]), IoError> {
    let mut tally = Tally::new(std::io::sink());
    encode_pass(source, compression, &mut tally, buffer, work, receipt)?;
    Ok((tally.written, *tally.hasher.finalize().as_bytes()))
}

/// Build a section descriptor from `source` in two bounded passes.
///
/// The descriptor equals [`section_descriptor_with_policy`] of the same bytes, but no
/// on-wire or logical buffer larger than a chunk is held: the pass hashes and counts
/// the compressed stream as zstd produces it. The working set is leased from
/// `work.ctx` first.
///
/// # Errors
///
/// Cancellation, a refused lease, a source that fails or has the wrong length.
pub fn describe_streaming(
    id: impl Into<String>,
    content_type: impl Into<String>,
    source: &dyn LogicalSource,
    policy: CompressPolicy,
    work: &StreamWork<'_>,
) -> Result<(SectionDescriptor, StreamReceipt), IoError> {
    let mut lease = work.admit(STREAM_ENCODER_WORKSPACE_BYTES)?;
    let mut receipt = StreamReceipt {
        leased_bytes: lease.bytes(),
        chunk_bytes: work.chunk_bytes,
        ..StreamReceipt::default()
    };
    let logical_len = source.logical_len();
    let wants_zstd = match policy {
        CompressPolicy::Never => false,
        CompressPolicy::Always => true,
        CompressPolicy::Auto => logical_len >= crate::error::lossless_u64(AUTO_COMPRESS_MIN_BYTES),
    };
    let mut chosen = (None, None);
    if wants_zstd {
        let (len, hash) =
            measure_pass(source, Some(COMPRESSION_ZSTD), lease.buffer(), work, &mut receipt)?;
        let keep = policy == CompressPolicy::Always
            || u128::from(len) * 100 < u128::from(logical_len) * 95;
        if keep {
            chosen = (Some(COMPRESSION_ZSTD.to_owned()), Some((len, hash)));
        }
    }
    let (compression, (on_wire_len, hash)) = match chosen {
        (Some(name), Some(measured)) => (Some(name), measured),
        _ => (None, measure_pass(source, None, lease.buffer(), work, &mut receipt)?),
    };
    receipt.on_wire_bytes = on_wire_len;
    let descriptor = SectionDescriptor {
        id: id.into(),
        content_type: content_type.into(),
        encoding_version: 1,
        required: true,
        compression,
        compressed_size: on_wire_len,
        uncompressed_size: logical_len,
        blake3: hash,
        logical_schema: "cbor.v1".into(),
    };
    Ok((descriptor, receipt))
}

/// Write the container for `manifest`, streaming each section from its source.
///
/// Header and manifest are written first, then each section is compressed straight
/// into `w` while its length and checksum are recomputed and compared with the
/// descriptor. A section that does not reproduce its descriptor fails with
/// [`IoError::ChecksumMismatch`] after its bytes were sent, so write to a target that
/// can be discarded. Nothing is written when the lease is refused or the context is
/// already cancelled.
///
/// # Errors
///
/// Count, size or checksum disagreement with the manifest, cancellation, a refused
/// lease, or an output failure.
pub fn write_streaming<W: Write>(
    mut w: W,
    manifest: &ArtifactManifest,
    sources: &[&dyn LogicalSource],
    work: &StreamWork<'_>,
) -> Result<StreamReceipt, IoError> {
    if manifest.sections.len() != sources.len() {
        return Err(IoError::ManifestMismatch {
            message: "section count != manifest section count",
        });
    }
    for (desc, source) in manifest.sections.iter().zip(sources) {
        if desc.uncompressed_size != source.logical_len() {
            return Err(IoError::ManifestMismatch { message: "section logical size mismatch" });
        }
        u32::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
    }
    let mut lease = work.admit(STREAM_ENCODER_WORKSPACE_BYTES)?;
    let mut receipt = StreamReceipt {
        leased_bytes: lease.bytes(),
        chunk_bytes: work.chunk_bytes,
        ..StreamReceipt::default()
    };
    let mut manifest_buf = Vec::new();
    ciborium::into_writer(&*manifest.sealed()?, &mut manifest_buf).map_err(IoError::cbor_encode)?;
    let manifest_len = u32::try_from(manifest_buf.len()).map_err(|_| IoError::TooLarge)?;
    w.write_all(MAGIC)?;
    w.write_all(&CONTAINER_VERSION.to_le_bytes())?;
    w.write_all(&manifest_len.to_le_bytes())?;
    w.write_all(&manifest_buf)?;
    for (desc, source) in manifest.sections.iter().zip(sources) {
        let len = u32::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
        w.write_all(&len.to_le_bytes())?;
        let mut tally = Tally::new(&mut w);
        encode_pass(
            *source,
            desc.compression.as_deref(),
            &mut tally,
            lease.buffer(),
            work,
            &mut receipt,
        )?;
        if tally.written != desc.compressed_size {
            return Err(IoError::ManifestMismatch { message: "section compressed size mismatch" });
        }
        if tally.hasher.finalize().as_bytes() != &desc.blake3 {
            return Err(IoError::ChecksumMismatch { section: desc.id.clone() });
        }
        receipt.largest_write = receipt.largest_write.max(tally.largest_write);
        receipt.on_wire_bytes += tally.written;
    }
    w.flush()?;
    Ok(receipt)
}

impl EncodedArtifact {
    /// [`write_streaming`] of this artifact's logical sections.
    ///
    /// Unlike [`Self::write_to`] it holds no compressed copy of any section: each is
    /// compressed into `w` chunk by chunk under `work`.
    ///
    /// # Errors
    ///
    /// The errors of [`write_streaming`], and a section id that differs from the manifest.
    pub fn write_streaming<W: Write>(
        &self,
        w: W,
        work: &StreamWork<'_>,
    ) -> Result<StreamReceipt, IoError> {
        for (desc, section) in self.manifest.sections.iter().zip(&self.sections) {
            if desc.id != section.id {
                return Err(IoError::ManifestMismatch { message: "section id mismatch" });
            }
        }
        let sources: Vec<&dyn LogicalSource> =
            self.sections.iter().map(|section| &section.data as &dyn LogicalSource).collect();
        write_streaming(w, &self.manifest, &sources, work)
    }
}

/// Read a container, handing the logical bytes of wanted sections to `on_chunk`.
///
/// Every section is hashed in chunks before it is decoded, so a wanted section is
/// delivered only after its on-wire checksum verified, and an unwanted one is verified
/// and skipped. Zstandard decoding is capped by the descriptor's `uncompressed_size`
/// and by [`StreamWork::decoder_window_log_max`]. No section is held whole.
///
/// # Errors
///
/// Bad header or manifest, a size or checksum disagreement, a frame that expands past its
/// declared size, cancellation, a refused lease, or an error from `on_chunk`.
pub fn read_streaming<R: Read + Seek>(
    mut r: R,
    work: &StreamWork<'_>,
    wants: impl Fn(&SectionDescriptor) -> bool,
    mut on_chunk: impl FnMut(&SectionDescriptor, &[u8]) -> Result<(), IoError>,
) -> Result<(ArtifactManifest, StreamReceipt), IoError> {
    let mut lease = work.admit(work.decoder_workspace_bytes())?;
    let mut receipt = StreamReceipt {
        leased_bytes: lease.bytes(),
        chunk_bytes: work.chunk_bytes,
        ..StreamReceipt::default()
    };
    let (manifest, _) = read_header_and_manifest(&mut r)?;
    for desc in &manifest.sections {
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let len = u64::from(u32::from_le_bytes(len_buf));
        if len != desc.compressed_size {
            return Err(IoError::ManifestMismatch { message: "section size mismatch" });
        }
        if len > crate::error::lossless_u64(work.max_section_bytes) {
            return Err(IoError::TooLarge);
        }
        let start = r.stream_position()?;
        verify_pass(&mut r, desc, len, lease.buffer(), work, &mut receipt)?;
        if wants(desc) {
            r.seek(SeekFrom::Start(start))?;
            let mut limited = (&mut r).take(len);
            decode_pass(&mut limited, desc, lease.buffer(), work, &mut receipt, &mut on_chunk)?;
            r.seek(SeekFrom::Start(start + len))?;
        }
    }
    Ok((manifest, receipt))
}

/// Hash `len` on-wire bytes in chunks and compare with the descriptor.
fn verify_pass<R: Read>(
    r: &mut R,
    desc: &SectionDescriptor,
    len: u64,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    receipt: &mut StreamReceipt,
) -> Result<(), IoError> {
    let mut hasher = blake3::Hasher::new();
    let mut remaining = len;
    while remaining > 0 {
        work.tick()?;
        let want = usize::try_from(remaining).unwrap_or(usize::MAX).min(buffer.len());
        r.read_exact(&mut buffer[..want])?;
        hasher.update(&buffer[..want]);
        remaining -= crate::error::lossless_u64(want);
        receipt.chunks += 1;
        receipt.largest_read = receipt.largest_read.max(want);
    }
    receipt.on_wire_bytes += len;
    if hasher.finalize().as_bytes() != &desc.blake3 {
        return Err(IoError::ChecksumMismatch { section: desc.id.clone() });
    }
    Ok(())
}

fn decode_pass<R: Read>(
    on_wire: &mut R,
    desc: &SectionDescriptor,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    receipt: &mut StreamReceipt,
    on_chunk: &mut impl FnMut(&SectionDescriptor, &[u8]) -> Result<(), IoError>,
) -> Result<(), IoError> {
    let cap = desc.uncompressed_size;
    let mut delivered = 0u64;
    let decompress = |error: std::io::Error| IoError::Decompress {
        section: desc.id.clone(),
        message: error.to_string(),
    };
    let mut deliver = |source: &mut dyn Read| -> Result<(), IoError> {
        loop {
            work.tick()?;
            let taken = source.read(buffer)?;
            if taken == 0 {
                return Ok(());
            }
            delivered += crate::error::lossless_u64(taken);
            if delivered > cap {
                return Err(IoError::Decompress {
                    section: desc.id.clone(),
                    message: format!("decompressed size exceeds declared uncompressed_size {cap}"),
                });
            }
            on_chunk(desc, &buffer[..taken])?;
            receipt.chunks += 1;
            receipt.largest_read = receipt.largest_read.max(taken);
            receipt.logical_bytes += crate::error::lossless_u64(taken);
        }
    };
    match desc.compression.as_deref() {
        None => deliver(on_wire)?,
        Some(COMPRESSION_ZSTD) => {
            let mut decoder = zstd::Decoder::new(on_wire).map_err(decompress)?;
            decoder.window_log_max(work.decoder_window_log_max).map_err(decompress)?;
            deliver(&mut decoder)?;
        }
        Some(other) => return Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
    if delivered != cap {
        return Err(IoError::Decompress {
            section: desc.id.clone(),
            message: format!("logical size {delivered} != uncompressed_size {cap}"),
        });
    }
    Ok(())
}

/// BLAKE3 digest of a byte string under a domain `tag`, fed in pieces.
///
/// The framing is `CanonicalWriter::digest_bytes(tag, bytes)`: the tag and the byte
/// count as length words, then the bytes, so the byte count must be known before the
/// first piece. [`Self::finish`] refuses a stream that delivered a different count.
#[derive(Clone, Debug)]
pub struct DomainDigest {
    hasher: blake3::Hasher,
    expected: u64,
    seen: u64,
}

impl DomainDigest {
    /// Start a digest of exactly `len` bytes under `tag`.
    #[must_use]
    pub fn new(tag: &str, len: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&crate::error::lossless_u64(tag.len()).to_le_bytes());
        hasher.update(tag.as_bytes());
        hasher.update(&len.to_le_bytes());
        Self { hasher, expected: len, seen: 0 }
    }

    /// Feed the next piece.
    pub fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
        self.seen += crate::error::lossless_u64(bytes.len());
    }

    /// The digest, or `None` when the pieces did not total the declared length.
    #[must_use]
    pub fn finish(&self) -> Option<[u8; 32]> {
        (self.seen == self.expected).then(|| *self.hasher.finalize().as_bytes())
    }
}

/// Compress `source` into `sink` in chunks with Zstandard at the container's level,
/// ticking `work` between chunks. Returns the logical bytes read.
///
/// # Errors
///
/// Cancellation, a read failure, or a write failure.
pub fn zstd_compress_chunked<R: Read, W: Write>(
    source: &mut R,
    sink: &mut W,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
) -> Result<u64, IoError> {
    let mut receipt = StreamReceipt::default();
    let mut encoder =
        zstd::stream::write::Encoder::new(&mut *sink, ZSTD_LEVEL).map_err(|e| zstd_io(&e))?;
    let moved = pump(source, &mut encoder, buffer, work, u64::MAX, &mut receipt)?;
    encoder.finish().map_err(|e| zstd_io(&e))?;
    Ok(moved)
}

/// Decompress a Zstandard stream into `on_chunk` in chunks, refusing more than
/// `max_logical` bytes as they arrive and any window past
/// [`StreamWork::decoder_window_log_max`]. Returns the logical bytes delivered.
///
/// # Errors
///
/// Cancellation, a corrupt frame, output past `max_logical`, or an error from `on_chunk`.
pub fn zstd_decompress_chunked<R: Read>(
    on_wire: R,
    max_logical: u64,
    buffer: &mut [u8],
    work: &StreamWork<'_>,
    mut on_chunk: impl FnMut(&[u8]) -> Result<(), IoError>,
) -> Result<u64, IoError> {
    let corrupt = |error: std::io::Error| IoError::Decompress {
        section: "stream".into(),
        message: error.to_string(),
    };
    let mut decoder = zstd::Decoder::new(on_wire).map_err(corrupt)?;
    decoder.window_log_max(work.decoder_window_log_max).map_err(corrupt)?;
    let mut total = 0u64;
    loop {
        work.tick()?;
        let taken = decoder.read(buffer).map_err(corrupt)?;
        if taken == 0 {
            return Ok(total);
        }
        total += crate::error::lossless_u64(taken);
        if total > max_logical {
            return Err(IoError::Decompress {
                section: "stream".into(),
                message: format!("decompressed size exceeds declared size {max_logical}"),
            });
        }
        on_chunk(&buffer[..taken])?;
    }
}

/// Bytes a streamed encode leases for its codec on top of two chunk buffers.
#[must_use]
pub const fn stream_encoder_workspace_bytes() -> u64 {
    STREAM_ENCODER_WORKSPACE_BYTES
}

/// Bytes a streamed decode under the default window leases for its codec on top of two chunk
/// buffers. [`StreamWork::decoder_workspace_bytes`] is the figure for a configured window.
#[must_use]
pub const fn stream_decoder_workspace_bytes() -> u64 {
    STREAM_DECODER_WORKSPACE_BYTES
}
