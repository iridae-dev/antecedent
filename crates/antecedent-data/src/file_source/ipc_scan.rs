//! Decode cost of each Arrow IPC record batch, read from message metadata alone.
//!
//! An IPC reader materializes a record batch at the size its writer chose: the
//! whole message body is read into one buffer (every column, projected or not),
//! compressed buffers are inflated into new allocations next to it, and each
//! array node adds structure overhead. The admission pool must hold that
//! before the reader runs, so the cost is derived here from the message
//! flatbuffers (the file footer's block table, or the message framing of a
//! stream) without reading a single body byte, except the eight-byte
//! uncompressed-length prefix of each buffer of a compressed batch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use arrow_ipc::{MessageHeader, root_as_footer, root_as_message};

use super::error::FileSourceError;
use super::provider::{format_error, open_file};

/// Bytes charged per array node for the array structure, its `Arc`s and the
/// offset or validity bookkeeping a decoder builds next to the data buffers.
pub(crate) const NODE_OVERHEAD_BYTES: u64 = 256;

/// Largest message metadata or footer read; a larger one is a corrupt length.
const MAX_METADATA_BYTES: u64 = 256 << 20;
const CONTINUATION: u32 = 0xFFFF_FFFF;
const MAGIC: &[u8; 6] = b"ARROW1";

/// Decode cost of every record batch of a file, in order, and of its dictionaries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct IpcCosts {
    /// Bytes the reader holds while it decodes each record batch.
    pub batches: Vec<u64>,
    /// Bytes of all dictionary batches, which stay resident while the reader lives.
    pub dictionaries: u64,
}

pub(crate) fn scan_file(path: &Path) -> Result<IpcCosts, FileSourceError> {
    let mut file = open_file(path)?;
    let length = file.metadata().map_err(|error| io(path, &error))?.len();
    if length < 12 + 6 {
        return Err(format_error(path, &"the file is shorter than an Arrow IPC file trailer"));
    }
    let mut trailer = [0_u8; 10];
    read_at(&mut file, length - 10, &mut trailer, path)?;
    if &trailer[4..] != MAGIC {
        return Err(format_error(path, &"the file does not end with the Arrow IPC magic"));
    }
    let footer_len =
        u64::from(u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]));
    if footer_len == 0 || footer_len > length - 10 || footer_len > MAX_METADATA_BYTES {
        return Err(format_error(path, &"the footer length is outside the file"));
    }
    let mut footer_bytes = vec![0_u8; usize_of(footer_len, path)?];
    read_at(&mut file, length - 10 - footer_len, &mut footer_bytes, path)?;
    let footer = root_as_footer(&footer_bytes).map_err(|error| format_error(path, &error))?;
    let mut costs = IpcCosts::default();
    if let Some(dictionaries) = footer.dictionaries() {
        for block in dictionaries.iter() {
            let cost = block_cost(&mut file, block.offset(), block.bodyLength(), length, path)?;
            costs.dictionaries = checked_sum(costs.dictionaries, cost, path)?;
        }
    }
    if let Some(blocks) = footer.recordBatches() {
        for block in blocks.iter() {
            let cost = block_cost(&mut file, block.offset(), block.bodyLength(), length, path)?;
            costs.batches.push(cost);
        }
    }
    Ok(costs)
}

pub(crate) fn scan_stream(path: &Path) -> Result<IpcCosts, FileSourceError> {
    let mut file = open_file(path)?;
    let length = file.metadata().map_err(|error| io(path, &error))?.len();
    let mut costs = IpcCosts::default();
    let mut position = 0_u64;
    while position + 4 <= length {
        let (message_start, meta_len) = read_prefix(&mut file, position, length, path)?;
        if meta_len == 0 {
            break;
        }
        let (kind, cost, body) = message_cost(&mut file, message_start, meta_len, length, path)?;
        let next = message_start
            .checked_add(meta_len)
            .and_then(|end| end.checked_add(body))
            .filter(|end| *end <= length)
            .ok_or_else(|| format_error(path, &"a message body runs past the end of the file"))?;
        match kind {
            Kind::Batch => costs.batches.push(cost),
            Kind::Dictionary => {
                costs.dictionaries = checked_sum(costs.dictionaries, cost, path)?;
            }
            Kind::Other => {}
        }
        position = next;
    }
    Ok(costs)
}

#[derive(Clone, Copy)]
enum Kind {
    Batch,
    Dictionary,
    Other,
}

/// Start of the metadata and its byte length, from the framing at `position`.
fn read_prefix(
    file: &mut File,
    position: u64,
    length: u64,
    path: &Path,
) -> Result<(u64, u64), FileSourceError> {
    let mut word = [0_u8; 4];
    read_at(file, position, &mut word, path)?;
    let first = u32::from_le_bytes(word);
    let (start, len) = if first == CONTINUATION {
        if position + 8 > length {
            return Err(format_error(path, &"a message prefix is cut off"));
        }
        read_at(file, position + 4, &mut word, path)?;
        (position + 8, u32::from_le_bytes(word))
    } else {
        (position + 4, first)
    };
    Ok((start, u64::from(len)))
}

fn read_metadata(
    file: &mut File,
    meta_start: u64,
    meta_len: u64,
    length: u64,
    path: &Path,
) -> Result<Vec<u8>, FileSourceError> {
    if meta_len > MAX_METADATA_BYTES
        || !meta_start.checked_add(meta_len).is_some_and(|end| end <= length)
    {
        return Err(format_error(path, &"a message's metadata runs past the end of the file"));
    }
    let mut bytes = vec![0_u8; usize_of(meta_len, path)?];
    read_at(file, meta_start, &mut bytes, path)?;
    Ok(bytes)
}

/// Cost of the message framed at `offset` in a file-format block.
fn block_cost(
    file: &mut File,
    offset: i64,
    declared_body: i64,
    length: u64,
    path: &Path,
) -> Result<u64, FileSourceError> {
    let offset = u64::try_from(offset)
        .map_err(|_| format_error(path, &"a footer block has a negative offset"))?;
    let declared_body = u64::try_from(declared_body)
        .map_err(|_| format_error(path, &"a footer block has a negative body length"))?;
    if offset + 4 > length {
        return Err(format_error(path, &"a footer block lies outside the file"));
    }
    let (meta_start, meta_len) = read_prefix(file, offset, length, path)?;
    let (_, cost, actual) = message_cost(file, meta_start, meta_len, length, path)?;
    // The footer's declared body must agree with the message, or a hostile
    // footer could understate what the reader will allocate.
    if actual != declared_body {
        return Err(format_error(path, &"a footer block disagrees with its message body length"));
    }
    Ok(cost)
}

fn message_cost(
    file: &mut File,
    meta_start: u64,
    meta_len: u64,
    length: u64,
    path: &Path,
) -> Result<(Kind, u64, u64), FileSourceError> {
    let bytes = read_metadata(file, meta_start, meta_len, length, path)?;
    let message = root_as_message(&bytes).map_err(|error| format_error(path, &error))?;
    let body = u64::try_from(message.bodyLength())
        .map_err(|_| format_error(path, &"a message records a negative body length"))?;
    let body_start = meta_start
        .checked_add(meta_len)
        .ok_or_else(|| format_error(path, &"a message offset overflows"))?;
    let (kind, batch) = if message.header_type() == MessageHeader::RecordBatch {
        (Kind::Batch, message.header_as_record_batch())
    } else if message.header_type() == MessageHeader::DictionaryBatch {
        (Kind::Dictionary, message.header_as_dictionary_batch().and_then(|d| d.data()))
    } else {
        return Ok((Kind::Other, 0, body));
    };
    let Some(batch) = batch else {
        return Err(format_error(path, &"a batch message carries no record batch header"));
    };
    let nodes = batch.nodes().map_or(0, |nodes| nodes.len() as u64);
    let mut inflated = 0_u64;
    if batch.compression().is_some() {
        if let Some(buffers) = batch.buffers() {
            for buffer in buffers.iter() {
                let (offset, len) = (buffer.offset(), buffer.length());
                let (Ok(offset), Ok(len)) = (u64::try_from(offset), u64::try_from(len)) else {
                    return Err(format_error(path, &"a buffer has a negative extent"));
                };
                if len < 8 {
                    continue;
                }
                let at = body_start
                    .checked_add(offset)
                    .filter(|at| at.checked_add(8).is_some_and(|end| end <= length))
                    .ok_or_else(|| format_error(path, &"a buffer lies outside the file"))?;
                let mut prefix = [0_u8; 8];
                read_at(file, at, &mut prefix, path)?;
                // A length of -1 marks a buffer stored uncompressed.
                let uncompressed = u64::try_from(i64::from_le_bytes(prefix)).unwrap_or(len - 8);
                inflated = checked_sum(inflated, uncompressed, path)?;
            }
        }
    }
    let structure = nodes
        .checked_mul(NODE_OVERHEAD_BYTES)
        .ok_or_else(|| format_error(path, &"an array node count overflows"))?;
    let cost = [meta_len, body, inflated, structure]
        .into_iter()
        .try_fold(0_u64, |sum, part| sum.checked_add(part))
        .ok_or_else(|| format_error(path, &"a batch decode cost overflows"))?;
    Ok((kind, cost, body))
}

fn read_at(
    file: &mut File,
    position: u64,
    buffer: &mut [u8],
    path: &Path,
) -> Result<(), FileSourceError> {
    file.seek(SeekFrom::Start(position)).map_err(|error| io(path, &error))?;
    file.read_exact(buffer)
        .map_err(|_| format_error(path, &"the file ends inside an Arrow IPC message"))
}

fn checked_sum(left: u64, right: u64, path: &Path) -> Result<u64, FileSourceError> {
    left.checked_add(right).ok_or_else(|| format_error(path, &"a decode cost overflows"))
}

fn usize_of(value: u64, path: &Path) -> Result<usize, FileSourceError> {
    usize::try_from(value).map_err(|_| format_error(path, &"a metadata length overflows usize"))
}

fn io(path: &Path, error: &std::io::Error) -> FileSourceError {
    FileSourceError::Io { path: path.to_path_buf(), message: error.to_string() }
}
