//! Binary metadata encoding and declaration validation for stream artifacts.

use super::{
    DevicePayload, Producer, Reader, StreamArtifactError, StreamMetadata, trailing_bytes, truncated,
};

const MAX_ARTIFACT_ID_LEN: usize = 256;
const PRODUCER_CPU: u8 = 1;
const PRODUCER_ACCELERATOR: u8 = 2;
const DEVICE_ABSENT: u8 = 0;
const DEVICE_MATERIALIZED: u8 = 1;
const DEVICE_LIVE: u8 = 2;

pub(super) fn encode_metadata(metadata: &StreamMetadata) -> Result<Vec<u8>, StreamArtifactError> {
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

pub(super) fn validate_declaration(
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

pub(super) fn decode_metadata(bytes: &[u8]) -> Result<StreamMetadata, StreamArtifactError> {
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
