// SPDX-License-Identifier: MIT OR Apache-2.0
//! Structural heap quotes plus explicit typed-slot supplements, independent of JSON spelling.
use antecedent_core::{ExecutionContext, execution::AdmissionLease};
use serde::{
    Serialize,
    ser::{
        self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
        SerializeTupleStruct, SerializeTupleVariant,
    },
};
use std::{fmt, sync::Arc};

#[derive(Debug)]
pub(crate) struct QuoteError(String);
impl fmt::Display for QuoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for QuoteError {}
impl ser::Error for QuoteError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}
fn add(a: u64, b: u64) -> Result<u64, QuoteError> {
    a.checked_add(b).ok_or_else(|| QuoteError("retained allocation size overflow".into()))
}
fn inline<T: ?Sized>(value: &T) -> Result<u64, QuoteError> {
    u64::try_from(std::mem::size_of_val(value)).map_err(|e| QuoteError(e.to_string()))
}
struct Quote;
struct Fields(u64);
impl Fields {
    fn heap<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.0 = add(self.0, value.serialize(Quote)?)?;
        Ok(())
    }
    fn slot<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        // Serde may pass a reference as the sequence element. Include a normal
        // owned-container header here; larger inline enum slots require typed_slots.
        self.0 = add(self.0, inline(value)?.max(std::mem::size_of::<Vec<()>>() as u64))?;
        self.heap(value)
    }
}
impl ser::Serializer for Quote {
    type Ok = u64;
    type Error = QuoteError;
    type SerializeSeq = Fields;
    type SerializeTuple = Fields;
    type SerializeTupleStruct = Fields;
    type SerializeTupleVariant = Fields;
    type SerializeMap = Fields;
    type SerializeStruct = Fields;
    type SerializeStructVariant = Fields;
    fn serialize_bool(self, _value: bool) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<bool>() as u64)
    }
    fn serialize_i8(self, _value: i8) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<i8>() as u64)
    }
    fn serialize_i16(self, _value: i16) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<i16>() as u64)
    }
    fn serialize_i32(self, _value: i32) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<i32>() as u64)
    }
    fn serialize_i64(self, _value: i64) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<i64>() as u64)
    }
    fn serialize_u8(self, _value: u8) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<u8>() as u64)
    }
    fn serialize_u16(self, _value: u16) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<u16>() as u64)
    }
    fn serialize_u32(self, _value: u32) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<u32>() as u64)
    }
    fn serialize_u64(self, _value: u64) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<u64>() as u64)
    }
    fn serialize_f32(self, _value: f32) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<f32>() as u64)
    }
    fn serialize_f64(self, _value: f64) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<f64>() as u64)
    }
    fn serialize_char(self, _value: char) -> Result<u64, QuoteError> {
        Ok(std::mem::size_of::<char>() as u64)
    }
    fn serialize_str(self, value: &str) -> Result<u64, QuoteError> {
        add(std::mem::size_of::<String>() as u64, inline(value)?)
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<u64, QuoteError> {
        inline(value)
    }
    fn serialize_none(self) -> Result<u64, QuoteError> {
        Ok(0)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<u64, QuoteError> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<u64, QuoteError> {
        Ok(0)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<u64, QuoteError> {
        Ok(0)
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
    ) -> Result<u64, QuoteError> {
        Ok(0)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<u64, QuoteError> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<u64, QuoteError> {
        value.serialize(self)
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Fields, QuoteError> {
        // Serde may erase the referent behind iterator items. Charge the collection
        // header and primitive fields as well as structural payloads; concrete large
        // failed slots receive a separate typed quote at their production owner.
        Ok(Fields(std::mem::size_of::<Vec<()>>() as u64))
    }
    fn serialize_tuple(self, _len: usize) -> Result<Fields, QuoteError> {
        Ok(Fields(0))
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Fields, QuoteError> {
        Ok(Fields(0))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Fields, QuoteError> {
        Ok(Fields(0))
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Fields, QuoteError> {
        Ok(Fields(64))
    }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Fields, QuoteError> {
        Ok(Fields(0))
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Fields, QuoteError> {
        Ok(Fields(0))
    }
}
impl SerializeSeq for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.slot(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
impl SerializeTuple for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.slot(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
impl SerializeTupleStruct for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.slot(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
impl SerializeTupleVariant for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.slot(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
impl SerializeStruct for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), QuoteError> {
        self.heap(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
impl SerializeStructVariant for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), QuoteError> {
        self.heap(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
impl SerializeMap for Fields {
    type Ok = u64;
    type Error = QuoteError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.slot(value)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), QuoteError> {
        self.slot(value)
    }
    fn end(self) -> Result<u64, QuoteError> {
        Ok(self.0)
    }
}
/// Structural heap quote plus normal owned-container headers and capacity growth.
/// Serde erases some concrete sequence element types, so callers add `typed_slots`
/// for large inline enums and failed Option slots. This is not an allocator measurement.
pub(crate) fn allocation_bytes<T: Serialize + ?Sized>(value: &T) -> Result<u64, QuoteError> {
    add(
        inline(value)?,
        value
            .serialize(Quote)?
            .checked_mul(2)
            .ok_or_else(|| QuoteError("retained heap size overflow".into()))?,
    )
}
/// Actual inline element storage, including the full width of unpopulated Option slots.
pub(crate) fn typed_slots<T>(values: &[T]) -> Result<u64, QuoteError> {
    u64::try_from(values.len())
        .ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<T>() as u64))
        .ok_or_else(|| QuoteError("retained typed slot size overflow".into()))
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RetainedAllocation(pub(crate) Vec<Arc<AdmissionLease>>);
impl PartialEq for RetainedAllocation {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl RetainedAllocation {
    pub(crate) fn acquire(
        bytes: u64,
        ctx: &ExecutionContext,
    ) -> Result<Self, antecedent_core::execution::AdmissionError> {
        let lease = ctx.acquire_admission(0, bytes)?;
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::RetainedBytes,
            bytes,
        );
        Ok(Self(vec![Arc::new(lease)]))
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.0.iter().map(|lease| lease.remainder().bytes()).sum()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn erased_sequence_references_still_cover_records_and_empty_nested_collections() {
        #[derive(Clone, Serialize)]
        struct Epoch {
            step: u64,
            score: f64,
            rejected: u64,
        }
        let trace = vec![Epoch { step: 0, score: 0.0, rejected: 0 }; 1000];
        assert!(allocation_bytes(&trace).unwrap() >= typed_slots(&trace).unwrap());
        let nested = vec![Vec::<f64>::new(); 1000];
        assert!(allocation_bytes(&nested).unwrap() >= typed_slots(&nested).unwrap());
        let names = vec![String::new(); 1000];
        assert!(allocation_bytes(&names).unwrap() >= typed_slots(&names).unwrap());
    }

    #[test]
    fn zero_numeric_vectors_are_charged_by_width() {
        let values = vec![0u64; 1000];
        assert!(allocation_bytes(&values).unwrap() >= 8000);
    }
}
