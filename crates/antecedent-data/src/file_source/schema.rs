//! Schema comparison, decode-width estimates and directory listing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use super::plan::*;

/// Schema of a table source and the per-column decode estimate.
#[derive(Clone, Debug)]
pub(super) struct SchemaView {
    pub(super) schema: SchemaRef,
    pub(super) columns: Vec<FileColumn>,
    pub(super) decode_per_row: Vec<Option<u64>>,
    pub(super) reader_bytes: u64,
}

impl SchemaView {
    pub(super) fn new(schema: SchemaRef, decode_per_row: Vec<Option<u64>>, reader_bytes: u64) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|field| FileColumn {
                name: Arc::from(field.name().as_str()),
                kind: FileColumnKind::of(field),
                nullable: field.is_nullable(),
                arrow_type: Arc::from(field.data_type().to_string()),
            })
            .collect();
        Self { schema, columns, decode_per_row, reader_bytes }
    }

    pub(super) fn plan(
        &self,
        closure: &ColumnClosure,
        options: ReadOptions,
    ) -> Result<ReadPlan, FileSourceError> {
        if options.chunk_rows == 0 {
            return Err(zero_chunk());
        }
        if closure.columns().is_empty() {
            return Err(FileSourceError::InvalidOptions {
                detail: "the required-column closure names no column".into(),
            });
        }
        let mut file_index = Vec::with_capacity(closure.columns().len());
        let mut resolved = Vec::with_capacity(closure.columns().len());
        for required in closure.columns() {
            let name = required.role.as_str();
            let mut matches =
                self.columns.iter().enumerate().filter(|(_, column)| &*column.name == name);
            let Some((index, column)) = matches.next() else {
                return Err(FileSourceError::UnknownColumn {
                    column: name.to_owned(),
                    available: self.columns.iter().map(|column| column.name.to_string()).collect(),
                });
            };
            if matches.next().is_some() {
                return Err(FileSourceError::SchemaMismatch {
                    path: PathBuf::new(),
                    detail: format!("column name '{name}' occurs more than once"),
                });
            }
            if let FileColumnKind::Unsupported { .. } = column.kind {
                return Err(convert::unsupported_type(name, self.schema.field(index).data_type()));
            }
            file_index.push(index);
            resolved.push(ColumnRole::new(name));
        }
        closure.project(&resolved)?;
        let mut projection = file_index.clone();
        projection.sort_unstable();
        let batch_position =
            file_index.iter().map(|index| projection.binary_search(index).unwrap_or(0)).collect();
        let mut decode_bytes_per_row = 0_u64;
        let mut owned_bytes_per_row = 0_u64;
        let mut columns = Vec::with_capacity(file_index.len());
        for &index in &file_index {
            let column = self.columns[index].clone();
            let arrow = arrow_decode_width(self.schema.field(index).data_type(), options);
            let recorded = self.decode_per_row.get(index).copied().flatten().unwrap_or(0);
            decode_bytes_per_row = decode_bytes_per_row.saturating_add(arrow.max(recorded));
            owned_bytes_per_row =
                owned_bytes_per_row.saturating_add(column.kind.owned_width_bytes().unwrap_or(8));
            columns.push(column);
        }
        Ok(ReadPlan {
            closure: closure.clone(),
            columns,
            file_index,
            projection,
            batch_position,
            chunk_rows: options.chunk_rows,
            decode_bytes_per_row,
            owned_bytes_per_row,
            reader_bytes: self.reader_bytes,
        })
    }
}

pub(super) fn arrow_decode_width(data_type: &DataType, options: ReadOptions) -> u64 {
    let text = options.text_bytes_per_row;
    match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Utf8 => 4_u64.saturating_add(text),
        DataType::LargeUtf8 => 8_u64.saturating_add(text),
        DataType::Utf8View => 16_u64.saturating_add(text),
        DataType::Dictionary(key, _) => arrow_decode_width(key, options).saturating_add(text),
        _ => 8,
    }
}

pub(super) fn same_schema(first: &Schema, next: &Schema, path: &Path) -> Result<(), FileSourceError> {
    if first.fields().len() != next.fields().len() {
        return Err(FileSourceError::SchemaMismatch {
            path: path.to_path_buf(),
            detail: format!(
                "{} columns where the first file has {}",
                next.fields().len(),
                first.fields().len()
            ),
        });
    }
    for (left, right) in first.fields().iter().zip(next.fields().iter()) {
        if left.name() != right.name() || left.data_type() != right.data_type() {
            return Err(FileSourceError::SchemaMismatch {
                path: path.to_path_buf(),
                detail: format!(
                    "column {} {} where the first file has {} {}",
                    right.name(),
                    right.data_type(),
                    left.name(),
                    left.data_type()
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn list_dataset(directory: &Path) -> Result<Vec<PathBuf>, FileSourceError> {
    let io = |error: std::io::Error| FileSourceError::Io {
        path: directory.to_path_buf(),
        message: error.to_string(),
    };
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(io)? {
        let path = entry.map_err(io)?.path();
        let hidden = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.') || name.starts_with('_'));
        if hidden || !path.is_file() {
            continue;
        }
        if TableFileFormat::from_path(&path) == Some(TableFileFormat::Parquet) {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(FileSourceError::EmptyDataset { path: directory.to_path_buf() });
    }
    Ok(paths)
}
