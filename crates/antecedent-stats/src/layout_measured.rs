pub(crate) const MEASURED: &[MeasuredPoint] = &[
    dense(LayoutOp::Gram, 64, 8, Product::Faer, 4),
    dense(LayoutOp::Gram, 4096, 128, Product::Faer, 16),
    sparse(128, 16, 4, Product::Reference, 1),
    sparse(8192, 1024, 48, Product::Faer, 16),
    dense(LayoutOp::ColumnPivotedQr, 32, 8, Product::Faer, 10),
    dense(LayoutOp::ColumnPivotedQr, 64, 16, Product::Faer, 1),
    dense(LayoutOp::ColumnPivotedQr, 128, 16, Product::Faer, 1),
    dense(LayoutOp::ColumnPivotedQr, 256, 32, Product::Faer, 1),
    dense(LayoutOp::ColumnPivotedQr, 512, 32, Product::Faer, 1),
    dense(LayoutOp::ColumnPivotedQr, 1024, 64, Product::Faer, 1),
    dense(LayoutOp::ColumnPivotedQr, 2048, 64, Product::Faer, 1),
];
