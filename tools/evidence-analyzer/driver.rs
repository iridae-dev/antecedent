//! Compiler inventory for evidence earning. Not a complete dependency analyzer.
//!
//! As a workspace compiler wrapper with `ANTECEDENT_ANALYZER_OUTPUT` set, it writes one
//! `body` line per ordinary function body (name, file, byte span and the body's text, so
//! a stale span is detectable). Without it, it also prints every item's source and the
//! query dependency edges for `scripts/evidence_analyzer_probe.py`, then stops.
#![feature(rustc_private)]
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_hir;

use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;

fn emit(line: String) {
    use std::io::Write;
    if let Ok(directory) = std::env::var("ANTECEDENT_ANALYZER_OUTPUT") {
        let path = std::path::Path::new(&directory).join(format!("{}.tsv", std::process::id()));
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(file, "{line}").unwrap();
    } else { println!("{line}"); }
}
struct Inventory;
impl Callbacks for Inventory {
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        // The tab-separated source payload is hex encoded, avoiding ambiguous
        // escaping and depending on no crates beyond the matching compiler.
        let diagnostic = std::env::var_os("ANTECEDENT_ANALYZER_OUTPUT").is_none();
        let hex = |text: &str| -> String { text.as_bytes().iter().map(|b| format!("{b:02x}")).collect() };
        for id in tcx.hir_crate_items(()).definitions() {
            if diagnostic {
                let span = tcx.hir_span_with_body(tcx.local_def_id_to_hir_id(id));
                let source = tcx.sess.source_map().span_to_snippet(span).unwrap_or_default();
                emit(format!("item\t{}\t{}\t{}", tcx.def_path_str(id.to_def_id()), span.from_expansion(), hex(&source)));
            }
            if matches!(tcx.def_kind(id), rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn)
                && !tcx.is_const_fn(id.to_def_id())
            {
                if let Some(body) = tcx.hir_maybe_body_owned_by(id) {
                    let span = body.value.span;
                    // Async/opaque-return bodies can determine a public hidden
                    // type and affect compilation without executing. Keep them.
                    let item_span = tcx.hir_span_with_body(tcx.local_def_id_to_hir_id(id));
                    let header = tcx.sess.source_map().span_to_snippet(item_span.with_hi(span.lo())).unwrap_or_default();
                    if header.contains("impl") || header.contains("async") { continue; }
                    if !span.from_expansion() {
                        let file = tcx.sess.source_map().lookup_source_file(span.lo());
                        let path = file.name.prefer_local_unconditionally().to_string();
                        let path = std::fs::canonicalize(&path).map(|p| p.to_string_lossy().into_owned()).unwrap_or(path);
                        let text = tcx.sess.source_map().span_to_snippet(span).unwrap_or_default();
                        emit(format!("body\t{}\t{}\t{}\t{}\t{}", tcx.def_path_str(id.to_def_id()),
                                 hex(&path), span.lo().0 - file.start_pos.0,
                                 span.hi().0 - file.start_pos.0, hex(&text)));
                    }
                }
            }
        }
        if !diagnostic {
            return Compilation::Continue;
        }
        // Snapshot first: looking up source identities while holding the graph
        // lock can itself invoke compiler queries. Unmapped nodes remain
        // explicit; dropping them would manufacture a completeness claim.
        let edges = std::cell::RefCell::new(Vec::new());
        tcx.dep_graph.with_retained_dep_graph(|graph| {
            edges.borrow_mut().extend(graph.edges().into_iter().map(|(a, b)| (*a, *b)));
        });
        for (source, target) in edges.into_inner() {
            let source_item = source.extract_def_id(tcx).map(|id| tcx.def_path_str(id));
            let target_item = target.extract_def_id(tcx).map(|id| tcx.def_path_str(id));
            emit(format!("edge\t{:?}\t{}\t{:?}\t{}", source.kind,
                     source_item.as_deref().unwrap_or("<unmapped>"), target.kind,
                     target_item.as_deref().unwrap_or("<unmapped>")));
        }
        Compilation::Stop
    }
}
fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| std::path::Path::new(a).file_stem().is_some_and(|s| s == "rustc")) { args.remove(1); }
    run_compiler(&args, &mut Inventory);
}
