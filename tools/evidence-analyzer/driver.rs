//! Compiler dependency inventory for evidence earning.
//!
//! As the workspace compiler wrapper with `ANTECEDENT_ANALYZER_OUTPUT` set, it writes
//! one JSON object per line describing every workspace item of the crate it compiles:
//!
//! * `mod`: a module, its file and the inline `mod` chain inside that file.
//! * `glob`: a glob import, the module it is in and the module it imports from.
//! * `item`: an item, keyed by the outermost call-site position of its name (or of
//!   its definition when it has none), with what it refers to:
//!   - `deps` from its signature, header, fields or type, `bdeps` from its body.
//!     Each is `[relation, key]`: `fn` (a function's signature), `cfn` (a function
//!     evaluated at compile time), `full` (the whole item) or `head` (an impl
//!     header and its associated constants and types);
//!   - `lookups`: `[module, name]` for every name resolved in a module scope;
//!   - `probes`: method resolutions another declaration could silently change;
//!   - `observes`: it reads source positions (`file!`, `line!`, `column!`,
//!     `Location`, panic hooks or backtraces);
//!   - `includes`: it embeds a file (`include!`, `include_str!`, `include_bytes!`).
//!
//! Keys and files are workspace-relative; a reference outside the workspace (a
//! registry crate or the standard library) has no key and is not recorded. Byte
//! offsets are into the file. Coverage, not this inventory, says what executed.
#![feature(rustc_private)]
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_span;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_INDEX, DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, HirId};
use rustc_interface::interface;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeckResults};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::{Span, Symbol, SyntaxContext, kw};

fn js(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn pairs(set: &BTreeSet<(String, String)>) -> String {
    let items: Vec<String> = set.iter().map(|(a, b)| format!("[{},{}]", js(a), js(b))).collect();
    format!("[{}]", items.join(","))
}

struct Paths {
    roots: Vec<String>,
}

impl Paths {
    fn new() -> Self {
        let mut roots = vec!["/antecedent-source".to_string()];
        if let Ok(root) = std::env::var("ANTECEDENT_SOURCE_ROOT") {
            roots.push(root);
        }
        if let Ok(cwd) = std::env::current_dir() {
            roots.push(cwd.to_string_lossy().into_owned());
            if let Ok(real) = std::fs::canonicalize(&cwd) {
                roots.push(real.to_string_lossy().into_owned());
            }
        }
        Paths { roots }
    }

    /// The workspace-relative path, or `None` outside the workspace.
    fn relative(&self, name: &str) -> Option<String> {
        if !name.starts_with('/') {
            if name.starts_with('<') || name.contains("..") {
                return None;
            }
            return Some(name.trim_start_matches("./").to_string());
        }
        for root in &self.roots {
            if let Some(rest) = name.strip_prefix(root.as_str()) {
                if let Some(rest) = rest.strip_prefix('/') {
                    return Some(rest.to_string());
                }
            }
        }
        None
    }
}

struct Inventory {
    paths: Paths,
}

struct Scope<'tcx> {
    tcx: TyCtxt<'tcx>,
    paths: &'tcx Paths,
    files: BTreeMap<rustc_span::BytePos, Option<(String, u32)>>,
}

impl<'tcx> Scope<'tcx> {
    /// Workspace file and offset of a span's outermost call site.
    fn place(&mut self, span: Span) -> Option<(String, u32)> {
        let span = span.source_callsite();
        let source_map = self.tcx.sess.source_map();
        let file = source_map.lookup_source_file(span.lo());
        let base = self.files.entry(file.start_pos).or_insert_with(|| {
            let name = file.name.prefer_local_unconditionally().to_string();
            self.paths.relative(&name).map(|rel| (rel, 0))
        });
        base.as_ref().map(|(rel, _)| (rel.clone(), span.lo().0 - file.start_pos.0))
    }

    fn key(&mut self, def_id: DefId) -> Option<String> {
        let span = self.tcx.def_ident_span(def_id).unwrap_or_else(|| self.tcx.def_span(def_id));
        self.place(span).map(|(file, lo)| format!("{file}:{lo}"))
    }

    fn module_key(&mut self, def_id: DefId) -> Option<String> {
        let root = DefId { krate: def_id.krate, index: CRATE_DEF_INDEX };
        let (file, _) = self.place(self.tcx.def_span(root))?;
        Some(format!("{file}#{}", self.tcx.def_path(def_id).to_string_no_crate_verbose()))
    }
}

struct Collect<'a, 'tcx> {
    scope: &'a mut Scope<'tcx>,
    results: Option<&'tcx TypeckResults<'tcx>>,
    module: DefId,
    const_context: bool,
    in_body: bool,
    deps: BTreeSet<(String, String)>,
    bdeps: BTreeSet<(String, String)>,
    lookups: BTreeSet<(String, String)>,
    probes: BTreeSet<String>,
    observes: bool,
    includes: bool,
    contexts: HashSet<SyntaxContext>,
}

const POSITION_PATHS: &[&str] = &[
    "core::panic::location::Location",
    "std::panic::set_hook",
    "std::panic::take_hook",
    "std::panic::update_hook",
    "std::panic::PanicHookInfo",
    "std::panicking::set_hook",
    "std::panicking::take_hook",
    "std::panicking::update_hook",
    "std::backtrace::Backtrace",
    "core::panic::panic_info::PanicInfo",
];

impl<'a, 'tcx> Collect<'a, 'tcx> {
    fn new(scope: &'a mut Scope<'tcx>, module: DefId) -> Self {
        Collect {
            scope,
            results: None,
            module,
            const_context: false,
            in_body: false,
            deps: BTreeSet::new(),
            bdeps: BTreeSet::new(),
            lookups: BTreeSet::new(),
            probes: BTreeSet::new(),
            observes: false,
            includes: false,
            contexts: HashSet::new(),
        }
    }

    fn emit_dep(&mut self, relation: &str, def_id: DefId) {
        if let Some(key) = self.scope.key(def_id) {
            let set = if self.in_body { &mut self.bdeps } else { &mut self.deps };
            set.insert((relation.to_string(), key));
        }
    }

    fn observer(&mut self, def_id: DefId) {
        let tcx = self.scope.tcx;
        if !matches!(tcx.crate_name(def_id.krate).as_str(), "core" | "std" | "alloc") {
            return;
        }
        let path = tcx.def_path_str(def_id);
        if POSITION_PATHS.iter().any(|p| path.starts_with(p)) {
            self.observes = true;
        }
    }

    fn def(&mut self, def_id: DefId) {
        let tcx = self.scope.tcx;
        let kind = tcx.def_kind(def_id);
        if !def_id.is_local() {
            self.observer(def_id);
        }
        let (relation, target) = match kind {
            DefKind::Fn | DefKind::AssocFn => (if self.const_context { "cfn" } else { "fn" }, def_id),
            DefKind::Ctor(CtorOf::Struct, _) | DefKind::Variant => ("full", tcx.parent(def_id)),
            DefKind::Ctor(CtorOf::Variant, _) => ("full", tcx.parent(tcx.parent(def_id))),
            DefKind::Field => {
                let parent = tcx.parent(def_id);
                if tcx.def_kind(parent) == DefKind::Variant { ("full", tcx.parent(parent)) } else { ("full", parent) }
            }
            DefKind::Struct
            | DefKind::Enum
            | DefKind::Union
            | DefKind::TyAlias
            | DefKind::Trait
            | DefKind::TraitAlias
            | DefKind::ForeignTy
            | DefKind::Const { .. }
            | DefKind::Static { .. }
            | DefKind::AssocConst { .. }
            | DefKind::AssocTy
            | DefKind::Macro(_) => ("full", def_id),
            DefKind::Impl { .. } => ("head", def_id),
            // The hidden type of `impl Trait` is decided by its function's body.
            DefKind::OpaqueTy => ("full", tcx.parent(def_id)),
            _ => return,
        };
        self.emit_dep(relation, target);
        if matches!(kind, DefKind::AssocFn | DefKind::AssocConst { .. } | DefKind::AssocTy) {
            let container = tcx.parent(def_id);
            match tcx.def_kind(container) {
                DefKind::Impl { .. } => self.emit_dep("head", container),
                DefKind::Trait => {
                    self.emit_dep("full", container);
                    // Generic code reads the selected impl's constant or type, which
                    // no executed body shows: every impl's item is an input.
                    if !matches!(kind, DefKind::AssocFn) {
                        let name = tcx.item_name(def_id);
                        for impl_id in tcx.all_impls(container) {
                            for item in tcx.associated_items(impl_id).filter_by_name_unhygienic(name) {
                                self.emit_dep("full", item.def_id);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn res(&mut self, res: Res) {
        match res {
            Res::Def(_, def_id) => self.def(def_id),
            Res::SelfTyAlias { alias_to, .. } | Res::SelfCtor(alias_to) => self.def(alias_to),
            _ => {}
        }
    }

    fn ty(&mut self, ty: Ty<'tcx>) {
        for arg in ty.walk() {
            let Some(t) = arg.as_type() else { continue };
            match *t.kind() {
                ty::Adt(adt, _) => self.def(adt.did()),
                ty::FnDef(def_id, _) | ty::Foreign(def_id) => self.def(def_id),
                ty::Alias(alias) => self.def(alias.kind.def_id()),
                ty::Dynamic(predicates, ..) => {
                    if let Some(principal) = predicates.principal_def_id() {
                        self.def(principal);
                    }
                }
                _ => {}
            }
        }
    }

    fn span(&mut self, span: Span) {
        let mut span = span;
        while span.from_expansion() {
            if !self.contexts.insert(span.ctxt()) {
                return;
            }
            let data = span.ctxt().outer_expn_data();
            if let ExpnKind::Macro(kind, name) = data.kind {
                if let Some(def_id) = data.macro_def_id {
                    self.def(def_id);
                }
                if kind == MacroKind::Bang {
                    match name.as_str() {
                        "file" | "line" | "column" => self.observes = true,
                        "include" | "include_str" | "include_bytes" => self.includes = true,
                        _ => {}
                    }
                }
            }
            span = data.call_site;
        }
    }

    /// Head of a receiver type for method probing: the nominal type after peeling references.
    fn head(&mut self, ty: Ty<'tcx>) -> (String, String) {
        let mut ty = ty;
        while let ty::Ref(_, inner, _) = *ty.kind() {
            ty = inner;
        }
        let key = |s: &mut Self, d: DefId| s.scope.key(d).unwrap_or_default();
        match *ty.kind() {
            ty::Adt(adt, _) => {
                let name = self.scope.tcx.item_name(adt.did()).to_string();
                (key(self, adt.did()), name)
            }
            ty::Dynamic(predicates, ..) => match predicates.principal_def_id() {
                Some(principal) => {
                    let name = self.scope.tcx.item_name(principal).to_string();
                    (key(self, principal), name)
                }
                None => (String::new(), "*".to_string()),
            },
            ty::Foreign(def_id) => {
                let name = self.scope.tcx.item_name(def_id).to_string();
                (key(self, def_id), name)
            }
            ty::Bool => (String::new(), "bool".into()),
            ty::Char => (String::new(), "char".into()),
            ty::Int(i) => (String::new(), i.name_str().into()),
            ty::Uint(u) => (String::new(), u.name_str().into()),
            ty::Float(f) => (String::new(), f.name_str().into()),
            ty::Str => (String::new(), "str".into()),
            ty::Slice(_) => (String::new(), "[]".into()),
            ty::Array(..) => (String::new(), "[;]".into()),
            ty::Tuple(_) => (String::new(), "()".into()),
            ty::RawPtr(..) => (String::new(), "*ptr".into()),
            ty::FnPtr(..) => (String::new(), "fn".into()),
            ty::Never => (String::new(), "!".into()),
            // A type parameter, projection, opaque or inference variable: any impl may apply.
            _ => (String::new(), "*".into()),
        }
    }

    fn probe(&mut self, name: Symbol, steps: Vec<Ty<'tcx>>, resolved: DefId, id: HirId, by_value: bool) {
        let tcx = self.scope.tcx;
        let in_trait = tcx.def_kind(tcx.parent(resolved)) == DefKind::Trait;
        // An inherent method found at the first probe step is never preempted; a
        // trait method found there only by an inherent item of the same name.
        if !in_trait && by_value {
            return;
        }
        let early = !by_value;
        let mut heads = BTreeSet::new();
        for step in steps {
            heads.insert(self.head(step));
        }
        let mut traits = BTreeSet::new();
        let mut resolved_trait = BTreeSet::new();
        if in_trait {
            let t = tcx.parent(resolved);
            resolved_trait.insert((self.scope.key(t).unwrap_or_default(), tcx.item_name(t).to_string()));
        }
        if let Some(candidates) = tcx.in_scope_traits(id) {
            for candidate in candidates {
                if (!in_trait || candidate.def_id != tcx.parent(resolved))
                    && tcx.associated_items(candidate.def_id).filter_by_name_unhygienic(name).next().is_some()
                {
                    let t = candidate.def_id;
                    traits.insert((self.scope.key(t).unwrap_or_default(), tcx.item_name(t).to_string()));
                }
            }
        }
        self.probes.insert(format!(
            "{{\"name\":{},\"early\":{},\"heads\":{},\"resolved\":{},\"traits\":{}}}",
            js(name.as_str()), early, pairs(&heads), pairs(&resolved_trait), pairs(&traits)
        ));
    }

    fn lookup(&mut self, module: DefId, name: Symbol) {
        if let Some(key) = self.scope.module_key(module) {
            self.lookups.insert((key, name.to_string()));
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Collect<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;
    type MaybeTyCtxt = TyCtxt<'tcx>;

    fn maybe_tcx(&mut self) -> TyCtxt<'tcx> {
        self.scope.tcx
    }

    fn visit_nested_body(&mut self, id: hir::BodyId) {
        let tcx = self.scope.tcx;
        let owner = tcx.hir_body_owner_def_id(id);
        let typeck = !matches!(tcx.def_kind(owner), DefKind::AnonConst);
        let results = if typeck { Some(tcx.typeck_body(id)) } else { self.results };
        let saved = (self.results, self.const_context, self.in_body);
        self.results = results;
        self.const_context = tcx.hir_body_const_context(owner).is_some();
        self.in_body = true;
        let body = tcx.hir_body(id);
        self.visit_body(body);
        (self.results, self.const_context, self.in_body) = saved;
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        self.span(e.span);
        if let Some(results) = self.results {
            if let Some(ty) = results.expr_ty_opt(e) {
                self.ty(ty);
            }
            for adjustment in results.expr_adjustments(e) {
                self.ty(adjustment.target);
            }
            match e.kind {
                hir::ExprKind::MethodCall(segment, receiver, ..) => {
                    if let Some(def_id) = results.type_dependent_def_id(e.hir_id) {
                        self.def(def_id);
                        let adjustments = results.expr_adjustments(receiver);
                        let mut steps = vec![results.expr_ty(receiver)];
                        steps.extend(adjustments.iter().filter(|a| matches!(a.kind, Adjust::Deref(_))).map(|a| a.target));
                        // A reborrow (`&*r`) is the first probe step's by-value candidate.
                        let by_value = results.expr_ty_adjusted(receiver) == results.expr_ty(receiver);
                        self.probe(segment.ident.name, steps, def_id, e.hir_id, by_value);
                    }
                }
                hir::ExprKind::Binary(..) | hir::ExprKind::Unary(..) | hir::ExprKind::AssignOp(..) | hir::ExprKind::Index(..) => {
                    if let Some(def_id) = results.type_dependent_def_id(e.hir_id) {
                        self.def(def_id);
                    }
                }
                _ => {}
            }
        }
        intravisit::walk_expr(self, e);
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        self.span(p.span);
        if let Some(results) = self.results {
            if let Some(ty) = results.node_type_opt(p.hir_id) {
                self.ty(ty);
            }
        }
        intravisit::walk_pat(self, p);
    }

    fn visit_qpath(&mut self, qpath: &'tcx hir::QPath<'tcx>, id: HirId, span: Span) {
        self.span(span);
        if let (hir::QPath::TypeRelative(_, segment), Some(results)) = (qpath, self.results) {
            if let Some(def_id) = results.type_dependent_def_id(id) {
                self.def(def_id);
                if let Some(self_ty) = results.node_args_opt(id).and_then(|args| args.types().next()) {
                    self.probe(segment.ident.name, vec![self_ty], def_id, id, true);
                }
            }
        }
        intravisit::walk_qpath(self, qpath, id);
    }

    fn visit_path(&mut self, path: &hir::Path<'tcx>, _id: HirId) {
        self.span(path.span);
        self.res(path.res);
        let segments = path.segments;
        if let Some(first) = segments.first() {
            let name = first.ident.name;
            // Locals, generic parameters and `Self` resolve lexically, never in a module.
            let lexical = matches!(
                first.res,
                Res::Local(_)
                    | Res::SelfTyParam { .. }
                    | Res::SelfTyAlias { .. }
                    | Res::SelfCtor(_)
                    | Res::Def(DefKind::TyParam | DefKind::ConstParam | DefKind::LifetimeParam, _)
            );
            if !lexical
                && !path.is_global()
                && ![kw::Crate, kw::SelfLower, kw::Super, kw::SelfUpper, kw::PathRoot, kw::DollarCrate].contains(&name)
            {
                self.lookup(self.module, name);
            }
        }
        for pair in segments.windows(2) {
            if let Res::Def(DefKind::Mod, module) = pair[0].res {
                self.lookup(module, pair[1].ident.name);
            }
        }
        for segment in segments {
            self.res(segment.res);
        }
        intravisit::walk_path(self, path);
    }
}

fn emit(line: String) {
    use std::io::Write;
    if let Ok(directory) = std::env::var("ANTECEDENT_ANALYZER_OUTPUT") {
        let path = std::path::Path::new(&directory).join(format!("{}.jsonl", std::process::id()));
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        file.write_all(line.as_bytes()).unwrap();
    } else {
        print!("{line}");
    }
}

fn item_module(tcx: TyCtxt<'_>, id: LocalDefId) -> DefId {
    tcx.parent_module_from_def_id(id).to_def_id()
}

impl Callbacks for Inventory {
    fn after_analysis<'tcx>(&mut self, _compiler: &interface::Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        // SAFETY of the borrow: `paths` outlives the analysis of this crate.
        let paths: &'tcx Paths = unsafe { &*(&self.paths as *const Paths) };
        let mut scope = Scope { tcx, paths, files: BTreeMap::new() };
        let mut out = String::new();
        // Modules: the file a module's items are in and its inline `mod` chain there.
        let modules = std::iter::once(rustc_hir::def_id::CRATE_DEF_ID)
            .chain(tcx.hir_crate_items(()).definitions().filter(|d| tcx.def_kind(*d) == DefKind::Mod));
        for module in modules {
            let inner = tcx.hir_get_module(rustc_hir::def_id::LocalModDefId::new_unchecked(module)).0.spans.inner_span;
            let Some((file, _)) = scope.place(inner) else { continue };
            let Some(key) = scope.module_key(module.to_def_id()) else { continue };
            let mut chain = Vec::new();
            let mut current = module;
            let mut parent_key = None;
            while current != rustc_hir::def_id::CRATE_DEF_ID {
                let parent = tcx.parent_module_from_def_id(current).to_local_def_id();
                let parent_inner = tcx.hir_get_module(rustc_hir::def_id::LocalModDefId::new_unchecked(parent)).0.spans.inner_span;
                if parent_key.is_none() {
                    parent_key = scope.module_key(parent.to_def_id());
                }
                let same_file = scope.place(parent_inner).map(|(f, _)| f) == Some(file.clone());
                let own_inner = tcx.hir_get_module(rustc_hir::def_id::LocalModDefId::new_unchecked(current)).0.spans.inner_span;
                let own_file = scope.place(own_inner).map(|(f, _)| f);
                if own_file != Some(file.clone()) || !same_file {
                    break;
                }
                chain.push(tcx.item_name(current.to_def_id()).to_string());
                current = parent;
            }
            chain.reverse();
            let chain: Vec<String> = chain.iter().map(|c| js(c)).collect();
            let name = if module == rustc_hir::def_id::CRATE_DEF_ID { String::new() } else { tcx.item_name(module.to_def_id()).to_string() };
            out.push_str(&format!(
                "{{\"t\":\"mod\",\"key\":{},\"file\":{},\"chain\":[{}],\"parent\":{},\"name\":{}}}\n",
                js(&key), js(&file), chain.join(","),
                parent_key.map(|k| js(&k)).unwrap_or_else(|| "null".into()), js(&name)
            ));
        }
        for id in tcx.hir_crate_items(()).free_items() {
            let item = tcx.hir_item(id);
            if let hir::ItemKind::Use(path, hir::UseKind::Glob) = item.kind {
                let module = item_module(tcx, item.owner_id.def_id);
                for res in path.res.iter() {
                    if let Some(Res::Def(DefKind::Mod | DefKind::Enum | DefKind::Trait, source)) = res {
                        if let (Some(m), Some(s)) = (scope.module_key(module), scope.module_key(*source)) {
                            out.push_str(&format!("{{\"t\":\"glob\",\"mod\":{},\"src\":{}}}\n", js(&m), js(&s)));
                        }
                    }
                }
            }
        }
        for id in tcx.hir_crate_items(()).owners() {
            let def_id = id.def_id;
            let kind = tcx.def_kind(def_id);
            if matches!(kind, DefKind::Mod | DefKind::ExternCrate | DefKind::GlobalAsm | DefKind::ForeignMod) {
                continue;
            }
            let Some(key) = scope.key(def_id.to_def_id()) else { continue };
            let module = item_module(tcx, def_id);
            let mut collect = Collect::new(&mut scope, module);
            match tcx.hir_owner_node(id) {
                hir::OwnerNode::Item(item) => collect.visit_item(item),
                hir::OwnerNode::ImplItem(item) => collect.visit_impl_item(item),
                hir::OwnerNode::TraitItem(item) => collect.visit_trait_item(item),
                hir::OwnerNode::ForeignItem(item) => collect.visit_foreign_item(item),
                _ => continue,
            }
            // An associated item's container; an impl's associated constants and types.
            if matches!(kind, DefKind::AssocFn | DefKind::AssocConst { .. } | DefKind::AssocTy) {
                let container = tcx.parent(def_id.to_def_id());
                let relation = if tcx.def_kind(container) == DefKind::Trait { "full" } else { "head" };
                collect.emit_dep(relation, container);
            }
            if let DefKind::Impl { .. } = kind {
                for item in tcx.associated_items(def_id).in_definition_order() {
                    if !matches!(tcx.def_kind(item.def_id), DefKind::AssocFn) {
                        collect.emit_dep("full", item.def_id);
                    }
                }
            }
            let probes: Vec<String> = collect.probes.iter().cloned().collect();
            let module_key = collect.scope.module_key(module).unwrap_or_default();
            let line = format!(
                "{{\"t\":\"item\",\"key\":{},\"kind\":{},\"mod\":{},\"deps\":{},\"bdeps\":{},\"lookups\":{},\"probes\":[{}],\"observes\":{},\"includes\":{}}}\n",
                js(&key), js(&format!("{kind:?}")), js(&module_key), pairs(&collect.deps), pairs(&collect.bdeps),
                pairs(&collect.lookups), probes.join(","), collect.observes, collect.includes
            );
            out.push_str(&line);
        }
        emit(out);
        Compilation::Continue
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| std::path::Path::new(a).file_stem().is_some_and(|s| s == "rustc")) {
        args.remove(1);
    }
    run_compiler(&args, &mut Inventory { paths: Paths::new() });
}
