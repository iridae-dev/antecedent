"""What a measurement can enter: regions, macro expansion, reachability, and the selection.

The inventories are written by hand in the compiler driver's format, so these checks
need no compiler.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evidence_items as ei
import evidence_objects
import evidence_manifests as em
import evidence_reach as er
import rust_items as ri
import rust_macros

LIB = "crates/c/src/lib.rs"
ROUTES = "crates/c/src/routes.rs"
TEST = "crates/c/tests/t.rs"

LIB_SOURCE = """pub mod routes;

pub enum Query { Alpha(AlphaQuery), Beta(BetaQuery) }
pub struct AlphaQuery { pub x: f64 }
pub struct BetaQuery { pub y: f64 }

#[derive(Clone, Copy)]
pub enum Route { Alpha, Beta }

pub fn classify(query: &Query) -> Route {
    match query {
        Query::Alpha(_) => Route::Alpha,
        Query::Beta(_) => Route::Beta,
    }
}

pub fn run(query: &Query) -> f64 {
    match classify(query) {
        Route::Alpha => routes::alpha(query),
        Route::Beta => routes::beta(query),
    }
}
"""
ROUTES_SOURCE = """use crate::Query;

pub fn alpha(query: &Query) -> f64 {
    let Query::Alpha(q) = query else { return 0.0 };
    q.x + shared(1.0)
}

pub fn beta(query: &Query) -> f64 {
    let Query::Beta(q) = query else { return 0.0 };
    invert(q.y)
}

fn invert(y: f64) -> f64 { 1.0 / y }

pub fn shared(v: f64) -> f64 { v }
"""
TEST_SOURCE = """use c::{AlphaQuery, Query};

#[test]
fn measures_alpha() {
    let query = Query::Alpha(AlphaQuery { x: 1.0 });
    assert!(c::run(&query) > 0.0);
}
"""
SOURCES = {LIB: LIB_SOURCE, ROUTES: ROUTES_SOURCE, TEST: TEST_SOURCE}


def key(rel: str, needle: str) -> str:
    """A driver key: the byte offset of an item's name."""
    source = SOURCES[rel]
    return f"{rel}:{len(source[:source.index(needle)].encode())}"


def inventory() -> ei.Inventory:
    lib, routes, test = f"{LIB}#", f"{LIB}#::routes", f"{TEST}#"
    query, alpha_q, beta_q = key(LIB, "Query {"), key(LIB, "AlphaQuery {"), key(LIB, "BetaQuery {")
    route, classify, run = key(LIB, "Route {"), key(LIB, "classify"), key(LIB, "run(")
    alpha, beta = key(ROUTES, "alpha"), key(ROUTES, "beta")
    invert, shared = key(ROUTES, "invert(y"), key(ROUTES, "shared(v")

    def item(k, mod, deps=(), bdeps=()):
        return {"t": "item", "key": k, "kind": "Fn", "mod": mod, "deps": list(deps), "bdeps": list(bdeps),
                "lookups": [], "probes": [], "observes": False, "includes": False}
    records = [
        {"t": "mod", "key": lib, "file": LIB, "chain": [], "parent": None, "name": ""},
        {"t": "mod", "key": routes, "file": ROUTES, "chain": [], "parent": lib, "name": "routes"},
        {"t": "mod", "key": test, "file": TEST, "chain": [], "parent": None, "name": ""},
        item(key(TEST, "measures_alpha"), test, bdeps=[["fn", run], ["full", query], ["full", alpha_q]]),
        item(run, lib, deps=[["full", query]],
             bdeps=[["fn", classify], ["fn", alpha], ["fn", beta], ["full", route]]),
        item(classify, lib, deps=[["full", query], ["full", route]]),
        item(alpha, routes, deps=[["full", query]], bdeps=[["fn", shared]]),
        item(beta, routes, deps=[["full", query]], bdeps=[["fn", invert]]),
        item(invert, routes), item(shared, routes), item(route, lib), item(alpha_q, lib), item(beta_q, lib),
        item(query, lib, deps=[["full", alpha_q], ["full", beta_q]]),
    ]
    return ei.Inventory.read(json.dumps(r) for r in records)


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        for rel, text in (*SOURCES.items(), ("crates/c/Cargo.toml", '[package]\nname = "c"\n')):
            self.write(rel, text)
        ei._PARSED.clear()
        em._FILE_DIGESTS.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        em._FILE_DIGESTS.clear()

    def earn(self, *, narrow, executed=(("measures_alpha", TEST, "assert!"), ("run", LIB, "match classify"),
                                        ("classify", LIB, "match query"), ("alpha", ROUTES, "q.x +"),
                                        ("shared", ROUTES, "{ v }"))):
        selection = ei.Selection(self.root, inventory(), callee_bodies=True, narrow=narrow)
        for _, rel, needle in executed:
            selection.execute(rel, SOURCES[rel].index(needle))
        entries, tables = selection.entries(em.Checker(self.root, {}), ["crates/c"], "crates/c")
        evidence_objects.write(self.root, tables)
        return selection, entries

    def current(self, entries):
        checker = em.Checker(self.root, {})
        names = next(e for e in entries if e["kind"] == "rust_names")
        return all(checker.items.items_current(e, names) for e in entries if e["kind"] == "rust_items") and all(
            checker.items.guards_current(e) for e in entries if e["kind"] == "rust_guards")


def body(source: str, name: str, self_type=None) -> er.Body:
    file = ri.File(source)
    node = next(n for n in er._walk(file.root) if n.kind == "fn" and n.name == name)
    return er.Body(file, node, self_type or er.self_type(node))


class RegionTests(unittest.TestCase):
    def test_a_match_arm_is_gated_by_the_variants_its_pattern_requires(self):
        found = body(LIB_SOURCE, "run")
        gates = [region.gate for region in found.regions[1:]]
        self.assertEqual(gates, [(frozenset({("Route", "Alpha")}),), (frozenset({("Route", "Beta")}),)])
        self.assertIn("alpha", found.regions[1].idents)
        self.assertNotIn("beta", found.regions[0].idents | found.regions[1].idents)
        # A pattern tests a value: it mentions no constructor.
        self.assertEqual(found.regions[0].variants, set())

    def test_an_alternation_requires_only_what_every_side_does(self):
        found = body("fn f(a: A, b: B) { match (a, b) { (A::X | A::Y, B::Z(_)) => g(), _ => h() } }", "f")
        self.assertEqual(found.regions[1].gate, (frozenset({("B", "Z")}),))
        self.assertEqual(found.regions[2].gate, (frozenset(),))

    def test_a_let_else_gates_the_statements_after_it(self):
        found = body(ROUTES_SOURCE, "beta")
        self.assertEqual(found.regions[1].gate, (frozenset({("Query", "Beta")}),))
        self.assertIn("invert", found.regions[1].idents)
        self.assertNotIn("invert", found.regions[0].idents)

    def test_a_let_chain_gates_its_block_and_a_matches_pattern_builds_nothing(self):
        source = ("fn f(q: &Q, on: bool) { if on && let Q::A(a) = q && a.ready() { g(a) } "
                  "if matches!(q, Q::B(_) | Q::C) { h() } if on || matches!(q, Q::D) { i() } }")
        found = body(source, "f")
        self.assertEqual(found.regions[1].gate, (frozenset({("Q", "A")}),))
        self.assertEqual({v for region in found.regions for v in region.variants if v[0] == "Q"}, set())

    def test_an_expression_mentions_the_variant_it_builds(self):
        found = body("fn f(x: u8) -> Q { if x == 0 { Q::A(x) } else { other(Q::B) } }", "f")
        self.assertEqual(found.regions[0].variants & {("Q", "A"), ("Q", "B")}, {("Q", "A"), ("Q", "B")})

    def test_an_arm_this_module_cannot_split_closes_nothing(self):
        # A comma inside generic arguments ends the arm early: the match is left whole.
        found = body("fn f(q: Q) -> u8 { match q { Q::A => g::<u8, u8>(), Q::B => 1 } }", "f")
        self.assertEqual(len(found.regions), 1)
        self.assertIn(("Q", "A"), found.regions[0].variants)

    def test_an_impl_inside_a_body_makes_the_whole_body_count(self):
        found = body("fn f(q: Q) -> u8 { impl Local { fn make() -> Self { Self { n: Q::B } } } "
                     "match q { Q::A => 1, _ => 2 } }", "f")
        self.assertEqual(len(found.regions), 1)
        self.assertIn("Local", found.regions[0].builds)

    def test_parameters_name_the_structs_a_caller_must_hold(self):
        source = ("struct S; impl S { fn f(&self, a: &crate::q::A, b: Option<B>, c: Box<C>, d: D<'_>, e: impl E) {} }")
        # `Option<B>` needs no `B`; a generic type is named by its head.
        self.assertEqual(body(source, "f").gate_types, ["S", "A", "Option", "C", "D"])


class TypeTests(unittest.TestCase):
    def types(self, source: str) -> er.Types:
        file = ri.File(source)

        def expanded(_, node):
            definitions = [file.kept[n.start:n.end] for n in er._walk(file.root)
                           if n.kind == "macro_rules" and n.name == node.name]
            text = rust_macros.expand(definitions[0], file.kept[node.start:node.end]) if definitions else None
            return ri.File(text) if text else None
        return er.Types.read({"crates/c/src/lib.rs": file}, expanded)

    def test_only_source_built_types_can_close_a_region(self):
        found = self.types(
            "#[derive(Clone, Debug)] enum A { X, Y(u8) }\n"
            "#[derive(Deserialize)] enum B { X }\n"       # built by a derive
            "enum C { #[default] X, Y }\n"                # built by `Default`
            "#[repr(u8)] enum D { X }\n"                  # built by a cast
            "struct S { a: u8 }\nstruct Unit;\n#[derive(Default)] struct T { a: u8 }\n")
        self.assertEqual(found.enums, {"A": {"X", "Y"}})
        self.assertEqual(found.structs, {"S"})

    def test_a_second_name_is_what_it_renames(self):
        found = self.types("enum A { X }\nuse a::A as K;\nstruct S { a: u8 }\ntype Alias<'a> = S;\n"
                           "struct P { a: u8 }\ntype F = fn(&P) -> u8;\n")
        self.assertEqual(found.renames["K"], {"A"})
        self.assertEqual(found.renames["Alias"], {"S"})
        self.assertNotIn("K", found.nominal)
        self.assertIn("P", found.structs)  # a function type names no constructor

    def test_declarations_inside_a_function_body_count(self):
        found = self.types("enum A { X }\nenum B { X }\nstruct S { a: u8 }\n"
                           "fn f() { use crate::A as Kind; let _ = Kind::X; }\n"
                           "fn g() { struct S { b: u8 } let _ = S { b: 1 }; }\n"
                           "fn h() { type Short = B; let _ = Short::X; }\n")
        self.assertEqual(found.renames["Kind"], {"A"})
        self.assertEqual(found.renames["Short"], {"B"})
        self.assertNotIn("S", found.structs)   # two types of one name
        self.assertIn("A", found.enums)

    def test_an_alias_of_a_generic_struct_holds_the_arguments_it_stores(self):
        found = self.types("struct Op<'a, Q, E = ()> { query: Q, extra: Option<E>, name: &'a str }\n"
                           "struct AQ { a: u8 }\nstruct BQ { b: u8 }\nstruct Extra { c: u8 }\n"
                           "type AOp<'a> = Op<'a, AQ, Extra>;\ntype BOp = Op<'static, Vec<BQ>>;\n")
        self.assertEqual(found.contains, {"AOp": {"AQ"}})   # `Extra` is behind an `Option`

    def test_a_definition_a_macro_writes_is_read_from_its_expansion(self):
        source = ("macro_rules! table { ($($v:ident($p:ty)),+ $(,)?) => { pub enum Q { $($v($p),)+ } "
                  "$(impl From<$p> for Q { fn from(q: $p) -> Self { Self::$v(q) } })+ }; }\n"
                  "table! { A(AQ), B(BQ) }\nstruct AQ { a: u8 }\nstruct BQ { b: u8 }\n")
        self.assertEqual(self.types(source).enums["Q"], {"A", "B"})
        # Without the expansion the enum is open.
        self.assertNotIn("Q", er.Types.read([ri.File(source)]).enums)


class MacroTests(unittest.TestCase):
    DEFINITION = ("macro_rules! table { ($(#[$m:meta])* $name:ident { $($v:ident => $s:literal),+ $(,)? }) => { "
                  "$(#[$m])* pub enum $name { $($v,)+ } "
                  "impl $name { pub const ALL: &[$name] = &[$(Self::$v),+]; "
                  "pub fn name(self) -> &'static str { match self { $(Self::$v => $s),+ } } } }; }")

    def test_a_table_macro_expands_to_its_items(self):
        text = rust_macros.expand(self.DEFINITION, '#[derive(Clone)] table! { Kind { A => "a", B => "b", } }')
        file = ri.File(text)
        self.assertEqual(list(file.by_locator),
                         ["enum:Kind#0", "impl:Kind#0", "impl:Kind#0/const:ALL#0", "impl:Kind#0/fn:name#0"])
        self.assertIn('Self :: B => "b"', text)

    def test_the_first_matching_rule_is_used(self):
        definition = "macro_rules! m { (@one $a:ident) => { fn $a() {} }; ($($a:ident),*) => { $(struct $a;)* }; }"
        self.assertEqual(rust_macros.expand(definition, "m!(@one f);"), "fn f ( ) { }")
        self.assertEqual(rust_macros.expand(definition, "m!(A, B);"), "struct A ; struct B ;")

    def test_what_is_not_expanded_exactly_is_not_expanded(self):
        self.assertIsNone(rust_macros.expand(self.DEFINITION, "table! { Kind { A => b } }"))       # no rule matches
        self.assertIsNone(rust_macros.expand("macro_rules! m { ($a:ident) => { ${concat($a, _x)} }; }", "m!(a);"))
        self.assertIsNone(rust_macros.expand("macro_rules! m { ($($a:ident),*; $($b:ident),*) => { $($a $b)* }; }",
                                             "m!(a, b; c);"))                                 # lengths disagree


class ReachTests(Workspace):
    def graph(self) -> er.Graph:
        return er.Graph(lambda rel: ei.parse(self.root / rel) if (self.root / rel).is_file() else None, inventory())

    def test_a_route_whose_variant_is_never_built_is_not_entered(self):
        graph = self.graph()
        found = er.reach(graph, {(TEST, "fn:measures_alpha#0")})
        entered = {locator for _, locator in found.entered}
        self.assertEqual(entered, {"fn:measures_alpha#0", "fn:run#0", "fn:classify#0", "fn:alpha#0", "fn:shared#0"})
        self.assertFalse(found.follows((LIB, "fn:run#0"), (ROUTES, "fn:beta#0")))
        self.assertTrue(found.follows((LIB, "fn:run#0"), (ROUTES, "fn:alpha#0")))
        self.assertIn(("Route", "Alpha"), found.variants)
        self.assertNotIn(("Route", "Beta"), found.variants)

    def test_building_the_variant_anywhere_that_runs_opens_the_route(self):
        self.edit(TEST, "assert!(c::run(&query) > 0.0);",
                  "assert!(c::run(&query) > 0.0);\n    let _ = Query::Beta(c::BetaQuery { y: 2.0 });")
        ei._PARSED.clear()
        found = er.reach(self.graph(), {(TEST, "fn:measures_alpha#0")})
        self.assertIn((ROUTES, "fn:invert#0"), found.entered)

    def test_a_function_is_entered_only_when_its_parameter_can_exist(self):
        source = ("pub struct Plan { pub n: u8 }\npub fn make() -> u8 { 1 }\n"
                  "pub fn with_plan(plan: &Plan) -> u8 { plan.n }\npub fn top() -> u8 { make() }\n")
        self.write("crates/c/src/p.rs", source)

        def at(needle):
            return f"crates/c/src/p.rs:{source.index(needle)}"
        records = [{"t": "mod", "key": "crates/c/src/p.rs#", "file": "crates/c/src/p.rs", "chain": [], "parent": None,
                    "name": ""}]
        def graph(resolved: bool) -> er.Graph:
            rows = list(records)
            for name, deps, bdeps in (("make()", [], []), ("Plan {", [], []),
                                      ("with_plan", [["full", at("Plan {")]] if resolved else [], []),
                                      ("top()", [], [["fn", at("make()")], ["fn", at("with_plan")]])):
                rows.append({"t": "item", "key": at(name), "kind": "Fn", "mod": "crates/c/src/p.rs#", "deps": deps,
                             "bdeps": bdeps, "lookups": [], "probes": [], "observes": False, "includes": False})
            return er.Graph(lambda rel: ei.parse(self.root / rel), ei.Inventory.read(json.dumps(r) for r in rows))
        # A parameter named `Plan` that the compiler did not resolve to the workspace's
        # `Plan` (another crate's type of that name) closes nothing.
        unresolved = er.reach(graph(False), {("crates/c/src/p.rs", "fn:top#0")})
        self.assertIn(("crates/c/src/p.rs", "fn:with_plan#0"), unresolved.entered)
        graph = graph(True)
        found = er.reach(graph, {("crates/c/src/p.rs", "fn:top#0")})
        self.assertEqual({locator for _, locator in found.entered}, {"fn:top#0", "fn:make#0"})
        # A function a run executed is entered whatever the analysis finds.
        again = er.reach(graph, {("crates/c/src/p.rs", "fn:top#0")}, executed={("crates/c/src/p.rs", "fn:with_plan#0")})
        self.assertIn(("crates/c/src/p.rs", "fn:with_plan#0"), again.entered)


class SelectionTests(Workspace):
    def test_a_sampled_run_binds_every_named_function_unless_narrowed(self):
        broad, _ = self.earn(narrow=False)
        self.assertIn("full", broad.parts[ROUTES]["fn:invert#0"])
        narrowed, _ = self.earn(narrow=True)
        self.assertNotIn("fn:invert#0", narrowed.parts[ROUTES])
        self.assertNotIn("fn:beta#0", narrowed.parts[ROUTES])
        self.assertIn("full", narrowed.parts[ROUTES]["fn:alpha#0"])
        self.assertIn("full", narrowed.parts[ROUTES]["fn:shared#0"])
        # Every part the narrowed selection takes, the broad one takes too.
        for rel, parts in narrowed.parts.items():
            for locator, chosen in parts.items():
                self.assertTrue(chosen <= broad.parts[rel][locator] or "full" in broad.parts[rel][locator])

    def test_only_the_route_measured_owes_the_record(self):
        _, entries = self.earn(narrow=True)
        self.edit(ROUTES, "{ 1.0 / y }", "{ 2.0 / y }")           # the other route's function
        self.edit(ROUTES, "invert(q.y)", "invert(q.y) + 1.0")      # the other route's entry
        self.assertTrue(self.current(entries))
        self.edit(ROUTES, "pub fn shared(v: f64) -> f64 { v }", "pub fn shared(v: f64) -> f64 { v + 0.0 }")
        self.assertFalse(self.current(entries))

    def test_executed_code_the_analysis_closed_withdraws_the_narrowing(self):
        # A run that executed the other route (however its variant came to be) is believed.
        selection, _ = self.earn(narrow=True, executed=(
            ("measures_alpha", TEST, "assert!"), ("run", LIB, "match classify"), ("beta", ROUTES, "invert(q.y)")))
        self.assertIsNone(selection.reach)
        self.assertIn("full", selection.parts[ROUTES]["fn:invert#0"])
        self.assertTrue(any(reason.startswith("narrowing withdrawn") for reason in selection.reasons))


class IncludeTests(unittest.TestCase):
    def test_an_included_file_is_selected_by_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lib, helpers = "crates/c/src/lib.rs", "crates/c/src/helpers.rs"
            lib_source = 'pub fn top() -> u8 { used() }\n\ninclude!("helpers.rs");\n'
            helpers_source = "fn used() -> u8 { 1 }\n\nfn idle() -> u8 { 2 }\n"
            for rel, text in ((lib, lib_source), (helpers, helpers_source), ("crates/c/Cargo.toml", "[package]\n")):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text, encoding="utf-8")
            ei._PARSED.clear()
            module = f"{lib}#"
            records = [{"t": "mod", "key": module, "file": lib, "chain": [], "parent": None, "name": ""}]
            for rel, source, needle, bdeps in ((lib, lib_source, "top", [["fn", f"{helpers}:{helpers_source.index('used')}"]]),
                                               (helpers, helpers_source, "used", []), (helpers, helpers_source, "idle", [])):
                records.append({"t": "item", "key": f"{rel}:{source.index(needle)}", "kind": "Fn", "mod": module,
                                "deps": [], "bdeps": bdeps, "lookups": [[module, "used"]], "probes": [],
                                "observes": False, "includes": False})
            selection = ei.Selection(root, ei.Inventory.read(json.dumps(r) for r in records))
            selection.execute(lib, lib_source.index("used()"))
            selection.execute(helpers, helpers_source.index("{ 1 }"))
            entries, _ = selection.entries(em.Checker(root, {}), ["crates/c"], "crates/c")
            self.assertEqual(selection.whole, {})
            self.assertEqual(set(selection.parts[helpers]), {"fn:used#0"})
            self.assertIn("full", selection.parts[helpers]["fn:used#0"])
            self.assertIn("full", selection.parts[lib]["macro:include#0"])
            helper_entry = next(e for e in entries if e.get("path") == helpers)
            self.assertEqual(helper_entry["kind"], "rust_items")
            self.assertEqual(helper_entry["scopes"], [[[], False]])  # its names resolve in the including module


if __name__ == "__main__":
    unittest.main()
