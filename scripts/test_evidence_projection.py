"""Engineering mutation matrix for compiler-selected source body projections."""
import unittest

from evidence_projection import ProjectionChanged, capture, from_compiler_coverage, matches


class GuardedSourceTests(unittest.TestCase):
    SOURCE = '''macro_rules! scale { () => { 2 } }
const C: i32 = 2;
trait T { fn value(&self) -> i32; }
struct S;
impl T for S { fn value(&self) -> i32 { 2 } }
fn used() -> i32 { scale!() + C }
fn unused() -> i32 { 7 }
'''

    def projection(self, source=None):
        source = self.SOURCE if source is None else source
        start = source.index('{ 7 }')
        return capture(source, [(start, start + len('{ 7 }'))])

    def test_unrelated_same_file_body_edit_retains_projection(self):
        self.assertTrue(matches(self.SOURCE.replace('{ 7 }', '{ 8 }'), self.projection()))

    def test_selection_and_compile_time_inputs_are_guarded(self):
        changes = [
            ('=> { 2 }', '=> { 3 }'), ('const C: i32 = 2', 'const C: i32 = 3'),
            ('struct S;', 'struct S { x: i32 }'), ('impl T for S', 'impl Other for S'),
            ('fn unused()', 'fn unused(x: i32)'), ('scale!() + C', 'scale!() - C'),
            ('fn value(&self)', 'fn value(&mut self)'),
        ]
        for old, new in changes:
            with self.subTest(old=old):
                self.assertFalse(matches(self.SOURCE.replace(old, new), self.projection()))
        for appended in ('impl Other for S {}', 'use elsewhere::scale;', 'mod new;',
                         '#[cfg(feature="new")] fn added() {}', 'const NEW: i32 = 1;'):
            self.assertFalse(matches(self.SOURCE + appended, self.projection()))

    def test_new_scope_changing_constructs_cannot_hide_in_body(self):
        for body in ('{ impl T for S {} 7 }', '{ #[attr] 7 }', '{ export!(); 7 }',
                     '{ macro_rules! a { () => {} } 7 }', '{ const C:i32=7; C }',
                     '{ unsafe { 7 } }', '{ extern "C" { fn f(); } 7 }',
                     '{ fn nested() {} 7 }'):
            with self.subTest(body=body), self.assertRaises(ProjectionChanged):
                matches(self.SOURCE.replace('{ 7 }', body), self.projection())

    def test_layout_change_cannot_silently_change_later_line_column_macros(self):
        self.assertFalse(matches(self.SOURCE.replace('{ 7 }', '{\n7\n}'), self.projection()))
        self.assertFalse(matches(self.SOURCE.replace('{ 7 }', '{ 1000 }'), self.projection()))

    def test_literals_comments_and_nested_braces_cannot_break_body_boundaries(self):
        for body in ('{ let s="}"; 7 }', '{ /* } */ 7 }', '{ if true { 7 } else { 0 } }',
                     '{ let s=r#"} impl T"#; 7 }'):
            source = self.SOURCE.replace('{ 7 }', body)
            start = source.index(body)
            projection = capture(source, [(start, start + len(body))])
            self.assertTrue(matches(source, projection))
            self.assertFalse(matches(source + 'impl New for S {}', projection))

    def test_multiple_holes_preserve_intervening_guards(self):
        source = 'fn a() { 1 }\nfn b() { 2 }\nconst C:u8=1;\n'
        spans = [(source.index('{ 1 }'), source.index('{ 1 }')+5),
                 (source.index('{ 2 }'), source.index('{ 2 }')+5)]
        projection = capture(source, spans)
        self.assertTrue(matches(source.replace('{ 1 }','{ 3 }').replace('{ 2 }','{ 4 }'), projection))
        self.assertFalse(matches(source.replace('fn b()', 'fn c()'), projection))

    def test_invalid_and_overlapping_spans_are_rejected(self):
        with self.assertRaises(ProjectionChanged):
            capture(self.SOURCE, [(0, 2)])
        start = self.SOURCE.index('{ 7 }')
        with self.assertRaises(ProjectionChanged):
            capture(self.SOURCE, [(start,start+5), (start,start+5)])

    def test_unclosed_or_missing_body_is_rejected(self):
        with self.assertRaises(ProjectionChanged):
            matches(self.SOURCE.replace('{ 7 }','{ 7 '), self.projection())

    def test_coverage_admission_is_conservative_and_opaque_by_default(self):
        source = 'fn a() { 1 }\nfn b() { 7 }\n'
        first = source.index('{ 1 }')
        second = source.index('{ 7 }')
        bodies = {'a': (first, first+5), 'b': (second, second+5)}
        projection, reasons = from_compiler_coverage(source, bodies, {1})
        self.assertFalse(projection['layouts'])
        self.assertTrue(reasons)
        projection, _ = from_compiler_coverage(source, bodies, {1}, opaque=False)
        self.assertEqual(len(projection['layouts']), 1)
        self.assertTrue(matches(source.replace('{ 7 }','{ 8 }'), projection))
        self.assertFalse(matches(source.replace('{ 1 }','{ 2 }'), projection))

    def test_missing_coverage_and_unknown_attributes_retain_complete_source(self):
        source = '#[opaque]\nfn b() { 7 }\n'
        start = source.index('{ 7 }')
        for covered in (set(), {1}):
            projection, reasons = from_compiler_coverage(source, {'b':(start,start+5)}, covered, opaque=False)
            self.assertFalse(projection['layouts'])
            self.assertTrue(reasons)

    def test_scoped_derive_does_not_make_unrelated_function_opaque(self):
        source = '#[derive(serde::Serialize)]\nstruct S { x: u8 }\nfn unused() { 7 }\n'
        start = source.index('{ 7 }')
        opaque, _ = from_compiler_coverage(source, {'unused': (start, start + 5)}, set(),
                                            opaque=False, coverage_complete=True)
        self.assertFalse(opaque['layouts'])
        projection, _ = from_compiler_coverage(source, {'unused': (start, start + 5)}, set(),
                                                opaque=False, coverage_complete=True, scoped_derives=True)
        self.assertEqual(len(projection['layouts']), 1)
        self.assertTrue(matches(source.replace('{ 7 }', '{ 8 }'), projection))
        self.assertFalse(matches(source.replace('x: u8', 'x: u16'), projection))

    def test_function_and_enclosing_macro_attributes_stay_opaque(self):
        for source in ('#[instrument]\nfn unused() { 7 }\n',
                       '#[instrument]\nmod m { fn unused() { 7 } }\n',
                       '#[derive(custom)]\nfn unused() { 7 }\n'):
            start = source.index('{ 7 }')
            projection, _ = from_compiler_coverage(source, {'unused': (start, start + 5)}, set(),
                                                   opaque=False, coverage_complete=True, scoped_derives=True)
            self.assertFalse(projection['layouts'])

    def test_scoped_derive_helper_attributes_belong_to_their_item(self):
        source = ('#[derive(serde::Serialize)]\n#[serde(rename_all = "lowercase")]\n'
                  'enum E { #[serde(rename = "a")] A, B }\nstruct T(#[allow(dead_code)] u8);\nfn unused() { 7 }\n')
        start = source.index('{ 7 }')
        projection, _ = from_compiler_coverage(source, {'unused': (start, start + 5)}, set(),
                                               opaque=False, coverage_complete=True, scoped_derives=True)
        self.assertEqual(len(projection['layouts']), 1)
        self.assertFalse(matches(source.replace('rename = "a"', 'rename = "b"'), projection))

    def test_body_inside_macro_input_is_retained(self):
        for source in ('seq!(N in 0..2 { fn unused() { 7 } });\n', 'm! { fn unused() { 7 } }\n',
                       'macro_rules! m { () => { fn unused() { 7 } } }\n'):
            start = source.index('{ 7 }')
            projection, reasons = from_compiler_coverage(source, {'unused': (start, start + 5)}, set(),
                                                         opaque=False, coverage_complete=True)
            self.assertFalse(projection['layouts'], source)
            self.assertTrue(reasons)

    def test_hole_layout_preserves_every_later_position(self):
        # Rust embeds caller locations in unwrap/indexing/overflow panics and in
        # line!/column!/file!: no lexical check can prove later code position
        # independent, so a hole may never move a later line or column.
        source = 'fn unused() { 7 }\nfn used() { v[0].unwrap() }\n'
        start = source.index('{ 7 }')
        projection, _ = from_compiler_coverage(source, {'unused': (start, start + 5)}, {2},
                                               opaque=False)
        self.assertEqual(projection['version'], 1)
        self.assertTrue(matches(source.replace('{ 7 }', '{ 8 }'), projection))
        self.assertFalse(matches(source.replace('{ 7 }', '{\n7\n}'), projection))
        self.assertFalse(matches(source.replace('{ 7 }', '{ 77 }'), projection))
        # Same byte length, fewer characters: panic columns count characters.
        changed = source.replace('{ 7 }', '{ é}')
        self.assertEqual(len(changed.encode()), len(source.encode()))
        self.assertFalse(matches(changed, projection))


if __name__ == '__main__':
    unittest.main()
