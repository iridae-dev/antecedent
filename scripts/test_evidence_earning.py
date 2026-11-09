"""Production manifest integration using disposable engineering workspaces."""
import subprocess
import unittest
from unittest.mock import patch

import calibration_status as calibration
import scientific_records as scientific
from evidence_earning import Snapshot, binding, measurement_binding, probe_binding
from evidence_fingerprint import Fingerprinter
from evidence_manifests import Checker
from test_scientific_records import Fixture


class EarningTests(Fixture):
    def setUp(self):
        super().setUp()
        # Fixture tests exercise snapshot behavior, not the host's Cargo config.
        ambient = patch('evidence_earning.noncanonical_compile_environment', return_value=[])
        ambient.start()
        self.addCleanup(ambient.stop)
        self.write('tools/evidence-analyzer/pin.toml', 'release="1.85.1"\ncommit="abc"\nanalyzer="diagnostic"\n')

    def snapshot(self, kind='scientific', key='top::t::check_top', measurement=None, record_binding=None):
        row = self.row('top')
        fp = Fingerprinter(self.root, self.metadata, '', platform=False, environ={})
        record_binding = record_binding or binding(kind, key, 'recorded-fingerprint', measurement)
        with patch('subprocess.check_output', return_value='release: 1.85.1\ncommit-hash: abc\n'):
            return Snapshot(self.root, row, record_binding, fp)

    def test_earned_scientific_manifest_classifies_without_cargo_or_fingerprinting(self):
        snapshot = self.snapshot()
        ref = snapshot.publish()
        scientific.record_passes(self.root, [{**self.row('top'), 'selector':'top::t::check_top',
            'fingerprint':'recorded-fingerprint', 'dependency_manifest':ref}])
        with patch.object(scientific, 'Fingerprinter', side_effect=AssertionError('compiler discovery')):
            owed, current = scientific.classify(self.root, [self.row('top')])
        self.assertEqual((len(owed),len(current)), (0,1))
        self.write('crates/leaf/src/lib.rs', 'pub fn leaf()->u8 { 9 }')
        owed, current = scientific.classify(self.root, [self.row('top')])
        self.assertEqual((len(owed),len(current)), (1,0))

    def test_changed_input_cannot_be_published_as_a_pass(self):
        snapshot = self.snapshot()
        self.write('crates/top/src/lib.rs', 'pub fn top()->u8 { 99 }')
        with self.assertRaises(RuntimeError):
            snapshot.publish()

    def test_added_source_input_invalidates_but_unrelated_crate_does_not(self):
        snapshot = self.snapshot()
        ref = snapshot.publish()
        self.write('crates/other/src/lib.rs', 'pub fn other()->u8 { 99 }')
        self.assertTrue(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])
        self.write('crates/leaf/src/new.rs', 'pub const V:u8=1;')
        self.assertFalse(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])

    def test_calibration_and_probe_share_production_checker(self):
        rec = dict(id='c', fingerprint='recorded-fingerprint', calibration_sha='measured')
        measurement = self.snapshot(record_binding=measurement_binding(rec))
        ref = measurement.publish()
        rec['dependency_manifest']=ref
        with patch.object(calibration, 'tree_fingerprinter', side_effect=AssertionError('compiler discovery')):
            now = calibration.tree_fingerprints([rec], root=self.root)
        self.assertEqual(calibration.classify([rec], [], now)['c'].status, calibration.CURRENT)
        changed_measurement = {**rec, 'observed':0.01}
        changed_now = calibration.tree_fingerprints([changed_measurement], root=self.root)
        self.assertNotEqual(calibration.classify([changed_measurement],[],changed_now)['c'].status,calibration.CURRENT)
        self.write('crates/top/src/lib.rs', 'pub fn top()->u8 { 99 }')
        now = calibration.tree_fingerprints([rec], root=self.root)
        self.assertNotEqual(calibration.classify([rec], [], now)['c'].status, calibration.CURRENT)

    def test_probe_binding_preserves_exact_measurement_and_prefix(self):
        row = dict(record_id='c',fingerprint='f',calibration_sha='measured',sha='probed',k=100,bound=.03,result='pass')
        snapshot = self.snapshot(record_binding=probe_binding(row))
        ref = snapshot.publish()
        self.assertTrue(Checker(self.root).check(probe_binding(row),ref)['current'])
        self.assertFalse(Checker(self.root).check(probe_binding({**row,'k':320}),ref)['current'])

    def test_compiler_mismatch_cannot_earn(self):
        fp = Fingerprinter(self.root,self.metadata,'',environ={})
        with (
            patch('subprocess.check_output', return_value='release: 9\ncommit-hash: wrong\n'),
            self.assertRaises(RuntimeError),
        ):
            Snapshot(self.root,self.row('top'),binding('scientific','t','f'),fp)

    def test_earning_refuses_a_noncanonical_build_environment(self):
        for env in ({'RUSTFLAGS': '--cfg changed_scientific_path'}, {'CARGO_PROFILE_TEST_OPT_LEVEL': '0'},
                    {'ANTECEDENT_SCIENTIFIC_FEATURES': 'x'}):
            with patch.dict('os.environ', env), self.assertRaises(RuntimeError, msg=env):
                self.snapshot()

    def test_lint_levels_and_the_checking_environment_do_not_change_currency(self):
        with patch.dict('os.environ', {'RUSTFLAGS': '-D warnings'}):
            snapshot = self.snapshot()
            ref = snapshot.publish()
        with patch.dict('os.environ', {'RUSTFLAGS': '-D warnings --cfg ci'}):
            self.assertTrue(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])

    def test_a_shared_fingerprinter_keeps_each_rows_inputs_its_own(self):
        fp = Fingerprinter(self.root, self.metadata, '', platform=False, environ={})
        with patch('subprocess.check_output', return_value='release: 1.85.1\ncommit-hash: abc\n'):
            Snapshot(self.root, self.row('other'), binding('scientific', 'o', 'f'), fp)
            shared = Snapshot(self.root, self.row('leaf'), binding('scientific', 'l', 'f'), fp)
            alone = Snapshot(self.root, self.row('leaf'), binding('scientific', 'l', 'f'))
        paths = lambda snap: sorted(i.get('path', i.get('name')) for i in snap.manifest['inputs'])  # noqa: E731
        self.assertEqual(paths(shared), paths(alone))
        self.assertFalse(any('crates/other' in p for p in paths(shared)))

    def test_compile_time_code_anywhere_in_the_graph_keeps_complete_source(self):
        import evidence_earning
        from evidence_earning import select_items
        coverage = {str((self.root / 'crates/leaf/src/lib.rs').resolve()): {(1, 1)}}
        with (
            patch.dict(evidence_earning._COMPILE_TIME, {self.root: None}),
            self.assertRaisesRegex(RuntimeError, 'Cargo metadata'),
        ):
            self.snapshot()
        for graph in ({'leaf': {'a_derive'}, 'mid': set(), 'top': set()},
                      {'leaf': set(), 'mid': set(), 'top': {'build_script_dep'}}):
            required = sorted(set().union(*graph.values()))
            import json
            self.write('parity/fingerprint_inputs.toml', '[[input]]\nsource="crates/top/Cargo.toml"\n'
                       f'compile_time={json.dumps(required)}\ninputs=[]\nenv=[]\nwhy="assessed fixture macro"\n')
            with patch.dict(evidence_earning._COMPILE_TIME, {self.root: graph}):
                snapshot = self.snapshot()
            self.assertTrue(snapshot.opaque, graph)
            select_items(snapshot, self.root / 'no-inventory', coverage)
            self.assertFalse(any(i['kind'] == 'rust_items' for i in snapshot.manifest['inputs']))
            self.assertTrue(snapshot.manifest['fallbacks'])
        with patch.dict(evidence_earning._COMPILE_TIME, {self.root: {'leaf': set(), 'mid': set(), 'top': set()}}):
            self.assertFalse(self.snapshot().opaque)

    def test_a_new_default_build_script_invalidates_the_pass(self):
        snapshot = self.snapshot()
        ref = snapshot.publish()
        self.write('crates/leaf/build.rs', 'fn main() { println!("cargo:rustc-env=VALUE=3"); }')
        self.assertFalse(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            snapshot.publish()

    def test_unassessed_compile_time_reads_refuse_earning(self):
        import evidence_earning
        graph = {'leaf': set(), 'mid': set(), 'top': {'reader'}}
        with (
            patch.dict(evidence_earning._COMPILE_TIME, {self.root: graph}),
            self.assertRaisesRegex(RuntimeError, 'undeclared compile-time input boundary'),
        ):
            self.snapshot()

    def test_nix_environment_claim_does_not_waive_unassessed_inputs(self):
        import evidence_earning
        graph = {'leaf': set(), 'mid': set(), 'top': {'reader'}}
        with (
            patch.dict(evidence_earning._COMPILE_TIME, {self.root: graph}),
            patch.dict('os.environ', {'IN_NIX_SHELL': 'pure', 'NIX_BUILD_TOP': '/tmp/fake'}, clear=True),
            self.assertRaisesRegex(RuntimeError, 'undeclared compile-time input boundary'),
        ):
            self.snapshot()

    def test_reviewed_registry_assesses_the_host_compile_time_graph(self):
        import evidence_earning
        import nix_evidence_policy
        graph = {'leaf': set(), 'mid': set(), 'top': {'reader'}}
        self.write('nix/compile-time-reviewed.json', '{}')
        self.write('Cargo.lock', '')
        with patch.dict(evidence_earning._COMPILE_TIME, {self.root: graph}), \
             patch.dict(evidence_earning._HOST_UNITS, clear=True), \
             patch.object(nix_evidence_policy, 'host_compiler_graph', return_value={}), \
             patch.object(nix_evidence_policy, 'validate_units', return_value={}) as validate:
            snapshot = self.snapshot()
        validate.assert_called_once()
        self.assertTrue(snapshot.opaque)
        paths = {i.get('path') for i in snapshot.manifest['inputs']}
        self.assertIn('nix/compile-time-reviewed.json', paths)
        self.assertIn('Cargo.lock', paths)

    def test_declared_compile_time_data_and_missing_inputs_are_guarded(self):
        import evidence_earning
        self.write('parity/fingerprint_inputs.toml', '''[[input]]
source="crates/top/Cargo.toml"
compile_time=["reader"]
inputs=["crates/top/input.txt", "crates/top/optional.txt"]
env=[]
why="reader macro reads the consumer inputs"
''')
        self.write('crates/top/input.txt', '7')
        graph = {'leaf': set(), 'mid': set(), 'top': {'reader'}}
        with patch.dict(evidence_earning._COMPILE_TIME, {self.root: graph}):
            snapshot = self.snapshot()
        ref = snapshot.publish()
        self.assertTrue(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])
        self.write('crates/top/input.txt', '8')
        self.assertFalse(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])
        self.write('crates/top/input.txt', '7')
        self.write('crates/top/optional.txt', 'new input')
        self.assertFalse(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])

    def test_compaction_preserves_explicit_ignored_data_inputs(self):
        self.write('crates/top/src/lib.rs', 'pub const V:&str=include_str!("data/value.txt");')
        self.write('crates/top/src/data/value.txt', '7')
        self.write('.gitignore', 'value.txt\n')
        for args in (['init','-q'], ['add','-A']):
            subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True)
        snapshot = self.snapshot()
        ref = snapshot.publish()
        self.assertTrue(any(i.get('path') == 'crates/top/src/data/value.txt'
                            for i in snapshot.manifest['inputs']))
        self.write('crates/top/src/data/value.txt', '8')
        self.assertFalse(Checker(self.root).check(snapshot.manifest['record'], ref)['current'])

    def test_earning_refuses_untracked_inputs_and_ignores_ignored_files(self):
        git = lambda *args: subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True)  # noqa: E731
        self.write('.gitignore', '*.so\n')
        git('init', '-q')
        git('add', '-A')
        self.write('crates/leaf/src/native.so', 'build product')
        ref = self.snapshot().publish()
        self.write('crates/leaf/src/scratch.rs', 'pub fn scratch() {}')
        with self.assertRaisesRegex(RuntimeError, 'untracked'):
            self.snapshot()
        (self.root / 'crates/leaf/src/scratch.rs').unlink()
        (self.root / 'crates/leaf/src/native.so').unlink()
        self.assertTrue(Checker(self.root).check(self.snapshot().manifest['record'], ref)['current'])

    def test_changed_scientific_fixture_declaration_cannot_reuse_old_pass(self):
        snapshot=self.snapshot()
        ref=snapshot.publish()
        scientific.record_passes(self.root,[{**self.row('top'),'selector':'top::t::check_top',
            'fingerprint':'recorded-fingerprint','dependency_manifest':ref}])
        owed,current=scientific.classify(self.root,[self.row('top',fixtures=['conformance/new'])])
        self.assertEqual((len(owed),len(current)),(1,0))
        self.assertIn('scientific_declarations_changed',owed[0]['manifest_reasons'])

    def test_missing_fixture_becoming_present_invalidates(self):
        row=self.row('top', fixtures=['conformance/new'])
        fp=Fingerprinter(self.root,self.metadata,'',environ={})
        with patch('subprocess.check_output', return_value='release: 1.85.1\ncommit-hash: abc\n'):
            snapshot=Snapshot(self.root,row,binding('scientific','t','f'),fp)
        ref=snapshot.publish()
        self.write('conformance/new.json','{"truth":2}')
        self.assertFalse(Checker(self.root).check(snapshot.manifest['record'],ref)['current'])


if __name__ == '__main__':
    unittest.main()
