"""Tests for offline target graph reporting; these do not run evidence tests."""
from pathlib import Path
import tempfile
import unittest

import nix_compile_time_inventory as audit


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'Cargo.toml').write_text('[package]\nname="root"\nversion="1.0.0"\n', encoding="utf-8")
        (self.root / 'Cargo.lock').write_text('''version = 4
[[package]]
name = "root"
version = "1.0.0"
[[package]]
name = "macro"
version = "0.1.0"
source = "registry+https://example.test/index"
checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
''', encoding="utf-8")
        self.source = self.root / 'macro'
        self.source.mkdir()
        (self.source / 'lib.rs').write_text('let _ = std::env::var("TMPDIR");\n', encoding="utf-8")
        self.metadata = {'packages': [
            {'name': 'root', 'version': '1.0.0', 'source': None,
             'manifest_path': str(self.root / 'Cargo.toml'), 'targets': [{'kind': ['lib']}]},
            {'name': 'macro', 'version': '0.1.0', 'source': 'registry+https://example.test/index',
             'manifest_path': str(self.source / 'Cargo.toml'), 'targets': [{'kind': ['proc-macro']}]},
        ]}
        self.tree = '0|root v1.0.0 (/repo)\tdefault\n1|macro v0.1.0 (proc-macro)\ta\n1|macro v0.1.0\tb,a (*)\n'

    def test_target_features_and_each_package_remain_explicitly_unreviewed(self):
        result = audit.inventory(self.root, self.metadata, self.tree, 'root', 'aarch64-apple-darwin',
                                 ['forest'], test_name='fixture', hints=True)
        self.assertEqual(result['test_target'], 'fixture')
        self.assertEqual(result['requested_features'], ['forest'])
        macro = next(p for p in result['packages'] if p['name'] == 'macro')
        self.assertEqual(macro['cargo_tree_features'], ['a', 'b'])
        self.assertEqual(macro['compile_time_kinds'], ['proc-macro'])
        self.assertEqual(macro['parents'], [{'name': 'root', 'version': '1.0.0'}])
        self.assertEqual(macro['audit_status'], 'unreviewed')
        self.assertEqual(macro['source_search_hints']['temp'], ['lib.rs:1'])
        self.assertEqual(macro['source_search_hints']['env'], ['lib.rs:1'])

    def test_unresolvable_ambiguous_and_unlocked_graphs_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            audit.inventory(self.root, self.metadata, self.tree + '1|other v1.0.0\t\n', 'root', 't', [])
        self.metadata['packages'].append(dict(self.metadata['packages'][1]))
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            audit.inventory(self.root, self.metadata, self.tree, 'root', 't', [])
        self.metadata['packages'].pop()
        self.metadata['packages'][1]['source'] = 'git+https://example.test/repo'
        with self.assertRaisesRegex(ValueError, 'unsupported or unlocked'):
            audit.inventory(self.root, self.metadata, self.tree, 'root', 't', [])

    def test_malformed_or_empty_tree_and_symlinks_are_refused(self):
        for tree in ('', 'unparseable\t\n', 'macro v0.1.0'):
            with self.subTest(tree=tree), self.assertRaises(ValueError):
                audit.tree_packages(tree)
        (self.source / 'other.rs').symlink_to(self.source / 'lib.rs')
        with self.assertRaisesRegex(ValueError, 'linked package source'):
            audit.source_hints(self.source)

    def test_unit_graph_records_modes_features_and_dependency_indices(self):
        graph = {'version': 1, 'roots': [0], 'units': [
            {'pkg_id': f'root 1.0.0 (path+{self.root.resolve().as_uri()})',
             'mode': 'test', 'platform': 'aarch64-apple-darwin',
             'target': {'name': 'root', 'kind': ['lib'], 'src_path': str(self.root / 'src/lib.rs')},
             'features': [], 'dependencies': [{'index': 1}]},
            {'pkg_id': 'macro 0.1.0 (registry+https://example.test/index)',
             'mode': 'build', 'platform': None,
             'target': {'name': 'macro', 'kind': ['proc-macro'], 'src_path': str(self.source / 'lib.rs')},
             'features': ['b', 'a'], 'dependencies': []},
        ]}
        result = audit.compiler_units(self.metadata, graph)
        self.assertEqual(result['roots'], [0])
        self.assertEqual(result['units'][0]['dependencies'], [{'index': 1}])
        self.assertEqual(result['units'][1]['features'], ['a', 'b'])
        self.assertEqual(result['units'][1]['target_path'], 'lib.rs')
        self.assertTrue(all(u['audit_status'] == 'unreviewed' for u in result['units']))
        graph['units'][0]['dependencies'] = [{'index': 2}]
        with self.assertRaisesRegex(ValueError, 'dependency'):
            audit.compiler_units(self.metadata, graph)

    def test_unit_graph_refuses_unknown_version_and_package_identity(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            audit.compiler_units(self.metadata, {'version': 2, 'units': [], 'roots': []})
        graph = {'version': 1, 'roots': [0], 'units': [{'pkg_id': 'unknown 1.0.0 (registry+unknown)'}]}
        with self.assertRaisesRegex(ValueError, 'unresolvable'):
            audit.compiler_units(self.metadata, graph)

    def test_modern_cargo_package_ids_and_graph_comparison(self):
        self.metadata['packages'][0]['id'] = 'path+file:///project#1.0.0'
        graph = {'version': 1, 'roots': [0], 'units': [{
            'pkg_id': 'path+file:///project#1.0.0', 'mode': 'test', 'platform': None,
            'target': {'name': 'root', 'kind': ['lib'], 'src_path': str(self.root / 'src/lib.rs')},
            'features': [], 'dependencies': []}]}
        host = audit.compiler_units(self.metadata, graph)
        self.assertTrue(audit.compare_compiler_units(host, host)['match'])
        graph['units'][0]['features'] = ['extra']
        builder = audit.compiler_units(self.metadata, graph)
        diff = audit.compare_compiler_units(host, builder)
        self.assertFalse(diff['match'])
        self.assertEqual((diff['host_only_unit_edges'], diff['builder_only_unit_edges']), (1, 1))


if __name__ == '__main__':
    unittest.main()
