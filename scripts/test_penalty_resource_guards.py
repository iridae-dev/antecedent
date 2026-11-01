#!/usr/bin/env python3
"""Mutation tests for the bounded penalty-grid source contract."""

from pathlib import Path
import unittest

from check_limits_agreement import RIDGE_RESOURCE_FILES, ridge_configuration_guard_errors

ROOT = Path(__file__).resolve().parent.parent


class PenaltyGuardMutations(unittest.TestCase):
    def test_each_guard_fails_closed_when_removed(self):
        sources = {path: (ROOT / path).read_text(encoding="utf-8") for path in RIDGE_RESOURCE_FILES}
        self.assertEqual(ridge_configuration_guard_errors(sources), [])
        mutations = {
            RIDGE_RESOURCE_FILES[0]: (
                "Self::check_context_free_grid_len(lambdas.len())?;",
                "CONTEXT_FREE_GRID_POLICY_BYTES / CONTEXT_FREE_GRID_BYTES_PER_PENALTY",
                "checked_mul(CONTEXT_FREE_GRID_BYTES_PER_PENALTY)",
                "bytes <= CONTEXT_FREE_GRID_POLICY_BYTES",
                "CONTEXT_FREE_GRID_BYTES_PER_PENALTY: u64 = 4096",
                'write!(key, "{:016x}"',
                'write!(result, "{value}"',
            ),
            RIDGE_RESOURCE_FILES[1]: (
                "raw.len()",
                "PyOverflowError",
                "check_context_free_grid_len(count)",
                "try_reserve_exact(count)",
                "values.len() == count",
                "bounded_penalty_values(&raw",
            ),
            RIDGE_RESOURCE_FILES[2]: (
                'bounded_penalty_values(&raw, "penalties")?',
                "RidgeTuning::context_free_max_grid_len()",
                "wrap_pyfunction!(penalty_grid_max_items, module)",
            ),
            RIDGE_RESOURCE_FILES[3]: (
                "_native._penalty_grid_max_items()",
                "except OverflowError",
                "declared > limit",
                "len(result) >= limit",
            ),
            RIDGE_RESOURCE_FILES[4]: (
                "values = bounded_penalty_list(self.lambdas)",
                "for value in bounded_penalty_list(self.lambdas)",
            ),
            RIDGE_RESOURCE_FILES[5]: (
                "for p in bounded_penalty_list(penalties)",
            ),
            RIDGE_RESOURCE_FILES[6]: (
                "65,536 input entries per declaration",
                "no shared reservation or execution-context admission",
            ),
            RIDGE_RESOURCE_FILES[7]: (
                "65,536 input penalties per declaration",
                "not a shared reservation or an execution-context admission",
            ),
        }
        for path, markers in mutations.items():
            for marker in markers:
                source = sources[path]
                positions = [i for i in range(len(source)) if source.startswith(marker, i)]
                self.assertTrue(positions, (path, marker))
                for index in positions:
                    with self.subTest(path=path, marker=marker, index=index):
                        changed = dict(sources)
                        changed[path] = source[:index] + "MUTATED_GUARD" + source[index + len(marker):]
                        self.assertTrue(ridge_configuration_guard_errors(changed))

    def test_constructor_guard_cannot_follow_copy(self):
        sources = {path: (ROOT / path).read_text(encoding="utf-8") for path in RIDGE_RESOURCE_FILES}
        path = RIDGE_RESOURCE_FILES[0]
        source = sources[path]
        guard = "Self::check_context_free_grid_len(lambdas.len())?;"
        copy = "let mut sorted = allocate_penalty_grid(lambdas.len())?;"
        source = source.replace(guard, "", 1).replace(copy, copy + "\n        " + guard, 1)
        sources[path] = source
        self.assertTrue(ridge_configuration_guard_errors(sources))


if __name__ == "__main__":
    unittest.main()
