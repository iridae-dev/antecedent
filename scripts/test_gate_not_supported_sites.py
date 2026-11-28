import unittest

from gate_not_supported_sites import violations


class SupportGateTests(unittest.TestCase):
    def test_a_new_refusal_fails_even_in_an_existing_adapter(self):
        self.assertEqual(violations('fn adapt(r: NotSupported) { r.code(); }\n'
                                    'fn bypass() { reason_code! (\n"not_supported"); }'), [2])
        self.assertEqual(violations('Err(CausalError::Support { id: crate::support::SupportRefusal::NotSupported, message })'), [1])
        self.assertEqual(violations('Err(Error { code: "not_supported", message })'), [1])

    def test_a_typed_conversion_match_or_test_is_no_new_producer(self):
        for source in (
            'impl From<NotSupported> for CausalError { fn from(r: NotSupported) -> Self { Self::Support { id: SupportRefusal::NotSupported, message: r.message() } } }',
            'match err { Self::Support { id: SupportRefusal::NotSupported, message } => read(message) }',
            '#[cfg(test)] mod test { fn fixture() { reason_code!("not_supported"); } }',
            '// reason_code!("not_supported")',
        ):
            self.assertEqual(violations(source), [], source)


if __name__ == "__main__":
    unittest.main()
