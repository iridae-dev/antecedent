# Security Policy

## Supported Versions

<!-- generated:release:supported-lines:begin -->
| Version | Supported |
|---------|-----------|
| 2.4.x   | Yes |
| 2.3.x   | Yes |
| 2.2.x   | Yes |
| 2.1.x   | Yes |
| 2.0.x   | Yes |
| < 2.0   | No — upgrade to the latest release |
<!-- generated:release:supported-lines:end -->

Security fixes are made on `main` and released from there. Fixes are not backported to
earlier minor versions; the supported upgrade path is to the latest release.

## Callback execution and artifact decoding

Antecedent can call a provider function it does not own (an arbitrary law, posterior or
nested callback) and reads several portable artifacts produced by other processes. The
trust boundaries are:

- **Declarations never execute code.** A callback declaration, an interface contract, a
  transport handoff and a stored callback result are data. Reading, planning, verifying or
  re-deriving them never invokes a provider; planning takes no provider argument.
- **Execution is explicit and bounded.** A provider runs only through an explicit execution
  call under a stated policy: invocation counts, cooperative cancellation before and after the
  call, an output-size bound and checked output validation. A side-effecting callback is
  refused unless the policy allows side effects, and is retried only under a declared
  idempotent contract with an idempotency key and a bounded attempt count. Every invocation
  is recorded in the session ledger.
- **Limits.** The wall-clock bound is enforced after the call returns; an in-process closure
  cannot be preempted. Isolating an untrusted provider needs a separate process or sandbox
  that the caller supplies. Do not pass untrusted code as a provider.
- **Results are claims, not native facts.** A callback result is diagnostic unless its
  declaration and verification support more, and even then it is an external claim; it never
  becomes native trust, a joint law or an identification licence by metadata alone.
- **Decoders are bounded and re-derive.** Each artifact decoder enforces a byte cap before
  parsing, rejects unknown or newer versions and unknown required features, checks the
  container checksum, and recomputes identities and derived values from stored fields rather
  than trusting stored digests. A resealed semantic edit is refused. Decoders return typed
  errors and are fuzzed (`fuzz/`).

See `docs/security_review.md` for the per-artifact limits and the review notes.

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Use one of these private channels:
1. GitHub Security Advisories (preferred) for this repository.
2. If advisories are unavailable, contact repository maintainers directly via GitHub.

Include:
- Affected component/file
- Reproduction details or proof of concept
- Impact assessment
- Suggested remediation (if available)

## Response Process

Maintainers will aim to:
- Acknowledge reports within 3 business days.
- Provide a status update within 7 business days.
- Coordinate disclosure and fix release timing with the reporter when possible.

## Disclosure

After a fix is available, maintainers will publish a summary of:
- Affected versions
- Impact
- Mitigation/fix guidance
