# Experimental evidence build inputs

The pinned flake and target expression are prototypes. They are **not** used
by evidence earning and do not establish an evidence boundary. Do not run
scientific or calibration tests through them or treat their outputs as records.

`flake.lock` pins nixpkgs; `evidence-toolchain.nix` pins official Rust 1.85.1
and LLVM tools archives by SHA-256 for `aarch64-darwin`. The disposable
`evidence-fixture` builds and runs only a tiny engineering program:

```sh
nix --extra-experimental-features 'nix-command flakes' build .#evidence-fixture
```

This checks only that the pinned toolchain can compile and run the fixture. It
does not verify a sandbox. A regular `nix develop` shell also does not provide
the required build isolation.

The local daemon currently has `sandbox = false` and `sandbox-fallback = true`.
Untrusted client overrides for both settings are ignored. A disposable build
confirmed that `/etc/nix/nix.conf` from the host was visible. Before a sandbox
can be assessed, the administrator must configure the daemon with
`sandbox = true` and `sandbox-fallback = false` and restart it. This is a
system-wide change and has not been performed by this repository.

Those settings alone do not prove the desired boundary. In particular, Darwin
Nix sandbox defaults can expose mutable host paths such as `/private/tmp` and
system frameworks. The precise permitted read paths, network, clock, entropy,
Nix substitutes and daemon behavior must be reviewed and verified with
deliberately failing fixtures. Do not infer that a Nix store path or a caller's
environment variable proves a sandboxed local build.

`evidence-target.nix` is a prototype for building a specified Rust target
without running it. A disposable fixture compiled and produced a usable test
binary and coverage paths under an unsandboxed daemon; this validates mechanics
only, not an evidence boundary. It requires a caller-supplied source snapshot;
independent commit/source verification, dependency and environment bindings,
coverage source mapping, output verification and earning integration remain
open. The expression must not be used to issue evidence until those are fixed
and tested with disposable fixtures.
