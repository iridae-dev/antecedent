"""Reject ambient build overrides that the evidence boundary does not assess.

The workspace's own Cargo configuration is tracked separately. Cargo also
searches ancestor directories and CARGO_HOME, neither of which is represented
by the repository fingerprint; refuse those files until they have an explicit
input model. This is a fail-closed guard for known override channels, not proof
that arbitrary procedural macros are hermetic.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# These are checked against BUILD_ENV by evidence_manifests, including its
# handling for lint-only Rust flags. Do not silently allow other CARGO_* or
# RUSTC_* settings, such as compiler wrappers and per-target linkers.
_CHECKED_BUILD_VARIABLES = frozenset({
    "CARGO_BUILD_TARGET", "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_PROFILE_TEST_DEBUG", "CARGO_PROFILE_TEST_OPT_LEVEL",
    "RUSTC", "RUSTC_BOOTSTRAP", "RUSTFLAGS",
})
_EXACT_OVERRIDES = frozenset({
    "AR", "AS", "CC", "CXX", "CPP", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
    "LDFLAGS", "RANLIB", "ARFLAGS", "RANLIBFLAGS", "NVCC", "CXXSTDLIB",
    "CRATE_CC_NO_DEFAULTS", "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "SDKROOT", "DEVELOPER_DIR",
    "MACOSX_DEPLOYMENT_TARGET", "IPHONEOS_DEPLOYMENT_TARGET", "CROSS_COMPILE",
    "VIRTUAL_ENV", "CONDA_PREFIX", "PYENV_VERSION", "LLVM_CONFIG_PATH",
    "ZSTD_SYS_USE_PKG_CONFIG", "LIBC_CI", "LIBC_BUILD_VERBOSE",
})
_OVERRIDE_PREFIXES = (
    "CARGO_", "RUSTC_", "RUSTUP_", "PYO3_", "PYTHON", "PKG_CONFIG",
    "BINDGEN_", "CMAKE_", "DYLD_", "LD_", "CC_", "CXX_", "AR_",
    "CFLAGS_", "CXXFLAGS_", "CPPFLAGS_", "LDFLAGS_", "ARFLAGS_",
    "RANLIB_", "RANLIBFLAGS_", "NVCC_", "CXXSTDLIB_", "BLAKE3_", "RUST_LIBC_UNSTABLE_",
)
_CC_TARGETED_VARIABLES = frozenset({
    "CC", "CXX", "AR", "RANLIB", "CFLAGS", "CXXFLAGS", "ARFLAGS",
    "RANLIBFLAGS", "NVCC", "CXXSTDLIB",
})


def noncanonical_compile_environment(root: Path, environ: Mapping[str, str]) -> list[str]:
    """Describe known unassessed settings and Cargo configs without exposing values.

    This must be called before Cargo metadata and before a build. BUILD_ENV is
    validated separately, so its allowlisted names are not accepted unchecked.
    This checks the caller's environment; a Nix builder's separately constructed
    environment must be verified independently rather than assumed equivalent.
    """
    issues = [
        f"environment override {name}"
        for name in sorted(environ)
        if name not in _CHECKED_BUILD_VARIABLES
        and (name in _EXACT_OVERRIDES or name.startswith(_OVERRIDE_PREFIXES)
             or any(name == f"{kind}_{var}" for kind in ("HOST", "TARGET")
                    for var in _CC_TARGETED_VARIABLES))
    ]
    root = root.resolve()
    home = Path(environ.get("CARGO_HOME") or Path(environ.get("HOME", "~")).expanduser() / ".cargo")
    directories = {parent / ".cargo" for parent in root.parents}
    directories.add(home)
    for directory in sorted(directories, key=str):
        for name in ("config", "config.toml"):
            path = directory / name
            if path.exists() or path.is_symlink():
                issues.append(f"ambient Cargo config {path}")
    return issues
