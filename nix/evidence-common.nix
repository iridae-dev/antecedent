# The build environment shared by the dependency and evidence derivations. Both must
# compile third-party units with identical flags, environment and Cargo
# configuration, or the evidence build would find them stale and recompile them.
{ pkgs }:
let
  lib = pkgs.lib;
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
  targetArgs = t: if t.kind == "lib" then [ "--lib" ] else [ "--test" t.name ];
  featureArgs = t: if t.features == [] then [] else [ "--features" (builtins.concatStringsSep "," t.features) ];
  dir = t: "targets/${t.crate}/${t.kind}/${if t.kind == "lib" then "lib" else t.name}";
in
{
  inherit rust dir targetArgs featureArgs;
  # Cargo from nixpkgs uses store-provided curl/OpenSSL rather than the host
  # LibreSSL used by the upstream Cargo binary. The compiler remains pinned to
  # the separately verified official toolchain pinned in evidence-toolchain.nix.
  nativeBuildInputs = [ rust pkgs.cargo pkgs.python3 ];
  # librustc_driver links zlib; only the inventory driver uses it.
  buildInputs = [ pkgs.zlib ];
  # The vendored crates depend only on the lock contents, never on the source path.
  vendor = lockContents: pkgs.rustPlatform.importCargoLock { lockFileContents = lockContents; };
  environment = vendor: ''
    # Build scripts may reuse probe files in OUT_DIR. Never use an ambient or
    # shared scratch tree: Nix's per-build TMPDIR must belong to this builder.
    umask 077
    python3 - "$TMPDIR" <<'PY'
    import os, stat, sys
    path = sys.argv[1]
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit("Nix build scratch directory must be private to the builder")
    PY
    export HOME="$TMPDIR/home" CARGO_HOME="$TMPDIR/cargo" CARGO_TARGET_DIR="$TMPDIR/target"
    export RUSTC=${rust}/bin/rustc
    # The source-path remap is applied to workspace crates only, by the workspace
    # wrapper: a flag naming the per-build directory would make every
    # third-party unit's fingerprint differ between derivations.
    # Incremental compilation partitions code generation non-reproducibly (it
    # reorders coverage records); evidence builds are never incremental.
    export RUSTFLAGS="-C instrument-coverage" CARGO_PROFILE_TEST_OPT_LEVEL=2 CARGO_PROFILE_TEST_DEBUG=line-tables-only
    export CARGO_INCREMENTAL=0
    mkdir -m 700 "$HOME" "$CARGO_HOME"
    mkdir -p -m 700 "$CARGO_TARGET_DIR"
    # The private parents protect scratch; outputs must remain readable after
    # Nix canonicalizes ownership in the store.
    umask 022
    cat > "$CARGO_HOME/config.toml" <<'EOF'
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "${vendor}"
    [net]
    offline = true
    EOF
  '';
  # One target's own test build, exactly as it would run alone.
  build = t: messages: ''
    mkdir -p "${messages}/${dir t}"
    ${pkgs.cargo}/bin/cargo ${lib.escapeShellArgs ([ "test" "--locked" "--offline" "--no-run" "--message-format=json" "-p" t.crate ] ++ targetArgs t ++ featureArgs t)} > "${messages}/${dir t}/cargo-messages.jsonl" || {
      cat "${messages}/${dir t}/cargo-messages.jsonl" >&2
      exit 1
    }
  '';
}
