# Prototype target build. Its caller must supply an independently verified,
# immutable snapshot of the intended commit; this expression does not attest
# that source identity or that the Nix daemon actually used a sandbox.
{ src, crate, targetKind, targetName ? "", featuresJson ? "[]" }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  nixpkgsSource = builtins.fetchTree lock.nodes.nixpkgs.locked;
  pkgs = import nixpkgsSource { system = "aarch64-darwin"; };
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
  features = builtins.fromJSON featuresJson;
  vendor = pkgs.rustPlatform.importCargoLock { lockFile = src + "/Cargo.lock"; };
  targetArgs = if targetKind == "lib" then [ "--lib" ] else [ "--test" targetName ];
  featureArgs = if features == [] then [] else [ "--features" (builtins.concatStringsSep "," features) ];
  cargoArgs = [ "test" "--locked" "--offline" "--no-run" "--message-format=json" "-p" crate ] ++ targetArgs ++ featureArgs;
in
assert targetKind == "lib" || targetKind == "test";
assert targetKind != "test" || targetName != "";
assert builtins.isList features && builtins.all builtins.isString features;
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-target";
  version = "0";
  inherit src;
  nativeBuildInputs = [ rust pkgs.python3 ];
  dontConfigure = true;
  dontStrip = true;
  buildPhase = ''
    export HOME="$TMPDIR/home" CARGO_HOME="$TMPDIR/cargo" CARGO_TARGET_DIR="$TMPDIR/target"
    export RUSTFLAGS="-C instrument-coverage --remap-path-prefix=$PWD=${src}" CARGO_PROFILE_TEST_OPT_LEVEL=2 CARGO_PROFILE_TEST_DEBUG=line-tables-only
    mkdir -p "$HOME" "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<'EOF'
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "${vendor}"
    [net]
    offline = true
    EOF
    cargo ${pkgs.lib.escapeShellArgs cargoArgs} > "$TMPDIR/cargo-messages.jsonl" || {
      cat "$TMPDIR/cargo-messages.jsonl" >&2
      exit 1
    }
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    python3 - "$TMPDIR/cargo-messages.jsonl" "$out/bin/test" <<'PY'
    import json, shutil, sys
    executables = []
    for line in open(sys.argv[1], encoding="utf-8"):
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("profile", {}).get("test") is True
                and message.get("executable")):
            executables.append(message["executable"])
    if len(executables) != 1:
        raise SystemExit(f"expected one test executable, found {len(executables)}")
    shutil.copy2(executables[0], sys.argv[2])
    PY
    ln -s ${rust}/lib/rustlib/aarch64-apple-darwin/bin/llvm-profdata "$out/bin/llvm-profdata"
    ln -s ${rust}/lib/rustlib/aarch64-apple-darwin/bin/llvm-cov "$out/bin/llvm-cov"
    printf '%s\n' ${pkgs.lib.escapeShellArg (toString src)} > "$out/source-root"
    rustc -vV > "$out/rustc-version"
  '';
}
