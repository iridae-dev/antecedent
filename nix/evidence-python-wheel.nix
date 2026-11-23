# Build-only production Python wheel. This does not execute tests or attest
# sandbox policy, runtime dependencies, source provenance, or evidence.
{ src }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  pkgs = import (builtins.fetchTree lock.nodes.nixpkgs.locked) { system = "aarch64-darwin"; };
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
  python = pkgs.python313;
  vendor = pkgs.rustPlatform.importCargoLock { lockFile = src + "/Cargo.lock"; };
  project = builtins.fromTOML (builtins.readFile (src + "/python/pyproject.toml"));
  features = project.tool.maturin.features;
  featureArgs = if features == [] then [] else [ "--features" (builtins.concatStringsSep "," features) ];
in
assert builtins.isList features && builtins.all builtins.isString features;
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-python-wheel";
  version = "0";
  inherit src;
  nativeBuildInputs = [ rust pkgs.cargo pkgs.maturin python ];
  dontConfigure = true;
  dontStrip = true;
  buildPhase = ''
    export HOME="$TMPDIR/home" CARGO_HOME="$TMPDIR/cargo" CARGO_TARGET_DIR="$TMPDIR/target"
    export RUSTC=${rust}/bin/rustc CARGO=${pkgs.cargo}/bin/cargo PYO3_PYTHON=${python}/bin/python3.13
    export PATH=${pkgs.cargo}/bin:$PATH PYTHONNOUSERSITE=1
    export RUSTFLAGS="--remap-path-prefix=$PWD=${src}"
    mkdir -p "$HOME" "$CARGO_HOME" "$out/wheels"
    cat > "$CARGO_HOME/config.toml" <<'EOF'
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "${vendor}"
    [net]
    offline = true
    EOF
    cd python
    ${pkgs.cargo}/bin/cargo metadata --format-version=1 --locked --offline ${pkgs.lib.escapeShellArgs featureArgs} > "$TMPDIR/cargo-metadata.json"
    RUSTC_BOOTSTRAP=1 ${pkgs.cargo}/bin/cargo -Z unstable-options build --unit-graph --locked --offline --profile release --lib -p antecedent-py ${pkgs.lib.escapeShellArgs featureArgs} > "$TMPDIR/cargo-unit-graph.json"
    ${pkgs.maturin}/bin/maturin build --locked --offline --profile release --interpreter ${python}/bin/python3.13 --out "$out/wheels"
    cd ..
  '';
  installPhase = ''
    test "$(find "$out/wheels" -name '*.whl' -type f | wc -l | tr -d ' ')" = 1
    mkdir -p "$out/bin"
    ln -s ${python}/bin/python3.13 "$out/bin/python3.13"
    ${python}/bin/python3.13 -I - "$PWD" ${pkgs.lib.escapeShellArg (toString src)} "$TMPDIR" "$out" <<'PY'
    import json, pathlib, platform, sys, sysconfig
    workdir, source, temporary, out = sys.argv[1:]
    def remap(value):
        if isinstance(value, str):
            return value.replace(workdir, source).replace(temporary, '/__nix_build_tmp__')
        if isinstance(value, list): return [remap(item) for item in value]
        if isinstance(value, dict): return {key: remap(item) for key, item in value.items()}
        return value
    for name in ('cargo-metadata.json', 'cargo-unit-graph.json'):
        data = json.loads((pathlib.Path(temporary) / name).read_text(encoding='utf-8'))
        (pathlib.Path(out) / name).write_text(json.dumps(remap(data), sort_keys=True) + '\n', encoding='utf-8')
    config = {'executable': sys.executable, 'version': platform.python_version(),
              'implementation': sys.implementation.name,
              'soabi': sysconfig.get_config_var('SOABI'),
              'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
              'config_args': sysconfig.get_config_var('CONFIG_ARGS')}
    (pathlib.Path(out) / 'python-config.json').write_text(json.dumps(config, sort_keys=True) + '\n', encoding='utf-8')
    PY
    printf '%s\n' ${pkgs.lib.escapeShellArg (toString src)} > "$out/source-root"
    ${rust}/bin/rustc -vV > "$out/rustc-version"
    ${pkgs.cargo}/bin/cargo --version > "$out/cargo-version"
    ${pkgs.maturin}/bin/maturin --version > "$out/maturin-version"
  '';
}
