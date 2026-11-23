# Build-only runtime assembled from exact uv.lock wheels and a Nix-built wheel.
# The output does not execute scientific/calibration tests or attest evidence.
{ src, wheelOutput }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  pkgs = import (builtins.fetchTree lock.nodes.nixpkgs.locked) { system = "aarch64-darwin"; };
  lib = pkgs.lib;
  python = pkgs.python313.withPackages (p: [ p.pip ]);
  uv = builtins.fromTOML (builtins.readFile (src + "/python/uv.lock"));
  packages = builtins.filter (p: p.name != "antecedent") uv.package;
  suitable = w:
    let n = builtins.baseNameOf w.url;
        pure = lib.hasInfix "py3" n && lib.hasSuffix "-none-any.whl" n;
        darwin = lib.hasInfix "macosx_" n && (lib.hasInfix "arm64" n || lib.hasInfix "universal2" n);
        abi = lib.hasInfix "-cp313-cp313-" n || lib.hasInfix "-py3-none-" n
           || builtins.any (v: lib.hasInfix ("-" + v + "-abi3-") n) [ "cp38" "cp39" "cp310" "cp311" "cp312" "cp313" ];
    in pure || (darwin && abi);
  selected = map (p:
    let wheel = lib.findFirst suitable (throw "no pinned Python 3.13 Darwin wheel for ${p.name}") (p.wheels or []);
        file = builtins.baseNameOf wheel.url;
        sha = builtins.substring 7 64 wheel.hash;
    in assert lib.hasPrefix "sha256:" wheel.hash && builtins.stringLength sha == 64;
       { inherit (p) name version; inherit file sha; source = pkgs.fetchurl { url = wheel.url; sha256 = sha; }; }
  ) packages;
  expected = pkgs.writeText "antecedent-python-locked-packages.json" (builtins.toJSON (map (p: { inherit (p) name version file sha; }) selected));
  copies = lib.concatMapStringsSep "\n" (p: "cp ${p.source} \"$TMPDIR/wheels/${p.file}\"") selected;
in
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-python-runtime";
  version = "0";
  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;
  nativeBuildInputs = [ python ];
  installPhase = ''
    test "$(cat ${wheelOutput}/source-root)" = ${lib.escapeShellArg (toString src)}
    export HOME="$TMPDIR/home" PYTHONNOUSERSITE=1 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
    mkdir -p "$HOME" "$TMPDIR/wheels" "$out/site" "$out/bin"
    ${copies}
    cp ${wheelOutput}/wheels/*.whl "$TMPDIR/wheels/"
    ${python}/bin/python3.13 -m pip install --no-index --no-deps --no-compile --target "$out/site" "$TMPDIR"/wheels/*.whl
    # pip records temporary input paths in direct_url.json; these are not runtime
    # inputs and would vary by Nix build directory, so remove only those files.
    find "$out/site" -path '*/.dist-info/direct_url.json' -delete
    find "$out/site" -path '*/*.dist-info/direct_url.json' -delete
    cp ${expected} "$out/locked-packages.json"
    printf '%s\n' ${lib.escapeShellArg (toString src)} > "$out/source-root"
    printf '%s\n' ${lib.escapeShellArg (toString wheelOutput)} > "$out/wheel-output"
    ln -s ${python}/bin/python3.13 "$out/bin/python3.13"
    ${python}/bin/python3.13 -I -B - "$out/site" "$out/locked-packages.json" <<'PY'
    import importlib.metadata, json, sys
    site, expected = sys.argv[1:]
    sys.path.insert(0, site)
    for package in json.load(open(expected, encoding='utf-8')):
        actual = importlib.metadata.version(package['name'])
        if actual != package['version']:
            raise SystemExit(f"locked Python dependency differs: {package['name']}: {actual}")
    # Check transitive metadata, including the project's minimum versions,
    # against this exact isolated site rather than the host's Python packages.
    from pip._internal.cli.main import main
    if main(['check']) != 0:
        raise SystemExit('locked Python runtime has unsatisfied requirements')
    import antecedent._native
    PY
  '';
}
