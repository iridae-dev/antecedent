# Third-party dependencies of the scientific test targets, compiled once and reused.
#
# `depsSrc` holds only the workspace manifests, Cargo.lock, the Cargo configuration
# and a stub for every workspace target (`nix_evidence_deps.py` derives it from the
# committed tree), so this output stays the same across commits that change no
# manifest. Each target runs the same `cargo test --no-run` invocation as the
# evidence build, with the same flags, so the evidence build finds every
# third-party unit fresh and recompiles only workspace crates.
{ depsSrc, depsTargetsJson }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  pkgs = import (builtins.fetchTree lock.nodes.nixpkgs.locked) { system = "aarch64-darwin"; };
  common = import ./evidence-common.nix { inherit pkgs; };
  targets = builtins.fromJSON depsTargetsJson;
  vendor = common.vendor (builtins.unsafeDiscardStringContext (builtins.readFile (depsSrc + "/Cargo.lock")));
in
assert builtins.isList targets && targets != [];
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-deps";
  version = "0";
  src = depsSrc;
  nativeBuildInputs = common.nativeBuildInputs;
  buildInputs = common.buildInputs;
  dontConfigure = true;
  dontStrip = true;
  dontFixup = true;
  buildPhase = ''
    ${common.environment vendor}
    ${pkgs.lib.concatMapStrings (t: common.build t "$TMPDIR/messages") targets}
  '';
  installPhase = ''
    # Drop every workspace unit. Store files all carry the same timestamp, so a
    # stub's fingerprint could otherwise look fresh for real source; without it,
    # Cargo must compile each workspace crate in the evidence build.
    python3 - "$CARGO_TARGET_DIR" <<'PY'
    import json, pathlib, shutil, subprocess, sys
    target = pathlib.Path(sys.argv[1])
    metadata = json.loads(subprocess.check_output(
        ["${pkgs.cargo}/bin/cargo", "metadata", "--format-version=1", "--no-deps", "--offline", "--locked"], text=True))
    members = {p["name"] for p in metadata["packages"]}
    removed = 0
    for profile in (p for p in target.iterdir() if p.is_dir()):
        for kind in (".fingerprint", "build"):
            for entry in (profile / kind).glob("*") if (profile / kind).is_dir() else ():
                if entry.name.rsplit("-", 1)[0] in members:
                    shutil.rmtree(entry)
                    removed += 1
    if not removed:
        raise SystemExit("no workspace unit found to remove from the dependency build")
    PY
    mkdir -p "$out"
    cp -a "$CARGO_TARGET_DIR" "$out/target"
  '';
}
