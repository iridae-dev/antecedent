# Everything an evidence build reads besides the committed source: the pinned
# toolchain, the standard build environment and tools, and the crates of
# Cargo.lock. Evidence builds run offline and without substitutes, so a host
# realises this once, online (`scripts/evidence_platforms.py prepare`). The crates
# are the very derivation the recipes vendor from the same lock, and the build
# compiles, links and runs one small program with the pinned compiler, so a
# toolchain that cannot produce a working executable on this host stops here.
#
# It is not an evidence recipe: no record binds it, and what it realises is
# identified by content or by the locked nixpkgs.
{ system ? builtins.currentSystem
, pkgs ? import (builtins.fetchTree (builtins.fromJSON (builtins.readFile ../flake.lock)).nodes.nixpkgs.locked) { inherit system; }
, cargoLock ? ../Cargo.lock
}:
let
  # Each platform's recipes: Apple silicon's here, the Linux forms in ./linux.
  recipes = if pkgs.stdenv.hostPlatform.isDarwin then ./. else ./linux;
  common = import (recipes + "/evidence-common.nix") { inherit pkgs; };
in
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-inputs";
  version = "0";
  dontUnpack = true;
  nativeBuildInputs = common.nativeBuildInputs;
  buildInputs = common.buildInputs;
  vendor = common.vendor (builtins.unsafeDiscardStringContext (builtins.readFile cargoLock));
  dontConfigure = true;
  dontStrip = true;
  buildPhase = ''
    test -d "$vendor"
    cat > fixture.rs <<'RS'
    fn main() { assert_eq!(2 + 2, 4); }
    RS
    ${common.rust}/bin/rustc fixture.rs -o fixture
    ./fixture
  '';
  installPhase = ''
    mkdir -p "$out"
    ${common.rust}/bin/rustc -vV > "$out/rustc-version"
    ${pkgs.cargo}/bin/cargo --version > "$out/cargo-version"
  '';
}
