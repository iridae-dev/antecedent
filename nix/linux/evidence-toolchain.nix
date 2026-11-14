{ pkgs }:

# Official Rust release archives for the Linux evidence platforms: the release
# ../evidence-toolchain.nix pins, from the same dist directory. Their published
# SHA-256s (the `.sha256` file beside each archive) are held in
# evidence-toolchain.json, and Nix refuses a download that does not hash to its
# pin. An archive with no pin fails evaluation here, so nothing is ever built from
# an unchecked download.
let
  pins = builtins.fromJSON (builtins.readFile ./evidence-toolchain.json);
  triples = {
    x86_64-linux = "x86_64-unknown-linux-gnu";
    aarch64-linux = "aarch64-unknown-linux-gnu";
  };
  system = pkgs.stdenv.hostPlatform.system;
  triple = triples.${system} or (throw "no reviewed Linux evidence toolchain for ${system}");
  archive = component:
    let
      name = "${component}-${pins.release}-${triple}.tar.xz";
      sha256 = pins.targets.${triple}.${component};
    in
    if sha256 == null then
      throw "nix/linux/evidence-toolchain.json: no SHA-256 is pinned for ${name}"
    else
      pkgs.fetchurl {
        url = "https://static.rust-lang.org/dist/${pins.date}/${name}";
        inherit sha256;
      };
  rust = archive "rust";
  llvmTools = archive "llvm-tools";
  rustcDev = archive "rustc-dev";
in
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-rust";
  version = pins.release;
  src = rust;
  # The archives expect a conventional loader and library path, which the build
  # sandbox does not hold: patch their binaries to the store's.
  nativeBuildInputs = [ pkgs.xz pkgs.autoPatchelfHook ];
  buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
  dontConfigure = true;
  dontBuild = true;
  # Nix's generic strip phase damages the .rmeta members of Rust's rlibs.
  dontStrip = true;
  installPhase = ''
    runHook preInstall
    bash ./install.sh --prefix="$out" --disable-ldconfig
    mkdir -p "$TMPDIR/llvm-tools"
    tar -xJf ${llvmTools} -C "$TMPDIR/llvm-tools" --strip-components=1
    (cd "$TMPDIR/llvm-tools" && bash ./install.sh --prefix="$out" --disable-ldconfig)
    mkdir -p "$TMPDIR/rustc-dev"
    tar -xJf ${rustcDev} -C "$TMPDIR/rustc-dev" --strip-components=1
    (cd "$TMPDIR/rustc-dev" && bash ./install.sh --prefix="$out" --disable-ldconfig)
    runHook postInstall
  '';
  # The host triple, for recipes that name the toolchain's rustlib directory.
  passthru = { inherit triple; };
}
