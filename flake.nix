{
  description = "Antecedent pinned evidence build inputs and sandbox fixture";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/7525d999cd850b9a488817abc89c75dc733acf17";

  outputs = { nixpkgs, ... }:
    let
      # The platforms with a reviewed evidence policy (scripts/evidence_platforms.py).
      systems = [ "aarch64-darwin" "x86_64-linux" "aarch64-linux" ];
      packagesFor = system:
        let
          pkgs = import nixpkgs { inherit system; };
          rust = import
            (if pkgs.stdenv.hostPlatform.isDarwin
            then ./nix/evidence-toolchain.nix
            else ./nix/linux/evidence-toolchain.nix)
            { inherit pkgs; };
          # An ordinary engineering fixture: builds and runs only this tiny program.
          fixture = pkgs.runCommand "antecedent-evidence-boundary-fixture" {
            nativeBuildInputs = [ rust pkgs.stdenv.cc ];
          } ''
            cat > fixture.rs <<'RS'
            fn main() { assert_eq!(2 + 2, 4); }
            RS
            rustc fixture.rs -o fixture
            ./fixture
            mkdir -p "$out"
            rustc -vV > "$out/rustc-version"
          '';
        in {
          evidence-rust = rust;
          evidence-fixture = fixture;
        };
    in {
      packages = nixpkgs.lib.genAttrs systems packagesFor;
    };
}
