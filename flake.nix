{
  description = "python";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";

  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        venvPath = ".venv";
        buildInputs = [
          pkgs.zlib
        ];
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.python311
          ];
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"

            # if [ -d ${venvPath} ]; then
            #     . ${venvPath}/bin/activate
            # fi
          '';
        };
      }
    );
}

