{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pname = "world-modeling";

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

        pyprojectOverrides = _final: _prev: { };

        pkgs = nixpkgs.legacyPackages.${system}.appendOverlays [ self.overlays.default ];

        python = pkgs.python312;

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
          pkgs.lib.composeExtensions overlay pyprojectOverrides
        );
      in
      {
        packages.default = pythonSet.mkVirtualEnv "${pname}-env" { ${pname} = [ ]; };

        devShells.default =
          let
            editableOverlay = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; };

            editablePythonSet = pythonSet.overrideScope editableOverlay;

            virtualenv = editablePythonSet.mkVirtualEnv "${pname}-dev-env" { ${pname} = [ ]; };
          in
          pkgs.mkShell {
            packages = [
              #virtualenv
              pkgs.uv
              pkgs.ruff
              #pkgs.pylyzer
            ];
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              # waah waah this isnt right, that's not real nix waaaaaah
              LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib/";
            '';
          };
      }
    )
    // {
      overlays.default = final: prev: {
        ruff = prev.ruff.overrideAttrs (oldAttrs: {
          makeWrapperArgs = (oldAttrs.makeWrapperArgs or [ ]) ++ [
            "--suffix PATH : ${prev.lib.makeBinPath [ prev.ruff ]}"
          ];
        });

        pylyzer = prev.pylyzer.override {
          rustPlatform = prev.rustPlatform // {
            buildRustPackage =
              args:
              prev.rustPlatform.buildRustPackage (
                args
                // rec {
                  version = "0.0.68";

                  src = prev.fetchFromGitHub {
                    owner = "mtshiba";
                    repo = "pylyzer";
                    rev = "refs/tags/v${version}";
                    hash = "sha256-xeQDyj18L9jCftne9S79kWjrW0K7Nkx86Cy2aFqePfA=";
                  };

                  cargoLock = {
                    lockFile = "${src}/Cargo.lock";
                    outputHashes = {
                      "rustpython-ast-0.4.0" = "sha256-kMUuqOVFSvvSHOeiYMjWdsLnDu12RyQld3qtTyd5tAM=";
                    };
                  };
                }
              );
          };
        };
      };
    };
}
