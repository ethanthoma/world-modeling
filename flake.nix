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
        manifest = (pkgs.lib.importTOML ./pyproject.toml).project;
        name = manifest.name;
        version = manifest.version;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

        pyprojectOverrides =
          final: prev:
          let
            cudaPackageOverrides =
              pkgs.lib.genAttrs
                (pkgs.lib.concatMap
                  (pkg: [
                    "nvidia-${pkg}-cu11"
                    "nvidia-${pkg}-cu12"
                  ])
                  [
                    "cublas"
                    "cuda-cupti"
                    "cuda-curand"
                    "cuda-nvrtc"
                    "cuda-runtime"
                    "cudnn"
                    "cufft"
                    "curand"
                    "cusolver"
                    "cusparse"
                    "nccl"
                    "nvjitlink"
                    "nvtx"
                  ]
                )
                (
                  name:
                  prev.${name}.overrideAttrs (old: {
                    autoPatchelfIgnoreMissingDeps = true;
                    postFixup = ''
                      rm -rf $out/${final.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                      ln -sfn $out/${final.python.sitePackages}/nvidia/*/lib/lib*.so* $out/lib
                    '';
                  })
                );
          in
          {
            nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (attrs: {
              propagatedBuildInputs = attrs.propagatedBuildInputs or [ ] ++ [
                final.nvidia-cublas-cu12
              ];
            });

            nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (attrs: {
              propagatedBuildInputs = attrs.propagatedBuildInputs or [ ] ++ [
                final.nvidia-cublas-cu12
              ];
            });

            torch = prev.torch.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = true;

              propagatedBuildInputs = old.propagatedBuildInputs or [ ] ++ [
                final.numpy
                final.packaging
              ];
            });

            calver = prev.calver.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.wheel ];
            });

            setuptools-scm = prev.setuptools-scm.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.wheel ];
            });

            trove-classifiers = prev.trove-classifiers.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.wheel ];
            });

            pluggy = prev.pluggy.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.wheel ];
            });
          }
          // cudaPackageOverrides;

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ self.overlays.default ];
        };

        python = pkgs.python312;

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
          pkgs.lib.composeExtensions overlay pyprojectOverrides
        );

        venv = pythonSet.mkVirtualEnv "${name}-${version}-venv" { ${name} = [ ]; };
      in
      {
        packages.default = pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [ venv ];
          text = ''
            exec ${venv}/bin/${name} "$@"
          '';
        };

        devShells.default =
          let
            editableOverlay = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; };

            editablePythonSet = pythonSet.overrideScope editableOverlay;

            virtualenv = editablePythonSet.mkVirtualEnv "${name}-${version}-dev-venv" { ${name} = [ ]; };
          in
          pkgs.mkShell {
            packages = [
              virtualenv
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
          rustPlatform = final.rustPlatform // {
            buildRustPackage =
              args:
              final.rustPlatform.buildRustPackage (
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
