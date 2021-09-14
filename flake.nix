{
  description = "A simple flake for EmoNet";

  inputs.nixpkgs.url = "nixpkgs/release-21.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.pypi-deps-db = {
    url = "github:DavHau/pypi-deps-db";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.mach-nix.follows = "mach-nix";
  };
  inputs.mach-nix = {
    url = "github:DavHau/mach-nix";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.pypi-deps-db.follows = "pypi-deps-db";
    inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { self, nixpkgs, mach-nix, flake-utils, pypi-deps-db, ... }:
    let
      name = "emonet";
      missingLibs = pkgs: with pkgs; [
        cudatoolkit_11_2
        cudnn_cudatoolkit_11_2
        pkgs.stdenv.cc.cc
      ];
      commonOptions = {
        ignoreDataOutdated = true;
        python = "python38";
        src = ./.;
        providers.soundfile = "nixpkgs";
      };
      package = { pkgs ? import <nixpkgs> }: mach-nix.lib.${pkgs.system}.buildPythonApplication
        {
          inherit (commonOptions) ignoreDataOutdated python src providers;
          pname = name;
          version = "0.0.1";
          requirements = builtins.readFile ./requirements.txt;

          postInstall = ''
            wrapProgram $out/bin/EmoNet --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath (missingLibs pkgs)}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib"
          '';
          meta = with pkgs.lib; {
            description = "";
            homepage = "https://github.com/EIHW/EmoNet";
            license = licenses.gpl3;
            platforms = platforms.linux;
            inherit version;
          };
        };
      overlay = final: prev: {
        emonet = {
          emonet = package { pkgs = prev; };
          defaultPackage = package { pkgs = prev; };
        };
      };
      developmentEnv = { pkgs ? import <nixpkgs> }: mach-nix.lib.${pkgs.system}.buildPythonPackage {
        inherit (commonOptions) ignoreDataOutdated python src providers;
        pname = name;
        version = "0.0.1";
        requirements = builtins.readFile ./requirements.txt;
      };
      shell = { pkgs ? import <nixpkgs> }: pkgs.mkShell {
        name = "${name}-dev";
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (missingLibs pkgs)}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib:$LD_LIBRARY_PATH";
          unset SOURCE_DATE_EPOCH
        '';
        buildInputs = [
          (developmentEnv { inherit pkgs; })
        ];
      };
    in
    flake-utils.lib.simpleFlake
      {
        inherit self nixpkgs overlay shell name;
        config = { allowUnfree = true; };
        systems = [ "x86_64-linux" ];
      };
}
