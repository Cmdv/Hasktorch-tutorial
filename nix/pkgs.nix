pkgs: _: with pkgs; {
  hasktorchTutorialHaskellPackages = import ./haskell.nix {
    inherit
      lib
      stdenv
      pkgs
      haskell-nix
      buildPackages
      config
      ;
  };
}
