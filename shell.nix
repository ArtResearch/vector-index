{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # buildInputs are dependencies for the development environment.
  buildInputs = with pkgs; [
    # C++ compiler
    gcc

    # Build system
    cmake

    # Libraries
    boost
    opencv
    curl
    jemalloc
  ];
}
