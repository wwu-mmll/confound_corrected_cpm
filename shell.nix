{ pkgs ? import <nixpkgs> {
    config = {
      # Noetig fuer cudaPackages.* (CUDA EULA ist "unfree")
      allowUnfree = true;
      cudaSupport = true;
    };
  }
}:

pkgs.mkShell {
  name = "confound-corrected-cpm";

  buildInputs = with pkgs; [
  (poetry.overridePythonAttrs (old: {
    doCheck = false;
    doInstallCheck = false;
  }))
  python314

  stdenv.cc.cc.lib
  zlib

  cudaPackages.nsight_systems
  cudaPackages.nsight_compute
  ];

  shellHook = ''
    # libstdc++ fuer torch (siehe: ImportError libstdc++.so.6)
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

    # NVIDIA-Treiber-Runtime-Libs (siehe: torch.cuda.is_available() == False)
    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
      # torch.compile's Triton backend hardcodes /sbin/ldconfig to find
      # libcuda.so.1, which doesn't exist on NixOS -- point it at the
      # driver lib dir directly instead.
      export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
    fi

    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

    echo "confound_corrected_cpm dev shell aktiv."
    echo "  torch findet libstdc++ und CUDA-Treiber jetzt automatisch."
    echo "  nsys/ncu stehen zur Verfuegung: \$(which nsys), \$(which ncu)"
    echo ""
    echo "  Aktiviere das Poetry-venv mit: poetry shell"
    echo "  Oder direkt:                .venv/bin/python examples/profile_run.py"
  '';
}
