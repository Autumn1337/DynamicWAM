"""Side-effect-free namespace for the isolated WAN source files.

Import concrete symbols from their defining modules. In particular, importing
the package must not initialize CUDA through the upstream T5 implementation.
"""
