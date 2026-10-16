Vendored from https://github.com/ModelTC/LightVAE at 38b0587d47df9188e3e05973265352962b20ada4, comfyui directory. H3 encoding and decoding adapter.

Locally adapted to ComfyUI v0.41.0's native `encoder`, `decoder` and
`encoder_kept_blocks` constructor arguments. Flat LightVAE residual indices
are converted to native stage/block pairs before construction. Checkpoint
conversion and original-component fallback remain in this adapter.
