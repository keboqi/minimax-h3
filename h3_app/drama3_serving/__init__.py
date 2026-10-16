"""Optional Drama 3 backend served by vLLM-Omni in its own environment.

ComfyUI stays the default Drama 3 route. This package prepares an isolated
vLLM-Omni environment, runs its server only while a Drama 3 job holds the GPU,
and generates long scripts as speaker-based chunks in parallel.
"""
