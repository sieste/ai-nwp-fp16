"""Stub for torch_harmonics.attention's compiled C++/CUDA extension
(torch_harmonics/attention/optimized/attention_helpers.cpp). Building it
needs a real compile step (setup.py, BUILD_CPP/BUILD_CUDA=1) this project
hasn't done -- not needed for SFNO, which doesn't use torch-harmonics'
sphere-attention submodule (a different, newer model family in the same
package) at all. torch_harmonics/attention/__init__.py already has a safe,
non-fatal fallback when these report unavailable (`attention_kernels =
None`, a warning, nothing else) -- confirmed by reading it, not assumed.
"""


def cuda_kernels_is_available() -> bool:
    return False


def optimized_kernels_is_available() -> bool:
    return False
