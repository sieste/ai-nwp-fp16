"""Stub for torch_harmonics.disco's compiled C++/CUDA extension -- same
situation as attention_helpers.py in this directory (see its docstring).
Not needed for SFNO, which doesn't use torch-harmonics' discrete-
continuous convolution submodule either."""


def cuda_kernels_is_available() -> bool:
    return False


def optimized_kernels_is_available() -> bool:
    return False


def pack_psi_dense(*args, **kwargs):
    raise RuntimeError("disco_helpers stub: pack_psi_dense should be unreachable for SFNO")


def preprocess_psi(*args, **kwargs):
    raise RuntimeError("disco_helpers stub: preprocess_psi should be unreachable for SFNO")
