"""Running AI weather models in fp32, bf16 and fp16, and verifying them.

Run scripts from the repository root as modules, e.g.

    .venv/bin/python -m nwpprec.forecast --model pangu --precision fp16
"""

import os

# set before torch is imported: lets the CUDA allocator grow and release
# segments, avoiding out-of-memory errors from fragmentation (Aurora)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
