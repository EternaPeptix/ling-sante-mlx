# Copyright © 2023 Apple Inc.

import os
import sys

from ._version import __version__

if os.environ.get("LINGSANTE_CMDBUF", "1") not in ("0", ""):
    if "mlx.core" in sys.modules:
        print(
            "[mlx_lm] note: mlx.core imported before mlx_lm; LINGSANTE_CMDBUF buffer sizes not applied",
            file=sys.stderr,
        )
    else:
        os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "64")
        os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from .convert import convert
from .generate import batch_generate, generate, stream_generate
from .utils import load

__all__ = [
    "__version__",
    "convert",
    "batch_generate",
    "generate",
    "stream_generate",
    "load",
]
