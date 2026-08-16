from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_deterministic_runtime(seed: int, *, torch: Any) -> None:
    """Configure the frozen single-GPU run for strict repeatability."""
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before CUDA initialization"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def deterministic_runtime_snapshot(*, torch: Any) -> dict[str, Any]:
    """Return the settings bound into the GPU smoke report."""
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "flash_sdp_enabled": _backend_flag(torch, "flash_sdp_enabled"),
        "memory_efficient_sdp_enabled": _backend_flag(
            torch, "mem_efficient_sdp_enabled"
        ),
        "math_sdp_enabled": _backend_flag(torch, "math_sdp_enabled"),
    }


def _backend_flag(torch: Any, name: str) -> bool | None:
    method = getattr(torch.backends.cuda, name, None)
    return bool(method()) if callable(method) else None
