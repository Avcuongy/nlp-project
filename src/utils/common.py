from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and (optionally) PyTorch.

    Args:
        seed: Seed value to set.
        deterministic: If True and PyTorch is available, set deterministic flags.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        # PyTorch not installed or CUDA not available; silently continue
        pass
