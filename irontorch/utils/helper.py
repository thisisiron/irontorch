# -*- coding: utf-8 -*-
"""Helper utilities for training."""

import os
import random
import warnings

import torch
import numpy as np
from packaging import version


def check_library_version(
    cur_version: str, min_version: str, must_be_same: bool = False
) -> bool:
    """Check if current version meets minimum version requirement.

    Args:
        cur_version: Current version string.
        min_version: Minimum required version string.
        must_be_same: If True, versions must match exactly.

    Returns:
        True if version requirement is met.
    """
    current = version.parse(cur_version)
    minimum = version.parse(min_version)
    return (current == minimum) if must_be_same else (current >= minimum)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

    if deterministic:  # ensure reproducibility.
        torch.backends.cudnn.benchmark = False

        torch_2_0 = check_library_version(torch.__version__, "2.0.0")

        if torch_2_0:
            torch.use_deterministic_algorithms(
                True, warn_only=True
            )  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
        else:
            warnings.warn(
                "Torch version is below 2.0.0. "
                "Deterministic algorithms may not be fully supported.",
                RuntimeWarning,
            )

    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
