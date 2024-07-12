import os
import random

import torch
import numpy as np


TORCH_2_0 = check_version(torch.__version__, "2.0.0")


def set_seed(seed=42, deterministic=False):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

    if deterministic:  # ensure reproducibility.
        torch.backends.cudnn.benchmark = False 

        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
        else:
            pass  # TODO: add warning 

    else:
        torch.backends.cudnn.benchmark = True 
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
