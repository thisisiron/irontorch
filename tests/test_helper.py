import pytest
import torch
import numpy as np
import random
# Removed unused import: import os
from irontorch.utils.helper import set_seed, check_library_version


def test_set_seed():
    set_seed(42)
    random_val = random.random()
    np_val = np.random.rand()
    torch_val = torch.rand(1).item()

    set_seed(42)
    assert random_val == random.random()
    assert np_val == np.random.rand()
    assert torch_val == torch.rand(1).item()


def test_set_seed_deterministic():
    set_seed(42, deterministic=True)
    random_val = random.random()
    np_val = np.random.rand()
    torch_val = torch.rand(1).item()

    set_seed(42, deterministic=True)
    assert random_val == random.random()
    assert np_val == np.random.rand()
    assert torch_val == torch.rand(1).item()


def test_check_library_version():
    assert check_library_version("1.8.0", "1.7.0")
    assert check_library_version("1.8.0", "1.8.0")
    assert not check_library_version("1.7.0", "1.8.0")
    assert check_library_version("1.8.0", "1.8.0", must_be_same=True)
    assert not check_library_version("1.8.0", "1.7.0", must_be_same=True)


def test_check_library_version_edge_cases():
    # Testing with various version formats
    assert check_library_version("1.8", "1.7")
    assert check_library_version("1.8.0.1", "1.8.0")
    assert check_library_version("1.10.0", "1.9.0")
    assert not check_library_version("1.8.0.rc1", "1.8.0")  # Release candidate comparison
    
    # Test with identical versions
    assert check_library_version("1.0.0", "1.0.0")
