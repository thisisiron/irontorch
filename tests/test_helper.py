import unittest
import torch
import numpy as np
import random
import os
from irontorch.utils.helper import set_seed, check_library_version

class TestHelperFunctions(unittest.TestCase):

    def test_set_seed(self):
        set_seed(42)
        random_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        set_seed(42)
        self.assertEqual(random_val, random.random())
        self.assertEqual(np_val, np.random.rand())
        self.assertEqual(torch_val, torch.rand(1).item())

    def test_set_seed_deterministic(self):
        set_seed(42, deterministic=True)
        random_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        set_seed(42, deterministic=True)
        self.assertEqual(random_val, random.random())
        self.assertEqual(np_val, np.random.rand())
        self.assertEqual(torch_val, torch.rand(1).item())

    def test_check_library_version(self):
        self.assertTrue(check_library_version("1.8.0", "1.7.0"))
        self.assertTrue(check_library_version("1.8.0", "1.8.0"))
        self.assertFalse(check_library_version("1.7.0", "1.8.0"))
        self.assertTrue(check_library_version("1.8.0", "1.8.0", must_be_same=True))
        self.assertFalse(check_library_version("1.8.0", "1.7.0", must_be_same=True))

if __name__ == '__main__':
    unittest.main()
