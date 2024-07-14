import unittest
import random
import numpy as np
import torch

from irontorch import set_seed


class TestSetSeed(unittest.TestCase):

    def test_without_deterministic(self):
        # Test without deterministic
        set_seed(42, deterministic=False)
        random_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        set_seed(42, deterministic=False)
        self.assertEqual(random_val, random.random())
        self.assertEqual(np_val, np.random.rand())
        self.assertEqual(torch_val, torch.rand(1).item())

        # Test with a different seed
        set_seed(43, deterministic=False)
        self.assertNotEqual(random_val, random.random())
        self.assertNotEqual(np_val, np.random.rand())
        self.assertNotEqual(torch_val, torch.rand(1).item())

    def test_with_deterministic(self):
        # Test with deterministic
        set_seed(42, deterministic=True)
        random_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        set_seed(42, deterministic=True)
        self.assertEqual(random_val, random.random())
        self.assertEqual(np_val, np.random.rand())
        self.assertEqual(torch_val, torch.rand(1).item())

        # Test with a different seed
        set_seed(43, deterministic=True)
        self.assertNotEqual(random_val, random.random())
        self.assertNotEqual(np_val, np.random.rand())
        self.assertNotEqual(torch_val, torch.rand(1).item())


if __name__ == '__main__':
    unittest.main()
