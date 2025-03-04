import unittest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils import data
from irontorch.distributed import (
    get_rank,
    get_local_rank,
    is_primary,
    synchronize,
    get_world_size,
    get_local_process_group,
    reduce_dict,
    is_parallel,
    upwrap_parallel,
    get_data_sampler,
    create_local_process_group
)


class TestDistributedFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
        create_local_process_group(1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_get_rank(self):
        self.assertEqual(get_rank(), 0)

    def test_get_local_rank(self):
        self.assertEqual(get_local_rank(), 0)

    def test_is_primary(self):
        self.assertTrue(is_primary())

    def test_get_world_size(self):
        self.assertEqual(get_world_size(), 1)

    def test_synchronize(self):
        synchronize()
        self.assertTrue(True)  # If no exception, the test passes

    def test_get_local_process_group(self):
        self.assertIsNotNone(get_local_process_group())

    def test_reduce_dict(self):
        input_dict = {'a': torch.tensor(1.0), 'b': torch.tensor(2.0)}
        reduced_dict = reduce_dict(input_dict)
        self.assertEqual(reduced_dict['a'], torch.tensor(1.0))
        self.assertEqual(reduced_dict['b'], torch.tensor(2.0))

    def test_is_parallel(self):
        model = nn.Linear(2, 2)
        self.assertFalse(is_parallel(model))
        parallel_model = nn.parallel.DataParallel(model)
        self.assertTrue(is_parallel(parallel_model))

    def test_upwrap_parallel(self):
        model = nn.Linear(2, 2)
        parallel_model = nn.parallel.DataParallel(model)
        self.assertIs(upwrap_parallel(parallel_model), model)

    def test_get_data_sampler(self):
        dataset = data.TensorDataset(torch.tensor([[1, 2], [3, 4]]))
        sampler = get_data_sampler(dataset, shuffle=True, distributed=False)
        self.assertIsInstance(sampler, data.RandomSampler)
        dist_sampler = get_data_sampler(dataset, shuffle=True, distributed=True)
        self.assertIsInstance(dist_sampler, data.distributed.DistributedSampler)


if __name__ == '__main__':
    unittest.main()
