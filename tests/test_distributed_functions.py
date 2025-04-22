import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from irontorch.distributed.distributed import (
    is_primary,
    get_rank,
    get_world_size,
    is_parallel,
    upwrap_parallel,
    get_data_sampler,
    reduce_dict,
)


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return self.size


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


@pytest.fixture
def simple_model():
    return SimpleModel()


def test_is_primary():
    # In a non-distributed setting, is_primary should be True
    assert is_primary() is True


def test_get_rank():
    # In a non-distributed setting, rank should be 0
    assert get_rank() == 0


def test_get_world_size():
    # In a non-distributed setting, world_size should be 1
    assert get_world_size() == 1


def test_is_parallel():
    model = SimpleModel()
    assert not is_parallel(model)
    
    # Test with DataParallel
    if torch.cuda.device_count() > 1:
        dp_model = nn.DataParallel(model)
        assert is_parallel(dp_model)


def test_upwrap_parallel(simple_model):
    # Test non-parallel model
    assert upwrap_parallel(simple_model) is simple_model
    
    # Test with DataParallel
    if torch.cuda.device_count() > 1:
        dp_model = nn.DataParallel(simple_model)
        unwrapped = upwrap_parallel(dp_model)
        assert unwrapped is simple_model


def test_get_data_sampler(dummy_dataset):
    # Test with non-distributed and no shuffle
    sampler = get_data_sampler(dummy_dataset, shuffle=False, distributed=False)
    assert isinstance(sampler, torch.utils.data.SequentialSampler)
    
    # Test with non-distributed and shuffle
    sampler = get_data_sampler(dummy_dataset, shuffle=True, distributed=False)
    assert isinstance(sampler, torch.utils.data.RandomSampler)
    
    # Note: We can't test the distributed case without an initialized process group
    # That would require a multi-process test setup


def test_reduce_dict():
    # In a non-distributed setting, reduce_dict should return the input dict unchanged
    test_dict = {
        "loss": torch.tensor(1.0),
        "accuracy": torch.tensor(0.75)
    }
    
    reduced = reduce_dict(test_dict, average=True)
    assert torch.equal(reduced["loss"], test_dict["loss"])
    assert torch.equal(reduced["accuracy"], test_dict["accuracy"]) 