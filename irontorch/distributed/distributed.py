import torch
from torch import nn
from torch import distributed as dist
from torch.utils import data

from typing import Union, Optional


def is_primary() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_parallel(model: nn.Module) -> bool:
   """Determines whether the model is a parallel model (DP or DDP)."""
   return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def upwrap_parallel(model: nn.Module) -> bool:
   """Converts a parallel model (DP or DDP) to a non-parallel model."""
   return model.module if is_parallel(model) else model


def get_data_sampler(
    dataset: data.Dataset, 
    shuffle: bool, 
    distributed: bool
) -> Union[data.sampler.Sampler, data.distributed.DistributedSampler]:

    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
