import torch
from torch import nn
from torch import distributed as dist
from torch.utils import data

from typing import Union, Optional, Dict, Any


def is_primary() -> bool:
    return get_rank() == 0


# TODO: Return local rank in a multi-node setup
def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def synchronize() -> None:
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


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
