# -*- coding: utf-8 -*-
"""Communication utilities for distributed training."""

import functools
import torch
from torch import nn
from torch import distributed as dist
from torch.utils import data

from typing import Union, Dict


_LOCAL_PROCESS_GROUP = None


def is_primary() -> bool:
    """Check if current process is the primary (rank 0)."""
    return get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get the local rank of the current process within its machine."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """Create a process group that contains ranks within the same machine.

    Detectron2's launch() in engine/launch.py will call this function.
    If you start workers without launch(), you'll have to also call this.
    Otherwise utilities like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine.
            Typically the number of GPUs.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        start_rank = i * num_workers_per_machine
        end_rank = (i + 1) * num_workers_per_machine
        ranks_on_i = list(range(start_rank, end_rank))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """Get the local process group for the current machine.

    Returns:
        A torch process group which only includes processes that are on
        the same machine as the current process. This group can be useful
        for communication within a machine, e.g. a per-machine SyncBN.
    """
    assert _LOCAL_PROCESS_GROUP is not None
    return _LOCAL_PROCESS_GROUP


def get_world_size() -> int:
    """Get the total number of processes in the distributed group."""
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def synchronize() -> None:
    """Synchronize all processes in the distributed group."""
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def reduce_dict(
    input_dict: Dict[str, torch.Tensor], average: bool = True
) -> Dict[str, torch.Tensor]:
    """Reduce a dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors to reduce.
        average: Whether to average the values.

    Returns:
        Reduced dictionary of tensors.
    """
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
    """Determine whether the model is a parallel model (DP or DDP)."""
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def upwrap_parallel(model: nn.Module) -> nn.Module:
    """Convert a parallel model (DP or DDP) to a non-parallel model."""
    return model.module if is_parallel(model) else model


def get_data_sampler(
    dataset: data.Dataset, shuffle: bool, distributed: bool
) -> Union[data.sampler.Sampler, data.distributed.DistributedSampler]:
    """Get the appropriate data sampler based on distribution settings.

    Args:
        dataset: The dataset to sample from.
        shuffle: Whether to shuffle the data.
        distributed: Whether to use distributed sampling.

    Returns:
        Appropriate sampler for the settings.
    """
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
