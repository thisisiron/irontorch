import torch
from torch import distributed as dist
from torch.utils import data


def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def is_parallel(model):
   """Determines whether the model is a parallel model (DP or DDP)."""
   return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def upwrap_parallel(model):
   """Converts a parallel model (DP or DDP) to a non-parallel model."""
   return model.module if is_parallel(model) else model


def get_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
