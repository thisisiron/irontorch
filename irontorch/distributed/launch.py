# -*- coding: utf-8 -*-
"""Distributed training launch utilities."""

import os

import torch
from torch.distributed.launcher.api import elastic_launch

from typing import Callable, Tuple

from irontorch import distributed as dist


def set_omp_threads():
    """Set OMP_NUM_THREADS to 1 if not already set."""
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"


def run(fn: Callable, conf, args: Tuple = ()) -> None:
    """Launch distributed training.

    Args:
        fn: The function to run.
        conf: Configuration object with n_gpu and launch_config.
        args: Arguments to pass to the function.
    """
    num_gpus_per_node = conf.n_gpu
    num_nodes = 1
    world_size = num_nodes * num_gpus_per_node

    if world_size > 1:
        set_omp_threads()

        elastic_launch(config=conf.launch_config, entrypoint=elastic_worker)(
            fn, args, num_gpus_per_node
        )
    else:
        fn(*args)


def elastic_worker(fn: Callable, args: Tuple, num_gpus_per_node: int):
    """Worker function for elastic launch.

    Args:
        fn: The function to run.
        args: Arguments to pass to the function.
        num_gpus_per_node: Number of GPUs per node.
    """
    if not torch.cuda.is_available():
        raise OSError("CUDA is not available. Please check your environments")

    local_rank = int(os.environ["LOCAL_RANK"])
    num_gpus_per_node = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.distributed.init_process_group(
        backend="NCCL",
    )

    dist.synchronize()

    if num_gpus_per_node > torch.cuda.device_count():
        raise ValueError(
            f"Requested {num_gpus_per_node} GPUs, "
            f"but only {torch.cuda.device_count()} GPUs are available."
        )

    torch.cuda.set_device(local_rank)

    dist.create_local_process_group(num_gpus_per_node)

    fn(*args)
