import os

import torch
from torch import multiprocessing as mp
from torch.distributed.launcher.api import elastic_launch

from typing import Callable, Tuple

from irontorch import distributed as dist


def set_omp_threads():
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"


def run(
    fn: Callable,
    num_gpus_per_node: int,
    num_nodes: int = 1,
    conf: Tuple=()
) -> None:

    world_size = num_nodes * num_gpus_per_node

    if world_size > 1:
        set_omp_threads()

        elastic_launch(config=conf.launch_config, entrypoint=elastic_worker)(fn, conf, num_gpus_per_node)
    else:
        fn(conf)


def elastic_worker(fn: Callable, conf: Tuple, num_gpus_per_node: int):
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
            f"Requested {num_gpus} GPUs, but only {torch.cuda.device_count()} GPUs are available."
        )

    torch.cuda.set_device(local_rank)

    dist.create_local_process_group(num_gpus_per_node)

    fn(conf)

