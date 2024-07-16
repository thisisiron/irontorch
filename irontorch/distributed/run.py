import torch
from torch import distributed as dist
from torch import multiprocessing as mp


def set_omp_threads():
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"


def run(
    target_function: Callable,
    gpus_per_node: int,
    num_nodes: int = 1,
    conf=()
    ) -> None:

    world_size = num_nodes * gpus_per_node

    if world_size > 1:
        set_omp_threads()
    else:
        target_function(*conf)



