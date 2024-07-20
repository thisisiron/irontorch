import argparse
import sys
import os


import torch
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.launcher.api import LaunchConfig


def parse_min_max_nodes(n_node):
    ar = n_node.split(":")

    if len(ar) == 1:
        min_node = max_node = int(ar[0])

    elif len(ar) == 2:
        min_node, max_node = int(ar[0]), int(ar[1])

    else:
        raise ValueError(f'n_node={n_node} is not in "MIN:MAX" format')

    return min_node, max_node


def local_world_size(n_gpu):
    if isinstance(n_gpu, int):
        return n_gpu

    elif n_gpu == "cpu":
        return os.cpu_count()

    elif n_gpu == "gpu":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        return torch.cuda.device_count()

    else:
        raise ValueError(f"Unsupported n_proc value: {n_gpu}")


def get_rdzv_endpoint(args, max_node):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        dist_url = args.dist_url

        return dist_url

    return args.rdzv_endpoint


def elastic_config(args):
    min_node, max_node = parse_min_max_nodes(args.n_node)
    n_proc = local_world_size(args.n_proc)

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args, max_node)

    config = LaunchConfig(
        min_nodes=min_node,
        max_nodes=max_node,
        nproc_per_node=n_proc,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        log_dir=args.log_dir,
    )

    return config


def add_elastic_args(parser):
    parser.add_argument("--n_proc", type=str, default="1")
    parser.add_argument("--n_node", type=str, default="1:1")
    parser.add_argument("--node_rank", type=int, default=0)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"127.0.0.1:{port}")

    parser.add_argument("--rdzv_backend", type=str, default="static")
    parser.add_argument(
        "--rdzv_endpoint",
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv_id", type=str, default="none", help="User-defined group id."
    )
    parser.add_argument(
        "--rdzv_conf",
        type=str,
        default="",
        help="Additional rendezvous configuration (<key 1>=<value 1>, <key 2>=<value 2>, ...).",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Start a local standalone rendezvous backend",
    )

    parser.add_argument("--max_restarts", type=int, default=0)
    parser.add_argument("--monitor_interval", type=float, default=5)
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
    )
    parser.add_argument("--role", type=str, default="default")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("-r", "--redirects", type=str, default="0")
    parser.add_argument("-t", "--tee", type=str, default="0")

    return parser



def parse_and_load_config():
    parser = argparse.ArgumentParser()
    parser = preset_argparser()
    # TODO: add arguments
    parser = add_elastic_args(parser)
    args = parser.parse_args()

    launch_config = elastic_config(args)