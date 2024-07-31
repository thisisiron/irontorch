import os
import sys
import logging

from irontorch import distributed as dist


def get_logger(save_dir, name='main', distributed_rank=None, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_decimal(value):
    for i in range(10):
        if value >= 10 ** (-i) - 1e-10:
            return i

    return 10


class Logger:
    def __init__(self, save_dir, rank):
        self.logger = get_logger(save_dir, distributed_rank=rank)

    def log(self, step, **kwargs):
        panels = [f"step: {step}"]

        for k, v in kwargs.items():
            if isinstance(v, float):
                decimal = get_decimal(v) + 2
                v = round(v, decimal)
                panels.append(f"{k}: {v}")

            else:
                panels.append(f"{k}: {v}")
        self.logger.info("| ".join(panels))


class WandB:
    def __init__(
        self,
        project,
        group=None,
        name=None,
        notes=None,
        resume=None,
        tags=None,
        id=None,
    ):
        if dist.is_primary():
            import wandb

            wandb.init(
                project=project,
                group=group,
                name=name,
                notes=notes,
                resume=resume,
                tags=tags,
                id=id,
            )

            self.wandb = wandb

    def log(self, step, **kwargs):
        self.wandb.log(kwargs, step=step)

    def __del__(self):
        if dist.is_primary():
            self.wandb.finish()
