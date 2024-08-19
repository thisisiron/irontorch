import os
import sys
import logging
import functools

from termcolor import colored
from typing import Optional, Union

try:
    from rich.logging import RichHandler

except ImportError:
    RichHandler = None

from irontorch import distributed as dist


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args: str, **kwargs: Union[str, int]) -> None:
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def get_logger(
    save_dir: Optional[str],
    distributed_rank: Optional[int] = None,
    filename: str = 'log.txt',
    mode: str = 'rich',
    name: str = 'main',
    abbrev_name: Optional[str] = None,
    enable_propagation: bool = False
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = enable_propagation 

    if abbrev_name is None:
        abbrev_name = name

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    if mode == 'rich' and RichHandler is None:
        mode = 'color'

    if mode == 'rich':
        rh = RichHandler(level=logging.DEBUG, log_time_format='%m/%d %H:%M:%S')
        logger.addHandler(rh)
        formatter = rh.formatter

    elif mode == 'color':
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(abbrev_name),
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    elif mode == 'plain':
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
                '[%(asctime)s %(name)s]: %(levelname)s: %(message)s',
                datefmt='%m/%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_decimal(value: float) -> int:
    for i in range(10):
        if value >= 10 ** (-i) - 1e-10:
            return i

    return 10


class Logger:
    def __init__(self, save_dir: str, name: str, rank: int, mode: str) -> None:
        self.logger = get_logger(save_dir, name=name, distributed_rank=rank, mode=mode)

    def log(self, step: int, **kwargs: Union[str, float, int]) -> None:
        panels = [f'step: {step}']

        for k, v in kwargs.items():
            if isinstance(v, float):
                decimal = get_decimal(v) + 2
                v = round(v, decimal)
                panels.append(f'{k}: {v}')
            else:
                panels.append(f'{k}: {v}')
        self.logger.info(' | '.join(panels))

    def info(self, message: str) -> None:
        self.logger.info(message)


class WandB:
    def __init__(
        self,
        project: str,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        resume: Optional[str] = None,
        tags: Optional[list[str]] = None,
        id: Optional[str] = None,
    ) -> None:        
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

    def log(self, step: int, **kwargs: Union[str, float, int]) -> None:
        self.wandb.log(kwargs, step=step)

    def __del__(self) -> None:
        if dist.is_primary():
            self.wandb.finish()
