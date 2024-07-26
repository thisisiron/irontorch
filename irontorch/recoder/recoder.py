import os

from irontorch import distributed as dist


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
