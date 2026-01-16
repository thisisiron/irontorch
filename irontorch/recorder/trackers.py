# -*- coding: utf-8 -*-
"""Experiment tracking integrations."""

import logging
from typing import Optional, Dict, Any

from irontorch import distributed as dist

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger:
    """Wrapper for Weights & Biases logging, active only on primary process."""

    def __init__(
        self,
        project: str,
        config: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        resume: Optional[str] = None,
        tags: Optional[list[str]] = None,
        id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes WandB if on the primary process.

        Args:
            project: The name of the WandB project.
            config: A dictionary of hyperparameters to log.
            group: The name of the WandB group.
            name: The name of the WandB run.
            notes: Notes for the WandB run.
            resume: Whether to resume a previous run (
                "allow", "must", "never", "auto" or None).
            tags: A list of tags for the WandB run.
            id: A unique ID for the WandB run (for resuming).
            **kwargs: Additional arguments passed to wandb.init.
        """
        self.wandb_instance = None
        self.logger = logging.getLogger(__name__)

        if wandb is None:
            self.logger.warning(
                "wandb library not found. WandB logging disabled."
            )
            return

        if dist.is_primary():
            try:
                self.wandb_instance = wandb.init(
                    project=project,
                    config=config,
                    group=group,
                    name=name,
                    notes=notes,
                    resume=resume,
                    tags=tags,
                    id=id,
                    **kwargs,
                )
                self.logger.info(
                    f"WandB initialized for project '{project}'. "
                    f"Run: {self.wandb_instance.name}"
                )
            except Exception as e:
                self.logger.exception(f"Failed to initialize WandB: {e}")
                self.wandb_instance = None
        else:
            self.logger.debug(
                "WandB logging disabled for non-primary process."
            )

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Logs data to WandB if initialized.

        Args:
            data: A dictionary of data to log.
            step: The step number for the log entry.
        """
        if self.wandb_instance and dist.is_primary():
            try:
                self.wandb_instance.log(data, step=step)
            except Exception as e:
                self.logger.exception(f"Failed to log data to WandB: {e}")

    def finish(self) -> None:
        """Finishes the WandB run if initialized."""
        if self.wandb_instance and dist.is_primary():
            try:
                self.wandb_instance.finish()
                self.logger.info("WandB run finished.")
            except Exception as e:
                self.logger.exception(f"Error finishing WandB run: {e}")
            self.wandb_instance = None

    def __del__(self) -> None:
        # Ensure finish is called even if the user forgets
        self.finish()
