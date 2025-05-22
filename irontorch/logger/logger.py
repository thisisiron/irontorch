# -*- coding: utf-8 -*-
"""Logging setup for irontorch.

This module provides functions to configure logging based on a YAML file.
It supports console logging (using rich for better formatting) and
rotating file logging (in JSON format).
"""

import logging
import logging.config
import os
import sys
import yaml
from pathlib import Path
from typing import Optional, Union, Dict, Any

from irontorch import distributed as dist

try:
    import rich.logging  # Check if rich is installed
except ImportError:
    # Fallback or raise error if rich is essential
    pass

try:
    import wandb
except ImportError:
    wandb = None

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "logging_config.yaml"


def setup_logging(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    log_file_path: Optional[Union[str, Path]] = None,
    default_level: int = logging.INFO,
) -> None:
    """Setup logging configuration from a YAML file.

    Args:
        config_path: Path to the logging configuration YAML file.
        log_file_path: Optional path to override the log file specified in the config.
        default_level: Default logging level if config file is not found.
    """
    config_path = Path(config_path)
    if config_path.is_file():
        try:
            with open(config_path, "rt") as f:
                config = yaml.safe_load(f.read())

            # Override log file path if provided
            if log_file_path:
                log_file_path = Path(log_file_path)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                if "file" in config.get("handlers", {}):
                    config["handlers"]["file"]["filename"] = str(log_file_path)

            logging.config.dictConfig(config)
            logging.getLogger(__name__).debug(f"Logging configured from {config_path}")
            if log_file_path:
                logging.getLogger(__name__).debug(
                    f"Log file path set to {log_file_path}"
                )

        except Exception as e:
            print(
                f"Error loading logging configuration from {config_path}: {e}",
                file=sys.stderr,
            )
            print("Falling back to basic logging configuration.", file=sys.stderr)
            logging.basicConfig(level=default_level)
    else:
        print(
            f"Warning: Logging configuration file not found at {config_path}",
            file=sys.stderr,
        )
        print("Using basic logging configuration.", file=sys.stderr)
        logging.basicConfig(level=default_level)


# Initialize logging when the module is imported
# You might want to call setup_logging explicitly in your application entry point
# setup_logging()

# --- WandB Integration --- #


class WandbLogger:
    """A wrapper for Weights & Biases logging, active only on the primary process."""

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
        self.logger = logging.getLogger(__name__)  # Use standard logger

        if wandb is None:
            self.logger.warning("wandb library not found. WandB logging disabled.")
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
                    f"WandB initialized for project 	'{project}'. Run: {self.wandb_instance.name}"
                )
            except Exception as e:
                self.logger.exception(f"Failed to initialize WandB: {e}")
                self.wandb_instance = None
        else:
            self.logger.debug("WandB logging disabled for non-primary process.")

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


# Example usage (optional, usually called from application code):
# if __name__ == '__main__':
#     # Example: Call setup_logging at the start of your script
#     setup_logging(log_file_path="/tmp/my_app.log")
#     logger = logging.getLogger("irontorch.example") # Get logger by name
#     logger.info("This is an info message.")
#     logger.warning("This is a warning.")
#     logger.debug("This is a debug message (check file log).")
#     try:
#         1 / 0
#     except ZeroDivisionError:
#         logger.exception("Caught an exception!")

#     # Example WandB usage
#     if dist.is_primary(): # Ensure wandb init is called only once
#         wandb_logger = WandBLogger(project="test-project", name="my-run", config={"lr": 0.01})
#         wandb_logger.log({"loss": 0.5, "accuracy": 95.0}, step=1)
#         wandb_logger.log({"loss": 0.4}, step=2)
#         # wandb_logger.finish() # Called automatically by __del__
