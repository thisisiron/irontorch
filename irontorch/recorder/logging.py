# -*- coding: utf-8 -*-
"""Logging setup for irontorch.

This module provides functions to configure logging based on a YAML file.
It supports console logging (using rich for better formatting) and
rotating file logging (in JSON format).

Also provides DistributedLogger for distributed training environments.
"""

import logging
import logging.config
import sys
import yaml
from pathlib import Path
from typing import Optional, Union

from irontorch import distributed as dist

try:
    import rich.logging  # noqa: F401  # Check if rich is installed
except ImportError:
    pass

DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "logging_config.yaml"
)


def setup_logging(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    log_file_path: Optional[Union[str, Path]] = None,
    default_level: int = logging.INFO,
) -> None:
    """Setup logging configuration from a YAML file.

    Args:
        config_path: Path to the logging configuration YAML file.
        log_file_path: Optional path to override the log file
            specified in the config.
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
            logging.getLogger(__name__).debug(
                f"Logging configured from {config_path}"
            )
            if log_file_path:
                logging.getLogger(__name__).debug(
                    f"Log file path set to {log_file_path}"
                )

        except Exception as e:
            print(
                f"Error loading logging configuration from {config_path}: {e}",
                file=sys.stderr,
            )
            print(
                "Falling back to basic logging configuration.",
                file=sys.stderr
            )
            logging.basicConfig(level=default_level)
    else:
        print(
            f"Warning: Logging configuration file not found at {config_path}",
            file=sys.stderr,
        )
        print("Using basic logging configuration.", file=sys.stderr)
        logging.basicConfig(level=default_level)


class DistributedLogger:
    """Wrapper for logging.Logger that only logs on primary process.

    In distributed training, this prevents duplicate log messages from
    multiple processes. Use _all suffix methods (e.g., info_all) to
    log from all processes.

    Example:
        >>> import logging
        >>> from irontorch.recorder import make_distributed
        >>> logger = make_distributed(logging.getLogger(__name__))
        >>> logger.info("Only on rank 0")
        >>> logger.info_all("On all ranks")
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize DistributedLogger.

        Args:
            logger: The base logger to wrap.
        """
        self._logger = logger

    # Primary-only methods
    def debug(self, msg, *args, **kwargs) -> None:
        """Log debug message on primary process only."""
        if dist.is_primary():
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs) -> None:
        """Log info message on primary process only."""
        if dist.is_primary():
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs) -> None:
        """Log warning message on primary process only."""
        if dist.is_primary():
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs) -> None:
        """Log error message on primary process only."""
        if dist.is_primary():
            self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs) -> None:
        """Log critical message on primary process only."""
        if dist.is_primary():
            self._logger.critical(msg, *args, **kwargs)

    # All-ranks methods
    def debug_all(self, msg, *args, **kwargs) -> None:
        """Log debug message on all processes."""
        self._logger.debug(msg, *args, **kwargs)

    def info_all(self, msg, *args, **kwargs) -> None:
        """Log info message on all processes."""
        self._logger.info(msg, *args, **kwargs)

    def warning_all(self, msg, *args, **kwargs) -> None:
        """Log warning message on all processes."""
        self._logger.warning(msg, *args, **kwargs)

    def error_all(self, msg, *args, **kwargs) -> None:
        """Log error message on all processes."""
        self._logger.error(msg, *args, **kwargs)

    def critical_all(self, msg, *args, **kwargs) -> None:
        """Log critical message on all processes."""
        self._logger.critical(msg, *args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped logger."""
        return getattr(self._logger, name)


def make_distributed(logger: logging.Logger) -> DistributedLogger:
    """Wrap a logger to only log on primary process in distributed training.

    Args:
        logger: The base logger to wrap.

    Returns:
        A DistributedLogger that wraps the given logger.

    Example:
        >>> import logging
        >>> from irontorch.recorder import make_distributed
        >>> logger = make_distributed(logging.getLogger(__name__))
        >>> logger.info("This only prints on rank 0")
    """
    return DistributedLogger(logger)
