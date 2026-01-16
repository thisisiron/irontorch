# -*- coding: utf-8 -*-
"""Logging setup for irontorch.

This module provides functions to configure logging based on a YAML file.
It supports console logging (using rich for better formatting) and
rotating file logging (in JSON format).
"""

import logging
import logging.config
import sys
import yaml
from pathlib import Path
from typing import Optional, Union

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
