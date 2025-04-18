# -*- coding: utf-8 -*-
"""Recorder module for logging and WandB integration."""

from .recorder import setup_logging, WandbLogger

__all__ = ["setup_logging", "WandbLogger"]
