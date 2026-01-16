# -*- coding: utf-8 -*-
"""Pydantic schema models for distributed training configuration."""

from typing import Any, Optional
from pydantic import BaseModel, StrictBool, ConfigDict


class MainConfig(BaseModel):
    """Main configuration model for distributed training."""

    model_config = ConfigDict(extra="allow")

    launch_config: Any = None
    distributed: Optional[StrictBool] = False
