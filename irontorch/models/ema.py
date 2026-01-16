# -*- coding: utf-8 -*-
"""Exponential Moving Average (EMA) for model weights."""

import copy
from typing import Optional

import torch
import torch.nn as nn

from irontorch import distributed as dist


class ModelEMA:
    """Exponential Moving Average of model weights.

    EMA 모델은 원본과 별도로 유지되며, 검증/추론에 사용됩니다.
    학습 중에는 원본 모델을 사용하고, update()를 호출하여
    EMA 가중치를 업데이트합니다.

    Example:
        >>> model = MyModel().cuda()
        >>> ema = ModelEMA(model, decay=0.9999)
        >>>
        >>> for epoch in range(epochs):
        ...     model.train()
        ...     for batch in trainloader:
        ...         loss = model(batch)
        ...         loss.backward()
        ...         optimizer.step()
        ...         ema.update(model)
        ...
        ...     # Validation with EMA model
        ...     ema.module.eval()
        ...     val_loss = validate(ema.module)
        >>>
        >>> # Save EMA model
        >>> torch.save(ema.module.state_dict(), "model_ema.pt")

    References:
        - timm: https://github.com/huggingface/pytorch-image-models
        - ema-pytorch: https://github.com/lucidrains/ema-pytorch
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize ModelEMA.

        Args:
            model: Source model to create EMA from.
            decay: EMA decay rate. Higher values mean slower updates.
                   Typical values: 0.9999 (images), 0.999 (small models).
            device: Device to store EMA model. If None, uses same as model.
        """
        self.decay = decay
        self.device = device
        self.num_updates = 0

        # Create a separate copy of the model
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.module.requires_grad_(False)

        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the source model.

        Should be called after each optimizer step.
        Only updates on the primary process in distributed settings.

        Args:
            model: Source model with updated weights.
        """
        if not dist.is_primary():
            return

        self.num_updates += 1

        # Update parameters
        for ema_p, model_p in zip(
            self.module.parameters(),
            model.parameters()
        ):
            ema_p.data.mul_(self.decay)
            ema_p.data.add_(model_p.data, alpha=1 - self.decay)

        # Update buffers (e.g., BatchNorm running stats)
        for ema_b, model_b in zip(
            self.module.buffers(),
            model.buffers()
        ):
            ema_b.data.copy_(model_b.data)

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing.

        Returns:
            Dictionary containing decay, num_updates, and module state.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "module": self.module.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.decay = state_dict["decay"]
        self.num_updates = state_dict.get("num_updates", 0)
        self.module.load_state_dict(state_dict["module"])
