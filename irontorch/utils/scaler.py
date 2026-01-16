# -*- coding: utf-8 -*-
"""Gradient scaler for mixed precision training."""

import torch

from .clip_grad import dispatch_clip_grad


class GradScaler:
    """Gradient scaler with integrated gradient clipping."""

    state_dict_key = "amp_scaler"

    def __init__(self, mixed_precision=False):
        """Initialize the gradient scaler.

        Args:
            mixed_precision: Whether to enable mixed precision training.
        """
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=mixed_precision
        )

    def __call__(
        self,
        loss,
        optimizer,
        parameters=None,
        clip_grad=None,
        clip_mode="norm",
        need_update=True,
    ):
        """Perform backward pass with optional gradient clipping.

        Args:
            loss: Loss tensor to backpropagate.
            optimizer: Optimizer to update.
            parameters: Model parameters for gradient clipping.
            clip_grad: Gradient clipping value (None to disable).
            clip_mode: Gradient clipping mode ("norm", "value", "agc").
            need_update: Whether to step the optimizer.
        """
        optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self.grad_scaler.unscale_(optimizer)
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

    def state_dict(self):
        """Return the state dict of the gradient scaler."""
        return self.grad_scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load the state dict into the gradient scaler."""
        self.grad_scaler.load_state_dict(state_dict)
