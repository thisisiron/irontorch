# -*- coding: utf-8 -*-
"""Tests for ModelEMA."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from irontorch.models import ModelEMA


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.bn = nn.BatchNorm1d(5)

    def forward(self, x):
        return self.bn(self.linear(x))


class TestModelEMA:
    """Test suite for ModelEMA."""

    def test_init_creates_copy(self):
        """Test that ModelEMA creates a separate copy of the model."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9999)

        # Check that ema.module is a different object
        assert ema.module is not model

        # Check that weights are initially equal
        for ema_p, model_p in zip(ema.module.parameters(), model.parameters()):
            assert torch.equal(ema_p, model_p)

    def test_init_sets_eval_mode(self):
        """Test that EMA model is set to eval mode."""
        model = SimpleModel()
        model.train()

        ema = ModelEMA(model, decay=0.9999)

        assert not ema.module.training

    def test_init_disables_gradients(self):
        """Test that EMA model has gradients disabled."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9999)

        for param in ema.module.parameters():
            assert not param.requires_grad

    @patch("irontorch.models.ema.dist.is_primary", return_value=True)
    def test_update_changes_weights(self, mock_is_primary):
        """Test that update() changes EMA weights."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9)

        # Store initial EMA weights
        initial_weights = [p.clone() for p in ema.module.parameters()]

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param))

        # Update EMA
        ema.update(model)

        # Check that EMA weights changed
        for ema_p, initial_p in zip(ema.module.parameters(), initial_weights):
            assert not torch.equal(ema_p, initial_p)

    @patch("irontorch.models.ema.dist.is_primary", return_value=True)
    def test_update_applies_decay(self, mock_is_primary):
        """Test that update() applies correct decay formula."""
        model = SimpleModel()
        decay = 0.9
        ema = ModelEMA(model, decay=decay)

        # Store initial EMA weights
        initial_ema = [p.clone() for p in ema.module.parameters()]

        # Set model weights to known values
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        # Update EMA
        ema.update(model)

        # Check decay formula: ema = decay * ema + (1 - decay) * model
        for ema_p, initial_p in zip(ema.module.parameters(), initial_ema):
            expected = decay * initial_p + (1 - decay) * torch.ones_like(initial_p)
            assert torch.allclose(ema_p, expected)

    @patch("irontorch.models.ema.dist.is_primary", return_value=False)
    def test_update_skipped_non_primary(self, mock_is_primary):
        """Test that update() is skipped on non-primary processes."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9)

        # Store initial EMA weights
        initial_weights = [p.clone() for p in ema.module.parameters()]

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param))

        # Update EMA (should be skipped)
        ema.update(model)

        # Check that EMA weights did not change
        for ema_p, initial_p in zip(ema.module.parameters(), initial_weights):
            assert torch.equal(ema_p, initial_p)

    @patch("irontorch.models.ema.dist.is_primary", return_value=True)
    def test_update_copies_buffers(self, mock_is_primary):
        """Test that update() copies buffers (e.g., BatchNorm stats)."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9)

        # Run forward pass to update BatchNorm stats
        x = torch.randn(32, 10)
        model.train()
        model(x)

        # Update EMA
        ema.update(model)

        # Check that buffers are copied
        for ema_b, model_b in zip(ema.module.buffers(), model.buffers()):
            assert torch.equal(ema_b, model_b)

    def test_state_dict(self):
        """Test state_dict() returns correct structure."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9999)

        state = ema.state_dict()

        assert "decay" in state
        assert "num_updates" in state
        assert "module" in state
        assert state["decay"] == 0.9999
        assert state["num_updates"] == 0

    def test_load_state_dict(self):
        """Test load_state_dict() restores state correctly."""
        model = SimpleModel()
        ema1 = ModelEMA(model, decay=0.9999)

        # Modify EMA state
        with torch.no_grad():
            for param in ema1.module.parameters():
                param.fill_(0.5)
        ema1.num_updates = 100

        # Save state
        state = ema1.state_dict()

        # Create new EMA and load state
        ema2 = ModelEMA(model, decay=0.5)
        ema2.load_state_dict(state)

        # Check state restored
        assert ema2.decay == 0.9999
        assert ema2.num_updates == 100
        for p1, p2 in zip(ema1.module.parameters(), ema2.module.parameters()):
            assert torch.equal(p1, p2)

    @patch("irontorch.models.ema.dist.is_primary", return_value=True)
    def test_num_updates_increments(self, mock_is_primary):
        """Test that num_updates increments on each update."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9999)

        assert ema.num_updates == 0

        ema.update(model)
        assert ema.num_updates == 1

        ema.update(model)
        assert ema.num_updates == 2

    def test_device_parameter(self):
        """Test that device parameter moves EMA model."""
        model = SimpleModel()

        # Test with CPU (always available)
        ema = ModelEMA(model, decay=0.9999, device=torch.device("cpu"))

        for param in ema.module.parameters():
            assert param.device == torch.device("cpu")
