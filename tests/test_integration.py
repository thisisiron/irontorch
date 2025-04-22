"""Integration tests for irontorch modules."""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch

from irontorch.utils.helper import set_seed
from irontorch.utils.scaler import GradScaler
from irontorch.utils.clip_grad import dispatch_clip_grad


# Define a simple model, dataset, and training loop for integration tests
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def training_setup():
    """Setup a simple training environment."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Create model, optimizer, loss function
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Create dummy data
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    return model, optimizer, criterion, inputs, targets


def test_training_with_grad_scaler_and_clipping(training_setup):
    """Test integration of GradScaler with gradient clipping."""
    model, optimizer, criterion, inputs, targets = training_setup
    
    # Create grad scaler
    scaler = GradScaler(mixed_precision=True)
    
    # Training step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Use scaler to handle loss and optimize with gradient clipping
    scaler(loss, optimizer, parameters=model.parameters(), clip_grad=1.0)
    
    # Check that gradients were computed and clipped
    for param in model.parameters():
        assert param.grad is not None
        # In mixed precision mode with clipping, gradients should be scaled and clipped
        assert param.grad.norm() <= 1.0 * scaler.state_dict()["scale"] + 1e-5


def test_reproducibility_with_set_seed():
    """Test that set_seed ensures reproducibility across runs."""
    # First run
    set_seed(123)
    model1 = SimpleModel()
    rand1 = torch.rand(5, 10)
    
    # Second run with same seed
    set_seed(123)
    model2 = SimpleModel()
    rand2 = torch.rand(5, 10)
    
    # Check that model weights are identical
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    
    # Check that random tensors are identical
    assert torch.equal(rand1, rand2)


@patch('torch.nn.utils.clip_grad_norm_')
@patch('torch.nn.utils.clip_grad_value_')
def test_dispatch_clip_grad_integration(mock_clip_grad_value, mock_clip_grad_norm, training_setup):
    """Test that dispatch_clip_grad correctly delegates to appropriate PyTorch functions."""
    model, optimizer, criterion, inputs, targets = training_setup
    
    # Forward and backward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Test norm clipping
    dispatch_clip_grad(model.parameters(), value=1.0, mode="norm")
    mock_clip_grad_norm.assert_called_once()
    
    # Reset mocks
    mock_clip_grad_norm.reset_mock()
    mock_clip_grad_value.reset_mock()
    
    # Test value clipping
    dispatch_clip_grad(model.parameters(), value=0.1, mode="value")
    mock_clip_grad_value.assert_called_once()


def test_end_to_end_training_loop():
    """Test a simple end-to-end training loop with multiple components."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Create model, optimizer, loss function
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    scaler = GradScaler(mixed_precision=False)  # no mixed precision for simplicity
    
    # Create dummy data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # Store initial loss
    initial_outputs = model(inputs)
    initial_loss = criterion(initial_outputs, targets).item()
    
    # Run a few training steps
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        scaler(loss, optimizer, parameters=model.parameters(), clip_grad=1.0)
    
    # Compute final loss
    final_outputs = model(inputs)
    final_loss = criterion(final_outputs, targets).item()
    
    # Loss should decrease after training
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}" 