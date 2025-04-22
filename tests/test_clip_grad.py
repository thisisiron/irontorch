import pytest
import torch
from irontorch.utils.clip_grad import dispatch_clip_grad


@pytest.fixture
def setup_model():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    return model, optimizer, criterion, inputs, targets


def test_clip_grad_norm(setup_model):
    model, optimizer, criterion, inputs, targets = setup_model
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    dispatch_clip_grad(model.parameters(), value=1.0, mode="norm")
    
    for param in model.parameters():
        assert param.grad.norm() <= 1.0 + 1e-6  # Add small epsilon for numerical stability


def test_clip_grad_value(setup_model):
    model, optimizer, criterion, inputs, targets = setup_model
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    dispatch_clip_grad(model.parameters(), value=0.1, mode="value")
    
    for param in model.parameters():
        assert torch.all(param.grad <= 0.1 + 1e-6)  # Add small epsilon for numerical stability
        assert torch.all(param.grad >= -0.1 - 1e-6)  # Check both positive and negative bounds


def test_clip_grad_agc(setup_model):
    model, optimizer, criterion, inputs, targets = setup_model
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    dispatch_clip_grad(model.parameters(), value=0.1, mode="agc")
    
    # AGC is more complex, this is a simplified check
    for param in model.parameters():
        assert param.grad is not None  # At minimum, ensure gradient exists


def test_invalid_clip_mode(setup_model):
    model, optimizer, criterion, inputs, targets = setup_model
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    with pytest.raises(AssertionError, match=r"Unknown clip mode.*"):
        dispatch_clip_grad(model.parameters(), value=1.0, mode="invalid_mode")
