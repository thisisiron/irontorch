import pytest
import torch
from irontorch.utils.scaler import GradScaler


@pytest.fixture
def setup_scaler():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scaler = GradScaler(mixed_precision=True)
    
    return model, optimizer, criterion, scaler


def test_grad_scaler_step(setup_scaler):
    model, optimizer, criterion, scaler = setup_scaler
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    scaler(
        loss, optimizer, parameters=model.parameters(), clip_grad=1.0
    )
    
    # Verify gradients exist after scaling step
    for param in model.parameters():
        assert param.grad is not None


def test_grad_scaler_state_dict(setup_scaler):
    _, _, _, scaler = setup_scaler
    
    state_dict = scaler.state_dict()
    assert "scale" in state_dict
    assert "_growth_tracker" in state_dict
    assert "growth_factor" in state_dict
    assert "backoff_factor" in state_dict


def test_grad_scaler_load_state_dict(setup_scaler):
    _, _, _, scaler = setup_scaler
    
    state_dict = scaler.state_dict()
    new_scaler = GradScaler(mixed_precision=True)
    new_scaler.load_state_dict(state_dict)
    
    # Compare state dictionaries
    original_state = scaler.state_dict()
    loaded_state = new_scaler.state_dict()
    
    assert original_state["scale"] == loaded_state["scale"]
    assert original_state["_growth_tracker"] == loaded_state["_growth_tracker"]
    assert original_state["growth_factor"] == loaded_state["growth_factor"]
    assert original_state["backoff_factor"] == loaded_state["backoff_factor"]


def test_grad_scaler_with_no_mixed_precision():
    # Test with mixed_precision=False
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scaler = GradScaler(mixed_precision=False)
    
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Should work without errors when mixed_precision is False
    scaler(loss, optimizer, parameters=model.parameters(), clip_grad=1.0)
    
    # Verify gradients exist
    for param in model.parameters():
        assert param.grad is not None
