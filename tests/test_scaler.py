import unittest
import torch
from irontorch.utils.scaler import GradScaler


class TestGradScaler(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.scaler = GradScaler(mixed_precision=True)

    def test_grad_scaler_step(self):
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.scaler(loss, self.optimizer, parameters=self.model.parameters(), clip_grad=1.0)
        self.assertTrue(self.optimizer.param_groups[0]['params'][0].grad is not None)

    def test_grad_scaler_state_dict(self):
        state_dict = self.scaler.state_dict()
        self.assertIn('scale', state_dict)
        self.assertIn('growth_tracker', state_dict)

    def test_grad_scaler_load_state_dict(self):
        state_dict = self.scaler.state_dict()
        new_scaler = GradScaler(mixed_precision=True)
        new_scaler.load_state_dict(state_dict)
        self.assertEqual(self.scaler.state_dict(), new_scaler.state_dict())


if __name__ == '__main__':
    unittest.main()
