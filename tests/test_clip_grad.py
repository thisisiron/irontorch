import unittest
import torch
from irontorch.utils.clip_grad import dispatch_clip_grad


class TestClipGrad(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()

    def test_clip_grad_norm(self):
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        dispatch_clip_grad(self.model.parameters(), value=1.0, mode="norm")
        for param in self.model.parameters():
            self.assertTrue(param.grad.norm() <= 1.0)

    def test_clip_grad_value(self):
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        dispatch_clip_grad(self.model.parameters(), value=0.1, mode="value")
        for param in self.model.parameters():
            self.assertTrue(torch.all(param.grad <= 0.1))

    def test_clip_grad_agc(self):
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        dispatch_clip_grad(self.model.parameters(), value=0.1, mode="agc")
        for param in self.model.parameters():
            self.assertTrue(param.grad.norm() <= 0.1)


if __name__ == "__main__":
    unittest.main()
