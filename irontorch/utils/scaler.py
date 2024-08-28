import torch

from .clip_grad import dispatch_clip_grad


class GradScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, mixed_precision=False):
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    def __call__(
            self,
            loss,
            optimizer,
            parameters=None,
            clip_grad=None,
            clip_mode='norm',
            need_update=True,
    ):
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
        return self.grad_scaler.state_dict()

    def load_state_dict(self, state_dict):
        self.grad_scaler.load_state_dict(state_dict)
