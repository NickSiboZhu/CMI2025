# 文件路径: development/utils/scheduler.py

import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


class WarmupAndReduceLROnPlateau:
    """
    A scheduler wrapper that combines a linear warmup with ReduceLROnPlateau.

    This scheduler first warms up the learning rate from a low value to the
    initial learning rate over a specified number of epochs. After the warmup
    period, it transitions to using the ReduceLROnPlateau scheduler to adaptively
    adjust the learning rate based on a monitored metric.
    """
    def __init__(self, optimizer, warmup_epochs, plateau_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.initial_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch = 0

    def step(self, metrics=None):
        """
        Performs a step for both the warmup and the plateau scheduler.
        """
        self._epoch += 1
        
        if self._epoch <= self.warmup_epochs:
            # Manual linear warmup
            lr_scale = self._epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.initial_lrs[i] * lr_scale
        else:
            # After warmup, step the ReduceLROnPlateau scheduler
            if metrics is None:
                raise ValueError("Metrics must be provided for ReduceLROnPlateau after warmup.")
            self.plateau_scheduler.step(metrics)

    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        return {
            'epoch': self._epoch,
            'plateau_scheduler': self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self._epoch = state_dict['epoch']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])



def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)