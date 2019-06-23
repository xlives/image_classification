from typing import Dict, Any

from overrides import overrides
import torch

from core.common.checks import ConfigurationError
from core.training.scheduler import Scheduler


class LearningRateScheduler(Scheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    def get_values(self) -> None:
        raise NotImplementedError


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):

    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:  # pylint: disable=protected-access,super-init-not-called
        self.lr_scheduler = lr_scheduler

    def get_values(self):
        return self.lr_scheduler.get_lr()

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        self.lr_scheduler.step(epoch)

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        if metric is None:
            raise ConfigurationError("This learning rate scheduler requires "
                                     "a validation metric to compute the schedule and therefore "
                                     "must be used with a validation dataset.")
        self.lr_scheduler.step(metric, epoch)


