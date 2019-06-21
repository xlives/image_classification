import torch

from modules.nn.regularizers.regularizer import Regularizer


class L1Regularizer(Regularizer):
    """Represents a penalty proportional to the sum of the absolute values of the parameters"""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))


class L2Regularizer(Regularizer):
    """Represents a penalty proportional to the sum of squared values of the parameters"""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))
