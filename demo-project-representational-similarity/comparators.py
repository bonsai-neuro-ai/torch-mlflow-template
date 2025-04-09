from enum import Enum, auto

import torch
import abc


class Comparator(abc.ABC):
    @abc.abstractmethod
    def compare(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compare two tensors and return a scalar similarity score."""


class HSICEstimator(Enum):
    GRETTON2006 = auto()
    SONG2007 = auto()
    LANGE2022 = auto()


class LinearCKA(Comparator):
    def __init__(self, hsic_estimator: HSICEstimator = HSICEstimator.SONG2007):
        self.hsic_estimator = hsic_estimator

    def _compute_hsic(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        gram_x = torch.einsum("if,jf->ij", x, x)
        gram_y = torch.einsum("if,jf->ij", y, y)

        if self.hsic_estimator is HSICEstimator.GRETTON2006:
            return torch.sum(gram_x * gram_y) / (m * (m - 1))
        elif self.hsic_estimator is HSICEstimator.SONG2007:
            gram_x = gram_x * (1 - torch.eye(m, device=x.device))
            gram_y = gram_y * (1 - torch.eye(m, device=x.device))
            return (
                torch.sum(gram_x * gram_y)
                + torch.sum(gram_x)
                * torch.sum(gram_y)
                / ((m - 1) * (m - 2) - 2 / (m - 2) * torch.sum(gram_x @ gram_y))
            ) / (m * (m - 3))
        elif self.hsic_estimator is HSICEstimator.LANGE2022:
            i, j = torch.triu_indices(m, m, 1)
            return torch.sum(gram_x[i, j] * gram_y[i, j]) * 2 / (m * (m - 3))
        else:
            raise ValueError("Unknown HSIC estimator")

    def compare(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compare two tensors using Linear CKA."""
        x, y = torch.flatten(x, start_dim=1), torch.flatten(y, start_dim=1)

        hsic_xx = self._compute_hsic(x, x)
        hsic_yy = self._compute_hsic(y, y)
        hsic_xy = self._compute_hsic(x, y)

        return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
